//! hf_download
//!
//! A small, pure Rust library to download Hugging Face repositories (models, datasets, spaces)
//! over HTTP without requiring Git. Works with public and private repos, supports resuming,
//! and performs parallel ranged downloads for speed.
//!
//! - Parallel, ranged downloads
//! - Async and blocking APIs
//! - Works with private repos (set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`)
//! - Include/exclude file globs
//!
//! # Quick start (blocking)
//!
//! ```rust,no_run
//! use hf_download::{DownloadConfig, HfDownloader, RepoType, ProgressEvent};
//! use std::path::Path;
//!
//! # fn main() -> anyhow::Result<()> {
//! let cfg = DownloadConfig::default();
//! let dl = HfDownloader::new(cfg)?;
//! let summary = dl.blocking_download_repo(
//!     "hf-internal-testing/tiny-random-bert",
//!     RepoType::Model,
//!     "main",
//!     Path::new("./downloads/tiny-bert"),
//!     |evt| match evt {
//!         ProgressEvent::RepoDiscovered { num_files, total_bytes } => println!("{num_files} files / {total_bytes} bytes"),
//!         ProgressEvent::FileStarted { path, size } => println!("start {path} ({size} bytes)"),
//!         ProgressEvent::BytesTransferred { path, bytes } => println!("{path} +{bytes} bytes"),
//!         ProgressEvent::FileCompleted { path } => println!("done {path}"),
//!         ProgressEvent::FileFailed { path, error } => eprintln!("fail {path}: {error}"),
//!     }
//! )?;
//! println!(
//!     "Downloaded {} files ({} bytes)",
//!     summary.files_downloaded,
//!     summary.bytes_downloaded
//! );
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! - Endpoint: `HF_ENDPOINT` (default: `https://huggingface.co`)
//! - Auth: `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
//! - Concurrency/chunk sizing and include/exclude globs via [`DownloadConfig`]
//!
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use globset::{Glob, GlobSet, GlobSetBuilder};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, RANGE, USER_AGENT};
use reqwest::{Client, Url};
use serde::Deserialize;
use tokio::fs::{create_dir_all, rename, OpenOptions};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};
use rand::{thread_rng, Rng};

const DEFAULT_CHUNK_SIZE: usize = 8 * 1024 * 1024; // 8 MiB
const DEFAULT_MAX_CONCURRENT: usize = 8;
const DEFAULT_MAX_CONCURRENT_CHUNKS: usize = 8;
const BASE_WAIT_MS: usize = 300;
const MAX_WAIT_MS: usize = 10_000;

/// Type of Hugging Face repository to download.
#[derive(Clone, Debug)]
pub enum RepoType {
    /// Model repositories (e.g. `bert-base-uncased`).
    Model,
    /// Dataset repositories.
    Dataset,
    /// Space repositories.
    Space,
}

impl RepoType {
    fn as_str(&self) -> &'static str {
        match self {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        }
    }
}

/// Configuration for downloads.
#[derive(Clone, Debug)]
pub struct DownloadConfig {
    /// Maximum number of files downloaded in parallel.
    pub max_concurrent_files: usize,
    /// Chunk size used for ranged requests per task in bytes.
    pub chunk_size_bytes: usize,
    /// Maximum number of retries per operation.
    pub max_retries: usize,
    /// Maximum number of concurrently throttled failures (limits backoffs in flight).
    pub parallel_failures: usize,
    /// Maximum number of concurrent chunks per file.
    pub max_concurrent_chunks: usize,
    /// Whether to resume partially downloaded files (`*.part`).
    pub resume: bool,
    /// Optional list of glob patterns to include.
    pub include: Option<Vec<String>>,
    /// Optional list of glob patterns to exclude.
    pub exclude: Option<Vec<String>>,
    /// Base URL of the Hugging Face Hub API (e.g. `https://huggingface.co`).
    pub api_base: String,
    /// Bearer token for private repositories. Can also be provided via `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.
    pub auth_token: Option<String>,
    /// Optional custom User-Agent header.
    pub user_agent: Option<String>,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        let api_base = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
        let auth_token = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
        Self {
            max_concurrent_files: DEFAULT_MAX_CONCURRENT,
            chunk_size_bytes: DEFAULT_CHUNK_SIZE,
            max_retries: 5,
            parallel_failures: 2,
            max_concurrent_chunks: DEFAULT_MAX_CONCURRENT_CHUNKS,
            resume: true,
            include: None,
            exclude: None,
            api_base,
            auth_token,
            user_agent: None,
        }
    }
}

/// Progress events emitted during repository or file downloads.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Repository contents discovered (after listing the tree).
    RepoDiscovered { num_files: usize, total_bytes: u64 },
    /// A file is about to be downloaded.
    FileStarted { path: String, size: u64 },
    /// Bytes for a file were transferred (per chunk completion).
    BytesTransferred { path: String, bytes: usize },
    /// A file finished successfully.
    FileCompleted { path: String },
    /// A file failed with an error.
    FileFailed { path: String, error: String },
}

/// Summary returned after a repository download.
#[derive(Debug, Default, Clone)]
pub struct RepoSummary {
    /// Number of files successfully downloaded.
    pub files_downloaded: usize,
    /// Total bytes reported as downloaded.
    /// Note: depending on the specific progress accounting, this may be zero in some versions.
    pub bytes_downloaded: u64,
}

/// Downloader that talks to the Hugging Face Hub and performs parallel downloads.
#[derive(Clone)]
pub struct HfDownloader {
    client: Client,
    config: DownloadConfig,
}

impl HfDownloader {
    /// Create a new downloader using the provided configuration.
    pub fn new(config: DownloadConfig) -> Result<Self> {
        let mut default_headers = HeaderMap::new();
        let ua = config.user_agent.clone().unwrap_or_else(|| format!("hf-download/{}", env!("CARGO_PKG_VERSION")));
        default_headers.insert(USER_AGENT, HeaderValue::from_str(&ua).unwrap());
        let client = Client::builder()
            .default_headers(default_headers)
            .http2_keep_alive_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self { client, config })
    }

    /// Download an entire repository asynchronously.
    ///
    /// - `repo_id`: e.g. `bert-base-uncased` or `username/dataset`
    /// - `repo_type`: [`RepoType`] of the repository
    /// - `revision`: branch, tag, or commit hash (e.g. `main`)
    /// - `dest_dir`: destination directory for downloaded files
    /// - `progress`: callback to receive [`ProgressEvent`]s
    pub async fn download_repo<F: Fn(ProgressEvent) + Send + Sync + 'static + Clone>(
        &self,
        repo_id: &str,
        repo_type: RepoType,
        revision: &str,
        dest_dir: &Path,
        progress: F,
    ) -> Result<RepoSummary> {
        let (files, total_bytes) = self
            .retrying(|| self.list_repo_files(repo_id, &repo_type, revision))
            .await
            .context("list repo files")?;

        let (globset_include, globset_exclude) = build_globsets(&self.config.include, &self.config.exclude)?;
        // Optional prefix-based filtering first for speed
        let include_prefixes: Option<&[String]> = None; // reserved for future config
        let exclude_prefixes: Option<&[String]> = None; // reserved for future config
        let filtered: Vec<_> = files
            .into_iter()
            .filter(|f| {
                if let Some(prefixes) = include_prefixes { if !prefixes.iter().any(|p| f.path.starts_with(p)) { return false; } }
                if let Some(prefixes) = exclude_prefixes { if prefixes.iter().any(|p| f.path.starts_with(p)) { return false; } }
                if let Some(ref gs) = globset_include {
                    if !gs.is_match(&f.path) {
                        return false;
                    }
                }
                if let Some(ref gs) = globset_exclude {
                    if gs.is_match(&f.path) {
                        return false;
                    }
                }
                true
            })
            .collect();

        progress(ProgressEvent::RepoDiscovered {
            num_files: filtered.len(),
            total_bytes,
        });

        create_dir_all(dest_dir).await?;

        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_files));
        let mut tasks = FuturesUnordered::new();
        let user_progress = Arc::new(progress.clone());
        let total_bytes = Arc::new(AtomicU64::new(0));

        for file in filtered {
            let permit = semaphore.clone().acquire_owned().await?;
            let user_progress = user_progress.clone();
            let tot = total_bytes.clone();
            let this = self.clone();
            let repo_id = repo_id.to_string();
            let revision = revision.to_string();
            let repo_type = match repo_type { RepoType::Model => RepoType::Model, RepoType::Dataset => RepoType::Dataset, RepoType::Space => RepoType::Space };
            let dest_path = dest_dir.join(&file.path);
            tasks.push(tokio::spawn(async move {
                let _p = permit; // keep permit until task ends
                if let Some(parent) = dest_path.parent() { create_dir_all(parent).await.ok(); }
                let res = this.download_file(&repo_id, repo_type, &revision, &file.path, &dest_path, move |evt| {
                    if let ProgressEvent::BytesTransferred { bytes, .. } = &evt { tot.fetch_add(*bytes as u64, Ordering::Relaxed); }
                    user_progress(evt)
                }).await;
                (file.path, res)
            }));
        }

        let mut summary = RepoSummary::default();
        while let Some(res) = tasks.next().await {
            match res {
                Ok((path, Ok(()))) => {
                    summary.files_downloaded += 1;
                    user_progress(ProgressEvent::FileCompleted { path });
                }
                Ok((path, Err(err))) => {
                    user_progress(ProgressEvent::FileFailed { path, error: err.to_string() });
                }
                Err(join_err) => {
                    // spawn failure
                    user_progress(ProgressEvent::FileFailed { path: "<task>".into(), error: join_err.to_string() });
                }
            }
        }

        summary.bytes_downloaded = total_bytes.load(Ordering::Relaxed);
        Ok(summary)
    }

    /// Download a single file from a repository asynchronously.
    pub async fn download_file<F: Fn(ProgressEvent) + Send + Sync + 'static>(
        &self,
        repo_id: &str,
        repo_type: RepoType,
        revision: &str,
        remote_path: &str,
        dest_path: &Path,
        progress: F,
    ) -> Result<()> {
        // Resolve URL on hub
        let resolved = self.resolve_file_url(repo_id, &repo_type, revision, remote_path)?;

        // Prepare headers
        let mut headers = HeaderMap::new();
        if let Some(token) = self.config.auth_token.as_ref() {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", token)).unwrap(),
            );
        }

        // Probe to get redirected URL and size (with retries)
        let resp = self
            .retrying(|| async {
                let r = self
                    .client
                    .get(resolved.clone())
                    .headers(headers.clone())
                    .header(RANGE, "bytes=0-0")
                    .send()
                    .await?;
                let r = r.error_for_status()?;
                Ok::<_, anyhow::Error>(r)
            })
            .await?;

        let final_url = resp.url().clone();
        // If host changed, strip authorization to avoid leaking token to CDN
        let mut effective_headers = headers.clone();
        if resolved.host_str() != final_url.host_str() {
            effective_headers.remove(AUTHORIZATION);
        }

        // Parse size; if server didn't support range probing, fall back to Content-Length
        let status = resp.status();
        let (size, range_supported) = if let Some(cr) = resp.headers().get(CONTENT_RANGE) {
            let content_range = cr.to_str()?;
            let size: usize = content_range
                .split('/')
                .last()
                .ok_or_else(|| anyhow!("Invalid content-range"))?
                .parse()?;
            (size, true)
        } else if let Some(cl) = resp.headers().get(CONTENT_LENGTH) {
            let size: usize = cl.to_str()?.parse()?;
            (size, status.as_u16() == 206)
        } else {
            // Unknown size; we will stream to end without pre-allocation
            (0, false)
        };

        let tmp_path = dest_path.with_extension("part");
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&tmp_path)
            .await?;

        let mut existing = if self.config.resume {
            match tokio::fs::metadata(&tmp_path).await {
                Ok(meta) => meta.len() as usize,
                Err(_) => 0,
            }
        } else {
            0
        };

        let chunk_size = self.config.chunk_size_bytes;
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_chunks));
        let failures_guard = Arc::new(Semaphore::new(self.config.parallel_failures.max(1)));
        let mut tasks = FuturesUnordered::new();
        let progress = Arc::new(progress);

        if range_supported && size > 0 {
            // Resume completion checks
            if self.config.resume {
                if existing == size {
                    // already fully downloaded; rename into place
                    drop(file);
                    rename(&tmp_path, dest_path).await?;
                    progress(ProgressEvent::FileCompleted { path: remote_path.to_string() });
                    return Ok(());
                } else if existing > size {
                    // truncate corrupted .part file
                    file.set_len(size as u64).await?;
                    existing = 0;
                }
            }
            // Pre-size before spawning tasks to avoid races
            file.set_len(size as u64).await?;
            progress(ProgressEvent::FileStarted { path: remote_path.to_string(), size: size as u64 });

            for start in (0..size).step_by(chunk_size) {
                if existing > start { continue; }
                let stop = std::cmp::min(start + chunk_size - 1, size - 1);
                let client = self.client.clone();
                let url = final_url.clone();
                let headers = effective_headers.clone();
                let path = tmp_path.clone();
                let progress_cb = progress.clone();
                let failures_guard = failures_guard.clone();
                let remote_path_string = remote_path.to_string();
                let retries = self.config.max_retries;
                let permit_f = semaphore.clone().acquire_owned().await?;
                tasks.push(tokio::spawn(async move {
                    let _pf = permit_f;
                    let mut attempt = 0usize;
                    loop {
                        let range = format!("bytes={start}-{stop}");
                        match download_chunk(&client, url.clone(), headers.clone(), &path, start as u64, &range).await {
                            Ok(bytes) => {
                                progress_cb(ProgressEvent::BytesTransferred { path: remote_path_string.clone(), bytes });
                                return Ok::<(), anyhow::Error>(());
                            }
                            Err(e) => {
                                if attempt >= retries {
                                    return Err(anyhow!(e));
                                }
                                if let Ok(guard) = failures_guard.clone().try_acquire_owned() {
                                    let wait = jittered_backoff_ms(attempt);
                                    sleep(Duration::from_millis(wait as u64)).await;
                                    drop(guard);
                                }
                                attempt += 1;
                                continue;
                            }
                        }
                    }
                }));
            }
        } else {
            // Fallback: no range support -> stream single request to file
            progress(ProgressEvent::FileStarted { path: remote_path.to_string(), size: size as u64 });
            let mut attempt = 0usize;
            let retries = self.config.max_retries;
            loop {
                match self.client.get(final_url.clone()).headers(effective_headers.clone()).send().await {
                    Ok(resp) => {
                        if let Err(e) = resp.error_for_status_ref() { if attempt >= retries { return Err(anyhow!(e)); } let wait = jittered_backoff_ms(attempt); sleep(Duration::from_millis(wait as u64)).await; attempt += 1; continue; }
                        // If Content-Length available, pre-size
                        if let Some(cl) = resp.headers().get(CONTENT_LENGTH) {
                            if let Some(len) = cl.to_str().ok().and_then(|s| s.parse::<u64>().ok()) {
                                file.set_len(len).await.ok();
                            }
                        }
                        let mut stream = resp.bytes_stream();
                        while let Some(chunk) = stream.next().await { let chunk = chunk?; file.write_all(&chunk).await?; progress(ProgressEvent::BytesTransferred { path: remote_path.to_string(), bytes: chunk.len() }); }
                        break;
                    }
                    Err(e) => {
                        if attempt >= retries { return Err(anyhow!(e)); }
                        let wait = jittered_backoff_ms(attempt);
                        sleep(Duration::from_millis(wait as u64)).await;
                        attempt += 1;
                        continue;
                    }
                }
            }
        }

        drop(file);

        // Join all chunk tasks (if any)
        while let Some(res) = tasks.next().await { res??; }

        // Atomically move into place
        rename(&tmp_path, dest_path).await?;
        Ok(())
    }

    /// Build the Hub URL for a file's resolve endpoint.
    fn resolve_file_url(&self, repo_id: &str, repo_type: &RepoType, revision: &str, path: &str) -> Result<Url> {
        let base = self.config.api_base.trim_end_matches('/');
        // For models, resolve endpoint does not include "/models"; for datasets and spaces it does include the kind prefix.
        let url = match repo_type {
            RepoType::Model => format!(
                "{base}/{id}/resolve/{rev}/{path}",
                base = base,
                id = repo_id,
                rev = revision,
                path = path,
            ),
            RepoType::Dataset | RepoType::Space => format!(
                "{base}/{kind}/{id}/resolve/{rev}/{path}",
                base = base,
                kind = repo_type.as_str(),
                id = repo_id,
                rev = revision,
                path = path,
            ),
        };
        Ok(Url::parse(&url)?)
    }

    /// List files for a repository revision via the Hub API.
    async fn list_repo_files(&self, repo_id: &str, repo_type: &RepoType, revision: &str) -> Result<(Vec<RepoFile>, u64)> {
        let base = self.config.api_base.trim_end_matches('/');
        let url = format!(
            "{base}/api/{kind}/{id}/tree/{rev}?recursive=1",
            base = base,
            kind = repo_type.as_str(),
            id = repo_id,
            rev = revision,
        );
        let mut req = self.client.get(url);
        if let Some(token) = self.config.auth_token.as_ref() {
            req = req.header(AUTHORIZATION, format!("Bearer {}", token));
        }
        let resp = self.retrying(|| async {
            let r = req.try_clone().expect("clone request").send().await?;
            let r = r.error_for_status()?;
            Ok::<_, anyhow::Error>(r)
        }).await?;
        let entries: Vec<TreeEntry> = resp.json().await?;
        let mut files = Vec::new();
        let mut total: u64 = 0;
        for e in entries.into_iter() {
            if e.r#type == Some("file".into()) || e.r#type.is_none() {
                if let Some(s) = e.size { total += s as u64; }
                files.push(RepoFile { path: e.path });
            }
        }
        Ok((files, total))
    }
}

#[derive(Debug, Clone)]
struct RepoFile {
    path: String,
}

#[derive(Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(default)]
    size: Option<usize>,
    #[serde(rename = "type")]
    #[serde(default)]
    r#type: Option<String>,
}

fn jittered_backoff_ms(n: usize) -> usize {
    let jitter: usize = thread_rng().gen_range(0..=500);
    (BASE_WAIT_MS + n * n + jitter).min(MAX_WAIT_MS)
}

async fn download_chunk(
    client: &Client,
    url: Url,
    headers: HeaderMap,
    dest_path: &Path,
    offset: u64,
    range: &str,
) -> Result<usize> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(dest_path)
        .await?;
    file.seek(std::io::SeekFrom::Start(offset)).await?;
    let resp = client
        .get(url)
        .headers(headers)
        .header(RANGE, range)
        .send()
        .await?
        .error_for_status()?;
    let bytes = resp.bytes().await?;
    file.write_all(&bytes).await?;
    Ok(bytes.len())
}

fn build_globsets(include: &Option<Vec<String>>, exclude: &Option<Vec<String>>) -> Result<(Option<GlobSet>, Option<GlobSet>)> {
    let to_set = |patterns: &Option<Vec<String>>| -> Result<Option<GlobSet>> {
        if let Some(ref pats) = patterns {
            if pats.is_empty() { return Ok(None); }
            let mut builder = GlobSetBuilder::new();
            for p in pats {
                builder.add(Glob::new(p)?);
            }
            Ok(Some(builder.build()?))
        } else { Ok(None) }
    };
    Ok((to_set(include)?, to_set(exclude)?))
}

// Blocking helpers for convenience
impl HfDownloader {
    /// Blocking convenience wrapper for [`HfDownloader::download_repo`].
    pub fn blocking_download_repo<F: Fn(ProgressEvent) + Send + Sync + 'static + Clone>(
        &self,
        repo_id: &str,
        repo_type: RepoType,
        revision: &str,
        dest_dir: &Path,
        progress: F,
    ) -> Result<RepoSummary> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("build runtime")?
            .block_on(self.download_repo(repo_id, repo_type, revision, dest_dir, progress))
    }

    /// Blocking convenience wrapper for [`HfDownloader::download_file`].
    pub fn blocking_download_file<F: Fn(ProgressEvent) + Send + Sync + 'static>(
        &self,
        repo_id: &str,
        repo_type: RepoType,
        revision: &str,
        remote_path: &str,
        dest_path: &Path,
        progress: F,
    ) -> Result<()> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("build runtime")?
            .block_on(self.download_file(repo_id, repo_type, revision, remote_path, dest_path, progress))
    }
}

// Generic retry helper with jittered backoff
impl HfDownloader {
    async fn retrying<F, Fut, T>(&self, mut f: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0usize;
        loop {
            match f().await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempt >= self.config.max_retries { return Err(e); }
                    let wait = jittered_backoff_ms(attempt);
                    sleep(Duration::from_millis(wait as u64)).await;
                    attempt += 1;
                }
            }
        }
    }
}
