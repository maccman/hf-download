use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use globset::{Glob, GlobSet, GlobSetBuilder};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_RANGE, RANGE, USER_AGENT};
use reqwest::{Client, Url};
use serde::Deserialize;
use tokio::fs::{create_dir_all, rename, OpenOptions};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

const DEFAULT_CHUNK_SIZE: usize = 8 * 1024 * 1024; // 8 MiB
const DEFAULT_MAX_CONCURRENT: usize = 8;
const BASE_WAIT_MS: usize = 300;
const MAX_WAIT_MS: usize = 10_000;

#[derive(Clone, Debug)]
pub enum RepoType {
    Model,
    Dataset,
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

#[derive(Clone, Debug)]
pub struct DownloadConfig {
    pub max_concurrent_files: usize,
    pub chunk_size_bytes: usize,
    pub max_retries: usize,
    pub parallel_failures: usize,
    pub resume: bool,
    pub include: Option<Vec<String>>,
    pub exclude: Option<Vec<String>>,
    pub api_base: String,
    pub auth_token: Option<String>,
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
            resume: true,
            include: None,
            exclude: None,
            api_base,
            auth_token,
            user_agent: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProgressEvent {
    RepoDiscovered { num_files: usize, total_bytes: u64 },
    FileStarted { path: String, size: u64 },
    BytesTransferred { path: String, bytes: usize },
    FileCompleted { path: String },
    FileFailed { path: String, error: String },
}

#[derive(Debug, Default, Clone)]
pub struct RepoSummary {
    pub files_downloaded: usize,
    pub bytes_downloaded: u64,
}

#[derive(Clone)]
pub struct HfDownloader {
    client: Client,
    config: DownloadConfig,
}

impl HfDownloader {
    pub fn new(config: DownloadConfig) -> Result<Self> {
        let mut default_headers = HeaderMap::new();
        if let Some(ref ua) = config.user_agent {
            default_headers.insert(USER_AGENT, HeaderValue::from_str(ua).unwrap());
        }
        let client = Client::builder()
            .default_headers(default_headers)
            .http2_keep_alive_timeout(Duration::from_secs(15))
            .build()?;
        Ok(Self { client, config })
    }

    pub async fn download_repo<F: Fn(ProgressEvent) + Send + Sync + 'static>(
        &self,
        repo_id: &str,
        repo_type: RepoType,
        revision: &str,
        dest_dir: &Path,
        progress: F,
    ) -> Result<RepoSummary> {
        let (files, total_bytes) = self
            .list_repo_files(repo_id, &repo_type, revision)
            .await
            .context("list repo files")?;

        let (globset_include, globset_exclude) = build_globsets(&self.config.include, &self.config.exclude)?;
        let filtered: Vec<_> = files
            .into_iter()
            .filter(|f| {
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
        let progress = Arc::new(progress);

        for file in filtered {
            let permit = semaphore.clone().acquire_owned().await?;
            let progress = progress.clone();
            let this = self.clone();
            let repo_id = repo_id.to_string();
            let revision = revision.to_string();
            let repo_type = match repo_type { RepoType::Model => RepoType::Model, RepoType::Dataset => RepoType::Dataset, RepoType::Space => RepoType::Space };
            let dest_path = dest_dir.join(&file.path);
            tasks.push(tokio::spawn(async move {
                let _p = permit; // keep permit until task ends
                if let Some(parent) = dest_path.parent() { create_dir_all(parent).await.ok(); }
                let res = this.download_file(&repo_id, repo_type, &revision, &file.path, &dest_path, move |evt| progress(evt)).await;
                (file.path, res)
            }));
        }

        let mut summary = RepoSummary::default();
        while let Some(res) = tasks.next().await {
            match res {
                Ok((path, Ok(()))) => {
                    summary.files_downloaded += 1;
                    progress(ProgressEvent::FileCompleted { path });
                }
                Ok((path, Err(err))) => {
                    progress(ProgressEvent::FileFailed { path, error: err.to_string() });
                }
                Err(join_err) => {
                    // spawn failure
                    progress(ProgressEvent::FileFailed { path: "<task>".into(), error: join_err.to_string() });
                }
            }
        }

        Ok(summary)
    }

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

        // Probe to get redirected URL and size
        let resp = self
            .client
            .get(resolved.clone())
            .headers(headers.clone())
            .header(RANGE, "bytes=0-0")
            .send()
            .await?
            .error_for_status()?;

        let final_url = resp.url().clone();
        // If host changed, strip authorization to avoid leaking token to CDN
        let mut effective_headers = headers.clone();
        if resolved.host_str() != final_url.host_str() {
            effective_headers.remove(AUTHORIZATION);
        }

        // Parse size
        let content_range = resp
            .headers()
            .get(CONTENT_RANGE)
            .ok_or_else(|| anyhow!("No content length"))?
            .to_str()?;
        let size: usize = content_range
            .split('/')
            .last()
            .ok_or_else(|| anyhow!("Invalid content-range"))?
            .parse()?;

        let tmp_path = dest_path.with_extension("part");
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&tmp_path)
            .await?;

        let existing = if self.config.resume {
            match tokio::fs::metadata(&tmp_path).await {
                Ok(meta) => meta.len() as usize,
                Err(_) => 0,
            }
        } else {
            0
        };

        let chunk_size = self.config.chunk_size_bytes;
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_files));
        let failures_guard = Arc::new(Semaphore::new(self.config.parallel_failures.max(1)));
        let mut tasks = FuturesUnordered::new();
        let progress = Arc::new(progress);

        for start in (0..size).step_by(chunk_size) {
            // Skip already downloaded part (approximate resume by leading prefix)
            if existing > start { continue; }
            let stop = std::cmp::min(start + chunk_size - 1, size - 1);
            let client = self.client.clone();
            let url = final_url.clone();
            let headers = effective_headers.clone();
            let path = tmp_path.clone();
            let progress_cb = progress.clone();
            let failures_guard = failures_guard.clone();
            let remote_path_string = remote_path.to_string();
            let permit_f = semaphore.clone().acquire_owned().await?;
            tasks.push(tokio::spawn(async move {
                let _pf = permit_f; // hold permit
                let mut attempt = 0usize;
                loop {
                    let range = format!("bytes={start}-{stop}");
                    match download_chunk(&client, url.clone(), headers.clone(), &path, start as u64, &range).await {
                        Ok(bytes) => {
                            progress_cb(ProgressEvent::BytesTransferred { path: remote_path_string.clone(), bytes });
                            return Ok::<(), anyhow::Error>(());
                        }
                        Err(e) => {
                            if attempt >= 5 { // per-chunk cap independent from global config for simplicity
                                return Err(anyhow!(e));
                            }
                            // throttle parallel failures
                            if let Ok(guard) = failures_guard.clone().try_acquire_owned() {
                                let wait = backoff_ms(attempt);
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

        // Ensure file length and notify start
        file.set_len(size as u64).await?;
        progress(ProgressEvent::FileStarted { path: remote_path.to_string(), size: size as u64 });
        drop(file);

        // Join all chunk tasks
        while let Some(res) = tasks.next().await {
            res??;
        }

        // Atomically move into place
        rename(&tmp_path, dest_path).await?;
        Ok(())
    }

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
        let resp = req.send().await?.error_for_status()?;
        let entries: Vec<TreeEntry> = resp.json().await?;
        let mut files = Vec::new();
        let mut total: u64 = 0;
        for e in entries.into_iter() {
            if e.r#type == Some("file".into()) || e.r#type.is_none() {
                let size = e.size.unwrap_or(0) as u64;
                total += size;
                files.push(RepoFile { path: e.path, size });
            }
        }
        Ok((files, total))
    }
}

#[derive(Debug, Clone)]
struct RepoFile {
    path: String,
    size: u64,
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

fn backoff_ms(n: usize) -> usize { (BASE_WAIT_MS + n * n).min(MAX_WAIT_MS) }

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
    pub fn blocking_download_repo<F: Fn(ProgressEvent) + Send + Sync + 'static>(
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
