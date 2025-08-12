# hf_download

Pure Rust library to download Hugging Face repositories (models, datasets, spaces) over HTTP without Git.

- Parallel, ranged downloads for speed
- No Git required, no upload code
- Works with private repos (bearer token)
- Async and blocking APIs

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
hf_download = { git = "https://github.com/your/repo.git" }
```

Or once published:

```toml
[dependencies]
hf_download = "0.1"
```

## Quick start

```rust
use hf_download::{DownloadConfig, HfDownloader, RepoType, ProgressEvent};
use std::path::Path;

let cfg = DownloadConfig::default();
let dl = HfDownloader::new(cfg).unwrap();
let summary = dl.blocking_download_repo(
    "bert-base-uncased",
    RepoType::Model,
    "main",
    Path::new("/tmp/bert"),
    |evt| match evt {
        ProgressEvent::RepoDiscovered { num_files, total_bytes } => println!("{} files / {} bytes", num_files, total_bytes),
        ProgressEvent::BytesTransferred { path, bytes } => println!("{} +{}", path, bytes),
        ProgressEvent::FileCompleted { path } => println!("done {path}"),
        ProgressEvent::FileFailed { path, error } => eprintln!("fail {path}: {error}"),
        ProgressEvent::FileStarted { .. } => {}
    },
).unwrap();
println!(
    "Downloaded {} files ({} bytes)",
    summary.files_downloaded,
    summary.bytes_downloaded
);
```

## CLI example

Build and run the example binary:

```bash
cargo run --example download_repo -- model bert-base-uncased main ./downloads/bert
```

Arguments: `<repo_type: model|dataset|space> <repo_id> [revision] [dest_dir]`

## Configuration

- Auth token via `DownloadConfig.auth_token` or env `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.
- Endpoint via `DownloadConfig.api_base` or env `HF_ENDPOINT` (default: `https://huggingface.co`).
- Concurrency and chunk size via `DownloadConfig`.
- Include/exclude file globs to filter downloads.
- Resume enabled by default: downloads into `*.part` and renames atomically.

## How it works

- Lists repo files using `GET /api/{models|datasets|spaces}/{repo_id}/tree/{revision}?recursive=1`.
- Resolves file URLs via `/resolve/{revision}/{path}`.
- Performs parallel ranged `GET` requests with retries and backoff. Authorization is sent only to the hub, not to the CDN after redirect.

## License

MIT or Apache-2.0
