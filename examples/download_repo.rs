use hf_download::{DownloadConfig, HfDownloader, ProgressEvent, RepoType};
use std::env;
use std::path::PathBuf;

fn parse_repo_type(s: &str) -> RepoType {
    match s.to_ascii_lowercase().as_str() {
        "model" | "models" => RepoType::Model,
        "dataset" | "datasets" => RepoType::Dataset,
        "space" | "spaces" => RepoType::Space,
        other => {
            eprintln!("Unknown repo type '{}', defaulting to model", other);
            RepoType::Model
        }
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let repo_type = args.next().unwrap_or_else(|| "model".to_string());
    let repo_id = args.next().unwrap_or_else(|| {
        eprintln!("Usage: download_repo <repo_type:model|dataset|space> <repo_id> [revision] [dest_dir]");
        std::process::exit(1);
    });
    let revision = args.next().unwrap_or_else(|| "main".to_string());
    let dest_dir = PathBuf::from(args.next().unwrap_or_else(|| format!("./downloaded-{}", repo_id.replace('/', "-"))));

    let config = DownloadConfig::default();
    let downloader = HfDownloader::new(config).expect("init downloader");

    let ty = parse_repo_type(&repo_type);
    let res = downloader.blocking_download_repo(&repo_id, ty, &revision, &dest_dir.as_path(), |evt| match evt {
        ProgressEvent::RepoDiscovered { num_files, total_bytes } => {
            println!("Discovered {} files ({} bytes)", num_files, total_bytes);
        }
        ProgressEvent::FileStarted { path, size } => {
            println!("Starting {} ({} bytes)", path, size);
        }
        ProgressEvent::BytesTransferred { path, bytes } => {
            println!("{} +{} bytes", path, bytes);
        }
        ProgressEvent::FileCompleted { path } => println!("Completed {}", path),
        ProgressEvent::FileFailed { path, error } => eprintln!("Failed {}: {}", path, error),
    });

    match res {
        Ok(summary) => println!(
            "Done: {} files downloaded ({} bytes) to {}",
            summary.files_downloaded,
            summary.bytes_downloaded,
            dest_dir.display()
        ),
        Err(err) => {
            eprintln!("Error: {}", err);
            std::process::exit(1);
        }
    }
}
