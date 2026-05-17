use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    process::{Output, Stdio},
    sync::{
        LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use regex::Regex;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
};
use tracing::{error, info, warn};

const DEFAULT_DATA_DIR: &str = "data";
const DEFAULT_WHISPER_BIN: &str = "nix run nixpkgs#openai-whisper --";
const DEFAULT_WHISPER_MODEL: &str = "large";
const DEFAULT_AUDIO_FORMAT: &str = "m4a";
const STREAMED_OUTPUT_CAPTURE_LIMIT: usize = 64 * 1024;
#[cfg(not(test))]
const STREAM_READER_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const STREAM_READER_DRAIN_TIMEOUT: Duration = Duration::from_millis(100);
const YOUTUBE_VIDEO_ID_LEN: usize = 11;
#[cfg(not(target_os = "linux"))]
const NON_LINUX_STALE_TEMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static SLUG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\p{Alphabetic}\p{Number}]+").expect("slug regex compiles"));

#[derive(Parser, Debug)]
#[command(name = "youtube-archiver")]
#[command(about = "Archive YouTube audio, transcribe it with Whisper, and emit llm-wiki markdown")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download, transcribe, and emit wiki markdown for a video, channel, or playlist URL.
    Ingest(IngestArgs),
    /// Print a per-video state table.
    Status(DataDirArgs),
    /// List archived videos as JSON.
    List(DataDirArgs),
}

#[derive(Args, Debug)]
struct IngestArgs {
    url: String,

    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,

    #[arg(long, default_value = DEFAULT_WHISPER_MODEL)]
    whisper_model: String,

    #[arg(
        long,
        default_value = DEFAULT_WHISPER_BIN,
        help = "Whisper command prefix. The default expands to: nix run nixpkgs#openai-whisper --"
    )]
    whisper_bin: String,

    #[arg(
        long = "whisper-arg",
        allow_hyphen_values = true,
        num_args = 1,
        value_name = "ARG",
        help = "Extra argument passed verbatim to the Whisper command; repeat for multiple args. Values starting with '-' are passed through."
    )]
    whisper_args: Vec<String>,

    #[arg(long, value_parser = parse_positive_usize)]
    limit: Option<usize>,

    #[arg(long, default_value = DEFAULT_AUDIO_FORMAT)]
    audio_format: String,

    #[arg(long)]
    force: bool,
}

#[derive(Args, Debug)]
struct DataDirArgs {
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid positive integer: {err}"))?;
    if parsed == 0 {
        Err("value must be greater than 0".to_owned())
    } else {
        Ok(parsed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Video,
    Channel,
    Playlist,
}

#[derive(Debug, Clone)]
struct VideoMetadata {
    video_id: String,
    url: String,
    channel_id: Option<String>,
    channel_title: Option<String>,
    uploader: Option<String>,
    title: Option<String>,
    upload_date: Option<String>,
    duration: Option<u64>,
    tags: Vec<String>,
}

struct WhisperConfig<'a> {
    bin: &'a str,
    model: &'a str,
    extra_args: &'a [String],
}

struct AbortOnDrop<T> {
    handle: Option<tokio::task::JoinHandle<T>>,
}

impl<T> AbortOnDrop<T> {
    fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self {
            handle: Some(handle),
        }
    }

    async fn join(mut self) -> std::result::Result<T, tokio::task::JoinError> {
        let handle = self.handle.take().expect("join handle exists");
        handle.await
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoRow {
    video_id: String,
    url: String,
    channel_id: Option<String>,
    channel_title: Option<String>,
    uploader: Option<String>,
    title: Option<String>,
    upload_date: Option<String>,
    duration: Option<u64>,
    tags: Vec<String>,
    downloaded_at: Option<String>,
    transcribed_at: Option<String>,
    wiki_emitted_at: Option<String>,
    whisper_model: Option<String>,
    audio_path: Option<String>,
    transcript_path: Option<String>,
    wiki_path: Option<String>,
    error: Option<String>,
}

struct Ledger {
    conn: Connection,
    data_dir: PathBuf,
}

impl Ledger {
    fn open(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .with_context(|| format!("create data dir {}", data_dir.display()))?;
        let data_dir = data_dir.to_path_buf();
        let db_path = data_dir.join("state.sqlite");
        let conn = Connection::open(&db_path)
            .with_context(|| format!("open ledger {}", db_path.display()))?;
        configure_connection(&conn)?;
        let ledger = Self { conn, data_dir };
        ledger.init()?;
        Ok(ledger)
    }

    fn open_read_only(data_dir: &Path) -> Result<Option<Self>> {
        let db_path = data_dir.join("state.sqlite");
        if !db_path.exists() {
            return Ok(None);
        }

        let conn = Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("open ledger read-only {}", db_path.display()))?;
        configure_connection(&conn)?;
        Ok(Some(Self {
            conn,
            data_dir: data_dir.to_path_buf(),
        }))
    }

    #[cfg(test)]
    fn open_in_memory() -> Result<Self> {
        let ledger = Self {
            conn: Connection::open_in_memory().context("open in-memory ledger")?,
            data_dir: std::env::current_dir()
                .context("read current directory for in-memory ledger")?
                .join(DEFAULT_DATA_DIR),
        };
        configure_connection(&ledger.conn)?;
        ledger.init()?;
        Ok(ledger)
    }

    fn init(&self) -> Result<()> {
        self.conn
            .execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    channel_id TEXT,
                    channel_title TEXT,
                    uploader TEXT,
                    title TEXT,
                    upload_date TEXT,
                    duration INTEGER,
                    tags TEXT,
                    downloaded_at TEXT,
                    transcribed_at TEXT,
                    wiki_emitted_at TEXT,
                    whisper_model TEXT,
                    audio_path TEXT,
                    transcript_path TEXT,
                    wiki_path TEXT,
                    error TEXT
                );
                "#,
            )
            .context("initialize ledger schema")?;
        self.ensure_column("uploader")?;
        self.ensure_column("upload_date")?;
        self.ensure_column("duration")?;
        self.ensure_column("tags")?;
        Ok(())
    }

    fn ensure_column(&self, name: &str) -> Result<()> {
        let alter_sql = match name {
            "uploader" => "ALTER TABLE videos ADD COLUMN uploader TEXT",
            "upload_date" => "ALTER TABLE videos ADD COLUMN upload_date TEXT",
            "duration" => "ALTER TABLE videos ADD COLUMN duration INTEGER",
            "tags" => "ALTER TABLE videos ADD COLUMN tags TEXT",
            _ => bail!("unsupported ledger migration column {name:?}"),
        };

        let mut stmt = self
            .conn
            .prepare("PRAGMA table_info(videos)")
            .context("inspect ledger schema")?;
        let mut rows = stmt.query([]).context("read ledger schema")?;
        while let Some(row) = rows.next().context("read ledger schema row")? {
            let column_name: String = row.get(1).context("read ledger column name")?;
            if column_name == name {
                return Ok(());
            }
        }

        self.conn
            .execute(alter_sql, [])
            .with_context(|| format!("add ledger column {name}"))?;
        Ok(())
    }

    fn ensure_video(&self, video_id: &str, url: &str) -> Result<()> {
        self.conn
            .execute(
                r#"
                INSERT INTO videos (video_id, url)
                VALUES (?1, ?2)
                ON CONFLICT(video_id) DO UPDATE SET url = excluded.url
                "#,
                params![video_id, url],
            )
            .with_context(|| format!("upsert ledger row for {video_id}"))?;
        Ok(())
    }

    fn upsert_metadata(&self, metadata: &VideoMetadata) -> Result<()> {
        let duration = metadata
            .duration
            .map(i64::try_from)
            .transpose()
            .context("duration does not fit sqlite integer")?;
        let tags = serde_json::to_string(&metadata.tags).context("serialize metadata tags")?;
        self.conn
            .execute(
                r#"
                INSERT INTO videos (
                    video_id, url, channel_id, channel_title, uploader, title,
                    upload_date, duration, tags
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                ON CONFLICT(video_id) DO UPDATE SET
                    url = excluded.url,
                    channel_id = excluded.channel_id,
                    channel_title = excluded.channel_title,
                    uploader = excluded.uploader,
                    title = excluded.title,
                    upload_date = excluded.upload_date,
                    duration = excluded.duration,
                    tags = excluded.tags
                "#,
                params![
                    metadata.video_id,
                    metadata.url,
                    metadata.channel_id,
                    metadata.channel_title,
                    metadata.uploader,
                    metadata.title,
                    metadata.upload_date,
                    duration,
                    tags
                ],
            )
            .with_context(|| format!("upsert metadata for {}", metadata.video_id))?;
        Ok(())
    }

    fn mark_downloaded(&self, video_id: &str, audio_path: &Path) -> Result<()> {
        let audio_path = self.path_to_ledger_string(audio_path)?;
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET downloaded_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    audio_path = ?2,
                    transcribed_at = NULL,
                    wiki_emitted_at = NULL,
                    error = NULL
                WHERE video_id = ?1
                "#,
                params![video_id, audio_path],
            )
            .with_context(|| format!("mark {video_id} downloaded"))?;
        Ok(())
    }

    fn mark_transcribed(
        &self,
        video_id: &str,
        whisper_model: &str,
        transcript_path: &Path,
    ) -> Result<()> {
        let transcript_path = self.path_to_ledger_string(transcript_path)?;
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET transcribed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    whisper_model = ?2,
                    transcript_path = ?3,
                    -- Preserve wiki_path so re-emission can remove stale slug paths.
                    wiki_emitted_at = NULL,
                    error = NULL
                WHERE video_id = ?1
                "#,
                params![video_id, whisper_model, transcript_path],
            )
            .with_context(|| format!("mark {video_id} transcribed"))?;
        Ok(())
    }

    fn invalidate_transcription_outputs(&self, video_id: &str) -> Result<()> {
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET transcribed_at = NULL,
                    wiki_emitted_at = NULL
                WHERE video_id = ?1
                "#,
                params![video_id],
            )
            .with_context(|| format!("invalidate transcription outputs for {video_id}"))?;
        Ok(())
    }

    fn mark_wiki_emitted(&self, video_id: &str, wiki_path: &Path) -> Result<()> {
        let wiki_path = self.path_to_ledger_string(wiki_path)?;
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET wiki_emitted_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    wiki_path = ?2,
                    error = NULL
                WHERE video_id = ?1
                "#,
                params![video_id, wiki_path],
            )
            .with_context(|| format!("mark {video_id} wiki emitted"))?;
        Ok(())
    }

    fn mark_error(&self, video_id: &str, err: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE videos SET error = ?2 WHERE video_id = ?1",
                params![video_id, err],
            )
            .with_context(|| format!("record error for {video_id}"))?;
        Ok(())
    }

    fn clear_stale_error(&self, video_id: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE videos SET error = NULL WHERE video_id = ?1 AND error IS NOT NULL",
                params![video_id],
            )
            .with_context(|| format!("clear stale error for {video_id}"))?;
        Ok(())
    }

    fn row(&self, video_id: &str) -> Result<Option<VideoRow>> {
        self.conn
            .query_row(
                r#"
                SELECT video_id, url, channel_id, channel_title, uploader, title,
                       upload_date, duration, tags,
                       downloaded_at, transcribed_at, wiki_emitted_at,
                       whisper_model, audio_path, transcript_path, wiki_path, error
                FROM videos
                WHERE video_id = ?1
                "#,
                params![video_id],
                row_from_sql,
            )
            .optional()
            .with_context(|| format!("read ledger row for {video_id}"))
    }

    fn rows(&self) -> Result<Vec<VideoRow>> {
        let mut stmt = self
            .conn
            .prepare(
                r#"
                SELECT video_id, url, channel_id, channel_title, uploader, title,
                       upload_date, duration, tags,
                       downloaded_at, transcribed_at, wiki_emitted_at,
                       whisper_model, audio_path, transcript_path, wiki_path, error
                FROM videos
                ORDER BY video_id
                "#,
            )
            .context("prepare ledger list query")?;
        let mut query_rows = stmt
            .query_map([], row_from_sql)
            .context("query ledger rows")?;
        let mut rows = Vec::new();
        for row in &mut query_rows {
            match row {
                Ok(row) => rows.push(row),
                Err(err) => {
                    warn!(error = %err, "skipping corrupt ledger row");
                }
            }
        }
        Ok(rows)
    }

    fn path_to_ledger_string(&self, path: &Path) -> Result<String> {
        path_to_ledger_string(&self.data_dir, path)
    }
}

fn configure_connection(conn: &Connection) -> Result<()> {
    conn.busy_timeout(Duration::from_secs(30))
        .context("configure sqlite busy timeout")?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match Cli::parse().command {
        Commands::Ingest(args) => ingest(args).await,
        Commands::Status(args) => status(args).await,
        Commands::List(args) => list(args),
    }
}

async fn ingest(args: IngestArgs) -> Result<()> {
    validate_whisper_config(&args.whisper_bin, &args.whisper_args)?;

    let ledger = Ledger::open(&args.data_dir)?;
    let mode = classify_youtube_url(&args.url);
    info!(?mode, url = %args.url, "resolving input URL");
    let video_ids = resolve_video_ids(&args.url, mode, args.limit).await?;

    if video_ids.is_empty() {
        bail!("yt-dlp did not return any video IDs for {}", args.url);
    }

    ensure_resolved_videos(&ledger, &video_ids)?;

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut failed_video_ids = Vec::new();

    for video_id in video_ids {
        match process_video(&args, &ledger, &video_id).await {
            Ok(()) => {
                succeeded += 1;
                info!(%video_id, "video processed");
            }
            Err(err) => {
                failed += 1;
                failed_video_ids.push(video_id.clone());
                let message = format!("{err:#}");
                error!(%video_id, error = %message, "video failed");
                if let Err(err) = ledger.mark_error(&video_id, &message) {
                    warn!(%video_id, error = %err, original_error = %message, "failed to record video error in ledger");
                }
            }
        }
    }

    if succeeded == 0 {
        bail!(
            "every video failed ({failed} failure(s)): {}",
            failed_video_ids.join(", ")
        );
    }

    Ok(())
}

fn ensure_resolved_videos(ledger: &Ledger, video_ids: &[String]) -> Result<()> {
    for video_id in video_ids {
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
    }
    Ok(())
}

async fn process_video(args: &IngestArgs, ledger: &Ledger, video_id: &str) -> Result<()> {
    let metadata = load_or_fetch_metadata(&args.data_dir, ledger, video_id, args.force).await?;

    let audio_path = download_audio(
        &args.data_dir,
        ledger,
        video_id,
        &args.audio_format,
        args.force,
    )
    .await
    .with_context(|| format!("download audio for {video_id}"))?;

    transcribe_audio(
        &args.data_dir,
        ledger,
        video_id,
        &audio_path,
        WhisperConfig {
            bin: &args.whisper_bin,
            model: &args.whisper_model,
            extra_args: &args.whisper_args,
        },
        args.force,
    )
    .await
    .with_context(|| format!("transcribe {video_id}"))?;

    emit_wiki_article(&args.data_dir, ledger, &metadata, args.force)
        .await
        .with_context(|| format!("emit wiki markdown for {video_id}"))?;

    if let Err(err) = ledger.clear_stale_error(video_id) {
        warn!(%video_id, error = %err, "failed to clear stale ledger error after successful processing");
    }
    Ok(())
}

async fn resolve_video_ids(
    url: &str,
    mode: InputMode,
    limit: Option<usize>,
) -> Result<Vec<String>> {
    let args = resolve_video_ids_args(url, mode, limit);
    let output = run_checked("yt-dlp", &args).await?;
    let stdout = String::from_utf8(output.stdout).context("yt-dlp emitted non-UTF8 video IDs")?;
    Ok(collect_valid_resolved_video_ids(&stdout, limit))
}

fn collect_valid_resolved_video_ids(stdout: &str, limit: Option<usize>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut ids = Vec::new();

    for line in stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if !is_valid_youtube_video_id(line) {
            warn!(
                error = %invalid_video_id_error(line),
                "skipping invalid video ID returned by yt-dlp"
            );
            continue;
        }
        if seen.insert(line.to_owned()) {
            ids.push(line.to_owned());
        }
        if limit.is_some_and(|max| ids.len() >= max) {
            break;
        }
    }

    ids
}

fn is_valid_youtube_video_id(video_id: &str) -> bool {
    video_id.len() == YOUTUBE_VIDEO_ID_LEN
        && video_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn invalid_video_id_error(video_id: &str) -> String {
    format!("invalid YouTube video ID {video_id:?}; expected [A-Za-z0-9_-]{{11}}")
}

fn resolve_video_ids_args(url: &str, mode: InputMode, limit: Option<usize>) -> Vec<String> {
    let mut args = vec![
        "--flat-playlist".to_owned(),
        "--print".to_owned(),
        "id".to_owned(),
    ];
    if mode == InputMode::Video {
        args.push("--no-playlist".to_owned());
    }
    if let Some(limit) = limit {
        args.extend(["--playlist-end".to_owned(), limit.to_string()]);
    }
    args.push("--".to_owned());
    args.push(url.to_owned());
    args
}

async fn load_or_fetch_metadata(
    data_dir: &Path,
    ledger: &Ledger,
    video_id: &str,
    force: bool,
) -> Result<VideoMetadata> {
    let media_dir = data_dir.join("media").join(video_id);
    let info_path = media_dir.join("info.json");

    if !force && let Some(metadata) = load_cached_metadata(video_id, &info_path).await? {
        ledger.upsert_metadata(&metadata)?;
        return Ok(metadata);
    }

    fs::create_dir_all(&media_dir)
        .await
        .with_context(|| format!("create {}", media_dir.display()))?;

    let url = canonical_video_url(video_id);
    let args = vec![
        "-j".to_owned(),
        "--no-playlist".to_owned(),
        "--".to_owned(),
        url,
    ];
    let output = run_checked("yt-dlp", &args).await?;
    let value: Value = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parse yt-dlp metadata for {video_id}"))?;
    atomic_write(&info_path, &output.stdout).await?;
    let metadata = metadata_from_value(video_id, &value);
    ledger.upsert_metadata(&metadata)?;
    Ok(metadata)
}

async fn load_cached_metadata(video_id: &str, info_path: &Path) -> Result<Option<VideoMetadata>> {
    let bytes = match fs::read(info_path).await {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| format!("read existing {}", info_path.display()));
        }
    };

    if bytes.is_empty() {
        warn!(path = %info_path.display(), "cached metadata is empty; refetching");
        return Ok(None);
    }

    match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => Ok(Some(metadata_from_value(video_id, &value))),
        Err(err) => {
            warn!(path = %info_path.display(), error = %err, "cached metadata is invalid JSON; refetching");
            Ok(None)
        }
    }
}

async fn download_audio(
    data_dir: &Path,
    ledger: &Ledger,
    video_id: &str,
    audio_format: &str,
    force: bool,
) -> Result<PathBuf> {
    if let Some(row) = ledger.row(video_id)?
        && should_skip_download_async(data_dir, &row, audio_format, force).await
    {
        let audio_path = row
            .audio_path
            .expect("checked by should_skip_download_async");
        return Ok(ledger_path_to_fs_path(data_dir, &audio_path));
    }

    let media_dir = data_dir.join("media").join(video_id);
    fs::create_dir_all(&media_dir)
        .await
        .with_context(|| format!("create {}", media_dir.display()))?;
    cleanup_stage_temp_dirs(&media_dir, ".download").await?;
    let tmp_dir = media_dir.join(unique_temp_name(".download"));
    fs::create_dir_all(&tmp_dir)
        .await
        .with_context(|| format!("create {}", tmp_dir.display()))?;

    let output_template = yt_dlp_audio_output_template(&tmp_dir);
    let url = canonical_video_url(video_id);
    let args = vec![
        "-f".to_owned(),
        "bestaudio".to_owned(),
        "--extract-audio".to_owned(),
        "--audio-format".to_owned(),
        audio_format.to_owned(),
        "--no-playlist".to_owned(),
        "-o".to_owned(),
        output_template,
        "--".to_owned(),
        url,
    ];

    let result: Result<PathBuf> = async {
        run_checked_stream_output("yt-dlp", &args).await?;
        let downloaded = find_audio_file(&tmp_dir, audio_format).await?;
        let extension = downloaded
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or(audio_format);
        let final_path = media_dir.join(format!("audio.{extension}"));
        fs::rename(&downloaded, &final_path)
            .await
            .with_context(|| format!("move audio to {}", final_path.display()))?;
        remove_stale_audio_files(&media_dir, &final_path).await?;
        sync_parent_dir(&final_path).await?;
        Ok(final_path)
    }
    .await;

    let cleanup = fs::remove_dir_all(&tmp_dir).await;
    let final_path = match result {
        Ok(final_path) => {
            if let Err(err) = cleanup {
                warn!(path = %tmp_dir.display(), error = %err, "failed to remove temporary download dir");
            }
            final_path
        }
        Err(err) => {
            let _ = cleanup;
            return Err(err);
        }
    };

    ledger.mark_downloaded(video_id, &final_path)?;
    Ok(final_path)
}

async fn transcribe_audio(
    data_dir: &Path,
    ledger: &Ledger,
    video_id: &str,
    audio_path: &Path,
    whisper: WhisperConfig<'_>,
    force: bool,
) -> Result<PathBuf> {
    if let Some(row) = ledger.row(video_id)?
        && should_skip_transcription_async(data_dir, &row, whisper.model, force).await
    {
        let transcript_path = row
            .transcript_path
            .expect("checked by should_skip_transcription_async");
        return Ok(ledger_path_to_fs_path(data_dir, &transcript_path));
    }

    let transcript_dir = data_dir.join("transcripts").join(video_id);
    fs::create_dir_all(&transcript_dir)
        .await
        .with_context(|| format!("create {}", transcript_dir.display()))?;
    cleanup_stage_temp_dirs(&transcript_dir, ".whisper").await?;
    let tmp_dir = transcript_dir.join(unique_temp_name(".whisper"));
    fs::create_dir_all(&tmp_dir)
        .await
        .with_context(|| format!("create {}", tmp_dir.display()))?;

    let (program, mut args) = split_command_prefix(whisper.bin)?;
    args.extend([
        path_to_string(audio_path),
        "--model".to_owned(),
        whisper.model.to_owned(),
    ]);
    args.extend(whisper.extra_args.iter().cloned());
    args.extend([
        "--output_dir".to_owned(),
        path_to_string(&tmp_dir),
        "--output_format".to_owned(),
        "all".to_owned(),
    ]);

    let result: Result<PathBuf> = async {
        run_checked_stream_output(&program, &args).await?;
        let output_stem = audio_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("audio");
        let (whisper_json, whisper_txt) = find_whisper_outputs(&tmp_dir, output_stem).await?;
        let final_json = transcript_dir.join("transcript.json");
        let final_txt = transcript_dir.join("transcript.txt");
        ledger.invalidate_transcription_outputs(video_id)?;
        replace_transcript_pair(&whisper_json, &whisper_txt, &final_json, &final_txt).await?;
        Ok(final_json)
    }
    .await;

    let cleanup = fs::remove_dir_all(&tmp_dir).await;
    let final_json = match result {
        Ok(final_json) => {
            if let Err(err) = cleanup {
                warn!(path = %tmp_dir.display(), error = %err, "failed to remove temporary whisper dir");
            }
            final_json
        }
        Err(err) => {
            let _ = cleanup;
            return Err(err);
        }
    };

    ledger.mark_transcribed(video_id, whisper.model, &final_json)?;
    Ok(final_json)
}

async fn emit_wiki_article(
    data_dir: &Path,
    ledger: &Ledger,
    metadata: &VideoMetadata,
    force: bool,
) -> Result<PathBuf> {
    let wiki_path = wiki_path_for_metadata(data_dir, metadata);
    let previous_wiki_path = if let Some(row) = ledger.row(&metadata.video_id)? {
        if should_skip_wiki_async(data_dir, &row, force).await {
            let row_wiki_path = row.wiki_path.expect("checked by should_skip_wiki_async");
            if ledger_path_matches(data_dir, &row_wiki_path, &wiki_path)? {
                return Ok(ledger_path_to_fs_path(data_dir, &row_wiki_path));
            }
            Some(row_wiki_path)
        } else {
            row.wiki_path
        }
    } else {
        None
    };

    let transcript_txt = data_dir
        .join("transcripts")
        .join(&metadata.video_id)
        .join("transcript.txt");
    let transcript = fs::read_to_string(&transcript_txt)
        .await
        .with_context(|| format!("read {}", transcript_txt.display()))?;

    let article = render_wiki_markdown(metadata, &transcript);
    atomic_write(&wiki_path, article.as_bytes()).await?;
    remove_stale_wiki_article(data_dir, previous_wiki_path.as_deref(), &wiki_path).await?;
    ledger.mark_wiki_emitted(&metadata.video_id, &wiki_path)?;
    Ok(wiki_path)
}

fn wiki_path_for_metadata(data_dir: &Path, metadata: &VideoMetadata) -> PathBuf {
    let channel_slug = slugify(
        metadata
            .channel_title
            .as_deref()
            .or(metadata.uploader.as_deref())
            .or(metadata.channel_id.as_deref())
            .unwrap_or("unknown-channel"),
    );
    data_dir
        .join("wiki")
        .join(channel_slug)
        .join(format!("{}.md", metadata.video_id))
}

fn ledger_path_matches(data_dir: &Path, ledger_path: &str, path: &Path) -> Result<bool> {
    Ok(normalize_ledger_path_string(data_dir, ledger_path)?
        == path_to_ledger_string(data_dir, path)?)
}

async fn status(args: DataDirArgs) -> Result<()> {
    let rows = Ledger::open_read_only(&args.data_dir)?
        .map(|ledger| ledger.rows())
        .transpose()?
        .unwrap_or_default();

    println!(
        "{:<14} {:<10} {:<11} {:<10} {:<7} title",
        "video_id", "download", "transcribe", "wiki", "error"
    );
    for row in rows {
        let download = download_state(&args.data_dir, &row).await;
        let transcript = transcript_state(&args.data_dir, &row).await;
        let wiki = wiki_state(&args.data_dir, &row).await;
        println!(
            "{:<14} {:<10} {:<11} {:<10} {:<7} {}",
            row.video_id,
            download,
            transcript,
            wiki,
            if row.error.is_some() { "yes" } else { "-" },
            row.title.as_deref().unwrap_or("-")
        );
    }

    Ok(())
}

fn list(args: DataDirArgs) -> Result<()> {
    let rows = list_rows(&args.data_dir)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&rows).context("serialize archived video rows")?
    );
    Ok(())
}

fn list_rows(data_dir: &Path) -> Result<Vec<VideoRow>> {
    let rows = Ledger::open_read_only(data_dir)?
        .map(|ledger| ledger.rows())
        .transpose()?
        .unwrap_or_default();
    Ok(rows
        .into_iter()
        .filter(|row| is_archived_row(data_dir, row))
        .collect::<Vec<_>>())
}

fn is_archived_row(data_dir: &Path, row: &VideoRow) -> bool {
    row.downloaded_at.is_some()
        && row.transcribed_at.is_some()
        && row.wiki_emitted_at.is_some()
        && row.error.is_none()
        && row.audio_path.as_deref().is_some_and(|path| {
            let path = ledger_path_to_fs_path(data_dir, path);
            path.is_file()
        })
        && row.transcript_path.as_deref().is_some_and(|path| {
            let json_path = ledger_path_to_fs_path(data_dir, path);
            let txt_path = json_path.with_file_name("transcript.txt");
            json_path.is_file() && txt_path.is_file()
        })
        && row.wiki_path.as_deref().is_some_and(|path| {
            let path = ledger_path_to_fs_path(data_dir, path);
            path.is_file()
        })
}

fn row_from_sql(row: &rusqlite::Row<'_>) -> rusqlite::Result<VideoRow> {
    let duration: Option<i64> = row.get(7)?;
    let tags_json: Option<String> = row.get(8)?;
    let tags = match tags_json.as_deref() {
        Some(value) => serde_json::from_str(value).map_err(|err| {
            rusqlite::Error::FromSqlConversionFailure(8, rusqlite::types::Type::Text, Box::new(err))
        })?,
        None => Vec::new(),
    };
    Ok(VideoRow {
        video_id: row.get(0)?,
        url: row.get(1)?,
        channel_id: row.get(2)?,
        channel_title: row.get(3)?,
        uploader: row.get(4)?,
        title: row.get(5)?,
        upload_date: row.get(6)?,
        duration: duration.and_then(|value| u64::try_from(value).ok()),
        tags,
        downloaded_at: row.get(9)?,
        transcribed_at: row.get(10)?,
        wiki_emitted_at: row.get(11)?,
        whisper_model: row.get(12)?,
        audio_path: row.get(13)?,
        transcript_path: row.get(14)?,
        wiki_path: row.get(15)?,
        error: row.get(16)?,
    })
}

fn classify_youtube_url(url: &str) -> InputMode {
    let (path, query) = split_url_path_and_query(url);
    let path = path.to_ascii_lowercase();
    let (host, path) = split_url_host_and_path(&path);
    let segments = path_segments(path);

    if (host.is_some_and(is_youtu_be_host) && !segments.is_empty())
        || (segments.first() == Some(&"watch") && query_param_has_value(query, "v"))
        || is_canonical_video_path(&segments)
    {
        InputMode::Video
    } else if segments.first() == Some(&"playlist") || query_param_has_value(query, "list") {
        InputMode::Playlist
    } else {
        InputMode::Channel
    }
}

fn split_url_path_and_query(url: &str) -> (&str, Option<&str>) {
    let without_fragment = url.split_once('#').map_or(url, |(head, _)| head);
    without_fragment
        .split_once('?')
        .map_or((without_fragment, None), |(path, query)| {
            (path, Some(query))
        })
}

fn split_url_host_and_path(path_or_url: &str) -> (Option<&str>, &str) {
    if let Some((_, rest)) = path_or_url.split_once("://") {
        return split_authority_and_path(rest);
    }
    if let Some(rest) = path_or_url.strip_prefix("//") {
        return split_authority_and_path(rest);
    }

    let path_or_url = path_or_url.trim_start_matches('/');
    if let Some((authority, path)) = path_or_url.split_once('/') {
        let host = host_name(authority);
        if is_youtube_host(host) {
            return (Some(host), path);
        }
    }

    (None, path_or_url)
}

fn split_authority_and_path(authority_and_path: &str) -> (Option<&str>, &str) {
    match authority_and_path.split_once('/') {
        Some((authority, path)) => (Some(host_name(authority)), path),
        None => (Some(host_name(authority_and_path)), ""),
    }
}

fn host_name(authority: &str) -> &str {
    authority
        .rsplit('@')
        .next()
        .unwrap_or(authority)
        .split(':')
        .next()
        .unwrap_or(authority)
}

fn path_segments(path: &str) -> Vec<&str> {
    path.split('/').filter(|part| !part.is_empty()).collect()
}

fn is_canonical_video_path(segments: &[&str]) -> bool {
    matches!(
        segments,
        ["shorts", _, ..] | ["live", _, ..] | ["embed", _, ..] | ["v", _, ..] | ["e", _, ..]
    )
}

fn is_youtube_host(host: &str) -> bool {
    host == "youtube.com" || host.ends_with(".youtube.com") || is_youtu_be_host(host)
}

fn is_youtu_be_host(host: &str) -> bool {
    host == "youtu.be" || host.ends_with(".youtu.be")
}

fn query_param_has_value(query: Option<&str>, key: &str) -> bool {
    query
        .into_iter()
        .flat_map(|query| query.split('&'))
        .map(|part| part.split_once('=').unwrap_or((part, "")))
        .any(|(name, value)| name.eq_ignore_ascii_case(key) && !value.is_empty())
}

fn metadata_from_value(video_id: &str, value: &Value) -> VideoMetadata {
    VideoMetadata {
        video_id: video_id.to_owned(),
        url: string_field(value, &["webpage_url", "original_url"])
            .unwrap_or_else(|| canonical_video_url(video_id)),
        channel_id: string_field(value, &["channel_id", "uploader_id"]),
        channel_title: string_field(value, &["channel", "channel_title"]),
        uploader: string_field(value, &["uploader"]),
        title: string_field(value, &["title"]),
        upload_date: string_field(value, &["upload_date"]),
        duration: duration_field(value),
        tags: tags_field(value),
    }
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        value
            .get(*name)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

fn duration_field(value: &Value) -> Option<u64> {
    let duration = value.get("duration")?;
    if let Some(duration) = duration.as_u64() {
        return Some(duration);
    }
    duration
        .as_f64()
        .filter(|duration| *duration >= 0.0)
        .map(|duration| duration.round() as u64)
}

fn tags_field(value: &Value) -> Vec<String> {
    value
        .get("tags")
        .and_then(Value::as_array)
        .map(|tags| {
            tags.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|tag| !tag.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn render_wiki_markdown(metadata: &VideoMetadata, transcript: &str) -> String {
    let title = metadata.title.as_deref().unwrap_or(&metadata.video_id);
    let channel = metadata
        .channel_title
        .as_deref()
        .or(metadata.uploader.as_deref())
        .or(metadata.channel_id.as_deref())
        .unwrap_or("Unknown Channel");
    let uploader = metadata.uploader.as_deref().unwrap_or(channel);
    let upload_date = metadata.upload_date.as_deref();

    let mut output = String::new();
    output.push_str("---\n");
    output.push_str(&format!("title: {}\n", yaml_string(title)));
    output.push_str(&format!("channel: {}\n", yaml_string(channel)));
    output.push_str(&format!("uploader: {}\n", yaml_string(uploader)));
    let upload_date = upload_date.map_or_else(|| "null".to_owned(), yaml_string);
    output.push_str(&format!("upload_date: {}\n", upload_date));
    output.push_str(&format!(
        "duration: {}\n",
        metadata
            .duration
            .map_or_else(|| "null".to_owned(), |duration| duration.to_string())
    ));
    output.push_str(&format!("url: {}\n", yaml_string(&metadata.url)));
    output.push_str(&format!("video_id: {}\n", yaml_string(&metadata.video_id)));
    output.push_str("tags:");
    if metadata.tags.is_empty() {
        output.push_str(" []\n");
    } else {
        output.push('\n');
        for tag in &metadata.tags {
            output.push_str(&format!("  - {}\n", yaml_string(tag)));
        }
    }
    output.push_str("---\n\n");
    output.push_str(transcript.trim());
    output.push('\n');
    output
}

fn yaml_string(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a string cannot fail")
}

fn slugify(value: &str) -> String {
    let lower = value.to_lowercase();
    let slug = SLUG_RE.replace_all(&lower, "-");
    let slug = slug.trim_matches('-');

    if slug.is_empty() {
        "unknown-channel".to_owned()
    } else {
        slug.to_owned()
    }
}

#[cfg(test)]
fn should_skip_download(data_dir: &Path, row: &VideoRow, audio_format: &str, force: bool) -> bool {
    !force
        && row.downloaded_at.is_some()
        && row.audio_path.as_deref().is_some_and(|path| {
            audio_path_matches_format(path, audio_format)
                && ledger_path_to_fs_path(data_dir, path).is_file()
        })
}

async fn should_skip_download_async(
    data_dir: &Path,
    row: &VideoRow,
    audio_format: &str,
    force: bool,
) -> bool {
    if force || row.downloaded_at.is_none() {
        return false;
    }

    let Some(path) = row.audio_path.as_deref() else {
        return false;
    };
    audio_path_matches_format(path, audio_format)
        && async_fs_path_is_file(ledger_path_to_fs_path(data_dir, path)).await
}

fn audio_path_matches_format(path: &str, audio_format: &str) -> bool {
    let Some(expected_extension) = expected_audio_extension(audio_format) else {
        return !audio_format.trim_start_matches('.').is_empty();
    };

    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(&expected_extension))
}

fn expected_audio_extension(audio_format: &str) -> Option<String> {
    let audio_format = audio_format.trim_start_matches('.');
    if audio_format.is_empty() || audio_format.eq_ignore_ascii_case("best") {
        return None;
    }

    let audio_format = audio_format.to_ascii_lowercase();
    let extension = match audio_format.as_str() {
        "aac" | "alac" | "m4a" => "m4a",
        "vorbis" => "ogg",
        other => other,
    };
    Some(extension.to_owned())
}

#[cfg(test)]
fn should_skip_transcription(
    data_dir: &Path,
    row: &VideoRow,
    whisper_model: &str,
    force: bool,
) -> bool {
    if force || row.transcribed_at.is_none() || row.whisper_model.as_deref() != Some(whisper_model)
    {
        return false;
    }

    row.transcript_path.as_deref().is_some_and(|path| {
        let json_path = ledger_path_to_fs_path(data_dir, path);
        let txt_path = json_path.with_file_name("transcript.txt");
        json_path.is_file() && txt_path.is_file()
    })
}

async fn should_skip_transcription_async(
    data_dir: &Path,
    row: &VideoRow,
    whisper_model: &str,
    force: bool,
) -> bool {
    if force || row.transcribed_at.is_none() || row.whisper_model.as_deref() != Some(whisper_model)
    {
        return false;
    }

    let Some(path) = row.transcript_path.as_deref() else {
        return false;
    };
    let json_path = ledger_path_to_fs_path(data_dir, path);
    let txt_path = json_path.with_file_name("transcript.txt");
    async_fs_path_is_file(json_path).await && async_fs_path_is_file(txt_path).await
}

#[cfg(test)]
fn should_skip_wiki(data_dir: &Path, row: &VideoRow, force: bool) -> bool {
    !force
        && row.wiki_emitted_at.is_some()
        && row
            .wiki_path
            .as_deref()
            .is_some_and(|path| ledger_path_to_fs_path(data_dir, path).is_file())
}

async fn should_skip_wiki_async(data_dir: &Path, row: &VideoRow, force: bool) -> bool {
    !force
        && row.wiki_emitted_at.is_some()
        && async_ledger_path_is_file(data_dir, row.wiki_path.as_deref()).await
}

async fn async_ledger_path_is_file(data_dir: &Path, path: Option<&str>) -> bool {
    match path {
        Some(path) => async_fs_path_is_file(ledger_path_to_fs_path(data_dir, path)).await,
        None => false,
    }
}

async fn async_fs_path_is_file(path: PathBuf) -> bool {
    match fs::metadata(path).await {
        Ok(metadata) => metadata.is_file(),
        Err(_) => false,
    }
}

async fn download_state(data_dir: &Path, row: &VideoRow) -> &'static str {
    if row.downloaded_at.is_none() {
        "-"
    } else if async_ledger_path_is_file(data_dir, row.audio_path.as_deref()).await {
        "done"
    } else {
        "missing"
    }
}

async fn wiki_state(data_dir: &Path, row: &VideoRow) -> &'static str {
    if row.wiki_emitted_at.is_none() {
        "-"
    } else if should_skip_wiki_async(data_dir, row, false).await {
        "done"
    } else {
        "missing"
    }
}

async fn transcript_state(data_dir: &Path, row: &VideoRow) -> &'static str {
    if row.transcribed_at.is_none() {
        "-"
    } else if should_skip_transcription_async(
        data_dir,
        row,
        row.whisper_model.as_deref().unwrap_or(""),
        false,
    )
    .await
    {
        "done"
    } else {
        "missing"
    }
}

async fn run_checked(program: &str, args: &[String]) -> Result<Output> {
    let output = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .output()
        .await
        .with_context(|| format!("run {}", format_command(program, args)))?;
    ensure_success(program, args, output)
}

async fn run_checked_stream_output(program: &str, args: &[String]) -> Result<Output> {
    let command = format_command(program, args);
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("run {command}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("capture stdout for {command}"))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("capture stderr for {command}"))?;

    let stdout_task = AbortOnDrop::new(tokio::spawn(async move {
        let mut captured = Vec::new();
        let mut truncated = false;
        let mut chunk = [0u8; 8192];
        let mut live_stdout = tokio::io::stdout();
        let mut live_stdout_failed = false;

        loop {
            let read = stdout.read(&mut chunk).await?;
            if read == 0 {
                break;
            }
            if !live_stdout_failed && let Err(err) = live_stdout.write_all(&chunk[..read]).await {
                live_stdout_failed = true;
                warn!(stream = "stdout", error = %err, "failed to write child output to live stream");
            }
            truncated |= push_captured_output(&mut captured, &chunk[..read]);
        }
        if !live_stdout_failed && let Err(err) = live_stdout.flush().await {
            warn!(stream = "stdout", error = %err, "failed to flush child output live stream");
        }

        if truncated {
            add_truncation_notice(&mut captured, "stdout");
        }

        Ok::<Vec<u8>, std::io::Error>(captured)
    }));
    let stderr_task = AbortOnDrop::new(tokio::spawn(async move {
        let mut captured = Vec::new();
        let mut truncated = false;
        let mut chunk = [0u8; 8192];
        let mut live_stderr = tokio::io::stderr();
        let mut live_stderr_failed = false;

        loop {
            let read = stderr.read(&mut chunk).await?;
            if read == 0 {
                break;
            }
            if !live_stderr_failed && let Err(err) = live_stderr.write_all(&chunk[..read]).await {
                live_stderr_failed = true;
                warn!(stream = "stderr", error = %err, "failed to write child output to live stream");
            }
            truncated |= push_captured_output(&mut captured, &chunk[..read]);
        }
        if !live_stderr_failed && let Err(err) = live_stderr.flush().await {
            warn!(stream = "stderr", error = %err, "failed to flush child output live stream");
        }

        if truncated {
            add_truncation_notice(&mut captured, "stderr");
        }

        Ok::<Vec<u8>, std::io::Error>(captured)
    }));

    let status = child
        .wait()
        .await
        .with_context(|| format!("wait for {command}"))?;
    let stdout = join_stream_reader(stdout_task, "stdout", &command).await?;
    let stderr = join_stream_reader(stderr_task, "stderr", &command).await?;

    let output = Output {
        status,
        stdout,
        stderr,
    };
    ensure_success(program, args, output)
}

async fn join_stream_reader(
    task: AbortOnDrop<std::io::Result<Vec<u8>>>,
    stream_name: &str,
    command: &str,
) -> Result<Vec<u8>> {
    tokio::time::timeout(STREAM_READER_DRAIN_TIMEOUT, task.join())
        .await
        .with_context(|| format!("timed out draining {stream_name} from {command}"))?
        .with_context(|| format!("join child {stream_name} reader"))?
        .with_context(|| format!("stream {stream_name} from {command}"))
}

fn push_captured_output(captured: &mut Vec<u8>, chunk: &[u8]) -> bool {
    if chunk.len() > STREAMED_OUTPUT_CAPTURE_LIMIT {
        captured.clear();
        captured.extend_from_slice(&chunk[chunk.len() - STREAMED_OUTPUT_CAPTURE_LIMIT..]);
        return true;
    }

    let mut truncated = false;
    if captured.len() + chunk.len() > STREAMED_OUTPUT_CAPTURE_LIMIT {
        let excess = captured.len() + chunk.len() - STREAMED_OUTPUT_CAPTURE_LIMIT;
        captured.drain(..excess);
        truncated = true;
    }
    captured.extend_from_slice(chunk);
    truncated
}

fn add_truncation_notice(captured: &mut Vec<u8>, stream_name: &str) {
    let mut notice =
        format!("[{stream_name} truncated to last {STREAMED_OUTPUT_CAPTURE_LIMIT} bytes]\n")
            .into_bytes();
    let payload_limit = STREAMED_OUTPUT_CAPTURE_LIMIT.saturating_sub(notice.len());
    if captured.len() > payload_limit {
        let excess = captured.len() - payload_limit;
        captured.drain(..excess);
    }
    notice.append(captured);
    *captured = notice;
}

fn ensure_success(program: &str, args: &[String], output: Output) -> Result<Output> {
    if output.status.success() {
        Ok(output)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stdout = stdout.trim();
        let mut details = Vec::new();
        if !stderr.is_empty() {
            details.push(format!("stderr: {stderr}"));
        }
        if !stdout.is_empty() {
            details.push(format!("stdout: {stdout}"));
        }
        let details = if details.is_empty() {
            String::new()
        } else {
            format!(": {}", details.join("; "))
        };
        bail!(
            "{} exited with {}{}",
            format_command(program, args),
            output.status,
            details,
        );
    }
}

fn split_command_prefix(command: &str) -> Result<(String, Vec<String>)> {
    let parts = shell_words::split(command).context("parse whisper command prefix")?;
    let (program, args) = parts
        .split_first()
        .ok_or_else(|| anyhow!("whisper command must not be empty"))?;
    Ok((program.to_owned(), args.to_vec()))
}

fn validate_whisper_config(bin: &str, extra_args: &[String]) -> Result<()> {
    let (_, prefix_args) = split_command_prefix(bin)?;
    validate_whisper_args_from("--whisper-bin", &prefix_args)?;
    validate_whisper_args(extra_args)
}

fn validate_whisper_args(args: &[String]) -> Result<()> {
    validate_whisper_args_from("--whisper-arg", args)
}

fn validate_whisper_args_from(source: &str, args: &[String]) -> Result<()> {
    for arg in args {
        let option = arg
            .split_once('=')
            .map_or(arg.as_str(), |(option, _)| option);
        if matches!(
            option,
            "-o" | "--output_dir" | "--output-dir" | "--output_format" | "--output-format"
        ) || is_combined_short_output_dir_arg(option)
        {
            bail!(
                "{source} includes {option}, which is managed by youtube-archiver; use --data-dir for archive location"
            );
        }
        if option == "--model" {
            bail!(
                "{source} includes {option}, which is managed by youtube-archiver; use --whisper-model instead"
            );
        }
    }
    Ok(())
}

fn is_combined_short_output_dir_arg(option: &str) -> bool {
    option.starts_with("-o") && !option.starts_with("--") && option.len() > 2
}

fn format_command(program: &str, args: &[String]) -> String {
    if args.is_empty() {
        program.to_owned()
    } else {
        format!("{program} {}", args.join(" "))
    }
}

async fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .with_context(|| format!("create {}", parent.display()))?;
    }
    cleanup_stale_temp_files_for(path).await?;

    let tmp_path = temp_path_for(path)?;
    let mut file = fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("create temp file {}", tmp_path.display()))?;
    if let Err(err) = file.write_all(bytes).await {
        let _ = fs::remove_file(&tmp_path).await;
        return Err(err).with_context(|| format!("write {}", tmp_path.display()));
    }
    if let Err(err) = file.sync_all().await {
        let _ = fs::remove_file(&tmp_path).await;
        return Err(err).with_context(|| format!("sync {}", tmp_path.display()));
    }
    drop(file);

    if let Err(err) = fs::rename(&tmp_path, path).await {
        let _ = fs::remove_file(&tmp_path).await;
        return Err(err).with_context(|| format!("rename temp file to {}", path.display()));
    }

    sync_parent_dir(path).await?;

    Ok(())
}

async fn sync_parent_dir(path: &Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };

    #[cfg(unix)]
    {
        let dir = fs::File::open(parent)
            .await
            .with_context(|| format!("open parent directory {}", parent.display()))?;
        dir.sync_all()
            .await
            .with_context(|| format!("sync parent directory {}", parent.display()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = parent;
    }

    Ok(())
}

async fn find_audio_file(tmp_dir: &Path, audio_format: &str) -> Result<PathBuf> {
    let expected_extension = expected_audio_extension(audio_format);
    let mut wrong_format = None;
    let mut entries = fs::read_dir(tmp_dir)
        .await
        .with_context(|| format!("read {}", tmp_dir.display()))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .with_context(|| format!("read entry in {}", tmp_dir.display()))?
    {
        let path = entry.path();
        if !entry
            .file_type()
            .await
            .with_context(|| format!("stat {}", path.display()))?
            .is_file()
        {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("audio.") {
            continue;
        }

        let extension = path.extension().and_then(|ext| ext.to_str());
        if expected_extension
            .as_deref()
            .is_none_or(|expected| extension.is_some_and(|ext| ext.eq_ignore_ascii_case(expected)))
        {
            return Ok(path);
        }
        wrong_format.get_or_insert(path);
    }

    if let (Some(path), Some(expected_extension)) = (wrong_format, expected_extension.as_deref()) {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("audio file");
        bail!(
            "yt-dlp produced {file_name} but not requested audio.{expected_extension} in {}",
            tmp_dir.display()
        );
    }

    if let Some(expected_extension) = expected_extension {
        bail!(
            "yt-dlp did not produce an audio.{expected_extension} file in {}",
            tmp_dir.display()
        )
    } else {
        bail!(
            "yt-dlp did not produce an audio file in {}",
            tmp_dir.display()
        )
    }
}

async fn remove_stale_audio_files(media_dir: &Path, keep_path: &Path) -> Result<()> {
    let mut entries = fs::read_dir(media_dir)
        .await
        .with_context(|| format!("read {}", media_dir.display()))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .with_context(|| format!("read entry in {}", media_dir.display()))?
    {
        let path = entry.path();
        if path == keep_path {
            continue;
        }
        if !entry
            .file_type()
            .await
            .with_context(|| format!("stat {}", path.display()))?
            .is_file()
        {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with("audio.") {
            fs::remove_file(&path)
                .await
                .with_context(|| format!("remove stale audio {}", path.display()))?;
        }
    }

    Ok(())
}

async fn remove_stale_wiki_article(
    data_dir: &Path,
    previous_path: Option<&str>,
    current_path: &Path,
) -> Result<()> {
    let Some(previous_path) = previous_path else {
        return Ok(());
    };
    let previous_path = normalize_ledger_path_string(data_dir, previous_path)?;
    let current_path = path_to_ledger_string(data_dir, current_path)?;
    if previous_path == current_path {
        return Ok(());
    }

    let previous_path = ledger_path_to_fs_path(data_dir, &previous_path);
    match fs::metadata(&previous_path).await {
        Ok(metadata) if metadata.is_file() => {
            fs::remove_file(&previous_path).await.with_context(|| {
                format!("remove stale wiki article {}", previous_path.display())
            })?;
            if let Some(parent) = previous_path.parent()
                && let Err(err) = fs::remove_dir(parent).await
                && err.kind() != std::io::ErrorKind::NotFound
                && err.kind() != std::io::ErrorKind::DirectoryNotEmpty
            {
                warn!(path = %parent.display(), error = %err, "failed to remove empty wiki directory");
            }
        }
        Ok(_) => {
            warn!(path = %previous_path.display(), "stale wiki path is not a regular file");
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(err)
                .with_context(|| format!("stat stale wiki article {}", previous_path.display()));
        }
    }

    Ok(())
}

async fn cleanup_stage_temp_dirs(parent: &Path, prefix: &str) -> Result<()> {
    let mut entries = fs::read_dir(parent)
        .await
        .with_context(|| format!("read {}", parent.display()))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .with_context(|| format!("read entry in {}", parent.display()))?
    {
        let path = entry.path();
        if !entry
            .file_type()
            .await
            .with_context(|| format!("stat {}", path.display()))?
            .is_dir()
        {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if should_remove_stage_temp_path(&path, file_name, prefix).await? {
            fs::remove_dir_all(&path)
                .await
                .with_context(|| format!("remove stale temp dir {}", path.display()))?;
        }
    }

    Ok(())
}

async fn cleanup_stale_temp_files_for(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("{} has no file name", path.display()))?;
    cleanup_stale_temp_files(parent, &format!(".{file_name}")).await
}

async fn cleanup_stale_temp_files(parent: &Path, prefix: &str) -> Result<()> {
    let mut entries = fs::read_dir(parent)
        .await
        .with_context(|| format!("read {}", parent.display()))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .with_context(|| format!("read entry in {}", parent.display()))?
    {
        let path = entry.path();
        if !entry
            .file_type()
            .await
            .with_context(|| format!("stat {}", path.display()))?
            .is_file()
        {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if should_remove_stage_temp_path(&path, file_name, prefix).await? {
            fs::remove_file(&path)
                .await
                .with_context(|| format!("remove stale temp file {}", path.display()))?;
        }
    }

    Ok(())
}

async fn should_remove_stage_temp_path(path: &Path, file_name: &str, prefix: &str) -> Result<bool> {
    let Some(pid) = stage_temp_path_pid(file_name, prefix) else {
        return Ok(false);
    };
    if pid == std::process::id() {
        return Ok(false);
    }

    should_remove_stage_temp_path_for_pid(path, pid).await
}

fn stage_temp_path_pid(file_name: &str, prefix: &str) -> Option<u32> {
    let rest = file_name.strip_prefix(prefix)?.strip_prefix('.')?;
    let rest = rest.strip_suffix(".tmp")?;
    let mut parts = rest.split('.');
    let pid = parts.next()?.parse::<u32>().ok()?;
    let mut has_timestamp = false;
    for part in parts {
        has_timestamp = true;
        if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
    }
    has_timestamp.then_some(pid)
}

#[cfg(target_os = "linux")]
async fn should_remove_stage_temp_path_for_pid(_path: &Path, pid: u32) -> Result<bool> {
    Ok(!process_matches_current_exe(pid))
}

#[cfg(not(target_os = "linux"))]
async fn should_remove_stage_temp_path_for_pid(path: &Path, _pid: u32) -> Result<bool> {
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat temp path {}", path.display()))?;
    Ok(metadata
        .modified()
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age >= NON_LINUX_STALE_TEMP_AGE))
}

#[cfg(target_os = "linux")]
fn process_matches_current_exe(pid: u32) -> bool {
    let proc_exe = Path::new("/proc").join(pid.to_string()).join("exe");
    let Ok(process_exe) = std::fs::read_link(proc_exe) else {
        return false;
    };
    let Ok(current_exe) = std::env::current_exe() else {
        return true;
    };
    process_exe == current_exe
}

async fn find_whisper_outputs(tmp_dir: &Path, preferred_stem: &str) -> Result<(PathBuf, PathBuf)> {
    let mut candidates: Vec<(String, Option<PathBuf>, Option<PathBuf>)> = Vec::new();
    let mut entries = fs::read_dir(tmp_dir)
        .await
        .with_context(|| format!("read {}", tmp_dir.display()))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .with_context(|| format!("read entry in {}", tmp_dir.display()))?
    {
        let path = entry.path();
        if !entry
            .file_type()
            .await
            .with_context(|| format!("stat {}", path.display()))?
            .is_file()
        {
            continue;
        }
        let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        let is_json = extension.eq_ignore_ascii_case("json");
        let is_txt = extension.eq_ignore_ascii_case("txt");
        if !is_json && !is_txt {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };

        let index = match candidates
            .iter()
            .position(|(candidate_stem, _, _)| candidate_stem == stem)
        {
            Some(index) => index,
            None => {
                candidates.push((stem.to_owned(), None, None));
                candidates.len() - 1
            }
        };
        let (_, json, txt) = &mut candidates[index];
        if is_json {
            *json = Some(path);
        } else {
            *txt = Some(path);
        }
    }

    for (stem, json, txt) in &candidates {
        if stem == preferred_stem {
            return match (json, txt) {
                (Some(json), Some(txt)) => Ok((json.clone(), txt.clone())),
                (Some(_), None) => bail!(
                    "whisper produced {preferred_stem}.json without matching {preferred_stem}.txt in {}",
                    tmp_dir.display()
                ),
                (None, Some(_)) => bail!(
                    "whisper produced {preferred_stem}.txt without matching {preferred_stem}.json in {}",
                    tmp_dir.display()
                ),
                (None, None) => unreachable!("preferred stem candidate has no transcript files"),
            };
        }
    }

    let complete_pairs = candidates
        .iter()
        .filter_map(|(_, json, txt)| Some((json.as_ref()?, txt.as_ref()?)))
        .collect::<Vec<_>>();

    match complete_pairs.as_slice() {
        [(json, txt)] => Ok(((*json).clone(), (*txt).clone())),
        [] => bail!(
            "whisper did not produce a matching .json/.txt transcript pair in {}",
            tmp_dir.display()
        ),
        pairs => bail!(
            "whisper produced {} transcript pairs in {} but none matched stem {preferred_stem}",
            pairs.len(),
            tmp_dir.display()
        ),
    }
}

async fn replace_transcript_pair(
    source_json: &Path,
    source_txt: &Path,
    final_json: &Path,
    final_txt: &Path,
) -> Result<()> {
    cleanup_stale_temp_files_for(final_json).await?;
    cleanup_stale_temp_files_for(final_txt).await?;

    let backup_json = temp_path_for(final_json)?;
    let backup_txt = temp_path_for(final_txt)?;
    let mut backed_up_json = false;
    let mut backed_up_txt = false;

    let backup_result: Result<()> = async {
        if fs::try_exists(final_txt).await.unwrap_or(false) {
            fs::rename(final_txt, &backup_txt)
                .await
                .with_context(|| format!("back up {}", final_txt.display()))?;
            backed_up_txt = true;
            sync_parent_dir(final_txt).await?;
        }
        if fs::try_exists(final_json).await.unwrap_or(false) {
            fs::rename(final_json, &backup_json)
                .await
                .with_context(|| format!("back up {}", final_json.display()))?;
            backed_up_json = true;
            sync_parent_dir(final_json).await?;
        }
        Ok(())
    }
    .await;

    if let Err(err) = backup_result {
        if backed_up_json {
            let _ = fs::rename(&backup_json, final_json).await;
        }
        if backed_up_txt {
            let _ = fs::rename(&backup_txt, final_txt).await;
        }
        if let Err(sync_err) = sync_pair_parent_dirs(final_json, final_txt).await {
            warn!(error = %sync_err, "failed to sync transcript directory after backup rollback");
        }
        return Err(err);
    }

    let result: Result<()> = async {
        fs::rename(source_json, final_json)
            .await
            .with_context(|| format!("move transcript JSON to {}", final_json.display()))?;
        sync_parent_dir(final_json).await?;
        fs::rename(source_txt, final_txt)
            .await
            .with_context(|| format!("move transcript text to {}", final_txt.display()))?;
        sync_parent_dir(final_txt).await?;
        Ok(())
    }
    .await;

    if let Err(err) = result {
        let _ = fs::remove_file(final_json).await;
        let _ = fs::remove_file(final_txt).await;
        if backed_up_json {
            let _ = fs::rename(&backup_json, final_json).await;
        }
        if backed_up_txt {
            let _ = fs::rename(&backup_txt, final_txt).await;
        }
        if let Err(sync_err) = sync_pair_parent_dirs(final_json, final_txt).await {
            warn!(error = %sync_err, "failed to sync transcript directory after replacement rollback");
        }
        return Err(err);
    }

    if backed_up_json {
        let _ = fs::remove_file(&backup_json).await;
    }
    if backed_up_txt {
        let _ = fs::remove_file(&backup_txt).await;
    }

    sync_pair_parent_dirs(final_json, final_txt).await?;

    Ok(())
}

async fn sync_pair_parent_dirs(first: &Path, second: &Path) -> Result<()> {
    sync_parent_dir(first).await?;
    if first.parent() != second.parent() {
        sync_parent_dir(second).await?;
    }
    Ok(())
}

fn temp_path_for(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("{} has no file name", path.display()))?;
    Ok(parent.join(unique_temp_name(&format!(".{file_name}"))))
}

fn unique_temp_name(prefix: &str) -> String {
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!("{prefix}.{}.{}.{}.tmp", std::process::id(), nanos, counter)
}

fn yt_dlp_audio_output_template(tmp_dir: &Path) -> String {
    let mut template = path_to_string(tmp_dir).replace('%', "%%");
    if !template.ends_with(std::path::MAIN_SEPARATOR) {
        template.push(std::path::MAIN_SEPARATOR);
    }
    template.push_str("audio.%(ext)s");
    template
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn path_to_ledger_string(data_dir: &Path, path: &Path) -> Result<String> {
    let data_dir = absolutize_path(data_dir)?;
    let path = absolutize_path(path)?;
    if let Ok(relative) = path.strip_prefix(&data_dir) {
        Ok(path_to_string(relative))
    } else {
        Ok(path_to_string(&path))
    }
}

fn normalize_ledger_path_string(data_dir: &Path, path: &str) -> Result<String> {
    let path = Path::new(path);
    if path.is_absolute() {
        path_to_ledger_string(data_dir, path)
    } else {
        Ok(path_to_string(path))
    }
}

fn ledger_path_to_fs_path(data_dir: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        data_dir.join(path)
    }
}

fn absolutize_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .context("read current directory for ledger path")?
            .join(path))
    }
}

fn canonical_video_url(video_id: &str) -> String {
    format!("https://www.youtube.com/watch?v={video_id}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn row_for_skip_tests() -> VideoRow {
        VideoRow {
            video_id: "abc123".to_owned(),
            url: canonical_video_url("abc123"),
            channel_id: None,
            channel_title: None,
            uploader: None,
            title: None,
            upload_date: None,
            duration: None,
            tags: Vec::new(),
            downloaded_at: None,
            transcribed_at: None,
            wiki_emitted_at: None,
            whisper_model: None,
            audio_path: None,
            transcript_path: None,
            wiki_path: None,
            error: None,
        }
    }

    fn write_test_file(path: &Path, bytes: &[u8]) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
        Ok(())
    }

    #[test]
    fn classifies_video_channel_and_playlist_urls() {
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("https://youtu.be/dQw4w9WgXcQ"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("youtu.be/dQw4w9WgXcQ"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/live/dQw4w9WgXcQ?list=PL123"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/v/dQw4w9WgXcQ"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/e/dQw4w9WgXcQ"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/watch?list=PL123&v=dQw4w9WgXcQ"),
            InputMode::Video
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/playlist?list=PL123"),
            InputMode::Playlist
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/@SomeChannel/videos"),
            InputMode::Channel
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/@SomeChannel/shorts"),
            InputMode::Channel
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/@SomeChannel/live"),
            InputMode::Channel
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/@SomeChannel/streams"),
            InputMode::Channel
        );
        assert_eq!(
            classify_youtube_url("https://www.youtube.com/user/SomeUser/videos"),
            InputMode::Channel
        );
    }

    #[test]
    fn resolve_video_ids_args_pass_limit_to_yt_dlp() {
        assert_eq!(
            resolve_video_ids_args(
                "https://www.youtube.com/playlist?list=PL123",
                InputMode::Playlist,
                Some(3)
            ),
            [
                "--flat-playlist",
                "--print",
                "id",
                "--playlist-end",
                "3",
                "--",
                "https://www.youtube.com/playlist?list=PL123"
            ]
        );
    }

    #[test]
    fn resolve_video_ids_args_disable_playlists_for_video_urls() {
        assert_eq!(
            resolve_video_ids_args(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
                InputMode::Video,
                None
            ),
            [
                "--flat-playlist",
                "--print",
                "id",
                "--no-playlist",
                "--",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123"
            ]
        );
    }

    #[test]
    fn collect_resolved_video_ids_rejects_malformed_ids() {
        let ids = collect_valid_resolved_video_ids(
            concat!(
                "dQw4w9WgXcQ\n",
                "../escape\n",
                "abc/defghij\n",
                "abc\0defghij\n",
                "too-short\n",
                "too-long-id12\n",
                "abc1234567_\n"
            ),
            None,
        );

        assert_eq!(ids, ["dQw4w9WgXcQ", "abc1234567_"]);
        assert!(!is_valid_youtube_video_id("../escape"));
        assert!(!is_valid_youtube_video_id("abc/defghij"));
        assert!(!is_valid_youtube_video_id("abc\0defghij"));
    }

    #[test]
    fn collect_resolved_video_ids_does_not_count_invalid_ids_toward_limit() {
        let ids =
            collect_valid_resolved_video_ids("../escape\ndQw4w9WgXcQ\nabc1234567_\n", Some(1));

        assert_eq!(ids, ["dQw4w9WgXcQ"]);
    }

    #[test]
    fn ensure_resolved_videos_inserts_all_rows() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let video_ids = vec!["dQw4w9WgXcQ".to_owned(), "abc1234567_".to_owned()];

        ensure_resolved_videos(&ledger, &video_ids)?;

        let rows = ledger.rows()?;
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row.downloaded_at.is_none()));
        assert!(rows.iter().all(|row| row.transcribed_at.is_none()));
        assert!(rows.iter().all(|row| row.wiki_emitted_at.is_none()));
        assert!(rows.iter().all(|row| row.error.is_none()));
        assert_eq!(rows[0].video_id, "abc1234567_");
        assert_eq!(rows[1].video_id, "dQw4w9WgXcQ");
        Ok(())
    }

    #[test]
    fn rejects_zero_limit() {
        let result = Cli::try_parse_from([
            "youtube-archiver",
            "ingest",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "--limit",
            "0",
        ]);

        assert!(result.is_err());
    }

    #[test]
    fn parses_hyphenated_whisper_args() {
        let cli = Cli::try_parse_from([
            "youtube-archiver",
            "ingest",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "--whisper-arg",
            "--language",
            "--whisper-arg",
            "en",
        ])
        .expect("hyphenated whisper arg should parse");

        let Commands::Ingest(args) = cli.command else {
            panic!("expected ingest command");
        };
        assert_eq!(args.whisper_args, ["--language", "en"]);
    }

    #[test]
    fn slugifies_channel_titles() {
        assert_eq!(slugify("The Rust Channel"), "the-rust-channel");
        assert_eq!(
            slugify("  Rust: Fast, Safe & Productive!  "),
            "rust-fast-safe-productive"
        );
        assert_eq!(
            slugify("\u{041a}\u{0430}\u{043d}\u{0430}\u{043b} Rust"),
            "\u{043a}\u{0430}\u{043d}\u{0430}\u{043b}-rust"
        );
        assert_eq!(slugify("!!!"), "unknown-channel");
    }

    #[test]
    fn renders_frontmatter_and_transcript_body() -> Result<()> {
        let metadata = VideoMetadata {
            video_id: "abc123".to_owned(),
            url: canonical_video_url("abc123"),
            channel_id: Some("channel-id".to_owned()),
            channel_title: Some("Rust Channel".to_owned()),
            uploader: Some("Uploader Name".to_owned()),
            title: Some("A \"quoted\" title".to_owned()),
            upload_date: Some("20260102".to_owned()),
            duration: Some(42),
            tags: vec!["rust".to_owned(), "cli tools".to_owned()],
        };

        let markdown = render_wiki_markdown(&metadata, "hello transcript\n");

        assert!(markdown.starts_with("---\n"));
        assert!(markdown.contains("title: \"A \\\"quoted\\\" title\"\n"));
        assert!(markdown.contains("channel: \"Rust Channel\"\n"));
        assert!(markdown.contains("uploader: \"Uploader Name\"\n"));
        assert!(markdown.contains("upload_date: \"20260102\"\n"));
        assert!(markdown.contains("duration: 42\n"));
        assert!(markdown.contains("url: \"https://www.youtube.com/watch?v=abc123\"\n"));
        assert!(markdown.contains("video_id: \"abc123\"\n"));
        assert!(markdown.contains("  - \"rust\"\n"));
        assert!(markdown.ends_with("\n\nhello transcript\n"));
        Ok(())
    }

    #[test]
    fn renders_channel_id_as_channel_fallback() -> Result<()> {
        let metadata = VideoMetadata {
            video_id: "abc123".to_owned(),
            url: canonical_video_url("abc123"),
            channel_id: Some("UC123".to_owned()),
            channel_title: None,
            uploader: None,
            title: None,
            upload_date: None,
            duration: None,
            tags: Vec::new(),
        };

        let markdown = render_wiki_markdown(&metadata, "hello");
        assert!(markdown.contains("upload_date: null\n"));
        assert!(markdown.contains("duration: null\n"));

        assert!(markdown.contains("channel: \"UC123\"\n"));
        assert!(markdown.contains("uploader: \"UC123\"\n"));
        Ok(())
    }

    #[test]
    fn parses_quoted_whisper_command_prefix() -> Result<()> {
        let (program, args) =
            split_command_prefix(r#""/tmp/bin/whisper tool" --initial_prompt "Alice Bob" --flag"#)?;

        assert_eq!(program, "/tmp/bin/whisper tool");
        assert_eq!(args, ["--initial_prompt", "Alice Bob", "--flag"]);
        Ok(())
    }

    #[test]
    fn rejects_managed_whisper_args_in_command_prefix() {
        let err = validate_whisper_config("whisper --output_dir /tmp/elsewhere", &[])
            .expect_err("output dir in whisper bin should be rejected");

        assert!(format!("{err:#}").contains("--whisper-bin"));
        assert!(format!("{err:#}").contains("--data-dir"));
    }

    #[test]
    fn rejects_whisper_output_args() {
        let args = vec!["--output_dir=/tmp/elsewhere".to_owned()];
        let err = validate_whisper_args(&args).expect_err("output dir should be rejected");

        assert!(format!("{err:#}").contains("--output_dir"));
    }

    #[test]
    fn rejects_whisper_short_output_dir_arg() {
        let args = vec!["-o".to_owned(), "/tmp/elsewhere".to_owned()];
        let err = validate_whisper_args(&args).expect_err("short output dir should be rejected");

        assert!(format!("{err:#}").contains("-o"));
    }

    #[test]
    fn rejects_whisper_combined_short_output_dir_arg() {
        let args = vec!["-omanaged-dir".to_owned()];
        let err =
            validate_whisper_args(&args).expect_err("combined short output dir should be rejected");

        assert!(format!("{err:#}").contains("-omanaged-dir"));
    }

    #[test]
    fn rejects_whisper_model_args() {
        let args = vec!["--model".to_owned(), "base".to_owned()];
        let err = validate_whisper_args(&args).expect_err("model should be rejected");

        assert!(format!("{err:#}").contains("--whisper-model"));
    }

    #[test]
    fn ledger_persists_frontmatter_metadata_fields() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let metadata = VideoMetadata {
            video_id: "abc123".to_owned(),
            url: canonical_video_url("abc123"),
            channel_id: Some("channel-id".to_owned()),
            channel_title: Some("Channel Title".to_owned()),
            uploader: Some("Uploader".to_owned()),
            title: Some("Title".to_owned()),
            upload_date: Some("20260102".to_owned()),
            duration: Some(123),
            tags: vec!["rust".to_owned(), "youtube".to_owned()],
        };

        ledger.upsert_metadata(&metadata)?;
        let row = ledger.row("abc123")?.expect("row exists");

        assert_eq!(row.uploader.as_deref(), Some("Uploader"));
        assert_eq!(row.upload_date.as_deref(), Some("20260102"));
        assert_eq!(row.duration, Some(123));
        assert_eq!(row.tags, vec!["rust".to_owned(), "youtube".to_owned()]);
        Ok(())
    }

    #[tokio::test]
    async fn cached_metadata_invalid_json_is_ignored() -> Result<()> {
        let dir = tempdir()?;
        let info_path = dir.path().join("info.json");
        fs::write(&info_path, b"not json").await?;

        let metadata = load_cached_metadata("dQw4w9WgXcQ", &info_path).await?;

        assert!(metadata.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn cached_metadata_loads_valid_json() -> Result<()> {
        let dir = tempdir()?;
        let info_path = dir.path().join("info.json");
        fs::write(
            &info_path,
            br#"{"title":"A Video","channel":"Rust Channel","tags":["rust"]}"#,
        )
        .await?;

        let metadata = load_cached_metadata("dQw4w9WgXcQ", &info_path)
            .await?
            .expect("valid metadata should load");

        assert_eq!(metadata.video_id, "dQw4w9WgXcQ");
        assert_eq!(metadata.title.as_deref(), Some("A Video"));
        assert_eq!(metadata.channel_title.as_deref(), Some("Rust Channel"));
        assert_eq!(metadata.tags, vec!["rust".to_owned()]);
        Ok(())
    }

    #[test]
    fn ledger_rejects_unknown_migration_columns() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let err = ledger
            .ensure_column("title; DROP TABLE videos")
            .expect_err("unknown migration column should be rejected");

        assert!(format!("{err:#}").contains("unsupported ledger migration column"));
        Ok(())
    }

    #[test]
    fn ledger_rejects_corrupt_tags_json() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        ledger.conn.execute(
            "INSERT INTO videos (video_id, url, tags) VALUES (?1, ?2, ?3)",
            params!["abc123", canonical_video_url("abc123"), "not json"],
        )?;

        let err = ledger.row("abc123").expect_err("corrupt tags should fail");

        assert!(format!("{err:#}").contains("expected ident"));
        Ok(())
    }

    #[test]
    fn ledger_rows_skip_corrupt_tags_json() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        ledger.conn.execute(
            "INSERT INTO videos (video_id, url, tags) VALUES (?1, ?2, ?3)",
            params!["bad", canonical_video_url("bad"), "not json"],
        )?;
        ledger.conn.execute(
            "INSERT INTO videos (video_id, url, title, tags) VALUES (?1, ?2, ?3, ?4)",
            params!["good", canonical_video_url("good"), "Good", "[\"rust\"]"],
        )?;

        let rows = ledger.rows()?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].video_id, "good");
        assert_eq!(rows[0].tags, vec!["rust".to_owned()]);
        Ok(())
    }

    #[tokio::test]
    async fn process_video_clears_stale_error_after_all_stages_skip() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "dQw4w9WgXcQ";
        let media_dir = dir.path().join("media").join(video_id);
        let transcript_dir = dir.path().join("transcripts").join(video_id);
        let wiki_path = dir
            .path()
            .join("wiki")
            .join("rust-channel")
            .join(format!("{video_id}.md"));
        let info_path = media_dir.join("info.json");
        let audio_path = media_dir.join("audio.m4a");
        let transcript_json = transcript_dir.join("transcript.json");
        let transcript_txt = transcript_dir.join("transcript.txt");

        write_test_file(
            &info_path,
            br#"{
                "id": "dQw4w9WgXcQ",
                "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "channel": "Rust Channel",
                "title": "Skipped Success"
            }"#,
        )?;
        write_test_file(&audio_path, b"audio")?;
        write_test_file(&transcript_json, b"{}")?;
        write_test_file(&transcript_txt, b"transcript")?;
        write_test_file(&wiki_path, b"wiki")?;

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_downloaded(video_id, &audio_path)?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;
        ledger.mark_wiki_emitted(video_id, &wiki_path)?;
        ledger.mark_error(video_id, "stale failure")?;

        let args = IngestArgs {
            url: canonical_video_url(video_id),
            data_dir: dir.path().to_path_buf(),
            whisper_model: "large".to_owned(),
            whisper_bin: DEFAULT_WHISPER_BIN.to_owned(),
            whisper_args: Vec::new(),
            limit: None,
            audio_format: "m4a".to_owned(),
            force: false,
        };

        process_video(&args, &ledger, video_id).await?;

        let row = ledger.row(video_id)?.expect("row exists");
        assert!(row.error.is_none());
        Ok(())
    }

    #[test]
    fn ledger_stores_artifact_paths_relative_to_data_dir() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "abc123";
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;

        ledger.mark_downloaded(video_id, &dir.path().join("media/abc123/audio.m4a"))?;
        let row = ledger.row(video_id)?.expect("row exists");

        let audio_path = row.audio_path.expect("audio path is set");
        assert_eq!(audio_path, "media/abc123/audio.m4a");
        Ok(())
    }

    #[test]
    fn marking_transcribed_invalidates_existing_wiki_timestamp_but_preserves_path() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_wiki_emitted(video_id, Path::new("data/wiki/channel/abc123.md"))?;

        let row = ledger.row(video_id)?.expect("row exists");
        assert!(row.wiki_emitted_at.is_some());
        assert!(row.wiki_path.is_some());

        ledger.mark_transcribed(
            video_id,
            "base",
            Path::new("data/transcripts/abc123/transcript.json"),
        )?;
        let row = ledger.row(video_id)?.expect("row exists");

        assert!(row.wiki_emitted_at.is_none());
        assert!(row.wiki_path.is_some());
        Ok(())
    }

    #[test]
    fn marking_downloaded_invalidates_downstream_timestamps() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_transcribed(
            video_id,
            "large",
            Path::new("data/transcripts/abc123/transcript.json"),
        )?;
        ledger.mark_wiki_emitted(video_id, Path::new("data/wiki/channel/abc123.md"))?;

        let row = ledger.row(video_id)?.expect("row exists");
        assert!(row.transcribed_at.is_some());
        assert!(row.wiki_emitted_at.is_some());

        ledger.mark_downloaded(video_id, Path::new("data/media/abc123/audio.opus"))?;
        let row = ledger.row(video_id)?.expect("row exists");

        assert!(row.downloaded_at.is_some());
        assert!(row.transcribed_at.is_none());
        assert!(row.wiki_emitted_at.is_none());
        Ok(())
    }

    #[test]
    fn invalidating_transcription_outputs_disables_transcription_skip() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        let transcript_json = dir.path().join("transcript.json");
        let transcript_txt = dir.path().join("transcript.txt");
        let wiki = dir.path().join("wiki/channel/abc123.md");
        std::fs::write(&transcript_json, b"{}")?;
        std::fs::write(&transcript_txt, b"text")?;

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;
        let row = ledger.row(video_id)?.expect("row exists");
        assert!(should_skip_transcription(dir.path(), &row, "large", false));
        assert!(row.wiki_emitted_at.is_some());

        ledger.invalidate_transcription_outputs(video_id)?;
        let row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_transcription(dir.path(), &row, "large", false));
        assert!(row.wiki_emitted_at.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn forced_transcription_failure_preserves_existing_archive_state() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "dQw4w9WgXcQ";
        let audio = dir.path().join("media/dQw4w9WgXcQ/audio.m4a");
        let transcript_json = dir.path().join("transcripts/dQw4w9WgXcQ/transcript.json");
        let transcript_txt = dir.path().join("transcripts/dQw4w9WgXcQ/transcript.txt");
        let wiki = dir.path().join("wiki/channel/dQw4w9WgXcQ.md");

        write_test_file(&audio, b"audio")?;
        write_test_file(&transcript_json, br#"{"old":true}"#)?;
        write_test_file(&transcript_txt, b"old transcript")?;
        write_test_file(&wiki, b"old wiki")?;

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_downloaded(video_id, &audio)?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;
        let before = ledger.row(video_id)?.expect("row exists");

        let err = transcribe_audio(
            dir.path(),
            &ledger,
            video_id,
            &audio,
            WhisperConfig {
                bin: "sh -c 'exit 9'",
                model: "large",
                extra_args: &[],
            },
            true,
        )
        .await
        .expect_err("forced transcription should fail");

        assert!(format!("{err:#}").contains("exited with"));
        let after = ledger.row(video_id)?.expect("row exists");
        assert_eq!(after.transcribed_at, before.transcribed_at);
        assert_eq!(after.wiki_emitted_at, before.wiki_emitted_at);
        assert_eq!(after.transcript_path, before.transcript_path);
        assert_eq!(after.wiki_path, before.wiki_path);
        assert_eq!(std::fs::read(&transcript_json)?, br#"{"old":true}"#);
        assert_eq!(std::fs::read(&transcript_txt)?, b"old transcript");
        Ok(())
    }

    #[tokio::test]
    async fn finds_single_whisper_output_pair_without_audio_stem() -> Result<()> {
        let dir = tempdir()?;
        let json = dir.path().join("custom-name.json");
        let txt = dir.path().join("custom-name.txt");
        fs::write(&json, b"{}").await?;
        fs::write(&txt, b"text").await?;

        assert_eq!(
            find_whisper_outputs(dir.path(), "audio").await?,
            (json, txt)
        );
        Ok(())
    }

    #[tokio::test]
    async fn whisper_output_pair_fallback_ignores_auxiliary_files() -> Result<()> {
        let dir = tempdir()?;
        let json = dir.path().join("custom-name.json");
        let txt = dir.path().join("custom-name.txt");
        fs::write(&json, br#"{"segments":[]}"#).await?;
        fs::write(&txt, b"text").await?;
        fs::write(dir.path().join("metadata.json"), b"{}").await?;

        assert_eq!(
            find_whisper_outputs(dir.path(), "audio").await?,
            (json, txt)
        );
        Ok(())
    }

    #[tokio::test]
    async fn whisper_output_pair_rejects_ambiguous_fallbacks() -> Result<()> {
        let dir = tempdir()?;
        fs::write(dir.path().join("first.json"), b"{}").await?;
        fs::write(dir.path().join("first.txt"), b"first").await?;
        fs::write(dir.path().join("second.json"), b"{}").await?;
        fs::write(dir.path().join("second.txt"), b"second").await?;

        let err = find_whisper_outputs(dir.path(), "audio")
            .await
            .expect_err("ambiguous transcript pairs should fail");

        assert!(format!("{err:#}").contains("produced 2 transcript pairs"));
        Ok(())
    }

    #[tokio::test]
    async fn whisper_output_pair_rejects_incomplete_preferred_stem() -> Result<()> {
        let dir = tempdir()?;
        fs::write(dir.path().join("audio.json"), b"{}").await?;
        fs::write(dir.path().join("fallback.json"), b"{}").await?;
        fs::write(dir.path().join("fallback.txt"), b"fallback").await?;

        let err = find_whisper_outputs(dir.path(), "audio")
            .await
            .expect_err("incomplete preferred transcript pair should fail");

        assert!(format!("{err:#}").contains("without matching audio.txt"));
        Ok(())
    }

    #[tokio::test]
    async fn find_audio_file_prefers_requested_extension() -> Result<()> {
        let dir = tempdir()?;
        let webm = dir.path().join("audio.webm");
        let m4a = dir.path().join("audio.m4a");
        fs::write(&webm, b"webm").await?;
        fs::write(&m4a, b"m4a").await?;

        assert_eq!(find_audio_file(dir.path(), "m4a").await?, m4a);
        Ok(())
    }

    #[tokio::test]
    async fn find_audio_file_accepts_yt_dlp_format_alias_extensions() -> Result<()> {
        let dir = tempdir()?;
        let m4a = dir.path().join("audio.m4a");
        fs::write(&m4a, b"m4a").await?;

        assert_eq!(find_audio_file(dir.path(), "aac").await?, m4a);

        let dir = tempdir()?;
        let m4a = dir.path().join("audio.m4a");
        fs::write(&m4a, b"m4a").await?;

        assert_eq!(find_audio_file(dir.path(), "alac").await?, m4a);

        let dir = tempdir()?;
        let ogg = dir.path().join("audio.ogg");
        fs::write(&ogg, b"ogg").await?;

        assert_eq!(find_audio_file(dir.path(), "vorbis").await?, ogg);
        Ok(())
    }

    #[tokio::test]
    async fn find_audio_file_rejects_unrequested_extension() -> Result<()> {
        let dir = tempdir()?;
        let webm = dir.path().join("audio.webm");
        fs::write(&webm, b"webm").await?;

        let err = find_audio_file(dir.path(), "m4a")
            .await
            .expect_err("wrong audio extension should be rejected");

        assert!(format!("{err:#}").contains("requested audio.m4a"));
        Ok(())
    }

    #[tokio::test]
    async fn replace_transcript_pair_swaps_both_outputs() -> Result<()> {
        let dir = tempdir()?;
        let source_json = dir.path().join("new.json");
        let source_txt = dir.path().join("new.txt");
        let final_json = dir.path().join("transcript.json");
        let final_txt = dir.path().join("transcript.txt");
        fs::write(&source_json, br#"{"new":true}"#).await?;
        fs::write(&source_txt, b"new text").await?;
        fs::write(&final_json, br#"{"old":true}"#).await?;
        fs::write(&final_txt, b"old text").await?;

        replace_transcript_pair(&source_json, &source_txt, &final_json, &final_txt).await?;

        assert_eq!(fs::read(&final_json).await?, br#"{"new":true}"#);
        assert_eq!(fs::read(&final_txt).await?, b"new text");
        assert!(!fs::try_exists(&source_json).await?);
        assert!(!fs::try_exists(&source_txt).await?);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn cleanup_stage_temp_dirs_removes_previous_stage_dirs() -> Result<()> {
        let dir = tempdir()?;
        let pid = u32::MAX;
        let stale = dir.path().join(format!(".download.{pid}.1.0.tmp"));
        let unrelated_dir = dir.path().join(format!(".whisper.{pid}.1.0.tmp"));
        let ordinary_file = dir.path().join(format!(".download.{pid}.2.0.tmp"));
        fs::create_dir(&stale).await?;
        fs::create_dir(&unrelated_dir).await?;
        fs::write(&ordinary_file, b"not a directory").await?;

        cleanup_stage_temp_dirs(dir.path(), ".download").await?;

        assert!(!fs::try_exists(&stale).await?);
        assert!(fs::try_exists(&unrelated_dir).await?);
        assert!(fs::try_exists(&ordinary_file).await?);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn cleanup_stage_temp_dirs_keeps_current_process_dirs() -> Result<()> {
        let dir = tempdir()?;
        let pid = std::process::id();
        let current = dir.path().join(format!(".download.{pid}.1.0.tmp"));
        fs::create_dir(&current).await?;

        cleanup_stage_temp_dirs(dir.path(), ".download").await?;

        assert!(fs::try_exists(&current).await?);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn process_match_check_uses_current_exe() {
        assert!(process_matches_current_exe(std::process::id()));
        assert!(!process_matches_current_exe(u32::MAX));
    }

    #[test]
    fn yt_dlp_audio_output_template_escapes_literal_percent_paths() {
        let tmp_dir = Path::new("data%dir").join("media").join("id%11");
        let template = yt_dlp_audio_output_template(&tmp_dir);

        assert!(template.contains("data%%dir"));
        assert!(template.contains("id%%11"));
        assert!(template.ends_with(&format!("{}audio.%(ext)s", std::path::MAIN_SEPARATOR)));
    }

    #[test]
    fn captured_stream_output_keeps_last_bytes() {
        let mut captured = vec![b'a'; STREAMED_OUTPUT_CAPTURE_LIMIT - 2];

        assert!(push_captured_output(&mut captured, b"bcde"));
        assert_eq!(captured.len(), STREAMED_OUTPUT_CAPTURE_LIMIT);
        assert_eq!(&captured[STREAMED_OUTPUT_CAPTURE_LIMIT - 4..], b"bcde");
    }

    #[test]
    fn captured_stream_output_exact_limit_is_not_truncated() {
        let mut captured = Vec::new();
        let chunk = vec![b'a'; STREAMED_OUTPUT_CAPTURE_LIMIT];

        assert!(!push_captured_output(&mut captured, &chunk));
        assert_eq!(captured.len(), STREAMED_OUTPUT_CAPTURE_LIMIT);
    }

    #[test]
    fn truncation_notice_keeps_capture_within_limit() {
        let mut captured = vec![b'a'; STREAMED_OUTPUT_CAPTURE_LIMIT];

        add_truncation_notice(&mut captured, "stdout");

        assert!(captured.starts_with(b"[stdout truncated to last "));
        assert_eq!(captured.len(), STREAMED_OUTPUT_CAPTURE_LIMIT);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streamed_command_times_out_when_background_process_keeps_pipe_open() {
        let args = vec!["-c".to_owned(), "sleep 2 & exit 0".to_owned()];

        let err = run_checked_stream_output("sh", &args)
            .await
            .expect_err("leaked stream pipe should time out");

        assert!(format!("{err:#}").contains("timed out draining"));
    }

    #[tokio::test]
    async fn stale_wiki_article_is_removed_when_target_changes() -> Result<()> {
        let dir = tempdir()?;
        let old_wiki = dir.path().join("wiki").join("old").join("abc123.md");
        let new_wiki = dir.path().join("wiki").join("new").join("abc123.md");
        atomic_write(&old_wiki, b"old").await?;
        atomic_write(&new_wiki, b"new").await?;

        remove_stale_wiki_article(dir.path(), Some(&path_to_string(&old_wiki)), &new_wiki).await?;

        assert!(!fs::try_exists(&old_wiki).await?);
        assert!(fs::try_exists(&new_wiki).await?);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn atomic_write_removes_stale_temp_files() -> Result<()> {
        let dir = tempdir()?;
        let target = dir.path().join("info.json");
        let stale = dir.path().join(format!(".info.json.{}.1.0.tmp", u32::MAX));
        fs::write(&stale, b"stale").await?;

        atomic_write(&target, b"fresh").await?;

        assert!(!fs::try_exists(&stale).await?);
        assert_eq!(fs::read(&target).await?, b"fresh");
        Ok(())
    }

    #[tokio::test]
    async fn emit_wiki_article_reemits_when_channel_slug_changes() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "abc123";
        let transcript_txt = dir
            .path()
            .join("transcripts")
            .join(video_id)
            .join("transcript.txt");
        let old_wiki = dir
            .path()
            .join("wiki")
            .join("old-channel")
            .join(format!("{video_id}.md"));
        atomic_write(&transcript_txt, b"hello transcript").await?;
        atomic_write(&old_wiki, b"old article").await?;
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_wiki_emitted(video_id, &old_wiki)?;

        let metadata = VideoMetadata {
            video_id: video_id.to_owned(),
            url: canonical_video_url(video_id),
            channel_id: None,
            channel_title: Some("New Channel".to_owned()),
            uploader: None,
            title: Some("Title".to_owned()),
            upload_date: None,
            duration: None,
            tags: Vec::new(),
        };

        let new_wiki = emit_wiki_article(dir.path(), &ledger, &metadata, false).await?;

        assert_eq!(
            new_wiki,
            dir.path()
                .join("wiki")
                .join("new-channel")
                .join(format!("{video_id}.md"))
        );
        assert!(!fs::try_exists(&old_wiki).await?);
        assert!(fs::try_exists(&new_wiki).await?);
        Ok(())
    }

    #[test]
    fn list_rows_only_include_completed_archived_videos() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let complete_id = "dQw4w9WgXcQ";
        let failed_id = "abc1234567_";
        let complete_audio = dir.path().join("media/dQw4w9WgXcQ/audio.m4a");
        let complete_transcript_json = dir.path().join("transcripts/dQw4w9WgXcQ/transcript.json");
        let complete_transcript_txt = dir.path().join("transcripts/dQw4w9WgXcQ/transcript.txt");
        let complete_wiki = dir.path().join("wiki/channel/dQw4w9WgXcQ.md");

        write_test_file(&complete_audio, b"audio")?;
        write_test_file(&complete_transcript_json, b"{}")?;
        write_test_file(&complete_transcript_txt, b"transcript")?;
        write_test_file(&complete_wiki, b"wiki")?;

        ledger.ensure_video(complete_id, &canonical_video_url(complete_id))?;
        ledger.mark_downloaded(complete_id, &complete_audio)?;
        ledger.mark_transcribed(complete_id, "large", &complete_transcript_json)?;
        ledger.mark_wiki_emitted(complete_id, &complete_wiki)?;
        ledger.ensure_video(failed_id, &canonical_video_url(failed_id))?;
        ledger.mark_error(failed_id, "download failed")?;
        drop(ledger);

        let rows = list_rows(dir.path())?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].video_id, complete_id);
        Ok(())
    }

    #[test]
    fn list_rows_require_archived_artifacts_to_exist() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "dQw4w9WgXcQ";
        let audio = dir.path().join("media/dQw4w9WgXcQ/audio.m4a");
        let transcript_json = dir.path().join("transcripts/dQw4w9WgXcQ/transcript.json");
        let wiki = dir.path().join("wiki/channel/dQw4w9WgXcQ.md");

        write_test_file(&audio, b"audio")?;
        write_test_file(&transcript_json, b"{}")?;
        write_test_file(&wiki, b"wiki")?;

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_downloaded(video_id, &audio)?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;
        drop(ledger);

        let rows = list_rows(dir.path())?;

        assert!(rows.is_empty());
        Ok(())
    }

    #[test]
    fn ledger_skip_logic_requires_timestamps_and_existing_files() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;

        let mut row = row_for_skip_tests();
        assert!(!should_skip_download(dir.path(), &row, "m4a", false));

        let missing_audio = dir.path().join("missing.m4a");
        ledger.mark_downloaded(video_id, &missing_audio)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_download(dir.path(), &row, "m4a", false));

        let audio_dir = dir.path().join("audio-dir.m4a");
        std::fs::create_dir(&audio_dir)?;
        ledger.mark_downloaded(video_id, &audio_dir)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_download(dir.path(), &row, "m4a", false));

        let audio = dir.path().join("audio.m4a");
        std::fs::write(&audio, b"audio")?;
        ledger.mark_downloaded(video_id, &audio)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(should_skip_download(dir.path(), &row, "m4a", false));
        assert!(should_skip_download(dir.path(), &row, "aac", false));
        assert!(should_skip_download(dir.path(), &row, "alac", false));
        assert!(should_skip_download(dir.path(), &row, "M4A", false));
        assert!(!should_skip_download(dir.path(), &row, "opus", false));
        assert!(!should_skip_download(dir.path(), &row, "m4a", true));

        let transcript_json = dir.path().join("transcript.json");
        std::fs::write(&transcript_json, b"{}")?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_transcription(dir.path(), &row, "large", false));

        let transcript_json_dir = dir.path().join("transcript-dir.json");
        let transcript_txt_dir = dir.path().join("transcript.txt");
        std::fs::create_dir(&transcript_json_dir)?;
        std::fs::create_dir(&transcript_txt_dir)?;
        ledger.mark_transcribed(video_id, "large", &transcript_json_dir)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_transcription(dir.path(), &row, "large", false));
        std::fs::remove_dir(&transcript_json_dir)?;
        std::fs::remove_dir(&transcript_txt_dir)?;

        let transcript_txt = dir.path().join("transcript.txt");
        std::fs::write(&transcript_txt, b"text")?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(should_skip_transcription(dir.path(), &row, "large", false));
        assert!(!should_skip_transcription(dir.path(), &row, "base", false));
        assert!(!should_skip_transcription(dir.path(), &row, "large", true));

        let wiki_dir = dir.path().join("wiki-dir.md");
        std::fs::create_dir(&wiki_dir)?;
        ledger.mark_wiki_emitted(video_id, &wiki_dir)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_wiki(dir.path(), &row, false));
        std::fs::remove_dir(&wiki_dir)?;

        let wiki = dir.path().join("wiki.md");
        std::fs::write(&wiki, b"wiki")?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(should_skip_wiki(dir.path(), &row, false));
        assert!(!should_skip_wiki(dir.path(), &row, true));

        Ok(())
    }
}
