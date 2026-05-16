use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use regex::Regex;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{fs, io::AsyncWriteExt, process::Command};
use tracing::{error, info};

const DEFAULT_DATA_DIR: &str = "data";
const DEFAULT_WHISPER_BIN: &str = "nix run nixpkgs#openai-whisper --";
const DEFAULT_WHISPER_MODEL: &str = "large";
const DEFAULT_AUDIO_FORMAT: &str = "m4a";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

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
    /// List ledger rows as JSON.
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

    #[arg(long)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoRow {
    video_id: String,
    url: String,
    channel_id: Option<String>,
    channel_title: Option<String>,
    title: Option<String>,
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
}

impl Ledger {
    fn open(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .with_context(|| format!("create data dir {}", data_dir.display()))?;
        let db_path = data_dir.join("state.sqlite");
        let conn = Connection::open(&db_path)
            .with_context(|| format!("open ledger {}", db_path.display()))?;
        let ledger = Self { conn };
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
        Ok(Some(Self { conn }))
    }

    #[cfg(test)]
    fn open_in_memory() -> Result<Self> {
        let ledger = Self {
            conn: Connection::open_in_memory().context("open in-memory ledger")?,
        };
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
                    title TEXT,
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
        self.conn
            .execute(
                r#"
                INSERT INTO videos (video_id, url, channel_id, channel_title, title)
                VALUES (?1, ?2, ?3, ?4, ?5)
                ON CONFLICT(video_id) DO UPDATE SET
                    url = excluded.url,
                    channel_id = excluded.channel_id,
                    channel_title = excluded.channel_title,
                    title = excluded.title
                "#,
                params![
                    metadata.video_id,
                    metadata.url,
                    metadata.channel_id,
                    metadata.channel_title,
                    metadata.title
                ],
            )
            .with_context(|| format!("upsert metadata for {}", metadata.video_id))?;
        Ok(())
    }

    fn mark_downloaded(&self, video_id: &str, audio_path: &Path) -> Result<()> {
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET downloaded_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    audio_path = ?2,
                    error = NULL
                WHERE video_id = ?1
                "#,
                params![video_id, path_to_string(audio_path)],
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
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET transcribed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    whisper_model = ?2,
                    transcript_path = ?3,
                    error = NULL
                WHERE video_id = ?1
                "#,
                params![video_id, whisper_model, path_to_string(transcript_path)],
            )
            .with_context(|| format!("mark {video_id} transcribed"))?;
        Ok(())
    }

    fn mark_wiki_emitted(&self, video_id: &str, wiki_path: &Path) -> Result<()> {
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET wiki_emitted_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    wiki_path = ?2,
                    error = NULL
                WHERE video_id = ?1
                "#,
                params![video_id, path_to_string(wiki_path)],
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

    fn row(&self, video_id: &str) -> Result<Option<VideoRow>> {
        self.conn
            .query_row(
                r#"
                SELECT video_id, url, channel_id, channel_title, title,
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
                SELECT video_id, url, channel_id, channel_title, title,
                       downloaded_at, transcribed_at, wiki_emitted_at,
                       whisper_model, audio_path, transcript_path, wiki_path, error
                FROM videos
                ORDER BY video_id
                "#,
            )
            .context("prepare ledger list query")?;
        let rows = stmt
            .query_map([], row_from_sql)
            .context("query ledger rows")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("read ledger rows")?;
        Ok(rows)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match Cli::parse().command {
        Commands::Ingest(args) => ingest(args).await,
        Commands::Status(args) => status(args),
        Commands::List(args) => list(args),
    }
}

async fn ingest(args: IngestArgs) -> Result<()> {
    let ledger = Ledger::open(&args.data_dir)?;
    let mode = classify_youtube_url(&args.url);
    info!(?mode, url = %args.url, "resolving input URL");
    let video_ids = resolve_video_ids(&args.url, args.limit).await?;

    if video_ids.is_empty() {
        bail!("yt-dlp did not return any video IDs for {}", args.url);
    }

    let mut succeeded = 0usize;
    let mut failed = 0usize;

    for video_id in video_ids {
        let url = canonical_video_url(&video_id);
        ledger.ensure_video(&video_id, &url)?;

        match process_video(&args, &ledger, &video_id).await {
            Ok(()) => {
                succeeded += 1;
                info!(%video_id, "video processed");
            }
            Err(err) => {
                failed += 1;
                let message = format!("{err:#}");
                error!(%video_id, error = %message, "video failed");
                ledger.mark_error(&video_id, &message)?;
            }
        }
    }

    if succeeded == 0 {
        bail!("every video failed ({failed} failure(s))");
    }

    Ok(())
}

async fn process_video(args: &IngestArgs, ledger: &Ledger, video_id: &str) -> Result<()> {
    let metadata = load_or_fetch_metadata(&args.data_dir, ledger, video_id, args.force).await?;
    ledger.upsert_metadata(&metadata)?;

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
        &args.whisper_bin,
        &args.whisper_model,
        args.force,
    )
    .await
    .with_context(|| format!("transcribe {video_id}"))?;

    emit_wiki_article(&args.data_dir, ledger, &metadata, args.force)
        .await
        .with_context(|| format!("emit wiki markdown for {video_id}"))?;

    Ok(())
}

async fn resolve_video_ids(url: &str, limit: Option<usize>) -> Result<Vec<String>> {
    let args = vec![
        "--flat-playlist".to_owned(),
        "--print".to_owned(),
        "id".to_owned(),
        url.to_owned(),
    ];
    let output = run_checked("yt-dlp", &args).await?;
    let stdout = String::from_utf8(output.stdout).context("yt-dlp emitted non-UTF8 video IDs")?;
    let mut seen = HashSet::new();
    let mut ids = Vec::new();

    for line in stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if seen.insert(line.to_owned()) {
            ids.push(line.to_owned());
        }
        if limit.is_some_and(|max| ids.len() >= max) {
            break;
        }
    }

    Ok(ids)
}

async fn load_or_fetch_metadata(
    data_dir: &Path,
    ledger: &Ledger,
    video_id: &str,
    force: bool,
) -> Result<VideoMetadata> {
    let media_dir = data_dir.join("media").join(video_id);
    let info_path = media_dir.join("info.json");

    if !force && info_path.exists() {
        let bytes = fs::read(&info_path)
            .await
            .with_context(|| format!("read existing {}", info_path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse existing {}", info_path.display()))?;
        return Ok(metadata_from_value(video_id, &value));
    }

    fs::create_dir_all(&media_dir)
        .await
        .with_context(|| format!("create {}", media_dir.display()))?;

    let url = canonical_video_url(video_id);
    let args = vec!["-j".to_owned(), "--no-playlist".to_owned(), url];
    let output = run_checked("yt-dlp", &args).await?;
    atomic_write(&info_path, &output.stdout).await?;
    let value: Value = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parse yt-dlp metadata for {video_id}"))?;
    let metadata = metadata_from_value(video_id, &value);
    ledger.upsert_metadata(&metadata)?;
    Ok(metadata)
}

async fn download_audio(
    data_dir: &Path,
    ledger: &Ledger,
    video_id: &str,
    audio_format: &str,
    force: bool,
) -> Result<PathBuf> {
    if let Some(row) = ledger
        .row(video_id)?
        .filter(|row| should_skip_download(row, force))
    {
        return Ok(PathBuf::from(
            row.audio_path.expect("checked by should_skip_download"),
        ));
    }

    let media_dir = data_dir.join("media").join(video_id);
    fs::create_dir_all(&media_dir)
        .await
        .with_context(|| format!("create {}", media_dir.display()))?;
    let tmp_dir = media_dir.join(unique_temp_name(".download"));
    fs::create_dir_all(&tmp_dir)
        .await
        .with_context(|| format!("create {}", tmp_dir.display()))?;

    let output_template = tmp_dir.join("audio.%(ext)s");
    let url = canonical_video_url(video_id);
    let args = vec![
        "-f".to_owned(),
        "bestaudio".to_owned(),
        "--extract-audio".to_owned(),
        "--audio-format".to_owned(),
        audio_format.to_owned(),
        "--no-playlist".to_owned(),
        "-o".to_owned(),
        path_to_string(&output_template),
        url,
    ];

    let result = run_checked("yt-dlp", &args).await;
    if let Err(err) = result {
        let _ = fs::remove_dir_all(&tmp_dir).await;
        return Err(err);
    }

    let downloaded = find_audio_file(&tmp_dir, audio_format).await?;
    let extension = downloaded
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or(audio_format);
    let final_path = media_dir.join(format!("audio.{extension}"));
    fs::rename(&downloaded, &final_path)
        .await
        .with_context(|| format!("move audio to {}", final_path.display()))?;
    let _ = fs::remove_dir_all(&tmp_dir).await;
    ledger.mark_downloaded(video_id, &final_path)?;
    Ok(final_path)
}

async fn transcribe_audio(
    data_dir: &Path,
    ledger: &Ledger,
    video_id: &str,
    audio_path: &Path,
    whisper_bin: &str,
    whisper_model: &str,
    force: bool,
) -> Result<PathBuf> {
    if let Some(row) = ledger
        .row(video_id)?
        .filter(|row| should_skip_transcription(row, whisper_model, force))
    {
        return Ok(PathBuf::from(
            row.transcript_path
                .expect("checked by should_skip_transcription"),
        ));
    }

    let transcript_dir = data_dir.join("transcripts").join(video_id);
    fs::create_dir_all(&transcript_dir)
        .await
        .with_context(|| format!("create {}", transcript_dir.display()))?;
    let tmp_dir = transcript_dir.join(unique_temp_name(".whisper"));
    fs::create_dir_all(&tmp_dir)
        .await
        .with_context(|| format!("create {}", tmp_dir.display()))?;

    let (program, mut args) = split_command_prefix(whisper_bin)?;
    args.extend([
        path_to_string(audio_path),
        "--model".to_owned(),
        whisper_model.to_owned(),
        "--output_dir".to_owned(),
        path_to_string(&tmp_dir),
        "--output_format".to_owned(),
        "all".to_owned(),
    ]);

    let result = run_checked(&program, &args).await;
    if let Err(err) = result {
        let _ = fs::remove_dir_all(&tmp_dir).await;
        return Err(err);
    }

    let output_stem = audio_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("audio");
    let whisper_json = tmp_dir.join(format!("{output_stem}.json"));
    let whisper_txt = tmp_dir.join(format!("{output_stem}.txt"));
    let final_json = transcript_dir.join("transcript.json");
    let final_txt = transcript_dir.join("transcript.txt");

    if !whisper_json.exists() || !whisper_txt.exists() {
        let _ = fs::remove_dir_all(&tmp_dir).await;
        bail!(
            "whisper completed but did not produce {} and {}",
            whisper_json.display(),
            whisper_txt.display()
        );
    }

    fs::rename(&whisper_json, &final_json)
        .await
        .with_context(|| format!("move transcript JSON to {}", final_json.display()))?;
    fs::rename(&whisper_txt, &final_txt)
        .await
        .with_context(|| format!("move transcript text to {}", final_txt.display()))?;
    let _ = fs::remove_dir_all(&tmp_dir).await;
    ledger.mark_transcribed(video_id, whisper_model, &final_json)?;
    Ok(final_json)
}

async fn emit_wiki_article(
    data_dir: &Path,
    ledger: &Ledger,
    metadata: &VideoMetadata,
    force: bool,
) -> Result<PathBuf> {
    if let Some(row) = ledger
        .row(&metadata.video_id)?
        .filter(|row| should_skip_wiki(row, force))
    {
        return Ok(PathBuf::from(
            row.wiki_path.expect("checked by should_skip_wiki"),
        ));
    }

    let transcript_txt = data_dir
        .join("transcripts")
        .join(&metadata.video_id)
        .join("transcript.txt");
    let transcript = fs::read_to_string(&transcript_txt)
        .await
        .with_context(|| format!("read {}", transcript_txt.display()))?;

    let channel_slug = slugify(
        metadata
            .channel_title
            .as_deref()
            .or(metadata.uploader.as_deref())
            .or(metadata.channel_id.as_deref())
            .unwrap_or("unknown-channel"),
    );
    let wiki_path = data_dir
        .join("wiki")
        .join(channel_slug)
        .join(format!("{}.md", metadata.video_id));
    let article = render_wiki_markdown(metadata, &transcript)?;
    atomic_write(&wiki_path, article.as_bytes()).await?;
    ledger.mark_wiki_emitted(&metadata.video_id, &wiki_path)?;
    Ok(wiki_path)
}

fn status(args: DataDirArgs) -> Result<()> {
    let rows = Ledger::open_read_only(&args.data_dir)?
        .map(|ledger| ledger.rows())
        .transpose()?
        .unwrap_or_default();

    println!(
        "{:<14} {:<10} {:<11} {:<10} {:<7} title",
        "video_id", "download", "transcribe", "wiki", "error"
    );
    for row in rows {
        println!(
            "{:<14} {:<10} {:<11} {:<10} {:<7} {}",
            row.video_id,
            stage_state(&row.downloaded_at, &row.audio_path),
            transcript_state(&row),
            stage_state(&row.wiki_emitted_at, &row.wiki_path),
            if row.error.is_some() { "yes" } else { "-" },
            row.title.as_deref().unwrap_or("-")
        );
    }

    Ok(())
}

fn list(args: DataDirArgs) -> Result<()> {
    let rows = Ledger::open_read_only(&args.data_dir)?
        .map(|ledger| ledger.rows())
        .transpose()?
        .unwrap_or_default();
    println!(
        "{}",
        serde_json::to_string_pretty(&rows).context("serialize ledger rows")?
    );
    Ok(())
}

fn row_from_sql(row: &rusqlite::Row<'_>) -> rusqlite::Result<VideoRow> {
    Ok(VideoRow {
        video_id: row.get(0)?,
        url: row.get(1)?,
        channel_id: row.get(2)?,
        channel_title: row.get(3)?,
        title: row.get(4)?,
        downloaded_at: row.get(5)?,
        transcribed_at: row.get(6)?,
        wiki_emitted_at: row.get(7)?,
        whisper_model: row.get(8)?,
        audio_path: row.get(9)?,
        transcript_path: row.get(10)?,
        wiki_path: row.get(11)?,
        error: row.get(12)?,
    })
}

fn classify_youtube_url(url: &str) -> InputMode {
    let lower = url.to_ascii_lowercase();
    if lower.contains("list=") || lower.contains("/playlist") {
        InputMode::Playlist
    } else if lower.contains("youtu.be/")
        || lower.contains("watch?v=")
        || lower.contains("/shorts/")
        || lower.contains("/embed/")
    {
        InputMode::Video
    } else {
        InputMode::Channel
    }
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

fn render_wiki_markdown(metadata: &VideoMetadata, transcript: &str) -> Result<String> {
    let title = metadata.title.as_deref().unwrap_or(&metadata.video_id);
    let channel = metadata
        .channel_title
        .as_deref()
        .or(metadata.uploader.as_deref())
        .unwrap_or("Unknown Channel");
    let uploader = metadata.uploader.as_deref().unwrap_or(channel);
    let upload_date = metadata.upload_date.as_deref();

    let mut output = String::new();
    output.push_str("---\n");
    output.push_str(&format!("title: {}\n", yaml_string(title)?));
    output.push_str(&format!("channel: {}\n", yaml_string(channel)?));
    output.push_str(&format!("uploader: {}\n", yaml_string(uploader)?));
    let upload_date = upload_date.map_or_else(|| Ok("null".to_owned()), yaml_string)?;
    output.push_str(&format!("upload_date: {}\n", upload_date));
    output.push_str(&format!(
        "duration: {}\n",
        metadata
            .duration
            .map_or_else(|| "null".to_owned(), |duration| duration.to_string())
    ));
    output.push_str(&format!("url: {}\n", yaml_string(&metadata.url)?));
    output.push_str(&format!("video_id: {}\n", yaml_string(&metadata.video_id)?));
    output.push_str("tags:");
    if metadata.tags.is_empty() {
        output.push_str(" []\n");
    } else {
        output.push('\n');
        for tag in &metadata.tags {
            output.push_str(&format!("  - {}\n", yaml_string(tag)?));
        }
    }
    output.push_str("---\n\n");
    output.push_str(transcript.trim());
    output.push('\n');
    Ok(output)
}

fn yaml_string(value: &str) -> Result<String> {
    serde_json::to_string(value).context("encode YAML string")
}

fn slugify(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let re = Regex::new("[^a-z0-9]+").expect("slug regex compiles");
    let slug = re.replace_all(&lower, "-");
    let slug = slug.trim_matches('-');

    if slug.is_empty() {
        "unknown-channel".to_owned()
    } else {
        slug.to_owned()
    }
}

fn should_skip_download(row: &VideoRow, force: bool) -> bool {
    !force
        && row.downloaded_at.is_some()
        && row
            .audio_path
            .as_deref()
            .is_some_and(|path| Path::new(path).exists())
}

fn should_skip_transcription(row: &VideoRow, whisper_model: &str, force: bool) -> bool {
    if force || row.transcribed_at.is_none() || row.whisper_model.as_deref() != Some(whisper_model)
    {
        return false;
    }

    row.transcript_path.as_deref().is_some_and(|path| {
        let json_path = Path::new(path);
        let txt_path = json_path.with_file_name("transcript.txt");
        json_path.exists() && txt_path.exists()
    })
}

fn should_skip_wiki(row: &VideoRow, force: bool) -> bool {
    !force
        && row.wiki_emitted_at.is_some()
        && row
            .wiki_path
            .as_deref()
            .is_some_and(|path| Path::new(path).exists())
}

fn stage_state(timestamp: &Option<String>, path: &Option<String>) -> &'static str {
    if timestamp.is_none() {
        "-"
    } else if path.as_deref().is_some_and(|path| Path::new(path).exists()) {
        "done"
    } else {
        "missing"
    }
}

fn transcript_state(row: &VideoRow) -> &'static str {
    if row.transcribed_at.is_none() {
        "-"
    } else if should_skip_transcription(row, row.whisper_model.as_deref().unwrap_or(""), false) {
        "done"
    } else {
        "missing"
    }
}

async fn run_checked(program: &str, args: &[String]) -> Result<std::process::Output> {
    let output = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("run {}", format_command(program, args)))?;

    if output.status.success() {
        Ok(output)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        bail!(
            "{} exited with {}{}{}",
            format_command(program, args),
            output.status,
            if stderr.is_empty() { "" } else { ": " },
            stderr
        );
    }
}

fn split_command_prefix(command: &str) -> Result<(String, Vec<String>)> {
    let mut parts = command.split_whitespace();
    let program = parts
        .next()
        .ok_or_else(|| anyhow!("whisper command must not be empty"))?
        .to_owned();
    Ok((program, parts.map(str::to_owned).collect()))
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

    Ok(())
}

async fn find_audio_file(tmp_dir: &Path, preferred_ext: &str) -> Result<PathBuf> {
    let mut fallback = None;
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

        if path.extension().and_then(|ext| ext.to_str()) == Some(preferred_ext) {
            return Ok(path);
        }
        fallback.get_or_insert(path);
    }

    fallback.ok_or_else(|| {
        anyhow!(
            "yt-dlp did not produce an audio file in {}",
            tmp_dir.display()
        )
    })
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
    format!(
        "{prefix}.{}.{}.tmp",
        std::process::id(),
        nanos + u128::from(counter)
    )
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
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
            title: None,
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
            classify_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
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
    }

    #[test]
    fn slugifies_channel_titles() {
        assert_eq!(slugify("The Rust Channel"), "the-rust-channel");
        assert_eq!(
            slugify("  Rust: Fast, Safe & Productive!  "),
            "rust-fast-safe-productive"
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

        let markdown = render_wiki_markdown(&metadata, "hello transcript\n")?;

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
    fn ledger_skip_logic_requires_timestamps_and_existing_files() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;

        let mut row = row_for_skip_tests();
        assert!(!should_skip_download(&row, false));

        let missing_audio = dir.path().join("missing.m4a");
        ledger.mark_downloaded(video_id, &missing_audio)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_download(&row, false));

        let audio = dir.path().join("audio.m4a");
        std::fs::write(&audio, b"audio")?;
        ledger.mark_downloaded(video_id, &audio)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(should_skip_download(&row, false));
        assert!(!should_skip_download(&row, true));

        let transcript_json = dir.path().join("transcript.json");
        std::fs::write(&transcript_json, b"{}")?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_transcription(&row, "large", false));

        let transcript_txt = dir.path().join("transcript.txt");
        std::fs::write(&transcript_txt, b"text")?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(should_skip_transcription(&row, "large", false));
        assert!(!should_skip_transcription(&row, "base", false));
        assert!(!should_skip_transcription(&row, "large", true));

        let wiki = dir.path().join("wiki.md");
        std::fs::write(&wiki, b"wiki")?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(should_skip_wiki(&row, false));
        assert!(!should_skip_wiki(&row, true));

        Ok(())
    }
}
