use std::{
    collections::HashSet,
    env, fmt, future,
    io::Write,
    path::{Component, Path, PathBuf},
    process::{ExitStatus, Output, Stdio},
    sync::{
        Arc, LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use fs4::{FileExt, TryLockError};
use regex::Regex;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    fs,
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::Command,
    sync::watch,
    time::Instant,
};
use tracing::{error, info, warn};

const DEFAULT_DATA_DIR: &str = "data";
const DEFAULT_WHISPER_BIN: &str = "nix run nixpkgs#openai-whisper --";
const DEFAULT_WIKI_INGEST_CMD: &str = "claude -p \"/wiki:ingest {path}\" --permission-mode acceptEdits --allowedTools \"Bash,Read,Write,Edit,Glob,Grep,Task\"";
const DEFAULT_WIKI_INGEST_WARNING: &str = "warning: using default wiki ingestion command with Claude Code --permission-mode acceptEdits and Bash/Read/Write/Edit/Glob/Grep/Task tools; review untrusted transcripts or override --wiki-ingest-cmd";
const DEFAULT_WHISPER_MODEL: &str = "large";
const DEFAULT_AUDIO_FORMAT: &str = "m4a";
const DEFAULT_WIKI_INGEST_TIMEOUT_SECS: u64 = 600;
const STREAMED_OUTPUT_CAPTURE_LIMIT: usize = 64 * 1024;
const WIKI_INGEST_STDERR_CAPTURE_LIMIT: usize = STREAMED_OUTPUT_CAPTURE_LIMIT;
const WIKI_INGEST_STDERR_LEDGER_LIMIT: usize = 4 * 1024;
const WIKI_INGEST_ERROR_PREFIX: &str = "wiki-ingest ";
#[cfg(not(test))]
const STREAM_READER_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const STREAM_READER_DRAIN_TIMEOUT: Duration = Duration::from_millis(100);
#[cfg(all(unix, not(test)))]
const PROCESS_GROUP_TERMINATE_GRACE: Duration = Duration::from_secs(2);
#[cfg(all(unix, test))]
const PROCESS_GROUP_TERMINATE_GRACE: Duration = Duration::from_millis(100);
const YOUTUBE_VIDEO_ID_LEN: usize = 11;
#[cfg(not(target_os = "linux"))]
const NON_LINUX_STALE_TEMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static SLUG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\p{Alphabetic}\p{Number}]+").expect("slug regex compiles"));
static MISSING_WIKI_PLUGIN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?ix)
        ^\s*(?:error:\s*)?
        (?:
            (?:unknown\s+(?:slash\s+)?command|no\s+such\s+command|command\s+not\s+found)\b[^\r\n]{0,120}/wiki:ingest\b
          | (?:slash\s+)?command\b[^\r\n]{0,120}/wiki:ingest\b[^\r\n]{0,80}(?:not\s+recognized|not\s+found|not\s+available|unavailable|unknown)
          | /wiki:ingest\b[^\r\n]{0,80}(?:not\s+recognized|not\s+found|not\s+available|unavailable|unknown|requires\s+(?:the\s+)?(?:wiki|llm-wiki)\s+plugin)
          | plugin\b[^\r\n]{0,60}\b(?:wiki|llm-wiki)\b[^\r\n]{0,60}\b(?:not\s+found|not\s+installed|not\s+enabled|disabled|missing)\b
          | \b(?:wiki|llm-wiki)\b[^\r\n]{0,60}\bplugin\b[^\r\n]{0,60}\b(?:not\s+found|not\s+installed|not\s+enabled|disabled|missing)\b
        )
        ",
    )
    .expect("missing wiki plugin regex compiles")
});

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
    /// Ingest emitted wiki markdown into llm-wiki.
    WikiIngest(WikiIngestCommandArgs),
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

    #[arg(
        long,
        help = "Run llm-wiki ingestion after each emitted wiki article using the configured wiki ingestion command",
        long_help = "Run llm-wiki ingestion after each emitted wiki article using the configured wiki ingestion command. Unless overridden, the default command runs Claude Code with --permission-mode acceptEdits and allows Bash,Read,Write,Edit,Glob,Grep,Task."
    )]
    auto_wiki_ingest: bool,

    #[command(flatten)]
    wiki_ingest: WikiIngestArgs,
}

#[derive(Args, Debug)]
struct DataDirArgs {
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,
}

#[derive(Args, Debug)]
struct WikiIngestCommandArgs {
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,

    #[command(flatten)]
    wiki_ingest: WikiIngestArgs,

    #[arg(
        long,
        allow_hyphen_values = true,
        num_args = 1,
        value_parser = parse_youtube_video_id,
        help = "Only ingest the emitted wiki article for this video ID"
    )]
    video_id: Option<String>,

    #[arg(
        long,
        help = "Retry rows whose last wiki-ingest attempt recorded an error; by default they are skipped"
    )]
    retry_errors: bool,

    #[arg(long, value_parser = parse_positive_usize, help = "Maximum number of pending wiki articles to ingest")]
    limit: Option<usize>,

    #[arg(
        long,
        help = "Re-ingest rows even when they were already marked ingested; requires an emitted wiki article"
    )]
    force: bool,
}

#[derive(Args, Debug, Clone)]
struct WikiIngestArgs {
    #[arg(
        long,
        value_name = "TEMPLATE",
        value_parser = parse_wiki_ingest_template,
        help = "Wiki ingestion command template containing {path}; used by wiki-ingest or ingest --auto-wiki-ingest",
        long_help = "Wiki ingestion command template containing {path}; overrides YTARCH_WIKI_INGEST_CMD. Values for {path}, {video_id}, {title}, and {channel_slug} are shell-escaped before parsing. The built-in default runs Claude Code with --permission-mode acceptEdits and allows Bash,Read,Write,Edit,Glob,Grep,Task. On the ingest command this option requires --auto-wiki-ingest."
    )]
    wiki_ingest_cmd: Option<String>,

    #[arg(
        long,
        help = "Working directory for the wiki ingestion command; default: <data-dir>/wiki"
    )]
    wiki_ingest_cwd: Option<PathBuf>,

    #[arg(
        long,
        value_parser = parse_positive_u64,
        help = "Wiki ingestion command timeout in seconds; default: 600"
    )]
    wiki_ingest_timeout_secs: Option<u64>,
}

impl WikiIngestArgs {
    fn has_cli_overrides(&self) -> bool {
        self.wiki_ingest_cmd.is_some()
            || self.wiki_ingest_cwd.is_some()
            || self.wiki_ingest_timeout_secs.is_some()
    }
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

fn parse_positive_u64(value: &str) -> std::result::Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|err| format!("invalid positive integer: {err}"))?;
    if parsed == 0 {
        Err("value must be greater than 0".to_owned())
    } else {
        Ok(parsed)
    }
}

fn parse_wiki_ingest_template(value: &str) -> std::result::Result<String, String> {
    validate_wiki_ingest_template(value)?;
    Ok(value.to_owned())
}

fn parse_youtube_video_id(value: &str) -> std::result::Result<String, String> {
    if is_valid_youtube_video_id(value) {
        Ok(value.to_owned())
    } else {
        Err(invalid_video_id_error(value))
    }
}

fn validate_wiki_ingest_template(value: &str) -> std::result::Result<(), String> {
    if !template_has_unescaped_path_token(value) {
        return Err("template must contain {path}".to_owned());
    }

    let values_a = WikiIngestTemplateValues {
        path: "/tmp/youtube archiver/wiki/channel-a/video \"quoted\".md".to_owned(),
        video_id: "abc123".to_owned(),
        title: "A \"quoted\" title".to_owned(),
        channel_slug: "channel-a".to_owned(),
    };
    let rendered_a = render_wiki_ingest_template(value, &values_a);
    let argv_a = shell_words::split(&rendered_a)
        .map_err(|err| format!("template must render to a shell-parseable command: {err}"))?;
    if argv_a.is_empty() {
        return Err("template must render to a non-empty command".to_owned());
    }

    // The preflight `command -v` check renders the template with the
    // first candidate row's metadata. If argv[0] varies per row (e.g.
    // a template like `~/scripts/{channel_slug}-ingest.sh {path}`),
    // preflight succeeds for one row and silently faults mid-batch on
    // a row with a different channel/title. Reject such templates.
    let values_b = WikiIngestTemplateValues {
        path: "/var/tmp/different/wiki/channel-b/other.md".to_owned(),
        video_id: "zyx987".to_owned(),
        title: "Different title".to_owned(),
        channel_slug: "channel-b".to_owned(),
    };
    let rendered_b = render_wiki_ingest_template(value, &values_b);
    let argv_b = shell_words::split(&rendered_b)
        .map_err(|err| format!("template must render to a shell-parseable command: {err}"))?;
    if argv_b.first() != argv_a.first() {
        return Err(
            "template's program (first token) must not reference {path}, {video_id}, {title}, or {channel_slug}; preflight cannot check a command that varies per video".to_owned(),
        );
    }

    Ok(())
}

fn template_has_unescaped_path_token(template: &str) -> bool {
    let mut rest = template;
    let mut state = ShellRenderState::default();
    while let Some(index) = rest.find('{') {
        let literal = &rest[..index];
        update_shell_render_state(&mut state, literal);
        let token_start = &rest[index..];
        if !state.escaped && token_start.starts_with("{path}") {
            return true;
        }
        update_shell_render_state(&mut state, "{");
        rest = &token_start[1..];
    }
    false
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

#[derive(Debug)]
struct WikiIngestConfig {
    template: String,
    uses_default_template: bool,
    cwd: PathBuf,
    create_cwd_for_preflight: bool,
    timeout: Duration,
}

#[derive(Debug)]
struct WikiIngestBatchOptions<'a> {
    video_id: Option<&'a str>,
    retry_errors: bool,
    limit: Option<usize>,
    force: bool,
    missing_plugin_hint_emitted: Option<&'a mut bool>,
}

#[derive(Debug, Default)]
struct WikiIngestBatchOutcome {
    succeeded: usize,
    skipped: usize,
    failed: usize,
}

#[derive(Debug)]
struct RenderedWikiIngestCommand {
    rendered: String,
    program: String,
    args: Vec<String>,
}

#[derive(Debug)]
struct WikiIngestTemplateValues {
    path: String,
    video_id: String,
    title: String,
    channel_slug: String,
}

#[derive(Debug)]
struct WikiIngestCommandOutput {
    status: ExitStatus,
    stderr: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct CommandProcessGroup {
    #[cfg(unix)]
    pgid: libc::pid_t,
}

#[derive(Debug, Clone, Copy)]
struct ShellRenderState {
    quote: ShellQuoteContext,
    escaped: bool,
}

impl Default for ShellRenderState {
    fn default() -> Self {
        Self {
            quote: ShellQuoteContext::Unquoted,
            escaped: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ShellQuoteContext {
    Unquoted,
    Single,
    Double,
}

#[derive(Debug)]
struct ExitCodeError {
    code: i32,
    message: String,
}

impl ExitCodeError {
    fn new(code: i32, message: String) -> Self {
        Self { code, message }
    }
}

impl fmt::Display for ExitCodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ExitCodeError {}

#[derive(Debug)]
struct InterruptedError;

impl fmt::Display for InterruptedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("wiki ingestion interrupted")
    }
}

impl std::error::Error for InterruptedError {}

fn is_interrupted_error(err: &anyhow::Error) -> bool {
    err.downcast_ref::<InterruptedError>().is_some()
}

#[derive(Clone)]
struct Interrupts {
    receiver: watch::Receiver<bool>,
}

impl Interrupts {
    fn install() -> (Self, AbortOnDrop<()>) {
        let (sender, receiver) = watch::channel(false);
        let task = AbortOnDrop::new(tokio::spawn(async move {
            wait_for_process_interrupt().await;
            let _ = sender.send(true);
        }));
        (Self { receiver }, task)
    }

    #[cfg(test)]
    fn inactive() -> Self {
        let (_sender, receiver) = watch::channel(false);
        Self { receiver }
    }

    #[cfg(test)]
    fn test_channel() -> (Self, watch::Sender<bool>) {
        let (sender, receiver) = watch::channel(false);
        (Self { receiver }, sender)
    }

    fn check(&self) -> Result<()> {
        if *self.receiver.borrow() {
            Err(InterruptedError.into())
        } else {
            Ok(())
        }
    }

    async fn wait(&self) {
        let mut receiver = self.receiver.clone();
        wait_for_interrupt(&mut receiver).await;
    }
}

async fn wait_for_interrupt(receiver: &mut watch::Receiver<bool>) {
    if *receiver.borrow() {
        return;
    }

    loop {
        match receiver.changed().await {
            Ok(()) if *receiver.borrow() => return,
            Ok(()) => {}
            Err(_) => future::pending::<()>().await,
        }
    }
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

    fn handle_mut(&mut self) -> &mut tokio::task::JoinHandle<T> {
        self.handle.as_mut().expect("join handle exists")
    }

    fn clear_completed(&mut self) {
        self.handle = None;
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}

#[derive(Debug)]
struct DataDirLock {
    path: PathBuf,
    description: &'static str,
    file: std::fs::File,
}

impl Drop for DataDirLock {
    fn drop(&mut self) {
        if let Err(err) = self.file.unlock() {
            warn!(path = %self.path.display(), description = self.description, error = %err, "failed to unlock data-dir lock");
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
    wiki_ingested_at: Option<String>,
    wiki_ingest_cmd: Option<String>,
    whisper_model: Option<String>,
    audio_path: Option<String>,
    transcript_path: Option<String>,
    wiki_path: Option<String>,
    error: Option<String>,
}

struct LedgerColumn {
    name: &'static str,
    create_sql: &'static str,
    migration_sql: Option<&'static str>,
}

const LEDGER_COLUMNS: &[LedgerColumn] = &[
    LedgerColumn {
        name: "video_id",
        create_sql: "video_id TEXT PRIMARY KEY",
        migration_sql: None,
    },
    LedgerColumn {
        name: "url",
        create_sql: "url TEXT NOT NULL",
        migration_sql: None,
    },
    LedgerColumn {
        name: "channel_id",
        create_sql: "channel_id TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "channel_title",
        create_sql: "channel_title TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "uploader",
        create_sql: "uploader TEXT",
        migration_sql: Some("ALTER TABLE videos ADD COLUMN uploader TEXT"),
    },
    LedgerColumn {
        name: "title",
        create_sql: "title TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "upload_date",
        create_sql: "upload_date TEXT",
        migration_sql: Some("ALTER TABLE videos ADD COLUMN upload_date TEXT"),
    },
    LedgerColumn {
        name: "duration",
        create_sql: "duration INTEGER",
        migration_sql: Some("ALTER TABLE videos ADD COLUMN duration INTEGER"),
    },
    LedgerColumn {
        name: "tags",
        create_sql: "tags TEXT",
        migration_sql: Some("ALTER TABLE videos ADD COLUMN tags TEXT"),
    },
    LedgerColumn {
        name: "downloaded_at",
        create_sql: "downloaded_at TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "transcribed_at",
        create_sql: "transcribed_at TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "wiki_emitted_at",
        create_sql: "wiki_emitted_at TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "wiki_ingested_at",
        create_sql: "wiki_ingested_at TEXT",
        migration_sql: Some("ALTER TABLE videos ADD COLUMN wiki_ingested_at TEXT"),
    },
    LedgerColumn {
        name: "wiki_ingest_cmd",
        create_sql: "wiki_ingest_cmd TEXT",
        migration_sql: Some("ALTER TABLE videos ADD COLUMN wiki_ingest_cmd TEXT"),
    },
    LedgerColumn {
        name: "whisper_model",
        create_sql: "whisper_model TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "audio_path",
        create_sql: "audio_path TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "transcript_path",
        create_sql: "transcript_path TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "wiki_path",
        create_sql: "wiki_path TEXT",
        migration_sql: None,
    },
    LedgerColumn {
        name: "error",
        create_sql: "error TEXT",
        migration_sql: None,
    },
];

fn create_videos_table_sql() -> String {
    let mut sql = "CREATE TABLE IF NOT EXISTS videos (\n".to_owned();
    for (index, column) in LEDGER_COLUMNS.iter().enumerate() {
        if index > 0 {
            sql.push_str(",\n");
        }
        sql.push_str("                    ");
        sql.push_str(column.create_sql);
    }
    sql.push_str("\n                );");
    sql
}

struct Ledger {
    conn: Connection,
    data_dir: PathBuf,
    select_columns: String,
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
        let mut ledger = Self {
            conn,
            data_dir,
            select_columns: String::new(),
        };
        ledger.init()?;
        ledger.select_columns = ledger.compute_select_columns_sql()?;
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
        let mut ledger = Self {
            conn,
            data_dir: data_dir.to_path_buf(),
            select_columns: String::new(),
        };
        ledger.select_columns = ledger.compute_select_columns_sql()?;
        Ok(Some(ledger))
    }

    #[cfg(test)]
    fn open_in_memory() -> Result<Self> {
        let ledger = Self {
            conn: Connection::open_in_memory().context("open in-memory ledger")?,
            data_dir: std::env::current_dir()
                .context("read current directory for in-memory ledger")?
                .join(DEFAULT_DATA_DIR),
            select_columns: String::new(),
        };
        configure_connection(&ledger.conn)?;
        let mut ledger = ledger;
        ledger.init()?;
        ledger.select_columns = ledger.compute_select_columns_sql()?;
        Ok(ledger)
    }

    fn init(&self) -> Result<()> {
        let create_sql = create_videos_table_sql();
        self.conn
            .execute_batch(&create_sql)
            .context("initialize ledger schema")?;
        for column in LEDGER_COLUMNS
            .iter()
            .filter(|column| column.migration_sql.is_some())
        {
            self.ensure_column(column.name)?;
        }
        Ok(())
    }

    fn ensure_column(&self, name: &str) -> Result<()> {
        let column = LEDGER_COLUMNS
            .iter()
            .find(|column| column.name == name && column.migration_sql.is_some())
            .ok_or_else(|| anyhow!("unsupported ledger migration column {name:?}"))?;

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
            .execute(
                column
                    .migration_sql
                    .expect("migration column has migration SQL"),
                [],
            )
            .with_context(|| format!("add ledger column {name}"))?;
        Ok(())
    }

    fn compute_select_columns_sql(&self) -> Result<String> {
        let existing_columns = self.video_column_names()?;
        let mut columns = Vec::with_capacity(LEDGER_COLUMNS.len());
        for column in LEDGER_COLUMNS {
            if existing_columns.contains(column.name) {
                columns.push(column.name.to_owned());
            } else if column.migration_sql.is_some() {
                columns.push(format!("NULL AS {}", column.name));
            } else {
                bail!(
                    "ledger schema corrupt: videos table is missing required column {}",
                    column.name
                );
            }
        }
        Ok(columns.join(", "))
    }

    fn video_column_names(&self) -> Result<HashSet<String>> {
        let mut stmt = self
            .conn
            .prepare("PRAGMA table_info(videos)")
            .context("inspect ledger schema")?;
        let mut rows = stmt.query([]).context("read ledger schema")?;
        let mut columns = HashSet::new();
        while let Some(row) = rows.next().context("read ledger schema row")? {
            columns.insert(row.get(1).context("read ledger column name")?);
        }
        Ok(columns)
    }

    #[cfg(test)]
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

    fn ensure_videos(&self, video_ids: &[String]) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("begin resolved video ledger transaction")?;
        {
            let mut stmt = tx
                .prepare(
                    r#"
                    INSERT INTO videos (video_id, url)
                    VALUES (?1, ?2)
                    ON CONFLICT(video_id) DO UPDATE SET url = excluded.url
                    "#,
                )
                .context("prepare resolved video ledger upsert")?;
            for video_id in video_ids {
                stmt.execute(params![video_id, canonical_video_url(video_id)])
                    .with_context(|| format!("upsert ledger row for {video_id}"))?;
            }
        }
        tx.commit()
            .context("commit resolved video ledger transaction")?;
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
        let wiki_error_pattern = wiki_ingest_error_like_pattern();
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET downloaded_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    audio_path = ?2,
                    transcribed_at = NULL,
                    wiki_emitted_at = NULL,
                    wiki_ingested_at = NULL,
                    wiki_ingest_cmd = NULL,
                    error = CASE
                        WHEN error LIKE ?3 THEN error
                        ELSE NULL
                    END
                WHERE video_id = ?1
                "#,
                params![video_id, audio_path, wiki_error_pattern],
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
        let wiki_error_pattern = wiki_ingest_error_like_pattern();
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET transcribed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    whisper_model = ?2,
                    transcript_path = ?3,
                    -- Preserve wiki_path so re-emission can remove stale slug paths.
                    wiki_emitted_at = NULL,
                    wiki_ingested_at = NULL,
                    wiki_ingest_cmd = NULL,
                    error = CASE
                        WHEN error LIKE ?4 THEN error
                        ELSE NULL
                    END
                WHERE video_id = ?1
                "#,
                params![video_id, whisper_model, transcript_path, wiki_error_pattern],
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
                    wiki_emitted_at = NULL,
                    wiki_ingested_at = NULL,
                    wiki_ingest_cmd = NULL
                WHERE video_id = ?1
                "#,
                params![video_id],
            )
            .with_context(|| format!("invalidate transcription outputs for {video_id}"))?;
        Ok(())
    }

    fn restore_transcription_outputs(&self, row: &VideoRow) -> Result<()> {
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET transcribed_at = ?2,
                    wiki_emitted_at = ?3,
                    whisper_model = ?4,
                    transcript_path = ?5,
                    wiki_path = ?6,
                    wiki_ingested_at = ?7,
                    wiki_ingest_cmd = ?8
                WHERE video_id = ?1
                "#,
                params![
                    row.video_id,
                    row.transcribed_at,
                    row.wiki_emitted_at,
                    row.whisper_model,
                    row.transcript_path,
                    row.wiki_path,
                    row.wiki_ingested_at,
                    row.wiki_ingest_cmd
                ],
            )
            .with_context(|| format!("restore transcription outputs for {}", row.video_id))?;
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
                    wiki_ingested_at = NULL,
                    wiki_ingest_cmd = NULL,
                    error = NULL
                WHERE video_id = ?1
                "#,
                params![video_id, wiki_path],
            )
            .with_context(|| format!("mark {video_id} wiki emitted"))?;
        Ok(())
    }

    fn mark_wiki_ingested(&self, row: &VideoRow, rendered_cmd: &str) -> Result<()> {
        let wiki_error_pattern = wiki_ingest_error_like_pattern();
        let updated = self
            .conn
            .execute(
                r#"
                UPDATE videos
                SET wiki_ingested_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    wiki_ingest_cmd = ?2,
                    error = CASE
                        WHEN error LIKE ?5 THEN NULL
                        ELSE error
                    END
                WHERE video_id = ?1
                  AND wiki_path IS ?3
                  AND wiki_emitted_at IS ?4
                "#,
                params![
                    &row.video_id,
                    rendered_cmd,
                    row.wiki_path.as_deref(),
                    row.wiki_emitted_at.as_deref(),
                    wiki_error_pattern
                ],
            )
            .with_context(|| format!("mark {} wiki ingested", row.video_id))?;
        if updated != 1 {
            bail!(
                "ledger row changed while wiki ingestion was running for {}",
                row.video_id
            );
        }
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

    fn clear_stale_non_wiki_ingest_error(&self, video_id: &str) -> Result<()> {
        let wiki_error_pattern = wiki_ingest_error_like_pattern();
        self.conn
            .execute(
                r#"
                UPDATE videos
                SET error = NULL
                WHERE video_id = ?1
                  AND error IS NOT NULL
                  AND error NOT LIKE ?2
                "#,
                params![video_id, wiki_error_pattern],
            )
            .with_context(|| format!("clear stale non-wiki-ingest error for {video_id}"))?;
        Ok(())
    }

    fn row(&self, video_id: &str) -> Result<Option<VideoRow>> {
        let select_columns = &self.select_columns;
        self.conn
            .query_row(
                &format!(
                    r#"
                    SELECT {select_columns}
                    FROM videos
                    WHERE video_id = ?1
                    "#
                ),
                params![video_id],
                row_from_sql,
            )
            .optional()
            .with_context(|| format!("read ledger row for {video_id}"))
    }

    fn rows(&self) -> Result<Vec<VideoRow>> {
        let select_columns = &self.select_columns;
        let mut stmt = self
            .conn
            .prepare(&format!(
                r#"
                SELECT {select_columns}
                FROM videos
                ORDER BY video_id
                "#
            ))
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

    fn wiki_ingest_rows(&self, retry_errors: bool, force: bool) -> Result<Vec<VideoRow>> {
        let select_columns = &self.select_columns;
        let include_ingested = force;
        let include_errors = force || retry_errors;
        let include_error_retries = retry_errors;
        let sql = format!(
            r#"
            SELECT {select_columns}
            FROM videos
            WHERE wiki_emitted_at IS NOT NULL
              AND (?1 OR wiki_ingested_at IS NULL OR (?3 AND error IS NOT NULL))
              AND (?2 OR error IS NULL)
            ORDER BY video_id
            "#
        );

        let mut stmt = self
            .conn
            .prepare(&sql)
            .context("prepare wiki ingest candidate query")?;
        let mut rows = Vec::new();
        let mut query_rows = stmt
            .query_map(
                params![include_ingested, include_errors, include_error_retries],
                row_from_sql,
            )
            .context("query wiki ingest candidate rows")?;
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
async fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let (interrupts, _interrupt_listener) = Interrupts::install();
    if let Err(err) = run_cli(&interrupts).await {
        if let Some(exit) = err.downcast_ref::<ExitCodeError>() {
            eprintln!("{exit}");
            std::process::exit(exit.code);
        }
        if is_interrupted_error(&err) {
            eprintln!("Interrupted");
            std::process::exit(130);
        }

        eprintln!("Error: {err:#}");
        std::process::exit(1);
    }
}

async fn run_cli(interrupts: &Interrupts) -> Result<()> {
    let mut command = Cli::parse().command;
    normalize_command_paths(&mut command)?;
    match command {
        Commands::Ingest(args) => ingest(args, interrupts).await,
        Commands::WikiIngest(args) => wiki_ingest(args, interrupts).await,
        Commands::Status(args) => status(args).await,
        Commands::List(args) => list(args).await,
    }
}

fn normalize_command_paths(command: &mut Commands) -> Result<()> {
    match command {
        Commands::Ingest(args) => normalize_data_dir(&mut args.data_dir),
        Commands::WikiIngest(args) => normalize_data_dir(&mut args.data_dir),
        Commands::Status(args) | Commands::List(args) => normalize_data_dir(&mut args.data_dir),
    }
}

fn normalize_data_dir(data_dir: &mut PathBuf) -> Result<()> {
    let original = data_dir.clone();
    *data_dir = absolutize_path(&original)
        .with_context(|| format!("resolve --data-dir {}", original.display()))?;
    Ok(())
}

async fn ingest(args: IngestArgs, interrupts: &Interrupts) -> Result<()> {
    validate_whisper_config(&args.whisper_bin, &args.whisper_args)?;
    reject_ignored_wiki_ingest_options(&args)?;
    std::fs::create_dir_all(&args.data_dir)
        .with_context(|| format!("create data dir {}", args.data_dir.display()))?;
    let _ingest_lock = acquire_ingest_lock(&args.data_dir)?;
    let wiki_ingest_config = if args.auto_wiki_ingest {
        Some(wiki_ingest_config(&args.data_dir, &args.wiki_ingest)?)
    } else {
        None
    };
    if let Some(config) = wiki_ingest_config.as_ref() {
        warn_if_using_default_wiki_ingest_command(config);
        preflight_wiki_ingest_config(&args.data_dir, config).await?;
    }

    let ledger = Ledger::open(&args.data_dir)?;
    let mode = classify_youtube_url(&args.url);
    info!(?mode, url = %args.url, "resolving input URL");
    let video_ids = resolve_video_ids(&args.url, mode, args.limit, interrupts).await?;

    if video_ids.is_empty() {
        bail!("yt-dlp did not return any video IDs for {}", args.url);
    }

    ensure_resolved_videos(&ledger, &video_ids)?;

    let mut succeeded = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut failed_video_ids = Vec::new();
    let mut missing_plugin_hint_emitted = false;

    for video_id in video_ids {
        interrupts.check()?;
        match process_video(
            &args,
            &ledger,
            &video_id,
            wiki_ingest_config.as_ref(),
            Some(&mut missing_plugin_hint_emitted),
            interrupts,
        )
        .await
        {
            Ok(ProcessVideoOutcome::Worked) => {
                succeeded += 1;
                info!(%video_id, "video processed");
            }
            Ok(ProcessVideoOutcome::Skipped) => {
                skipped += 1;
                info!(%video_id, "video already complete, nothing to do");
            }
            Err(err) => {
                if is_interrupted_error(&err) {
                    return Err(err);
                }
                failed += 1;
                failed_video_ids.push(video_id.clone());
                let message = format!("{err:#}");
                error!(%video_id, error = %message, "video failed");
                if should_preserve_recorded_wiki_ingest_error(&ledger, &video_id, &message) {
                    warn!(%video_id, error = %message, "preserving existing ledger error recorded by failed stage");
                } else if let Err(err) = ledger.mark_error(&video_id, &message) {
                    warn!(%video_id, error = %err, original_error = %message, "failed to record video error in ledger");
                }
            }
        }
    }

    interrupts.check()?;
    // Match `wiki-ingest` semantics: only bail when every video that
    // actually needed work failed. A run consisting entirely of skips
    // (with no failures) is a successful no-op.
    if succeeded == 0 && failed > 0 {
        bail!(
            "every video failed ({failed} failure(s), {skipped} already complete): {}",
            failed_video_ids.join(", ")
        );
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessVideoOutcome {
    /// At least one pipeline stage advanced the ledger.
    Worked,
    /// Every stage was already complete; this run was a no-op for this video.
    Skipped,
}

fn reject_ignored_wiki_ingest_options(args: &IngestArgs) -> Result<()> {
    if !args.auto_wiki_ingest && args.wiki_ingest.has_cli_overrides() {
        bail!("wiki ingestion options require --auto-wiki-ingest on ingest");
    }
    Ok(())
}

fn warn_if_using_default_wiki_ingest_command(config: &WikiIngestConfig) {
    if config.uses_default_template {
        eprintln!("{DEFAULT_WIKI_INGEST_WARNING}");
    }
}

fn should_preserve_recorded_wiki_ingest_error(
    ledger: &Ledger,
    video_id: &str,
    new_error: &str,
) -> bool {
    new_error.starts_with(&format!("{WIKI_INGEST_ERROR_PREFIX}{video_id}"))
        && ledger
            .row(video_id)
            .ok()
            .flatten()
            .is_some_and(|row| is_wiki_ingest_ledger_error(row.error.as_deref()))
}

fn ensure_resolved_videos(ledger: &Ledger, video_ids: &[String]) -> Result<()> {
    ledger.ensure_videos(video_ids)
}

async fn process_video(
    args: &IngestArgs,
    ledger: &Ledger,
    video_id: &str,
    wiki_ingest_config: Option<&WikiIngestConfig>,
    missing_plugin_hint_emitted: Option<&mut bool>,
    interrupts: &Interrupts,
) -> Result<ProcessVideoOutcome> {
    interrupts.check()?;
    // Snapshot stage state before the pipeline runs so the caller can
    // distinguish "every stage was already complete" from "actual work
    // happened" for exit-code accounting.
    let before = stage_progress_signature(ledger, video_id)?;

    let metadata =
        load_or_fetch_metadata(&args.data_dir, ledger, video_id, args.force, interrupts).await?;

    let audio_path = download_audio(
        &args.data_dir,
        ledger,
        video_id,
        &args.audio_format,
        args.force,
        interrupts,
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
        interrupts,
    )
    .await
    .with_context(|| format!("transcribe {video_id}"))?;

    emit_wiki_article(&args.data_dir, ledger, &metadata, args.force)
        .await
        .with_context(|| format!("emit wiki markdown for {video_id}"))?;
    interrupts.check()?;

    if let Some(config) = wiki_ingest_config {
        run_wiki_ingest_batch(
            &args.data_dir,
            ledger,
            config,
            WikiIngestBatchOptions {
                video_id: Some(video_id),
                // Auto-ingest only retries rows whose existing error was
                // recorded by the wiki-ingest stage itself; the engine
                // re-checks this per row. We pass `true` here so a row
                // with a prior wiki-ingest failure is eligible.
                retry_errors: true,
                limit: None,
                force: args.force,
                missing_plugin_hint_emitted,
            },
            interrupts,
        )
        .await
        .with_context(|| format!("wiki-ingest {video_id}"))?;
    }

    if let Err(err) = ledger.clear_stale_non_wiki_ingest_error(video_id) {
        warn!(%video_id, error = %err, "failed to clear stale ledger error after successful processing");
    }

    let after = stage_progress_signature(ledger, video_id)?;
    Ok(if after == before {
        ProcessVideoOutcome::Skipped
    } else {
        ProcessVideoOutcome::Worked
    })
}

/// Snapshot of every stage timestamp on a video row, used to detect
/// whether a pipeline pass actually advanced any stage.
#[derive(Debug, Default, PartialEq, Eq)]
struct StageProgressSignature {
    downloaded_at: Option<String>,
    transcribed_at: Option<String>,
    wiki_emitted_at: Option<String>,
    wiki_ingested_at: Option<String>,
}

fn stage_progress_signature(ledger: &Ledger, video_id: &str) -> Result<StageProgressSignature> {
    Ok(ledger
        .row(video_id)?
        .map(|row| StageProgressSignature {
            downloaded_at: row.downloaded_at,
            transcribed_at: row.transcribed_at,
            wiki_emitted_at: row.wiki_emitted_at,
            wiki_ingested_at: row.wiki_ingested_at,
        })
        .unwrap_or_default())
}

async fn resolve_video_ids(
    url: &str,
    mode: InputMode,
    limit: Option<usize>,
    interrupts: &Interrupts,
) -> Result<Vec<String>> {
    let args = resolve_video_ids_args(url, mode, limit);
    let output = run_checked("yt-dlp", &args, interrupts).await?;
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
    interrupts: &Interrupts,
) -> Result<VideoMetadata> {
    let media_dir = data_dir.join("media").join(video_id);
    let info_path = media_dir.join("info.json");

    let cached_metadata = if force {
        None
    } else {
        load_cached_metadata(video_id, &info_path).await?
    };
    if let Some(metadata) = cached_metadata {
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
    let output = run_checked("yt-dlp", &args, interrupts).await?;
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
        Ok(value) => match string_field(&value, &["id"]).as_deref() {
            Some(cached_id) if cached_id == video_id => {
                Ok(Some(metadata_from_value(video_id, &value)))
            }
            Some(cached_id) => {
                warn!(
                    path = %info_path.display(),
                    expected_video_id = %video_id,
                    cached_video_id = %cached_id,
                    "cached metadata video ID mismatch; refetching"
                );
                Ok(None)
            }
            None => {
                warn!(
                    path = %info_path.display(),
                    expected_video_id = %video_id,
                    "cached metadata missing video ID; refetching"
                );
                Ok(None)
            }
        },
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
    interrupts: &Interrupts,
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
        "bestaudio/best".to_owned(),
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
        run_checked_stream_output("yt-dlp", &args, interrupts).await?;
        let downloaded = find_audio_file(&tmp_dir, audio_format).await?;
        let extension = downloaded
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or(audio_format);
        let final_path = media_dir.join(format!("audio.{extension}"));
        fs::rename(&downloaded, &final_path)
            .await
            .with_context(|| format!("move audio to {}", final_path.display()))?;
        if let Err(err) = remove_stale_audio_files(&media_dir, &final_path).await {
            warn!(path = %media_dir.display(), error = %err, "failed to remove stale audio files");
        }
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
    interrupts: &Interrupts,
) -> Result<PathBuf> {
    let previous_row = ledger.row(video_id)?;
    if let Some(row) = previous_row.as_ref()
        && should_skip_transcription_async(data_dir, row, whisper.model, force).await
    {
        let transcript_path = row
            .transcript_path
            .as_deref()
            .expect("checked by should_skip_transcription_async");
        return Ok(ledger_path_to_fs_path(data_dir, transcript_path));
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
        run_checked_stream_output(&program, &args, interrupts).await?;
        let output_stem = audio_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("audio");
        let (whisper_json, whisper_txt) = find_whisper_outputs(&tmp_dir, output_stem).await?;
        let final_json = transcript_dir.join("transcript.json");
        let final_txt = transcript_dir.join("transcript.txt");
        ledger.invalidate_transcription_outputs(video_id)?;
        if let Err(err) =
            replace_transcript_pair(&whisper_json, &whisper_txt, &final_json, &final_txt).await
        {
            let restore_error = previous_row
                .as_ref()
                .and_then(|row| ledger.restore_transcription_outputs(row).err());
            if let Some(restore_err) = restore_error {
                warn!(%video_id, error = %restore_err, "failed to restore transcription ledger state after replacement failure");
            }
            return Err(err);
        }
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
        "{:<14} {:<10} {:<11} {:<10} {:<8} {:<7} title",
        "video_id", "download", "transcribe", "wiki", "ingest", "error"
    );
    for row in rows {
        let download = download_state(&args.data_dir, &row).await;
        let transcript = transcript_state(&args.data_dir, &row).await;
        let wiki = wiki_state(&args.data_dir, &row).await;
        let wiki_ingest = wiki_ingest_state(&args.data_dir, &row).await;
        println!(
            "{:<14} {:<10} {:<11} {:<10} {:<8} {:<7} {}",
            row.video_id,
            download,
            transcript,
            wiki,
            wiki_ingest,
            if row.error.is_some() { "yes" } else { "-" },
            row.title.as_deref().unwrap_or("-")
        );
    }

    Ok(())
}

async fn list(args: DataDirArgs) -> Result<()> {
    let rows = list_rows(&args.data_dir).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&rows).context("serialize archived video rows")?
    );
    Ok(())
}

async fn wiki_ingest(args: WikiIngestCommandArgs, interrupts: &Interrupts) -> Result<()> {
    let config = wiki_ingest_config(&args.data_dir, &args.wiki_ingest)?;
    warn_if_using_default_wiki_ingest_command(&config);
    preflight_wiki_ingest_config(&args.data_dir, &config).await?;
    let ledger = Ledger::open(&args.data_dir)?;
    run_wiki_ingest_batch(
        &args.data_dir,
        &ledger,
        &config,
        WikiIngestBatchOptions {
            video_id: args.video_id.as_deref(),
            retry_errors: args.retry_errors,
            limit: args.limit,
            force: args.force,
            missing_plugin_hint_emitted: None,
        },
        interrupts,
    )
    .await?;
    Ok(())
}

fn wiki_ingest_config(data_dir: &Path, args: &WikiIngestArgs) -> Result<WikiIngestConfig> {
    let (template, uses_default_template) = wiki_ingest_template(args.wiki_ingest_cmd.as_deref())?;
    let create_cwd_for_preflight = args.wiki_ingest_cwd.is_none();
    let cwd = match args.wiki_ingest_cwd.as_deref() {
        Some(cwd) => absolutize_path(cwd)
            .with_context(|| format!("resolve --wiki-ingest-cwd {}", cwd.display()))?,
        None => data_dir.join("wiki"),
    };
    Ok(WikiIngestConfig {
        template,
        uses_default_template,
        cwd,
        create_cwd_for_preflight,
        timeout: Duration::from_secs(
            args.wiki_ingest_timeout_secs
                .unwrap_or(DEFAULT_WIKI_INGEST_TIMEOUT_SECS),
        ),
    })
}

fn wiki_ingest_template(cli_template: Option<&str>) -> Result<(String, bool)> {
    let (template, uses_default_template) = match cli_template {
        Some(template) => (template.to_owned(), false),
        None => match env::var("YTARCH_WIKI_INGEST_CMD") {
            Ok(template) => (template, false),
            Err(_) => (DEFAULT_WIKI_INGEST_CMD.to_owned(), true),
        },
    };
    validate_wiki_ingest_template(&template).map_err(|err| anyhow!(err))?;
    Ok((template, uses_default_template))
}

fn wiki_ingest_error_like_pattern() -> String {
    format!("{WIKI_INGEST_ERROR_PREFIX}%")
}

fn acquire_ingest_lock(data_dir: &Path) -> Result<DataDirLock> {
    acquire_data_dir_lock_at(data_dir.join(".ingest.lock"), "ingest")
}

fn acquire_data_dir_lock_at(path: PathBuf, description: &'static str) -> Result<DataDirLock> {
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("open {description} lock {}", path.display()))?;

    match FileExt::try_lock(&file) {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            bail!(
                "{description} is already running (lock file: {})",
                path.display()
            );
        }
        Err(TryLockError::Error(err)) => {
            return Err(err).with_context(|| format!("lock {description} {}", path.display()));
        }
    };

    if let Err(err) = file.set_len(0) {
        warn!(path = %path.display(), description, error = %err, "failed to truncate data-dir lock metadata");
    }
    if let Err(err) = writeln!(file, "pid={}", std::process::id()) {
        warn!(path = %path.display(), description, error = %err, "failed to write data-dir lock metadata");
    }
    Ok(DataDirLock {
        path,
        description,
        file,
    })
}

fn acquire_wiki_ingest_lock(data_dir: &Path) -> Result<DataDirLock> {
    acquire_data_dir_lock_at(data_dir.join(".wiki-ingest.lock"), "wiki ingestion")
}

async fn run_wiki_ingest_batch(
    data_dir: &Path,
    ledger: &Ledger,
    config: &WikiIngestConfig,
    options: WikiIngestBatchOptions<'_>,
    interrupts: &Interrupts,
) -> Result<WikiIngestBatchOutcome> {
    let WikiIngestBatchOptions {
        video_id,
        retry_errors,
        limit,
        force,
        missing_plugin_hint_emitted,
    } = options;
    let rows =
        wiki_ingest_candidate_rows(data_dir, ledger, video_id, retry_errors, force, limit).await?;
    interrupts.check()?;
    let mut outcome = WikiIngestBatchOutcome::default();
    if rows.is_empty() {
        if let Some(video_id) = video_id {
            let already_ingested = match ledger.row(video_id)? {
                Some(row) => should_skip_wiki_ingest_async(data_dir, &row, force).await,
                None => false,
            };
            if already_ingested {
                info!(%video_id, "wiki article already ingested; pass --force to re-ingest");
            } else {
                info!(%video_id, "no wiki article pending ingestion");
            }
        } else {
            info!("no wiki articles pending ingestion");
        }
        return Ok(outcome);
    }

    let _lock = acquire_wiki_ingest_lock(data_dir)?;
    let preflight_command =
        render_wiki_ingest_preflight_command(data_dir, &config.template, Some(&rows[0]))?;
    preflight_wiki_ingest_command(
        &preflight_command.program,
        &config.cwd,
        config.create_cwd_for_preflight,
        config.uses_default_template,
    )
    .await?;

    let mut failed_video_ids = Vec::new();
    let mut attempted_invocations = 0usize;
    let mut local_missing_plugin_hint_emitted = false;
    let missing_plugin_hint_emitted =
        missing_plugin_hint_emitted.unwrap_or(&mut local_missing_plugin_hint_emitted);

    for row in rows {
        interrupts.check()?;
        let row = match ledger.row(&row.video_id) {
            Ok(Some(refreshed)) => refreshed,
            Ok(None) => row,
            Err(err) => {
                outcome.failed += 1;
                failed_video_ids.push(row.video_id.clone());
                let message = format!("{WIKI_INGEST_ERROR_PREFIX}failed: {}", one_line_error(&err));
                error!(video_id = %row.video_id, error = %message, "wiki ingestion failed");
                if let Err(err) = ledger.mark_error(&row.video_id, &message) {
                    warn!(video_id = %row.video_id, error = %err, original_error = %message, "failed to record wiki ingestion error in ledger");
                }
                continue;
            }
        };
        if !should_attempt_wiki_ingest_row(data_dir, &row, retry_errors, force).await {
            outcome.skipped += 1;
            info!(video_id = %row.video_id, "wiki ingestion skipped because the refreshed row is no longer pending");
            continue;
        }
        // Match `should_attempt_wiki_ingest_row`: only treat an existing
        // error as retry-eligible when it was recorded by the wiki-ingest
        // stage itself.
        let force_for_row =
            force || (retry_errors && is_wiki_ingest_ledger_error(row.error.as_deref()));
        if should_skip_wiki_ingest_async(data_dir, &row, force_for_row).await {
            outcome.skipped += 1;
            info!(video_id = %row.video_id, "wiki ingestion skipped");
            continue;
        }

        let result = run_wiki_ingest_row(data_dir, ledger, config, &row, interrupts).await;
        match result {
            Ok(RunWikiIngestRowOutcome::Succeeded) => {
                attempted_invocations += 1;
                outcome.succeeded += 1;
                info!(video_id = %row.video_id, "wiki ingestion completed");
            }
            Ok(RunWikiIngestRowOutcome::CommandFailed { status, stderr }) => {
                attempted_invocations += 1;
                outcome.failed += 1;
                failed_video_ids.push(row.video_id.clone());
                let stderr_for_hint = String::from_utf8_lossy(&stderr);
                let stderr_tail =
                    stderr_tail_one_line_limited(&stderr, WIKI_INGEST_STDERR_LEDGER_LIMIT);
                if should_emit_missing_wiki_plugin_hint(
                    config.uses_default_template,
                    attempted_invocations,
                    *missing_plugin_hint_emitted,
                    stderr_for_hint.as_ref(),
                ) {
                    eprintln!("{}", wiki_ingest_install_hint());
                    *missing_plugin_hint_emitted = true;
                }
                let message = format!(
                    "{WIKI_INGEST_ERROR_PREFIX}exited {}: {}",
                    exit_status_code(&status),
                    stderr_tail
                );
                error!(video_id = %row.video_id, error = %message, "wiki ingestion failed");
                if let Err(err) = ledger.mark_error(&row.video_id, &message) {
                    warn!(video_id = %row.video_id, error = %err, original_error = %message, "failed to record wiki ingestion error in ledger");
                }
            }
            Err(err) => {
                if is_interrupted_error(&err) {
                    return Err(err);
                }
                outcome.failed += 1;
                failed_video_ids.push(row.video_id.clone());
                let message = format!("{WIKI_INGEST_ERROR_PREFIX}failed: {}", one_line_error(&err));
                error!(video_id = %row.video_id, error = %message, "wiki ingestion failed");
                if let Err(err) = ledger.mark_error(&row.video_id, &message) {
                    warn!(video_id = %row.video_id, error = %err, original_error = %message, "failed to record wiki ingestion error in ledger");
                }
            }
        }
    }

    if outcome.succeeded == 0 && outcome.failed > 0 {
        bail!(
            "every wiki ingestion failed ({} failure(s)): {}",
            outcome.failed,
            failed_video_ids.join(", ")
        );
    }

    Ok(outcome)
}

#[derive(Debug)]
enum RunWikiIngestRowOutcome {
    Succeeded,
    CommandFailed { status: ExitStatus, stderr: Vec<u8> },
}

async fn run_wiki_ingest_row(
    data_dir: &Path,
    ledger: &Ledger,
    config: &WikiIngestConfig,
    row: &VideoRow,
    interrupts: &Interrupts,
) -> Result<RunWikiIngestRowOutcome> {
    let wiki_path = wiki_path_from_row(data_dir, row)?;
    if !async_fs_path_is_file(wiki_path.clone()).await {
        bail!("wiki file is missing on disk: {}", wiki_path.display());
    }

    let command = render_wiki_ingest_command(data_dir, &config.template, row)?;
    let output = run_wiki_ingest_command(&command, &config.cwd, config.timeout, interrupts).await?;
    if output.status.success() {
        ledger.mark_wiki_ingested(row, &command.rendered)?;
        Ok(RunWikiIngestRowOutcome::Succeeded)
    } else {
        Ok(RunWikiIngestRowOutcome::CommandFailed {
            status: output.status,
            stderr: output.stderr,
        })
    }
}

async fn wiki_ingest_candidate_rows(
    data_dir: &Path,
    ledger: &Ledger,
    video_id: Option<&str>,
    retry_errors: bool,
    force: bool,
    limit: Option<usize>,
) -> Result<Vec<VideoRow>> {
    let rows = match video_id {
        Some(video_id) => {
            let row = ledger
                .row(video_id)?
                .ok_or_else(|| anyhow!("video {video_id} is not in the ledger"))?;
            if row.wiki_emitted_at.is_none() {
                bail!(
                    "video {video_id} has no emitted wiki article; run ingest first to emit wiki markdown"
                );
            }
            if !force && !retry_errors && row.error.is_some() {
                bail!(
                    "video {video_id} has a recorded error; pass --retry-errors to retry it or --force to re-ingest"
                );
            }
            vec![row]
        }
        None => ledger.wiki_ingest_rows(retry_errors, force)?,
    };

    let mut candidates = Vec::new();
    for row in rows {
        // Candidate selection happens before the batch lock, so every row is
        // refreshed and checked again inside run_wiki_ingest_batch.
        if should_attempt_wiki_ingest_row(data_dir, &row, retry_errors, force).await {
            candidates.push(row);
            if candidates.len() == limit.unwrap_or(usize::MAX) {
                break;
            }
        }
    }
    Ok(candidates)
}

async fn should_attempt_wiki_ingest_row(
    data_dir: &Path,
    row: &VideoRow,
    retry_errors: bool,
    force: bool,
) -> bool {
    if row.wiki_emitted_at.is_none() {
        return false;
    }
    // Only retry rows whose existing error was actually recorded by the
    // wiki-ingest stage. Unrelated errors (stale download, transcript
    // corruption, ...) must not trigger a paid LLM re-invocation just
    // because the caller (e.g. `ingest --auto-wiki-ingest`) passes
    // `retry_errors = true`.
    let retry_eligible_error = retry_errors && is_wiki_ingest_ledger_error(row.error.as_deref());
    if !force && !retry_eligible_error && row.error.is_some() {
        return false;
    }
    if force || row.wiki_ingested_at.is_none() || retry_eligible_error {
        return true;
    }

    !should_skip_wiki_ingest_async(data_dir, row, false).await
}

fn render_wiki_ingest_command(
    data_dir: &Path,
    template: &str,
    row: &VideoRow,
) -> Result<RenderedWikiIngestCommand> {
    let values = wiki_ingest_template_values(data_dir, row)?;
    let rendered = render_wiki_ingest_template(template, &values);
    let argv = shell_words::split(&rendered).context("parse rendered wiki ingestion command")?;
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| anyhow!("wiki ingestion command must not be empty"))?;
    Ok(RenderedWikiIngestCommand {
        rendered,
        program: program.to_owned(),
        args: args.to_vec(),
    })
}

fn render_wiki_ingest_template(template: &str, values: &WikiIngestTemplateValues) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    let mut state = ShellRenderState::default();

    while let Some(index) = rest.find('{') {
        let literal = &rest[..index];
        rendered.push_str(literal);
        update_shell_render_state(&mut state, literal);
        let token_start = &rest[index..];
        if !state.escaped
            && let Some((token, value)) = wiki_ingest_template_token(token_start, values)
        {
            rendered.push_str(&quote_for_shell_context(value, state.quote));
            state.escaped = false;
            rest = &token_start[token.len()..];
        } else {
            rendered.push('{');
            update_shell_render_state(&mut state, "{");
            rest = &token_start[1..];
        }
    }
    rendered.push_str(rest);
    rendered
}

fn update_shell_render_state(state: &mut ShellRenderState, literal: &str) {
    for ch in literal.chars() {
        match state.quote {
            ShellQuoteContext::Unquoted => {
                if state.escaped {
                    state.escaped = false;
                } else if ch == '\\' {
                    state.escaped = true;
                } else if ch == '\'' {
                    state.quote = ShellQuoteContext::Single;
                } else if ch == '"' {
                    state.quote = ShellQuoteContext::Double;
                }
            }
            ShellQuoteContext::Single => {
                if ch == '\'' {
                    state.quote = ShellQuoteContext::Unquoted;
                }
            }
            ShellQuoteContext::Double => {
                if state.escaped {
                    state.escaped = false;
                } else if ch == '\\' {
                    state.escaped = true;
                } else if ch == '"' {
                    state.quote = ShellQuoteContext::Unquoted;
                }
            }
        }
    }
}

fn quote_for_shell_context(value: &str, context: ShellQuoteContext) -> String {
    match context {
        ShellQuoteContext::Unquoted => shell_words::quote(value).into_owned(),
        ShellQuoteContext::Single => value.replace('\'', r#"'\''"#),
        ShellQuoteContext::Double => {
            let mut escaped = String::with_capacity(value.len());
            for ch in value.chars() {
                if matches!(ch, '\\' | '"' | '$' | '`') {
                    escaped.push('\\');
                }
                escaped.push(ch);
            }
            escaped
        }
    }
}

fn wiki_ingest_template_token<'a>(
    value: &str,
    values: &'a WikiIngestTemplateValues,
) -> Option<(&'static str, &'a str)> {
    if value.starts_with("{path}") {
        Some(("{path}", &values.path))
    } else if value.starts_with("{video_id}") {
        Some(("{video_id}", &values.video_id))
    } else if value.starts_with("{title}") {
        Some(("{title}", &values.title))
    } else if value.starts_with("{channel_slug}") {
        Some(("{channel_slug}", &values.channel_slug))
    } else {
        None
    }
}

fn wiki_ingest_template_values(
    data_dir: &Path,
    row: &VideoRow,
) -> Result<WikiIngestTemplateValues> {
    let wiki_path = wiki_path_from_row(data_dir, row)?;
    let absolute_wiki_path = absolutize_path(&wiki_path)?;
    Ok(WikiIngestTemplateValues {
        path: path_to_string(&absolute_wiki_path),
        video_id: row.video_id.clone(),
        title: row.title.clone().unwrap_or_default(),
        channel_slug: channel_slug_from_wiki_path(&wiki_path),
    })
}

fn render_wiki_ingest_preflight_command(
    data_dir: &Path,
    template: &str,
    row: Option<&VideoRow>,
) -> Result<RenderedWikiIngestCommand> {
    let values = match row {
        Some(row) => wiki_ingest_preflight_template_values(data_dir, row)?,
        None => {
            let path = absolutize_path(&data_dir.join("wiki").join(".preflight.md"))?;
            WikiIngestTemplateValues {
                path: path_to_string(&path),
                video_id: "preflight".to_owned(),
                title: String::new(),
                channel_slug: String::new(),
            }
        }
    };
    let rendered = render_wiki_ingest_template(template, &values);
    let argv = shell_words::split(&rendered).context("parse rendered wiki ingestion command")?;
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| anyhow!("wiki ingestion command must not be empty"))?;
    Ok(RenderedWikiIngestCommand {
        rendered,
        program: program.to_owned(),
        args: args.to_vec(),
    })
}

fn wiki_ingest_preflight_template_values(
    data_dir: &Path,
    row: &VideoRow,
) -> Result<WikiIngestTemplateValues> {
    let wiki_path = row
        .wiki_path
        .as_deref()
        .map(|path| ledger_path_to_fs_path(data_dir, path))
        .unwrap_or_else(|| data_dir.join("wiki").join(".preflight.md"));
    let absolute_wiki_path = absolutize_path(&wiki_path)?;
    Ok(WikiIngestTemplateValues {
        path: path_to_string(&absolute_wiki_path),
        video_id: row.video_id.clone(),
        title: row.title.clone().unwrap_or_default(),
        channel_slug: channel_slug_from_wiki_path(&wiki_path),
    })
}

fn wiki_path_from_row(data_dir: &Path, row: &VideoRow) -> Result<PathBuf> {
    let path = row
        .wiki_path
        .as_deref()
        .ok_or_else(|| anyhow!("ledger row for {} has no wiki_path", row.video_id))?;
    Ok(ledger_path_to_fs_path(data_dir, path))
}

fn channel_slug_from_wiki_path(path: &Path) -> String {
    path.parent()
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_owned()
}

async fn preflight_wiki_ingest_config(data_dir: &Path, config: &WikiIngestConfig) -> Result<()> {
    let command = render_wiki_ingest_preflight_command(data_dir, &config.template, None)?;
    preflight_wiki_ingest_command(
        &command.program,
        &config.cwd,
        config.create_cwd_for_preflight,
        config.uses_default_template,
    )
    .await
}

async fn preflight_wiki_ingest_command(
    program: &str,
    cwd: &Path,
    create_cwd: bool,
    uses_default_template: bool,
) -> Result<()> {
    if !create_cwd {
        let metadata = fs::metadata(cwd)
            .await
            .with_context(|| format!("wiki ingestion cwd does not exist: {}", cwd.display()))?;
        if !metadata.is_dir() {
            bail!("wiki ingestion cwd is not a directory: {}", cwd.display());
        }
    }

    if !command_exists(program, cwd).await {
        return Err(ExitCodeError::new(
            3,
            format!(
                "error: wiki ingestion command not found: '{}'\n{}",
                program,
                wiki_ingest_command_not_found_hint(uses_default_template)
            ),
        )
        .into());
    }

    if create_cwd {
        fs::create_dir_all(cwd)
            .await
            .with_context(|| format!("create wiki ingestion cwd {}", cwd.display()))?;
    }

    Ok(())
}

async fn command_exists(program: &str, cwd: &Path) -> bool {
    let program_path = Path::new(program);
    if program_path.is_absolute() || program_has_path_separator(program) {
        return executable_path_exists(&resolve_wiki_ingest_program(program, cwd)).await;
    }

    let Some(paths) = env::var_os("PATH") else {
        return false;
    };
    for dir in env::split_paths(&paths) {
        if executable_path_exists(&dir.join(program)).await {
            return true;
        }
    }
    false
}

/// Resolve a `--wiki-ingest-cmd` program to the path we will hand to
/// `Command::new`. Keeps the resolution rule identical for preflight
/// (`command_exists`) and execution (`run_wiki_ingest_command`):
///
/// - Absolute path → use as-is.
/// - Relative path with a separator (e.g. `./ingest.sh`) → resolve
///   against `cwd` so spawn doesn't depend on Rust/OS quirks around
///   the `Command::new` + `current_dir` combination (Windows resolves
///   the executable before the cwd takes effect).
/// - Bare program (e.g. `claude`) → leave alone; the OS handles `PATH`
///   resolution.
fn resolve_wiki_ingest_program(program: &str, cwd: &Path) -> PathBuf {
    let program_path = Path::new(program);
    if program_path.is_absolute() {
        program_path.to_path_buf()
    } else if program_has_path_separator(program) {
        cwd.join(program_path)
    } else {
        program_path.to_path_buf()
    }
}

fn program_has_path_separator(program: &str) -> bool {
    #[cfg(windows)]
    {
        program.contains('/') || program.contains('\\')
    }
    #[cfg(not(windows))]
    {
        program.contains(std::path::MAIN_SEPARATOR)
    }
}

#[cfg(not(windows))]
async fn executable_path_exists(path: &Path) -> bool {
    is_executable_file(path).await
}

#[cfg(windows)]
async fn executable_path_exists(path: &Path) -> bool {
    for candidate in windows_executable_candidates(path) {
        if is_executable_file(&candidate).await {
            return true;
        }
    }
    false
}

#[cfg(unix)]
async fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    fs::metadata(path)
        .await
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}

#[cfg(all(not(unix), not(windows)))]
async fn is_executable_file(path: &Path) -> bool {
    fs::metadata(path)
        .await
        .is_ok_and(|metadata| metadata.is_file())
}

#[cfg(windows)]
async fn is_executable_file(path: &Path) -> bool {
    if !fs::metadata(path)
        .await
        .is_ok_and(|metadata| metadata.is_file())
    {
        return false;
    }
    windows_executable_extensions().iter().any(|extension| {
        path.extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| format!(".{value}").eq_ignore_ascii_case(extension))
    })
}

#[cfg(windows)]
fn windows_executable_candidates(path: &Path) -> Vec<PathBuf> {
    if path.extension().is_some() {
        return vec![path.to_path_buf()];
    }
    windows_executable_extensions()
        .into_iter()
        .map(|extension| {
            let mut candidate = path.as_os_str().to_owned();
            candidate.push(extension);
            PathBuf::from(candidate)
        })
        .collect()
}

#[cfg(windows)]
fn windows_executable_extensions() -> Vec<String> {
    env::var_os("PATHEXT")
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".to_owned())
        .split(';')
        .filter_map(|value| {
            let value = value.trim();
            if value.is_empty() {
                None
            } else if value.starts_with('.') {
                Some(value.to_owned())
            } else {
                Some(format!(".{value}"))
            }
        })
        .collect()
}

fn wiki_ingest_install_hint() -> &'static str {
    "hint: install Claude Code (https://docs.claude.com/en/docs/claude-code/quickstart)\n      then run: claude plugin install wiki@llm-wiki\n      or override with --wiki-ingest-cmd '<your command>'"
}

fn wiki_ingest_command_not_found_hint(uses_default_template: bool) -> &'static str {
    if uses_default_template {
        wiki_ingest_install_hint()
    } else {
        "hint: check --wiki-ingest-cmd or YTARCH_WIKI_INGEST_CMD and ensure the command is installed and on PATH"
    }
}

async fn run_wiki_ingest_command(
    command: &RenderedWikiIngestCommand,
    cwd: &Path,
    timeout: Duration,
    interrupts: &Interrupts,
) -> Result<WikiIngestCommandOutput> {
    let deadline = std::time::Instant::now()
        .checked_add(timeout)
        .map(Instant::from_std)
        .ok_or_else(|| anyhow!("wiki-ingest timeout is too large: {}s", timeout.as_secs()))?;

    // Resolve cwd-relative executables (e.g. `./ingest.sh`) to an
    // absolute path BEFORE handing off to Command::new. `Command::new`
    // combined with `.current_dir(cwd)` has platform-specific behavior
    // — on Windows the program is resolved before the cwd is applied,
    // so preflight (which checks `cwd.join(program)`) can pass while
    // spawn fails. Aligning the resolution rule between preflight and
    // spawn avoids that drift.
    let resolved_program = resolve_wiki_ingest_program(&command.program, cwd);
    let mut child_command = Command::new(&resolved_program);
    child_command
        .args(&command.args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_command_process_group(&mut child_command);
    let mut child = child_command
        .spawn()
        .with_context(|| format!("run {}", command.rendered))?;
    let process_group = match command_child_process_group(&child, &command.rendered) {
        Ok(process_group) => process_group,
        Err(err) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(err).with_context(|| {
                format!(
                    "configure wiki ingestion process group for {}",
                    command.rendered
                )
            });
        }
    };

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("capture stdout for {}", command.rendered))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("capture stderr for {}", command.rendered))?;
    let stdout_progress = Arc::new(AtomicU64::new(0));
    let stdout_reader_progress = Arc::clone(&stdout_progress);
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
            stdout_reader_progress.fetch_add(1, Ordering::Relaxed);
            if !live_stdout_failed {
                match live_stdout.write_all(&chunk[..read]).await {
                    Ok(()) => {}
                    Err(err) => {
                        live_stdout_failed = true;
                        warn!(stream = "stdout", error = %err, "failed to write child output to live stream");
                    }
                }
            }
            truncated |= push_captured_output(&mut captured, &chunk[..read]);
        }
        if !live_stdout_failed {
            match live_stdout.flush().await {
                Ok(()) => {}
                Err(err) => {
                    warn!(stream = "stdout", error = %err, "failed to flush child output live stream");
                }
            }
        }

        if truncated {
            add_truncation_notice(&mut captured, "stdout");
        }

        Ok::<Vec<u8>, std::io::Error>(captured)
    }));
    let stderr_progress = Arc::new(AtomicU64::new(0));
    let stderr_reader_progress = Arc::clone(&stderr_progress);
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
            stderr_reader_progress.fetch_add(1, Ordering::Relaxed);
            if !live_stderr_failed {
                match live_stderr.write_all(&chunk[..read]).await {
                    Ok(()) => {}
                    Err(err) => {
                        live_stderr_failed = true;
                        warn!(stream = "stderr", error = %err, "failed to write child output to live stream");
                    }
                }
            }
            truncated |= push_captured_output_with_limit(
                &mut captured,
                &chunk[..read],
                WIKI_INGEST_STDERR_CAPTURE_LIMIT,
            );
        }
        if !live_stderr_failed {
            match live_stderr.flush().await {
                Ok(()) => {}
                Err(err) => {
                    warn!(stream = "stderr", error = %err, "failed to flush child output live stream");
                }
            }
        }
        if truncated {
            add_truncation_notice(&mut captured, "stderr");
        }

        Ok::<Vec<u8>, std::io::Error>(captured)
    }));

    let status = tokio::select! {
        status = child.wait() => {
            match status {
                Ok(status) => {
                    terminate_command_process_group(process_group, "process exit", &command.rendered).await;
                    status
                }
                Err(err) => {
                    terminate_command_process_group(process_group, "wait error", &command.rendered).await;
                    return Err(err).with_context(|| format!("wait for {}", command.rendered));
                }
            }
        }
        () = tokio::time::sleep_until(deadline) => {
            kill_timed_out_child(&mut child, process_group, &command.rendered).await;
            let drain_deadline = Instant::now() + STREAM_READER_DRAIN_TIMEOUT;
            let (_stdout, stderr) = tokio::join!(
                join_stream_reader_until(
                    stdout_task,
                    "stdout",
                    &command.rendered,
                    &stdout_progress,
                    drain_deadline,
                ),
                join_stream_reader_until(
                    stderr_task,
                    "stderr",
                    &command.rendered,
                    &stderr_progress,
                    drain_deadline,
                )
            );
            let stderr = stderr.unwrap_or_default();
            bail!(
                "wiki-ingest timed out after {}s: {}",
                timeout.as_secs(),
                stderr_tail_one_line_limited(&stderr, WIKI_INGEST_STDERR_LEDGER_LIMIT)
            );
        }
        () = interrupts.wait() => {
            // tokio::select! makes no ordering guarantees, so an interrupt
            // arriving at the same instant the child exits 0 can win the
            // race. Before tearing the child down, see if it has actually
            // finished; if so, treat this as a normal completion so a
            // successful ingest isn't recorded as an interrupted failure.
            match child.try_wait() {
                Ok(Some(status)) => {
                    terminate_command_process_group(process_group, "interrupt-after-exit", &command.rendered).await;
                    status
                }
                _ => {
                    interrupt_command_child(&mut child, process_group, &command.rendered).await;
                    let drain_deadline = Instant::now() + STREAM_READER_DRAIN_TIMEOUT;
                    let (stdout, stderr) = tokio::join!(
                        join_stream_reader_until(
                            stdout_task,
                            "stdout",
                            &command.rendered,
                            &stdout_progress,
                            drain_deadline,
                        ),
                        join_stream_reader_until(
                            stderr_task,
                            "stderr",
                            &command.rendered,
                            &stderr_progress,
                            drain_deadline,
                        )
                    );
                    if stdout.is_err() || stderr.is_err() {
                        kill_command_process_group(process_group, "interrupt stream drain failure", &command.rendered);
                    }
                    return Err(InterruptedError.into());
                }
            }
        }
    };
    let drain_deadline = Instant::now() + STREAM_READER_DRAIN_TIMEOUT;
    let (stdout, stderr) = tokio::join!(
        join_stream_reader_until(
            stdout_task,
            "stdout",
            &command.rendered,
            &stdout_progress,
            drain_deadline,
        ),
        join_stream_reader_until(
            stderr_task,
            "stderr",
            &command.rendered,
            &stderr_progress,
            drain_deadline,
        )
    );
    let (_stdout, stderr) = match (stdout, stderr) {
        (Ok(stdout), Ok(stderr)) => (stdout, stderr),
        (stdout, stderr) => {
            kill_command_process_group(process_group, "stream drain failure", &command.rendered);
            if Instant::now() >= drain_deadline {
                let stderr = stderr.unwrap_or_default();
                bail!(
                    "timed out draining wiki-ingest output after process exit: {}",
                    stderr_tail_one_line_limited(&stderr, WIKI_INGEST_STDERR_LEDGER_LIMIT)
                );
            }
            (stdout?, stderr?)
        }
    };

    Ok(WikiIngestCommandOutput { status, stderr })
}

fn configure_command_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        command.process_group(0);
    }
}

fn command_child_process_group(
    child: &tokio::process::Child,
    command: &str,
) -> Result<Option<CommandProcessGroup>> {
    #[cfg(unix)]
    {
        let Some(raw_pid) = child.id() else {
            warn!(
                command,
                "child has no process id; process-group cleanup unavailable"
            );
            return Ok(None);
        };
        let pid = libc::pid_t::try_from(raw_pid).context("child pid does not fit pid_t")?;

        set_command_child_process_group_from_parent(pid)?;

        for _ in 0..5 {
            let actual_pgid = unsafe { libc::getpgid(pid) };
            if actual_pgid == pid {
                return Ok(Some(CommandProcessGroup { pgid: pid }));
            }
            if actual_pgid >= 0 {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }

            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::ESRCH) {
                warn!(pid, command, error = %err, "child exited before process group verification completed; retaining expected process group id");
                return Ok(Some(CommandProcessGroup { pgid: pid }));
            }
            return Err(err).with_context(|| format!("verify process group for {command}"));
        }

        let actual_pgid = unsafe { libc::getpgid(pid) };
        if actual_pgid == pid {
            Ok(Some(CommandProcessGroup { pgid: pid }))
        } else if actual_pgid >= 0 {
            bail!(
                "child was not started in its own process group for {command}: pid {pid}, pgid {actual_pgid}"
            );
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::ESRCH) {
                warn!(pid, command, error = %err, "child exited before process group verification completed; retaining expected process group id");
                Ok(Some(CommandProcessGroup { pgid: pid }))
            } else {
                Err(err).with_context(|| format!("verify process group for {command}"))
            }
        }
    }

    #[cfg(not(unix))]
    {
        let _ = child;
        let _ = command;
        Ok(None)
    }
}

#[cfg(unix)]
fn set_command_child_process_group_from_parent(pid: libc::pid_t) -> Result<()> {
    let result = unsafe { libc::setpgid(pid, pid) };
    if result == 0 {
        return Ok(());
    }

    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        Some(libc::EACCES | libc::EPERM | libc::ESRCH) => Ok(()),
        _ => Err(err).context("set child process group"),
    }
}

async fn kill_timed_out_child(
    child: &mut tokio::process::Child,
    process_group: Option<CommandProcessGroup>,
    command: &str,
) {
    kill_command_process_group(process_group, "timeout", command);
    let _ = child.start_kill();
    let _ = child.wait().await;
}

async fn interrupt_command_child(
    child: &mut tokio::process::Child,
    process_group: Option<CommandProcessGroup>,
    command: &str,
) {
    #[cfg(unix)]
    {
        if process_group.is_some() {
            signal_command_process_group(process_group, libc::SIGINT, "interrupt", command);
        } else {
            let _ = child.start_kill();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = process_group;
        let _ = command;
        let _ = child.start_kill();
    }

    match tokio::time::timeout(STREAM_READER_DRAIN_TIMEOUT, child.wait()).await {
        Ok(Ok(_)) => {}
        Ok(Err(err)) => {
            warn!(command, error = %err, "failed to wait for interrupted child");
        }
        Err(_) => {
            kill_timed_out_child(child, process_group, command).await;
        }
    }
}

async fn terminate_command_process_group(
    process_group: Option<CommandProcessGroup>,
    reason: &'static str,
    command: &str,
) {
    #[cfg(unix)]
    {
        if signal_command_process_group(process_group, libc::SIGTERM, reason, command) {
            tokio::time::sleep(PROCESS_GROUP_TERMINATE_GRACE).await;
            kill_command_process_group(process_group, reason, command);
        }
    }

    #[cfg(not(unix))]
    {
        let _ = process_group;
        let _ = reason;
        let _ = command;
    }
}

fn kill_command_process_group(
    process_group: Option<CommandProcessGroup>,
    reason: &'static str,
    command: &str,
) {
    #[cfg(unix)]
    signal_command_process_group(process_group, libc::SIGKILL, reason, command);

    #[cfg(not(unix))]
    {
        let _ = process_group;
        let _ = reason;
        let _ = command;
    }
}

#[cfg(unix)]
fn signal_command_process_group(
    process_group: Option<CommandProcessGroup>,
    signal: libc::c_int,
    reason: &'static str,
    command: &str,
) -> bool {
    let Some(process_group) = process_group else {
        return false;
    };
    let result = unsafe { libc::kill(-process_group.pgid, signal) };
    if result != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::ESRCH) {
            warn!(
                pgid = process_group.pgid,
                signal,
                reason,
                command,
                error = %err,
                "failed to signal child process group"
            );
        }
        return false;
    }
    true
}

#[cfg(unix)]
async fn wait_for_process_interrupt() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigint = match signal(SignalKind::interrupt()) {
        Ok(signal) => Some(signal),
        Err(err) => {
            warn!(error = %err, "failed to install SIGINT handler");
            None
        }
    };
    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(signal) => Some(signal),
        Err(err) => {
            warn!(error = %err, "failed to install SIGTERM handler");
            None
        }
    };

    match (&mut sigint, &mut sigterm) {
        (Some(sigint), Some(sigterm)) => {
            tokio::select! {
                _ = sigint.recv() => {}
                _ = sigterm.recv() => {}
            }
        }
        (Some(sigint), None) => {
            let _ = sigint.recv().await;
        }
        (None, Some(sigterm)) => {
            let _ = sigterm.recv().await;
        }
        (None, None) => future::pending::<()>().await,
    }
}

#[cfg(not(unix))]
async fn wait_for_process_interrupt() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        warn!(error = %err, "failed to install Ctrl-C handler");
        future::pending::<()>().await;
    }
}

fn exit_status_code(status: &ExitStatus) -> String {
    status
        .code()
        .map_or_else(|| status.to_string(), |code| code.to_string())
}

#[cfg(test)]
fn stderr_tail_one_line(stderr: &[u8]) -> String {
    stderr_tail_one_line_limited(stderr, stderr.len())
}

fn stderr_tail_one_line_limited(stderr: &[u8], limit: usize) -> String {
    let stderr = if limit == 0 || stderr.len() <= limit {
        stderr
    } else {
        &stderr[stderr.len() - limit..]
    };
    let tail = String::from_utf8_lossy(stderr);
    let tail = escaped_one_line(tail.trim());
    if tail.is_empty() {
        "(no stderr)".to_owned()
    } else {
        tail
    }
}

fn escaped_one_line(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                output.push_str(r"\n");
            }
            '\n' => output.push_str(r"\n"),
            '\t' => output.push_str(r"\t"),
            ch if ch.is_control() => output.push(' '),
            ch => output.push(ch),
        }
    }
    output
}

fn one_line_error(err: &anyhow::Error) -> String {
    format!("{err:#}")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_missing_wiki_plugin_error(stderr_tail: &str) -> bool {
    stderr_tail
        .lines()
        .any(|line| MISSING_WIKI_PLUGIN_RE.is_match(line))
}

fn should_emit_missing_wiki_plugin_hint(
    uses_default_template: bool,
    attempted_invocations: usize,
    already_emitted: bool,
    stderr_tail: &str,
) -> bool {
    uses_default_template
        && attempted_invocations == 1
        && !already_emitted
        && is_missing_wiki_plugin_error(stderr_tail)
}

fn is_wiki_ingest_ledger_error(error: Option<&str>) -> bool {
    error.is_some_and(|error| error.starts_with(WIKI_INGEST_ERROR_PREFIX))
}

async fn list_rows(data_dir: &Path) -> Result<Vec<VideoRow>> {
    let rows = Ledger::open_read_only(data_dir)?
        .map(|ledger| ledger.rows())
        .transpose()?
        .unwrap_or_default();
    let mut archived = Vec::new();
    for row in rows {
        if is_archived_row(data_dir, &row).await {
            archived.push(row);
        }
    }
    Ok(archived)
}

async fn is_archived_row(data_dir: &Path, row: &VideoRow) -> bool {
    row.downloaded_at.is_some()
        && row.transcribed_at.is_some()
        && row.wiki_emitted_at.is_some()
        && row
            .error
            .as_deref()
            .is_none_or(|error| is_wiki_ingest_ledger_error(Some(error)))
        && async_ledger_path_is_file(data_dir, row.audio_path.as_deref()).await
        && async_transcript_paths_exist(data_dir, row.transcript_path.as_deref()).await
        && async_ledger_path_is_file(data_dir, row.wiki_path.as_deref()).await
}

fn row_from_sql(row: &rusqlite::Row<'_>) -> rusqlite::Result<VideoRow> {
    let tags_column = row.as_ref().column_index("tags")?;
    let duration: Option<i64> = row.get("duration")?;
    let tags_json: Option<String> = row.get("tags")?;
    let tags = match tags_json.as_deref() {
        Some(value) => serde_json::from_str(value).map_err(|err| {
            rusqlite::Error::FromSqlConversionFailure(
                tags_column,
                rusqlite::types::Type::Text,
                Box::new(err),
            )
        })?,
        None => Vec::new(),
    };
    Ok(VideoRow {
        video_id: row.get("video_id")?,
        url: row.get("url")?,
        channel_id: row.get("channel_id")?,
        channel_title: row.get("channel_title")?,
        uploader: row.get("uploader")?,
        title: row.get("title")?,
        upload_date: row.get("upload_date")?,
        duration: duration.and_then(|value| u64::try_from(value).ok()),
        tags,
        downloaded_at: row.get("downloaded_at")?,
        transcribed_at: row.get("transcribed_at")?,
        wiki_emitted_at: row.get("wiki_emitted_at")?,
        wiki_ingested_at: row.get("wiki_ingested_at")?,
        wiki_ingest_cmd: row.get("wiki_ingest_cmd")?,
        whisper_model: row.get("whisper_model")?,
        audio_path: row.get("audio_path")?,
        transcript_path: row.get("transcript_path")?,
        wiki_path: row.get("wiki_path")?,
        error: row.get("error")?,
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

#[cfg(test)]
fn should_skip_wiki_ingest(data_dir: &Path, row: &VideoRow, force: bool) -> bool {
    !force
        && row.wiki_ingested_at.is_some()
        && row
            .wiki_path
            .as_deref()
            .is_some_and(|path| ledger_path_to_fs_path(data_dir, path).is_file())
}

async fn should_skip_wiki_ingest_async(data_dir: &Path, row: &VideoRow, force: bool) -> bool {
    !force
        && row.wiki_ingested_at.is_some()
        && async_ledger_path_is_file(data_dir, row.wiki_path.as_deref()).await
}

async fn async_ledger_path_is_file(data_dir: &Path, path: Option<&str>) -> bool {
    match path {
        Some(path) => async_fs_path_is_file(ledger_path_to_fs_path(data_dir, path)).await,
        None => false,
    }
}

async fn async_transcript_paths_exist(data_dir: &Path, path: Option<&str>) -> bool {
    let Some(path) = path else {
        return false;
    };
    let json_path = ledger_path_to_fs_path(data_dir, path);
    let txt_path = json_path.with_file_name("transcript.txt");
    async_fs_path_is_file(json_path).await && async_fs_path_is_file(txt_path).await
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

async fn wiki_ingest_state(data_dir: &Path, row: &VideoRow) -> &'static str {
    if is_wiki_ingest_ledger_error(row.error.as_deref()) {
        "error"
    } else if row.wiki_ingested_at.is_some() {
        if should_skip_wiki_ingest_async(data_dir, row, false).await {
            "done"
        } else {
            "missing"
        }
    } else if row.wiki_emitted_at.is_some()
        && async_ledger_path_is_file(data_dir, row.wiki_path.as_deref()).await
    {
        "pending"
    } else {
        "-"
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

fn spawn_capture_reader<R>(
    mut stream: R,
    progress: Arc<AtomicU64>,
    stream_name: &'static str,
) -> AbortOnDrop<std::io::Result<Vec<u8>>>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    AbortOnDrop::new(tokio::spawn(async move {
        let mut captured = Vec::new();
        let mut truncated = false;
        let mut chunk = [0u8; 8192];

        loop {
            let read = stream.read(&mut chunk).await?;
            if read == 0 {
                break;
            }
            progress.fetch_add(1, Ordering::Relaxed);
            truncated |= push_captured_output(&mut captured, &chunk[..read]);
        }

        if truncated {
            add_truncation_notice(&mut captured, stream_name);
        }

        Ok(captured)
    }))
}

async fn run_checked(program: &str, args: &[String], interrupts: &Interrupts) -> Result<Output> {
    let command = format_command(program, args);
    let mut child_command = Command::new(program);
    child_command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_command_process_group(&mut child_command);
    let mut child = child_command
        .spawn()
        .with_context(|| format!("run {command}"))?;
    let process_group = match command_child_process_group(&child, &command) {
        Ok(process_group) => process_group,
        Err(err) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(err).with_context(|| format!("configure process group for {command}"));
        }
    };

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("capture stdout for {command}"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("capture stderr for {command}"))?;
    let stdout_progress = Arc::new(AtomicU64::new(0));
    let stdout_task = spawn_capture_reader(stdout, Arc::clone(&stdout_progress), "stdout");
    let stderr_progress = Arc::new(AtomicU64::new(0));
    let stderr_task = spawn_capture_reader(stderr, Arc::clone(&stderr_progress), "stderr");

    let status = tokio::select! {
        status = child.wait() => {
            match status {
                Ok(status) => {
                    terminate_command_process_group(process_group, "process exit", &command).await;
                    status
                }
                Err(err) => {
                    terminate_command_process_group(process_group, "wait error", &command).await;
                    return Err(err).with_context(|| format!("wait for {command}"));
                }
            }
        }
        () = interrupts.wait() => {
            interrupt_command_child(&mut child, process_group, &command).await;
            let drain_deadline = Instant::now() + STREAM_READER_DRAIN_TIMEOUT;
            let _ = tokio::join!(
                join_stream_reader_until(
                    stdout_task,
                    "stdout",
                    &command,
                    &stdout_progress,
                    drain_deadline,
                ),
                join_stream_reader_until(
                    stderr_task,
                    "stderr",
                    &command,
                    &stderr_progress,
                    drain_deadline,
                )
            );
            return Err(InterruptedError.into());
        }
    };
    let stdout = join_stream_reader(stdout_task, "stdout", &command, &stdout_progress).await?;
    let stderr = join_stream_reader(stderr_task, "stderr", &command, &stderr_progress).await?;

    let output = Output {
        status,
        stdout,
        stderr,
    };
    ensure_success(program, args, output)
}

async fn run_checked_stream_output(
    program: &str,
    args: &[String],
    interrupts: &Interrupts,
) -> Result<Output> {
    let command = format_command(program, args);
    let mut child_command = Command::new(program);
    child_command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_command_process_group(&mut child_command);
    let mut child = child_command
        .spawn()
        .with_context(|| format!("run {command}"))?;
    let process_group = match command_child_process_group(&child, &command) {
        Ok(process_group) => process_group,
        Err(err) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(err).with_context(|| format!("configure process group for {command}"));
        }
    };
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("capture stdout for {command}"))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("capture stderr for {command}"))?;

    let stdout_progress = Arc::new(AtomicU64::new(0));
    let stdout_reader_progress = Arc::clone(&stdout_progress);
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
            stdout_reader_progress.fetch_add(1, Ordering::Relaxed);
            if !live_stdout_failed {
                match live_stdout.write_all(&chunk[..read]).await {
                    Ok(()) => {}
                    Err(err) => {
                        live_stdout_failed = true;
                        warn!(stream = "stdout", error = %err, "failed to write child output to live stream");
                    }
                }
            }
            truncated |= push_captured_output(&mut captured, &chunk[..read]);
        }
        if !live_stdout_failed {
            match live_stdout.flush().await {
                Ok(()) => {}
                Err(err) => {
                    warn!(stream = "stdout", error = %err, "failed to flush child output live stream");
                }
            }
        }

        if truncated {
            add_truncation_notice(&mut captured, "stdout");
        }

        Ok::<Vec<u8>, std::io::Error>(captured)
    }));
    let stderr_progress = Arc::new(AtomicU64::new(0));
    let stderr_reader_progress = Arc::clone(&stderr_progress);
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
            stderr_reader_progress.fetch_add(1, Ordering::Relaxed);
            if !live_stderr_failed {
                match live_stderr.write_all(&chunk[..read]).await {
                    Ok(()) => {}
                    Err(err) => {
                        live_stderr_failed = true;
                        warn!(stream = "stderr", error = %err, "failed to write child output to live stream");
                    }
                }
            }
            truncated |= push_captured_output(&mut captured, &chunk[..read]);
        }
        if !live_stderr_failed {
            match live_stderr.flush().await {
                Ok(()) => {}
                Err(err) => {
                    warn!(stream = "stderr", error = %err, "failed to flush child output live stream");
                }
            }
        }

        if truncated {
            add_truncation_notice(&mut captured, "stderr");
        }

        Ok::<Vec<u8>, std::io::Error>(captured)
    }));

    let status = tokio::select! {
        status = child.wait() => {
            match status {
                Ok(status) => {
                    terminate_command_process_group(process_group, "process exit", &command).await;
                    status
                }
                Err(err) => {
                    terminate_command_process_group(process_group, "wait error", &command).await;
                    return Err(err).with_context(|| format!("wait for {command}"));
                }
            }
        }
        () = interrupts.wait() => {
            interrupt_command_child(&mut child, process_group, &command).await;
            let drain_deadline = Instant::now() + STREAM_READER_DRAIN_TIMEOUT;
            let _ = tokio::join!(
                join_stream_reader_until(
                    stdout_task,
                    "stdout",
                    &command,
                    &stdout_progress,
                    drain_deadline,
                ),
                join_stream_reader_until(
                    stderr_task,
                    "stderr",
                    &command,
                    &stderr_progress,
                    drain_deadline,
                )
            );
            return Err(InterruptedError.into());
        }
    };
    let stdout = join_stream_reader(stdout_task, "stdout", &command, &stdout_progress).await?;
    let stderr = join_stream_reader(stderr_task, "stderr", &command, &stderr_progress).await?;

    let output = Output {
        status,
        stdout,
        stderr,
    };
    ensure_success(program, args, output)
}

async fn join_stream_reader(
    mut task: AbortOnDrop<std::io::Result<Vec<u8>>>,
    stream_name: &str,
    command: &str,
    stream_progress: &AtomicU64,
) -> Result<Vec<u8>> {
    let mut observed_progress = stream_progress.load(Ordering::Relaxed);

    loop {
        let drain_timeout = tokio::time::sleep(STREAM_READER_DRAIN_TIMEOUT);
        tokio::pin!(drain_timeout);

        tokio::select! {
            join_result = task.handle_mut() => {
                task.clear_completed();
                return join_result
                    .with_context(|| format!("join child {stream_name} reader"))?
                    .with_context(|| format!("stream {stream_name} from {command}"));
            }
            () = &mut drain_timeout => {
                let current_progress = stream_progress.load(Ordering::Relaxed);
                if current_progress == observed_progress {
                    bail!("timed out draining {stream_name} from {command}");
                }
                observed_progress = current_progress;
            }
        }
    }
}

async fn join_stream_reader_until(
    mut task: AbortOnDrop<std::io::Result<Vec<u8>>>,
    stream_name: &str,
    command: &str,
    stream_progress: &AtomicU64,
    deadline: Instant,
) -> Result<Vec<u8>> {
    let mut observed_progress = stream_progress.load(Ordering::Relaxed);

    loop {
        let drain_timeout = tokio::time::sleep(STREAM_READER_DRAIN_TIMEOUT);
        let command_timeout = tokio::time::sleep_until(deadline);
        tokio::pin!(drain_timeout);
        tokio::pin!(command_timeout);

        tokio::select! {
            join_result = task.handle_mut() => {
                task.clear_completed();
                return join_result
                    .with_context(|| format!("join child {stream_name} reader"))?
                    .with_context(|| format!("stream {stream_name} from {command}"));
            }
            () = &mut drain_timeout => {
                let current_progress = stream_progress.load(Ordering::Relaxed);
                if current_progress == observed_progress {
                    bail!("timed out draining {stream_name} from {command}");
                }
                observed_progress = current_progress;
            }
            () = &mut command_timeout => {
                bail!("timed out draining {stream_name} from {command}");
            }
        }
    }
}

fn push_captured_output(captured: &mut Vec<u8>, chunk: &[u8]) -> bool {
    push_captured_output_with_limit(captured, chunk, STREAMED_OUTPUT_CAPTURE_LIMIT)
}

fn push_captured_output_with_limit(captured: &mut Vec<u8>, chunk: &[u8], limit: usize) -> bool {
    if chunk.len() > limit {
        captured.clear();
        captured.extend_from_slice(&chunk[chunk.len() - limit..]);
        return true;
    }

    let mut truncated = false;
    if captured.len() + chunk.len() > limit {
        let excess = captured.len() + chunk.len() - limit;
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
            if let Some(parent) = previous_path.parent() {
                match fs::remove_dir(parent).await {
                    Ok(()) => {}
                    Err(err)
                        if err.kind() != std::io::ErrorKind::NotFound
                            && err.kind() != std::io::ErrorKind::DirectoryNotEmpty =>
                    {
                        warn!(path = %parent.display(), error = %err, "failed to remove empty wiki directory");
                    }
                    Err(_) => {}
                }
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

    let move_result: Result<()> = async {
        fs::rename(source_json, final_json)
            .await
            .with_context(|| format!("move transcript JSON to {}", final_json.display()))?;
        fs::rename(source_txt, final_txt)
            .await
            .with_context(|| format!("move transcript text to {}", final_txt.display()))?;
        Ok(())
    }
    .await;

    if let Err(err) = move_result {
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

    // The new pair is visible now. A directory sync failure is durability-related;
    // report it, but do not roll back data that was successfully renamed.
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
        Ok(path_to_string(&normalize_path_lexically(path)))
    }
}

fn ledger_path_to_fs_path(data_dir: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        normalize_path_lexically(path)
    } else {
        normalize_path_lexically(&data_dir.join(path))
    }
}

fn absolutize_path(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("read current directory for ledger path")?
            .join(path)
    };
    Ok(normalize_path_lexically(&path))
}

fn normalize_path_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                let last = normalized.components().next_back();
                match last {
                    Some(Component::Normal(_)) => {
                        normalized.pop();
                    }
                    Some(Component::ParentDir) | None => {
                        if !normalized.has_root() {
                            normalized.push("..");
                        }
                    }
                    Some(Component::Prefix(_) | Component::RootDir | Component::CurDir) => {}
                }
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        normalized
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
            wiki_ingested_at: None,
            wiki_ingest_cmd: None,
            whisper_model: None,
            audio_path: None,
            transcript_path: None,
            wiki_path: None,
            error: None,
        }
    }

    fn default_wiki_ingest_args() -> WikiIngestArgs {
        WikiIngestArgs {
            wiki_ingest_cmd: None,
            wiki_ingest_cwd: None,
            wiki_ingest_timeout_secs: None,
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
    fn parse_wiki_ingest_cmd_rejects_templates_without_path() {
        let result = Cli::try_parse_from([
            "youtube-archiver",
            "wiki-ingest",
            "--wiki-ingest-cmd",
            "claude -p /wiki:ingest",
        ]);

        assert!(result.is_err());
    }

    #[test]
    fn parse_wiki_ingest_cmd_rejects_escaped_path_token() {
        let result = Cli::try_parse_from([
            "youtube-archiver",
            "wiki-ingest",
            "--wiki-ingest-cmd",
            r"printf \{path}",
        ]);

        let err = result.expect_err("escaped path token should not count as a template token");
        assert!(err.to_string().contains("template must contain {path}"));
    }

    #[test]
    fn parse_wiki_ingest_cmd_rejects_shell_unparseable_templates() {
        let result = Cli::try_parse_from([
            "youtube-archiver",
            "wiki-ingest",
            "--wiki-ingest-cmd",
            "claude -p \"/wiki:ingest {path}",
        ]);

        let err = result.expect_err("unclosed quote should fail clap validation");
        assert!(err.to_string().contains("shell-parseable command"));
    }

    #[test]
    fn parses_wiki_ingest_limit() {
        let cli = Cli::try_parse_from(["youtube-archiver", "wiki-ingest", "--limit", "2"])
            .expect("wiki ingest limit should parse");

        let Commands::WikiIngest(args) = cli.command else {
            panic!("expected wiki-ingest command");
        };
        assert_eq!(args.limit, Some(2));
    }

    #[test]
    fn parses_hyphen_leading_wiki_ingest_video_id() {
        let cli = Cli::try_parse_from([
            "youtube-archiver",
            "wiki-ingest",
            "--video-id",
            "-abc1234567",
        ])
        .expect("hyphen-leading video id should parse");

        let Commands::WikiIngest(args) = cli.command else {
            panic!("expected wiki-ingest command");
        };
        assert_eq!(args.video_id.as_deref(), Some("-abc1234567"));
    }

    #[test]
    fn rejects_zero_wiki_ingest_limit() {
        let result = Cli::try_parse_from(["youtube-archiver", "wiki-ingest", "--limit", "0"]);

        assert!(result.is_err());
    }

    #[test]
    fn parses_auto_wiki_ingest_args_on_ingest() {
        let cli = Cli::try_parse_from([
            "youtube-archiver",
            "ingest",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "--auto-wiki-ingest",
            "--wiki-ingest-cmd",
            "sh -c \"test -f {path}\"",
            "--wiki-ingest-cwd",
            "/tmp/wiki",
            "--wiki-ingest-timeout-secs",
            "12",
        ])
        .expect("auto wiki ingest args should parse");

        let Commands::Ingest(args) = cli.command else {
            panic!("expected ingest command");
        };
        assert!(args.auto_wiki_ingest);
        assert_eq!(
            args.wiki_ingest.wiki_ingest_cmd.as_deref(),
            Some("sh -c \"test -f {path}\"")
        );
        assert_eq!(
            args.wiki_ingest.wiki_ingest_cwd.as_deref(),
            Some(Path::new("/tmp/wiki"))
        );
        assert_eq!(args.wiki_ingest.wiki_ingest_timeout_secs, Some(12));
    }

    #[test]
    fn rejects_wiki_ingest_args_that_ingest_would_ignore_without_auto_flag() {
        let cli = Cli::try_parse_from([
            "youtube-archiver",
            "ingest",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "--wiki-ingest-timeout-secs",
            "12",
        ])
        .expect("wiki ingest args should still parse without auto flag");

        let Commands::Ingest(args) = cli.command else {
            panic!("expected ingest command");
        };
        assert!(!args.auto_wiki_ingest);
        assert!(args.wiki_ingest.has_cli_overrides());
        assert_eq!(args.wiki_ingest.wiki_ingest_timeout_secs, Some(12));

        let err = reject_ignored_wiki_ingest_options(&args)
            .expect_err("ingest should reject wiki options without --auto-wiki-ingest");
        assert!(format!("{err:#}").contains("require --auto-wiki-ingest"));
    }

    #[test]
    fn render_wiki_ingest_template_substitutes_and_quotes_tokens() -> Result<()> {
        let values = WikiIngestTemplateValues {
            path: "/tmp/wiki dir/{video_id}/video's file.md".to_owned(),
            video_id: "abc123".to_owned(),
            title: "A \"quoted\" {path} title".to_owned(),
            channel_slug: "rust channel".to_owned(),
        };
        let template = "cmd {path} {video_id} {title} {channel_slug}";

        let rendered = render_wiki_ingest_template(template, &values);

        assert_eq!(
            rendered,
            format!(
                "cmd {} abc123 {} {}",
                shell_words::quote(&values.path),
                shell_words::quote(&values.title),
                shell_words::quote(&values.channel_slug)
            )
        );
        let argv = shell_words::split(&rendered)?;
        assert_eq!(argv[1], values.path);
        assert_eq!(argv[2], values.video_id);
        assert_eq!(argv[3], values.title);
        assert_eq!(argv[4], values.channel_slug);
        Ok(())
    }

    #[test]
    fn render_wiki_ingest_template_quotes_tokens_inside_quotes() -> Result<()> {
        let values = WikiIngestTemplateValues {
            path: "/tmp/wiki dir/video \"quoted\" $file.md".to_owned(),
            video_id: "abc123".to_owned(),
            title: "can't stop".to_owned(),
            channel_slug: "rust channel".to_owned(),
        };

        let rendered = render_wiki_ingest_template(
            "cmd \"/wiki:ingest {path}\" 'title {title}' {channel_slug}",
            &values,
        );
        let argv = shell_words::split(&rendered)?;

        assert!(rendered.contains(r#"'title can'\''t stop'"#));
        assert_eq!(argv[1], format!("/wiki:ingest {}", values.path));
        assert_eq!(argv[2], format!("title {}", values.title));
        assert_eq!(argv[3], values.channel_slug);
        Ok(())
    }

    #[test]
    fn render_wiki_ingest_template_supports_shell_scripts_with_positional_args() -> Result<()> {
        let values = WikiIngestTemplateValues {
            path: "/tmp/yta space/wiki/foo/abc123.md".to_owned(),
            video_id: "abc123".to_owned(),
            title: "quoted title'; touch nope".to_owned(),
            channel_slug: "foo".to_owned(),
        };

        let rendered = render_wiki_ingest_template(
            "sh -c 'test -f \"$1\" && printf %s \"$2\"' sh {path} {title}",
            &values,
        );
        let argv = shell_words::split(&rendered)?;

        assert_eq!(argv[0], "sh");
        assert_eq!(argv[1], "-c");
        assert_eq!(argv[2], "test -f \"$1\" && printf %s \"$2\"");
        assert_eq!(argv[3], "sh");
        assert_eq!(argv[4], values.path);
        assert_eq!(argv[5], values.title);
        Ok(())
    }

    #[test]
    fn render_wiki_ingest_template_leaves_escaped_tokens_literal() -> Result<()> {
        let values = WikiIngestTemplateValues {
            path: "/tmp/wiki dir/abc123.md".to_owned(),
            video_id: "abc123".to_owned(),
            title: "A Title".to_owned(),
            channel_slug: "foo".to_owned(),
        };

        let rendered = render_wiki_ingest_template(r#"cmd \{path} '{video_id}'"#, &values);
        let argv = shell_words::split(&rendered)?;

        assert_eq!(argv[1], "{path}");
        assert_eq!(argv[2], values.video_id);
        Ok(())
    }

    #[test]
    fn missing_wiki_plugin_classifier_matches_expected_phrases() {
        assert!(is_missing_wiki_plugin_error(
            "unknown command: /wiki:ingest"
        ));
        assert!(is_missing_wiki_plugin_error("No such command /wiki:ingest"));
        assert!(is_missing_wiki_plugin_error(
            "command not found: /wiki:ingest"
        ));
        assert!(is_missing_wiki_plugin_error(
            "unknown slash command: /wiki:ingest"
        ));
        assert!(is_missing_wiki_plugin_error(
            "Command '/wiki:ingest' is not recognized"
        ));
        assert!(is_missing_wiki_plugin_error(
            "slash command /wiki:ingest is not available"
        ));
        assert!(is_missing_wiki_plugin_error(
            "/wiki:ingest requires the wiki plugin"
        ));
        assert!(is_missing_wiki_plugin_error("plugin wiki not found"));
        assert!(is_missing_wiki_plugin_error(
            "llm-wiki plugin missing from this workspace"
        ));
        assert!(is_missing_wiki_plugin_error(
            "Plugin wiki was not installed for this session"
        ));
        assert!(is_missing_wiki_plugin_error(
            "startup complete\nerror: unknown command: /wiki:ingest\n"
        ));
        assert!(!is_missing_wiki_plugin_error("unknown command: status"));
        assert!(!is_missing_wiki_plugin_error("plugin calendar not found"));
        assert!(!is_missing_wiki_plugin_error(
            "failed to run /wiki:ingest after a transient network timeout"
        ));
        assert!(!is_missing_wiki_plugin_error(
            "transcript text: plugin wiki not found in the quoted source"
        ));
        assert!(!is_missing_wiki_plugin_error(
            "note: unknown command: /wiki:ingest appears in transcript text"
        ));
        assert!(!is_missing_wiki_plugin_error(
            "network timeout while reading file"
        ));
    }

    #[test]
    fn stderr_tail_preserves_line_structure_on_one_line() {
        let stderr = b"  first line\n  {\"error\":\"bad value\"}\r\nthird\tline  ";

        assert_eq!(
            stderr_tail_one_line_limited(stderr, stderr.len()),
            r#"first line\n  {"error":"bad value"}\nthird\tline"#
        );
    }

    #[test]
    fn missing_wiki_plugin_classifier_uses_more_than_ledger_stderr_tail() {
        let mut stderr = b"unknown command: /wiki:ingest ".to_vec();
        stderr.extend(vec![b'x'; WIKI_INGEST_STDERR_LEDGER_LIMIT + 1]);

        assert!(!is_missing_wiki_plugin_error(
            &stderr_tail_one_line_limited(&stderr, WIKI_INGEST_STDERR_LEDGER_LIMIT)
        ));
        assert!(is_missing_wiki_plugin_error(&stderr_tail_one_line(&stderr)));
    }

    #[test]
    fn missing_wiki_plugin_hint_only_applies_to_default_template() {
        assert!(should_emit_missing_wiki_plugin_hint(
            true,
            1,
            false,
            "unknown command: /wiki:ingest",
        ));
        assert!(!should_emit_missing_wiki_plugin_hint(
            false,
            1,
            false,
            "unknown command: /wiki:ingest",
        ));
        assert!(!should_emit_missing_wiki_plugin_hint(
            true,
            2,
            false,
            "unknown command: /wiki:ingest",
        ));
        assert!(!should_emit_missing_wiki_plugin_hint(
            true,
            1,
            true,
            "unknown command: /wiki:ingest",
        ));
    }

    #[test]
    fn default_wiki_ingest_template_keeps_slash_command_path_in_one_arg() -> Result<()> {
        let values = WikiIngestTemplateValues {
            path: "/tmp/wiki dir/video \"quoted\" file.md".to_owned(),
            video_id: "abc123".to_owned(),
            title: String::new(),
            channel_slug: "wiki-dir".to_owned(),
        };

        let rendered = render_wiki_ingest_template(DEFAULT_WIKI_INGEST_CMD, &values);
        let argv = shell_words::split(&rendered)?;

        assert_eq!(argv[0], "claude");
        assert_eq!(argv[1], "-p");
        assert_eq!(argv[2], format!("/wiki:ingest {}", values.path));
        let allowed_tools = argv
            .windows(2)
            .find_map(|pair| (pair[0] == "--allowedTools").then_some(pair[1].as_str()))
            .expect("default command should include --allowedTools");
        assert_eq!(allowed_tools, "Bash,Read,Write,Edit,Glob,Grep,Task");
        Ok(())
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
            br#"{"id":"dQw4w9WgXcQ","title":"A Video","channel":"Rust Channel","tags":["rust"]}"#,
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

    #[tokio::test]
    async fn cached_metadata_rejects_mismatched_video_id() -> Result<()> {
        let dir = tempdir()?;
        let info_path = dir.path().join("info.json");
        fs::write(
            &info_path,
            br#"{"id":"abc1234567_","title":"Different Video"}"#,
        )
        .await?;

        let metadata = load_cached_metadata("dQw4w9WgXcQ", &info_path).await?;

        assert!(metadata.is_none());
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
    fn ledger_migrates_wiki_ingest_columns() -> Result<()> {
        let dir = tempdir()?;
        std::fs::create_dir_all(dir.path())?;
        let conn = Connection::open(dir.path().join("state.sqlite"))?;
        conn.execute_batch(
            r#"
            CREATE TABLE videos (
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
            INSERT INTO videos (video_id, url)
            VALUES ('abc123', 'https://www.youtube.com/watch?v=abc123');
            "#,
        )?;
        drop(conn);

        let ledger = Ledger::open(dir.path())?;
        let row = ledger.row("abc123")?.expect("row exists");

        assert!(row.wiki_ingested_at.is_none());
        assert!(row.wiki_ingest_cmd.is_none());
        Ok(())
    }

    #[test]
    fn ledger_read_only_reads_legacy_rows_without_wiki_ingest_columns() -> Result<()> {
        let dir = tempdir()?;
        std::fs::create_dir_all(dir.path())?;
        let conn = Connection::open(dir.path().join("state.sqlite"))?;
        conn.execute_batch(
            r#"
            CREATE TABLE videos (
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
            INSERT INTO videos (video_id, url, title, tags)
            VALUES ('abc123', 'https://www.youtube.com/watch?v=abc123', 'Legacy', '["rust"]');
            "#,
        )?;
        drop(conn);

        let ledger = Ledger::open_read_only(dir.path())?.expect("ledger exists");
        let rows = ledger.rows()?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].title.as_deref(), Some("Legacy"));
        assert_eq!(rows[0].tags, vec!["rust".to_owned()]);
        assert!(rows[0].wiki_ingested_at.is_none());
        assert!(rows[0].wiki_ingest_cmd.is_none());
        Ok(())
    }

    #[test]
    fn ledger_read_only_reports_missing_required_columns() -> Result<()> {
        let dir = tempdir()?;
        let conn = Connection::open(dir.path().join("state.sqlite"))?;
        conn.execute_batch(
            r#"
            CREATE TABLE videos (
                video_id TEXT PRIMARY KEY
            );
            INSERT INTO videos (video_id) VALUES ('abc123');
            "#,
        )?;
        drop(conn);

        let err = match Ledger::open_read_only(dir.path()) {
            Ok(_) => bail!("missing required schema column should fail fast"),
            Err(err) => err,
        };

        assert!(format!("{err:#}").contains("missing required column url"));
        Ok(())
    }

    #[test]
    fn mark_wiki_ingested_sets_columns_and_clears_wiki_error() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_wiki_emitted(video_id, Path::new("wiki/foo/abc123.md"))?;
        ledger.mark_error(video_id, "wiki-ingest exited 1: previous failure")?;
        let row = ledger.row(video_id)?.expect("row exists");

        ledger.mark_wiki_ingested(&row, "claude -p '/wiki:ingest path'")?;
        let row = ledger.row(video_id)?.expect("row exists");

        assert!(row.wiki_ingested_at.is_some());
        assert_eq!(
            row.wiki_ingest_cmd.as_deref(),
            Some("claude -p '/wiki:ingest path'")
        );
        assert!(row.error.is_none());
        Ok(())
    }

    #[test]
    fn mark_wiki_ingested_preserves_unrelated_stage_error() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_wiki_emitted(video_id, Path::new("wiki/foo/abc123.md"))?;
        ledger.mark_error(video_id, "download failed")?;
        let row = ledger.row(video_id)?.expect("row exists");

        ledger.mark_wiki_ingested(&row, "claude -p '/wiki:ingest path'")?;
        let row = ledger.row(video_id)?.expect("row exists");

        assert!(row.wiki_ingested_at.is_some());
        assert_eq!(row.error.as_deref(), Some("download failed"));
        Ok(())
    }

    #[test]
    fn mark_wiki_ingested_rejects_changed_wiki_row() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_wiki_emitted(video_id, Path::new("wiki/old/abc123.md"))?;
        let old_row = ledger.row(video_id)?.expect("row exists");
        ledger.mark_wiki_emitted(video_id, Path::new("wiki/new/abc123.md"))?;

        let err = ledger
            .mark_wiki_ingested(&old_row, "claude -p '/wiki:ingest old'")
            .expect_err("stale wiki row should not be marked ingested");
        let row = ledger.row(video_id)?.expect("row exists");

        assert!(format!("{err:#}").contains("ledger row changed"));
        assert!(row.wiki_ingested_at.is_none());
        assert!(row.wiki_ingest_cmd.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn wiki_ingest_command_success_fails_when_ledger_update_fails() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "abc123";
        let wiki = dir.path().join("wiki/foo/abc123.md");
        write_test_file(&wiki, b"wiki")?;
        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;
        ledger.conn.execute_batch(
            r#"
            CREATE TRIGGER fail_wiki_ingested
            BEFORE UPDATE OF wiki_ingested_at ON videos
            BEGIN
                SELECT RAISE(FAIL, 'forced wiki_ingested_at failure');
            END;
            "#,
        )?;
        let config = WikiIngestConfig {
            template: "true {path}".to_owned(),
            uses_default_template: false,
            cwd: dir.path().join("wiki"),
            create_cwd_for_preflight: true,
            timeout: Duration::from_secs(5),
        };
        let interrupts = Interrupts::inactive();

        let err = run_wiki_ingest_batch(
            dir.path(),
            &ledger,
            &config,
            WikiIngestBatchOptions {
                video_id: None,
                retry_errors: false,
                limit: None,
                force: false,
                missing_plugin_hint_emitted: None,
            },
            &interrupts,
        )
        .await
        .expect_err("ledger update failure should fail the row");

        assert!(format!("{err:#}").contains("every wiki ingestion failed"));
        let row = ledger.row(video_id)?.expect("row exists");
        assert!(row.wiki_ingested_at.is_none());
        Ok(())
    }

    #[test]
    fn core_stage_markers_preserve_wiki_ingest_errors_until_wiki_reemit() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "abc123";
        let wiki_error = "wiki-ingest exited 1: previous failure";
        let audio_path = dir.path().join("media/abc123/audio.m4a");
        let transcript_path = dir.path().join("transcripts/abc123/transcript.json");
        let wiki_path = dir.path().join("wiki/channel/abc123.md");

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_error(video_id, wiki_error)?;
        ledger.mark_downloaded(video_id, &audio_path)?;
        assert_eq!(
            ledger.row(video_id)?.expect("row exists").error.as_deref(),
            Some(wiki_error)
        );

        ledger.mark_transcribed(video_id, "large", &transcript_path)?;
        assert_eq!(
            ledger.row(video_id)?.expect("row exists").error.as_deref(),
            Some(wiki_error)
        );

        ledger.mark_wiki_emitted(video_id, &wiki_path)?;
        assert_eq!(
            ledger.row(video_id)?.expect("row exists").error.as_deref(),
            None
        );
        Ok(())
    }

    #[test]
    fn preserves_recorded_wiki_ingest_error_only_for_wiki_ingest_wrapper_error() -> Result<()> {
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_error(video_id, "wiki-ingest exited 1: stderr details")?;

        assert!(should_preserve_recorded_wiki_ingest_error(
            &ledger,
            video_id,
            "wiki-ingest abc123: every wiki ingestion failed (1 failure(s)): abc123"
        ));
        assert!(!should_preserve_recorded_wiki_ingest_error(
            &ledger,
            video_id,
            "download audio for abc123: yt-dlp failed"
        ));
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

        assert!(err.to_string().contains("read ledger row for abc123"));
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
            auto_wiki_ingest: false,
            wiki_ingest: default_wiki_ingest_args(),
        };
        let interrupts = Interrupts::inactive();

        process_video(&args, &ledger, video_id, None, None, &interrupts).await?;

        let row = ledger.row(video_id)?.expect("row exists");
        assert!(row.error.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn process_video_preserves_wiki_ingest_error_until_retry_succeeds() -> Result<()> {
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
        let wiki_error = "wiki-ingest exited 1: previous failure";

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
        ledger.mark_error(video_id, wiki_error)?;

        let args = IngestArgs {
            url: canonical_video_url(video_id),
            data_dir: dir.path().to_path_buf(),
            whisper_model: "large".to_owned(),
            whisper_bin: DEFAULT_WHISPER_BIN.to_owned(),
            whisper_args: Vec::new(),
            limit: None,
            audio_format: "m4a".to_owned(),
            force: false,
            auto_wiki_ingest: false,
            wiki_ingest: default_wiki_ingest_args(),
        };
        let interrupts = Interrupts::inactive();

        process_video(&args, &ledger, video_id, None, None, &interrupts).await?;

        let row = ledger.row(video_id)?.expect("row exists");
        assert_eq!(row.error.as_deref(), Some(wiki_error));
        Ok(())
    }

    #[tokio::test]
    async fn process_video_preserves_stale_error_until_failed_stage_is_recorded() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "dQw4w9WgXcQ";
        let info_path = dir.path().join("media").join(video_id).join("info.json");
        std::fs::create_dir_all(&info_path)?;

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_error(video_id, "old failure")?;

        let args = IngestArgs {
            url: canonical_video_url(video_id),
            data_dir: dir.path().to_path_buf(),
            whisper_model: "large".to_owned(),
            whisper_bin: DEFAULT_WHISPER_BIN.to_owned(),
            whisper_args: Vec::new(),
            limit: None,
            audio_format: "m4a".to_owned(),
            force: false,
            auto_wiki_ingest: false,
            wiki_ingest: default_wiki_ingest_args(),
        };
        let interrupts = Interrupts::inactive();

        let err = process_video(&args, &ledger, video_id, None, None, &interrupts)
            .await
            .expect_err("directory metadata path should fail before any stage records an error");
        assert!(format!("{err:#}").contains("read existing"));
        assert_eq!(
            ledger.row(video_id)?.expect("row exists").error.as_deref(),
            Some("old failure")
        );

        ledger.mark_error(video_id, &format!("{err:#}"))?;
        assert_ne!(
            ledger.row(video_id)?.expect("row exists").error.as_deref(),
            Some("old failure")
        );
        Ok(())
    }

    #[tokio::test]
    async fn process_video_auto_wiki_ingests_after_emit() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "dQw4w9WgXcQ";
        let media_dir = dir.path().join("media").join(video_id);
        let transcript_dir = dir.path().join("transcripts").join(video_id);
        let info_path = media_dir.join("info.json");
        let audio_path = media_dir.join("audio.m4a");
        let transcript_json = transcript_dir.join("transcript.json");
        let transcript_txt = transcript_dir.join("transcript.txt");
        let counter = dir.path().join("auto-counter");

        write_test_file(
            &info_path,
            br#"{
                "id": "dQw4w9WgXcQ",
                "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "channel": "Rust Channel",
                "title": "Auto Wiki"
            }"#,
        )?;
        write_test_file(&audio_path, b"audio")?;
        write_test_file(&transcript_json, b"{}")?;
        write_test_file(&transcript_txt, b"transcript")?;

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_downloaded(video_id, &audio_path)?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;

        let template = format!(
            "sh -c 'test -f \"$1\" && printf x >> \"$2\"' sh {{path}} {}",
            shell_words::quote(&path_to_string(&counter))
        );
        let config = WikiIngestConfig {
            template,
            uses_default_template: false,
            cwd: dir.path().join("wiki"),
            create_cwd_for_preflight: true,
            timeout: Duration::from_secs(5),
        };
        let args = IngestArgs {
            url: canonical_video_url(video_id),
            data_dir: dir.path().to_path_buf(),
            whisper_model: "large".to_owned(),
            whisper_bin: DEFAULT_WHISPER_BIN.to_owned(),
            whisper_args: Vec::new(),
            limit: None,
            audio_format: "m4a".to_owned(),
            force: false,
            auto_wiki_ingest: true,
            wiki_ingest: default_wiki_ingest_args(),
        };
        let interrupts = Interrupts::inactive();

        let mut missing_plugin_hint_emitted = false;
        process_video(
            &args,
            &ledger,
            video_id,
            Some(&config),
            Some(&mut missing_plugin_hint_emitted),
            &interrupts,
        )
        .await?;

        let row = ledger.row(video_id)?.expect("row exists");
        assert!(row.wiki_emitted_at.is_some());
        assert!(row.wiki_ingested_at.is_some());
        assert!(row.error.is_none());
        assert_eq!(std::fs::read(&counter)?, b"x");
        Ok(())
    }

    #[tokio::test]
    async fn ingest_auto_wiki_ingest_preflights_before_resolving_videos() -> Result<()> {
        let dir = tempdir()?;
        let missing_command = format!("missing-yta-command-{} {{path}}", std::process::id());
        let args = IngestArgs {
            url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".to_owned(),
            data_dir: dir.path().to_path_buf(),
            whisper_model: "large".to_owned(),
            whisper_bin: DEFAULT_WHISPER_BIN.to_owned(),
            whisper_args: Vec::new(),
            limit: None,
            audio_format: "m4a".to_owned(),
            force: false,
            auto_wiki_ingest: true,
            wiki_ingest: WikiIngestArgs {
                wiki_ingest_cmd: Some(missing_command),
                wiki_ingest_cwd: None,
                wiki_ingest_timeout_secs: None,
            },
        };
        let interrupts = Interrupts::inactive();

        let err = ingest(args, &interrupts)
            .await
            .expect_err("missing wiki ingestion command should fail before yt-dlp");
        let exit = err
            .downcast_ref::<ExitCodeError>()
            .expect("missing command should use explicit exit code");

        assert_eq!(exit.code, 3);
        assert!(exit.message.contains("wiki ingestion command not found"));
        assert!(!dir.path().join("state.sqlite").exists());
        assert!(!dir.path().join("wiki").exists());
        Ok(())
    }

    #[tokio::test]
    async fn preflight_wiki_ingest_command_creates_default_cwd_after_command_is_found() -> Result<()>
    {
        let dir = tempdir()?;
        let cwd = dir.path().join("wiki");
        let program = path_to_string(&std::env::current_exe()?);

        preflight_wiki_ingest_command(&program, &cwd, true, false).await?;

        assert!(cwd.is_dir());
        Ok(())
    }

    #[tokio::test]
    async fn wiki_ingest_preflights_before_opening_ledger() -> Result<()> {
        let dir = tempdir()?;
        let missing_command = format!("missing-yta-command-{} {{path}}", std::process::id());
        let args = WikiIngestCommandArgs {
            data_dir: dir.path().to_path_buf(),
            wiki_ingest: WikiIngestArgs {
                wiki_ingest_cmd: Some(missing_command),
                wiki_ingest_cwd: None,
                wiki_ingest_timeout_secs: None,
            },
            video_id: None,
            retry_errors: false,
            limit: None,
            force: false,
        };
        let interrupts = Interrupts::inactive();

        let err = wiki_ingest(args, &interrupts)
            .await
            .expect_err("missing wiki ingestion command should fail before ledger open");
        let exit = err
            .downcast_ref::<ExitCodeError>()
            .expect("missing command should use explicit exit code");

        assert_eq!(exit.code, 3);
        assert!(exit.message.contains("wiki ingestion command not found"));
        assert!(!dir.path().join("state.sqlite").exists());
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn preflight_wiki_ingest_command_uses_configured_cwd() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir()?;
        let script = dir.path().join("ingest.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\nif [ \"$#\" -ne 0 ]; then exit 64; fi\nexit 0\n",
        )?;
        let mut permissions = std::fs::metadata(&script)?.permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions)?;

        preflight_wiki_ingest_command("./ingest.sh", dir.path(), false, false).await?;
        Ok(())
    }

    #[tokio::test]
    async fn preflight_wiki_ingest_command_rejects_missing_configured_cwd() -> Result<()> {
        let dir = tempdir()?;
        let missing = dir.path().join("missing-wiki-dir");

        let err = preflight_wiki_ingest_command("sh", &missing, false, false)
            .await
            .expect_err("missing configured cwd should fail preflight");

        assert!(format!("{err:#}").contains("wiki ingestion cwd does not exist"));
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
    fn ledger_path_normalization_handles_dot_segments() -> Result<()> {
        let dir = tempdir()?;
        let data_dir = dir.path().join(".").join("data");
        let wiki = dir.path().join("data").join("wiki/foo/abc123.md");

        assert_eq!(
            path_to_ledger_string(&data_dir, &wiki)?,
            "wiki/foo/abc123.md"
        );
        assert_eq!(
            ledger_path_to_fs_path(&data_dir, "./wiki/../wiki/foo/abc123.md"),
            wiki
        );
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
        let interrupts = Interrupts::inactive();

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
            &interrupts,
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

    #[cfg(unix)]
    #[tokio::test]
    async fn replacement_failure_preserves_existing_archive_state() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let video_id = "dQw4w9WgXcQ";
        let audio = dir.path().join("media/dQw4w9WgXcQ/audio.m4a");
        let transcript_dir = dir.path().join("transcripts/dQw4w9WgXcQ");
        let transcript_json = transcript_dir.join("transcript.json");
        let transcript_txt = transcript_dir.join("transcript.txt");
        let wiki = dir.path().join("wiki/channel/dQw4w9WgXcQ.md");
        let script = dir.path().join("fake-whisper.sh");

        write_test_file(&audio, b"audio")?;
        write_test_file(&transcript_json, br#"{"old":true}"#)?;
        write_test_file(&transcript_txt, b"old transcript")?;
        write_test_file(&wiki, b"old wiki")?;
        write_test_file(
            &script,
            br#"set -eu
out=
prev=
for arg in "$@"; do
    if [ "$prev" = "--output_dir" ]; then
        out="$arg"
        break
    fi
    prev="$arg"
done
mkdir -p "$out"
printf '{"new":true}' > "$out/audio.json"
printf 'new transcript' > "$out/audio.txt"
chmod 555 "$(dirname "$out")"
"#,
        )?;

        std::fs::set_permissions(&transcript_dir, std::fs::Permissions::from_mode(0o555))?;
        let probe = transcript_dir.join("permission-probe");
        let permissions_are_enforced = std::fs::write(&probe, b"x").is_err();
        std::fs::set_permissions(&transcript_dir, std::fs::Permissions::from_mode(0o755))?;
        let _ = std::fs::remove_file(&probe);
        if !permissions_are_enforced {
            return Ok(());
        }

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_downloaded(video_id, &audio)?;
        ledger.mark_transcribed(video_id, "large", &transcript_json)?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;
        let before = ledger.row(video_id)?.expect("row exists");
        let interrupts = Interrupts::inactive();

        let whisper_bin = format!("sh {}", shell_words::quote(&path_to_string(&script)));
        let result = transcribe_audio(
            dir.path(),
            &ledger,
            video_id,
            &audio,
            WhisperConfig {
                bin: &whisper_bin,
                model: "large",
                extra_args: &[],
            },
            true,
            &interrupts,
        )
        .await;
        let _ = std::fs::set_permissions(&transcript_dir, std::fs::Permissions::from_mode(0o755));

        let err = result.expect_err("replacement should fail");
        assert!(format!("{err:#}").contains("back up"));
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
    async fn run_checked_captures_stdout_and_stderr() -> Result<()> {
        let args = vec!["-c".to_owned(), "printf out; printf err >&2".to_owned()];
        let output = run_checked("sh", &args, &Interrupts::inactive()).await?;

        assert_eq!(output.stdout, b"out");
        assert_eq!(output.stderr, b"err");
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_checked_cleans_up_pipe_holding_descendant_after_child_exits() -> Result<()> {
        let args = vec!["-c".to_owned(), "sleep 2 & printf out; exit 0".to_owned()];
        let interrupts = Interrupts::inactive();

        let output = tokio::time::timeout(
            Duration::from_secs(2),
            run_checked("sh", &args, &interrupts),
        )
        .await
        .expect("process-group cleanup should bound output drain")?;

        assert_eq!(output.stdout, b"out");
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streamed_command_cleans_up_pipe_holding_descendant_after_child_exits() -> Result<()> {
        let args = vec!["-c".to_owned(), "sleep 2 & exit 0".to_owned()];
        let interrupts = Interrupts::inactive();

        let output = tokio::time::timeout(
            Duration::from_secs(2),
            run_checked_stream_output("sh", &args, &interrupts),
        )
        .await
        .expect("process-group cleanup should bound stream drain")?;

        assert!(output.status.success());
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streamed_command_cleans_up_silent_pipe_despite_other_pipe_progress() -> Result<()> {
        let script = concat!(
            "(for i in 1 2 3 4 5 6 7 8 9 10; do ",
            "printf tick >&2; sleep 0.05; ",
            "done) >/dev/null & sleep 1 & exit 0",
        );
        let args = vec!["-c".to_owned(), script.to_owned()];
        let interrupts = Interrupts::inactive();

        let result = tokio::time::timeout(
            STREAM_READER_DRAIN_TIMEOUT * 3,
            run_checked_stream_output("sh", &args, &interrupts),
        )
        .await;
        let output = result.expect("process-group cleanup should bound stream drain")?;

        assert!(output.status.success());
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streamed_command_stops_on_shared_interrupt() -> Result<()> {
        let args = vec!["-c".to_owned(), "sleep 5".to_owned()];
        let (interrupts, sender) = Interrupts::test_channel();
        let trigger = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = sender.send(true);
        });

        let err = tokio::time::timeout(
            Duration::from_secs(2),
            run_checked_stream_output("sh", &args, &interrupts),
        )
        .await
        .expect("shared interrupt should stop the command promptly")
        .expect_err("interrupted command should fail");
        trigger.await.context("join interrupt trigger")?;

        assert!(is_interrupted_error(&err));
        Ok(())
    }

    #[tokio::test]
    async fn stream_reader_drain_timeout_resets_on_progress() -> Result<()> {
        let progress = Arc::new(AtomicU64::new(0));
        let task_progress = Arc::clone(&progress);
        let task = AbortOnDrop::new(tokio::spawn(async move {
            for _ in 0..3 {
                tokio::time::sleep(STREAM_READER_DRAIN_TIMEOUT / 2).await;
                task_progress.fetch_add(1, Ordering::Relaxed);
            }
            Ok::<Vec<u8>, std::io::Error>(b"done".to_vec())
        }));

        let output = join_stream_reader(task, "stdout", "test command", &progress).await?;

        assert_eq!(output, b"done");
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wiki_ingest_rejects_timeout_too_large_for_deadline() -> Result<()> {
        let dir = tempdir()?;
        let command = RenderedWikiIngestCommand {
            rendered: "sh -c true".to_owned(),
            program: "sh".to_owned(),
            args: vec!["-c".to_owned(), "true".to_owned()],
        };
        let interrupts = Interrupts::inactive();

        let err = run_wiki_ingest_command(
            &command,
            dir.path(),
            Duration::from_secs(u64::MAX),
            &interrupts,
        )
        .await
        .expect_err("unrepresentable timeout should be rejected without panicking");

        assert!(format!("{err:#}").contains("timeout is too large"));
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wiki_ingest_cleans_up_pipe_holding_descendant_after_child_exits() -> Result<()> {
        let dir = tempdir()?;
        let script = "(while :; do printf x >&2; sleep 0.02; done) & exit 0";
        let command = RenderedWikiIngestCommand {
            rendered: format!("sh -c {}", shell_words::quote(script)),
            program: "sh".to_owned(),
            args: vec!["-c".to_owned(), script.to_owned()],
        };
        let interrupts = Interrupts::inactive();

        let output = tokio::time::timeout(
            Duration::from_secs(2),
            run_wiki_ingest_command(
                &command,
                dir.path(),
                Duration::from_millis(200),
                &interrupts,
            ),
        )
        .await
        .expect("wiki-ingest cleanup should bound stream drain")?;

        assert!(output.status.success());
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wiki_ingest_normal_exit_kills_child_process_group() -> Result<()> {
        let dir = tempdir()?;
        let counter = dir.path().join("counter");
        let script = format!(
            "(while :; do printf x >> {}; sleep 0.02; done) & exit 0",
            shell_words::quote(&path_to_string(&counter))
        );
        let command = RenderedWikiIngestCommand {
            rendered: format!("sh -c {}", shell_words::quote(&script)),
            program: "sh".to_owned(),
            args: vec!["-c".to_owned(), script],
        };
        let interrupts = Interrupts::inactive();

        let output = tokio::time::timeout(
            Duration::from_secs(2),
            run_wiki_ingest_command(
                &command,
                dir.path(),
                Duration::from_millis(200),
                &interrupts,
            ),
        )
        .await
        .expect("wiki-ingest cleanup should return promptly")?;

        assert!(output.status.success());
        tokio::time::sleep(Duration::from_millis(200)).await;
        let first_len = file_len_or_zero(&counter).await?;
        tokio::time::sleep(Duration::from_millis(250)).await;
        let second_len = file_len_or_zero(&counter).await?;
        assert_eq!(first_len, second_len);
        Ok(())
    }

    #[cfg(unix)]
    async fn file_len_or_zero(path: &Path) -> Result<u64> {
        match fs::metadata(path).await {
            Ok(metadata) => Ok(metadata.len()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(err) => Err(err).with_context(|| format!("stat {}", path.display())),
        }
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

    #[tokio::test]
    async fn list_rows_only_include_completed_archived_videos() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let complete_id = "dQw4w9WgXcQ";
        let failed_id = "abc1234567_";
        let core_error_id = "def1234567_";
        let complete_audio = dir.path().join("media/dQw4w9WgXcQ/audio.m4a");
        let complete_transcript_json = dir.path().join("transcripts/dQw4w9WgXcQ/transcript.json");
        let complete_transcript_txt = dir.path().join("transcripts/dQw4w9WgXcQ/transcript.txt");
        let complete_wiki = dir.path().join("wiki/channel/dQw4w9WgXcQ.md");
        let core_error_audio = dir.path().join("media/def1234567_/audio.m4a");
        let core_error_transcript_json = dir.path().join("transcripts/def1234567_/transcript.json");
        let core_error_transcript_txt = dir.path().join("transcripts/def1234567_/transcript.txt");
        let core_error_wiki = dir.path().join("wiki/channel/def1234567_.md");

        write_test_file(&complete_audio, b"audio")?;
        write_test_file(&complete_transcript_json, b"{}")?;
        write_test_file(&complete_transcript_txt, b"transcript")?;
        write_test_file(&complete_wiki, b"wiki")?;
        write_test_file(&core_error_audio, b"audio")?;
        write_test_file(&core_error_transcript_json, b"{}")?;
        write_test_file(&core_error_transcript_txt, b"transcript")?;
        write_test_file(&core_error_wiki, b"wiki")?;

        ledger.ensure_video(complete_id, &canonical_video_url(complete_id))?;
        ledger.mark_downloaded(complete_id, &complete_audio)?;
        ledger.mark_transcribed(complete_id, "large", &complete_transcript_json)?;
        ledger.mark_wiki_emitted(complete_id, &complete_wiki)?;
        ledger.mark_error(complete_id, "wiki-ingest exited 1: missing plugin")?;
        ledger.ensure_video(core_error_id, &canonical_video_url(core_error_id))?;
        ledger.mark_downloaded(core_error_id, &core_error_audio)?;
        ledger.mark_transcribed(core_error_id, "large", &core_error_transcript_json)?;
        ledger.mark_wiki_emitted(core_error_id, &core_error_wiki)?;
        ledger.mark_error(core_error_id, "download failed")?;
        ledger.ensure_video(failed_id, &canonical_video_url(failed_id))?;
        ledger.mark_error(failed_id, "download failed")?;
        drop(ledger);

        let rows = list_rows(dir.path()).await?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].video_id, complete_id);
        assert!(
            rows[0]
                .error
                .as_deref()
                .is_some_and(|error| is_wiki_ingest_ledger_error(Some(error)))
        );
        Ok(())
    }

    #[tokio::test]
    async fn list_rows_require_archived_artifacts_to_exist() -> Result<()> {
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

        let rows = list_rows(dir.path()).await?;

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

        assert!(!should_skip_wiki_ingest(dir.path(), &row, false));
        ledger.mark_wiki_ingested(&row, "claude -p /wiki:ingest")?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(should_skip_wiki_ingest(dir.path(), &row, false));
        assert!(!should_skip_wiki_ingest(dir.path(), &row, true));
        std::fs::remove_file(&wiki)?;
        assert!(!should_skip_wiki_ingest(dir.path(), &row, false));

        let wiki_dir = dir.path().join("wiki-ingest-dir.md");
        std::fs::create_dir(&wiki_dir)?;
        ledger.mark_wiki_emitted(video_id, &wiki_dir)?;
        row = ledger.row(video_id)?.expect("row exists");
        ledger.mark_wiki_ingested(&row, "claude -p /wiki:ingest")?;
        row = ledger.row(video_id)?.expect("row exists");
        assert!(!should_skip_wiki_ingest(dir.path(), &row, false));

        Ok(())
    }

    #[tokio::test]
    async fn wiki_ingest_engine_records_success_skip_force_and_failure() -> Result<()> {
        let dir = tempdir()?;
        let data_dir = dir.path().join("data dir");
        let ledger = Ledger::open(&data_dir)?;
        let video_id = "abc123";
        let wiki = data_dir.join("wiki").join("foo").join("abc123.md");
        let counter = dir.path().join("counter");
        write_test_file(&wiki, b"wiki")?;
        ledger.upsert_metadata(&VideoMetadata {
            video_id: video_id.to_owned(),
            url: canonical_video_url(video_id),
            channel_id: None,
            channel_title: Some("Foo".to_owned()),
            uploader: None,
            title: Some("A Title".to_owned()),
            upload_date: None,
            duration: None,
            tags: Vec::new(),
        })?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;

        let template = format!(
            "sh -c 'test -f \"$1\" && printf x >> \"$2\"' sh {{path}} {}",
            shell_words::quote(&path_to_string(&counter))
        );
        let config = WikiIngestConfig {
            template,
            uses_default_template: false,
            cwd: dir.path().join("wiki"),
            create_cwd_for_preflight: true,
            timeout: Duration::from_secs(5),
        };
        let interrupts = Interrupts::inactive();

        let outcome = run_wiki_ingest_batch(
            &data_dir,
            &ledger,
            &config,
            WikiIngestBatchOptions {
                video_id: None,
                retry_errors: false,
                limit: None,
                force: false,
                missing_plugin_hint_emitted: None,
            },
            &interrupts,
        )
        .await?;
        assert_eq!(outcome.succeeded, 1);
        assert_eq!(std::fs::read(&counter)?, b"x");
        let first = ledger.row(video_id)?.expect("row exists");
        let first_ingested_at = first.wiki_ingested_at.clone();
        assert!(first_ingested_at.is_some());
        assert!(first.wiki_ingest_cmd.as_deref().is_some_and(|cmd| {
            cmd.contains("test -f")
                && cmd.contains(&path_to_string(&absolutize_path(&wiki).unwrap()))
        }));

        let outcome = run_wiki_ingest_batch(
            &data_dir,
            &ledger,
            &config,
            WikiIngestBatchOptions {
                video_id: None,
                retry_errors: false,
                limit: None,
                force: false,
                missing_plugin_hint_emitted: None,
            },
            &interrupts,
        )
        .await?;
        assert_eq!(outcome.succeeded + outcome.skipped + outcome.failed, 0);
        assert_eq!(std::fs::read(&counter)?, b"x");
        assert_eq!(
            ledger.row(video_id)?.expect("row exists").wiki_ingested_at,
            first_ingested_at
        );

        tokio::time::sleep(Duration::from_secs(1)).await;
        let outcome = run_wiki_ingest_batch(
            &data_dir,
            &ledger,
            &config,
            WikiIngestBatchOptions {
                video_id: None,
                retry_errors: false,
                limit: None,
                force: true,
                missing_plugin_hint_emitted: None,
            },
            &interrupts,
        )
        .await?;
        assert_eq!(outcome.succeeded, 1);
        assert_eq!(std::fs::read(&counter)?, b"xx");
        let forced = ledger.row(video_id)?.expect("row exists");
        assert_ne!(forced.wiki_ingested_at, first_ingested_at);

        let failing = WikiIngestConfig {
            template: "false {path}".to_owned(),
            uses_default_template: false,
            cwd: dir.path().join("wiki"),
            create_cwd_for_preflight: true,
            timeout: Duration::from_secs(5),
        };
        let err = run_wiki_ingest_batch(
            &data_dir,
            &ledger,
            &failing,
            WikiIngestBatchOptions {
                video_id: None,
                retry_errors: true,
                limit: None,
                force: true,
                missing_plugin_hint_emitted: None,
            },
            &interrupts,
        )
        .await
        .expect_err("failing command should fail the batch");

        assert!(format!("{err:#}").contains("every wiki ingestion failed"));
        let failed = ledger.row(video_id)?.expect("row exists");
        assert!(
            failed
                .error
                .as_deref()
                .is_some_and(|error| error.starts_with("wiki-ingest exited 1:"))
        );
        assert_eq!(std::fs::read(&wiki)?, b"wiki");

        let outcome = run_wiki_ingest_batch(
            &data_dir,
            &ledger,
            &config,
            WikiIngestBatchOptions {
                video_id: None,
                retry_errors: true,
                limit: None,
                force: false,
                missing_plugin_hint_emitted: None,
            },
            &interrupts,
        )
        .await?;
        assert_eq!(outcome.succeeded, 1);
        assert_eq!(std::fs::read(&counter)?, b"xxx");
        let retried = ledger.row(video_id)?.expect("row exists");
        assert!(retried.error.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn wiki_ingest_batch_returns_ok_when_some_attempts_succeed() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let script = dir.path().join("ingest.sh");
        write_test_file(
            &script,
            br#"test -f "$2" || exit 9
if [ "$1" = "good123" ]; then
  exit 0
fi
echo "planned failure" >&2
exit 7
"#,
        )?;

        for video_id in ["bad456", "good123"] {
            let wiki = dir
                .path()
                .join("wiki")
                .join("foo")
                .join(format!("{video_id}.md"));
            write_test_file(&wiki, b"wiki")?;
            ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
            ledger.mark_wiki_emitted(video_id, &wiki)?;
        }

        let config = WikiIngestConfig {
            template: format!(
                "sh {} {{video_id}} {{path}}",
                shell_words::quote(&path_to_string(&script))
            ),
            uses_default_template: false,
            cwd: dir.path().join("wiki"),
            create_cwd_for_preflight: true,
            timeout: Duration::from_secs(5),
        };
        let interrupts = Interrupts::inactive();

        let outcome = run_wiki_ingest_batch(
            dir.path(),
            &ledger,
            &config,
            WikiIngestBatchOptions {
                video_id: None,
                retry_errors: false,
                limit: None,
                force: false,
                missing_plugin_hint_emitted: None,
            },
            &interrupts,
        )
        .await?;

        assert_eq!(outcome.succeeded, 1);
        assert_eq!(outcome.failed, 1);
        assert!(
            ledger
                .row("good123")?
                .expect("row exists")
                .wiki_ingested_at
                .is_some()
        );
        assert!(
            ledger
                .row("bad456")?
                .expect("row exists")
                .error
                .as_deref()
                .is_some_and(|error| error.starts_with("wiki-ingest exited 7:"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn wiki_ingest_batch_records_bad_first_row_without_aborting_preflight() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open(dir.path())?;
        let good_wiki = dir.path().join("wiki/foo/good123.md");
        write_test_file(&good_wiki, b"wiki")?;

        ledger.ensure_video("bad456", &canonical_video_url("bad456"))?;
        ledger.conn.execute(
            "UPDATE videos SET wiki_emitted_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE video_id = ?1",
            params!["bad456"],
        )?;
        ledger.ensure_video("good123", &canonical_video_url("good123"))?;
        ledger.mark_wiki_emitted("good123", &good_wiki)?;

        let config = WikiIngestConfig {
            template: "true {path}".to_owned(),
            uses_default_template: false,
            cwd: dir.path().join("wiki"),
            create_cwd_for_preflight: true,
            timeout: Duration::from_secs(5),
        };
        let interrupts = Interrupts::inactive();

        let outcome = run_wiki_ingest_batch(
            dir.path(),
            &ledger,
            &config,
            WikiIngestBatchOptions {
                video_id: None,
                retry_errors: false,
                limit: None,
                force: false,
                missing_plugin_hint_emitted: None,
            },
            &interrupts,
        )
        .await?;

        assert_eq!(outcome.succeeded, 1);
        assert_eq!(outcome.failed, 1);
        assert!(
            ledger
                .row("bad456")?
                .expect("row exists")
                .error
                .as_deref()
                .is_some_and(|error| error.contains("has no wiki_path"))
        );
        assert!(
            ledger
                .row("good123")?
                .expect("row exists")
                .wiki_ingested_at
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn wiki_ingest_lock_rejects_concurrent_batch() -> Result<()> {
        let dir = tempdir()?;
        let _lock = acquire_wiki_ingest_lock(dir.path())?;

        let err = acquire_wiki_ingest_lock(dir.path())
            .expect_err("second wiki ingestion lock should fail");

        assert!(format!("{err:#}").contains("already running"));
        Ok(())
    }

    #[test]
    fn ingest_lock_rejects_concurrent_run() -> Result<()> {
        let dir = tempdir()?;
        let _lock = acquire_ingest_lock(dir.path())?;

        let err = acquire_ingest_lock(dir.path()).expect_err("second ingest lock should fail");

        assert!(format!("{err:#}").contains("ingest is already running"));
        Ok(())
    }

    #[tokio::test]
    async fn refreshed_wiki_ingest_rows_must_still_be_pending_candidates() -> Result<()> {
        let dir = tempdir()?;
        let wiki = dir.path().join("wiki/foo/abc123.md");
        write_test_file(&wiki, b"wiki")?;
        let mut row = row_for_skip_tests();
        row.wiki_path = Some(path_to_ledger_string(dir.path(), &wiki)?);

        row.wiki_emitted_at = None;
        assert!(!should_attempt_wiki_ingest_row(dir.path(), &row, false, false).await);

        row.wiki_emitted_at = Some("2026-05-19T00:00:00Z".to_owned());
        assert!(should_attempt_wiki_ingest_row(dir.path(), &row, false, false).await);

        row.wiki_ingested_at = Some("2026-05-19T00:00:01Z".to_owned());
        assert!(!should_attempt_wiki_ingest_row(dir.path(), &row, false, false).await);
        assert!(should_attempt_wiki_ingest_row(dir.path(), &row, false, true).await);

        std::fs::remove_file(&wiki)?;
        assert!(should_attempt_wiki_ingest_row(dir.path(), &row, false, false).await);
        Ok(())
    }

    #[tokio::test]
    async fn wiki_ingest_candidates_reject_unknown_or_not_emitted_video_id() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open_in_memory()?;

        let err =
            wiki_ingest_candidate_rows(dir.path(), &ledger, Some("missing"), false, false, None)
                .await
                .expect_err("unknown video id should fail");
        assert!(format!("{err:#}").contains("not in the ledger"));

        ledger.ensure_video("abc123", &canonical_video_url("abc123"))?;
        let err =
            wiki_ingest_candidate_rows(dir.path(), &ledger, Some("abc123"), false, false, None)
                .await
                .expect_err("video without wiki article should fail");
        assert!(format!("{err:#}").contains("no emitted wiki article"));
        Ok(())
    }

    #[tokio::test]
    async fn wiki_ingest_candidates_skip_already_ingested_rows_until_force() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open_in_memory()?;
        let video_id = "abc123";
        let wiki = dir.path().join("wiki/foo/abc123.md");
        write_test_file(&wiki, b"wiki")?;

        ledger.ensure_video(video_id, &canonical_video_url(video_id))?;
        ledger.mark_wiki_emitted(video_id, &wiki)?;
        let row = ledger.row(video_id)?.expect("row exists");
        ledger.mark_wiki_ingested(&row, "claude -p /wiki:ingest")?;

        let rows =
            wiki_ingest_candidate_rows(dir.path(), &ledger, None, false, false, None).await?;
        assert!(rows.is_empty());

        std::fs::remove_file(&wiki)?;
        let rows =
            wiki_ingest_candidate_rows(dir.path(), &ledger, None, false, false, None).await?;
        assert!(rows.is_empty());

        std::fs::create_dir_all(&wiki)?;
        let rows =
            wiki_ingest_candidate_rows(dir.path(), &ledger, None, false, false, None).await?;
        assert!(rows.is_empty());

        let rows = wiki_ingest_candidate_rows(dir.path(), &ledger, None, false, true, None).await?;
        assert_eq!(
            rows.iter()
                .map(|row| row.video_id.as_str())
                .collect::<Vec<_>>(),
            ["abc123"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn wiki_ingest_candidates_filter_recorded_errors_without_retry() -> Result<()> {
        let dir = tempdir()?;
        let ledger = Ledger::open_in_memory()?;
        let unrelated_id = "abc123";
        let wiki_error_id = "def456";
        let clean_id = "ghi789";

        ledger.ensure_video(unrelated_id, &canonical_video_url(unrelated_id))?;
        ledger.mark_wiki_emitted(unrelated_id, &dir.path().join("wiki/foo/abc123.md"))?;
        ledger.mark_error(unrelated_id, "download failed")?;

        ledger.ensure_video(wiki_error_id, &canonical_video_url(wiki_error_id))?;
        ledger.mark_wiki_emitted(wiki_error_id, &dir.path().join("wiki/foo/def456.md"))?;
        ledger.mark_error(wiki_error_id, "wiki-ingest exited 1: missing plugin")?;

        ledger.ensure_video(clean_id, &canonical_video_url(clean_id))?;
        ledger.mark_wiki_emitted(clean_id, &dir.path().join("wiki/foo/ghi789.md"))?;

        let rows =
            wiki_ingest_candidate_rows(dir.path(), &ledger, None, false, false, None).await?;
        assert_eq!(
            rows.iter()
                .map(|row| row.video_id.as_str())
                .collect::<Vec<_>>(),
            ["ghi789"]
        );

        // retry_errors=true must only resurface rows whose existing
        // error came from the wiki-ingest stage itself — an unrelated
        // "download failed" error must NOT trigger a paid LLM call.
        let rows = wiki_ingest_candidate_rows(dir.path(), &ledger, None, true, false, None).await?;
        assert_eq!(
            rows.iter()
                .map(|row| row.video_id.as_str())
                .collect::<Vec<_>>(),
            ["def456", "ghi789"]
        );

        // force=true bypasses error filtering entirely.
        let rows = wiki_ingest_candidate_rows(dir.path(), &ledger, None, false, true, None).await?;
        assert_eq!(rows.len(), 3);

        let rows =
            wiki_ingest_candidate_rows(dir.path(), &ledger, None, true, true, Some(1)).await?;
        assert_eq!(
            rows.iter()
                .map(|row| row.video_id.as_str())
                .collect::<Vec<_>>(),
            ["abc123"]
        );

        let err = wiki_ingest_candidate_rows(
            dir.path(),
            &ledger,
            Some(wiki_error_id),
            false,
            false,
            None,
        )
        .await
        .expect_err("explicit errored video should explain retry requirement");
        assert!(format!("{err:#}").contains("pass --retry-errors"));

        let rows =
            wiki_ingest_candidate_rows(dir.path(), &ledger, Some(wiki_error_id), false, true, None)
                .await?;
        assert_eq!(
            rows.iter()
                .map(|row| row.video_id.as_str())
                .collect::<Vec<_>>(),
            ["def456"]
        );
        Ok(())
    }
}
