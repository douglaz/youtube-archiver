use std::{env, error::Error, path::Path, process::Command};

use rusqlite::{Connection, params};
use tempfile::tempdir;

const DEFAULT_E2E_URL: &str = "https://www.youtube.com/watch?v=jNQXAC9IVRw";
const DEFAULT_E2E_WHISPER_BIN: &str = "nix run nixpkgs#openai-whisper --";
const DEFAULT_E2E_WHISPER_MODEL: &str = "tiny";
const DEFAULT_E2E_AUDIO_FORMAT: &str = "m4a";

#[test]
fn e2e_ingest_status_and_list() -> Result<(), Box<dyn Error>> {
    if !e2e_enabled() {
        eprintln!("skipping e2e test; set YTARCH_E2E=1 to enable");
        return Ok(());
    }

    let url = env::var("YTARCH_E2E_URL").unwrap_or_else(|_| DEFAULT_E2E_URL.to_owned());
    let whisper_bin =
        env::var("YTARCH_E2E_WHISPER_BIN").unwrap_or_else(|_| DEFAULT_E2E_WHISPER_BIN.to_owned());
    let whisper_model = env::var("YTARCH_E2E_WHISPER_MODEL")
        .unwrap_or_else(|_| DEFAULT_E2E_WHISPER_MODEL.to_owned());
    let audio_format =
        env::var("YTARCH_E2E_AUDIO_FORMAT").unwrap_or_else(|_| DEFAULT_E2E_AUDIO_FORMAT.to_owned());
    let data_dir_guard = tempdir()?;
    let data_dir = data_dir_guard.path().to_string_lossy().into_owned();

    run_success(&[
        "ingest".to_owned(),
        url,
        "--data-dir".to_owned(),
        data_dir.clone(),
        "--limit".to_owned(),
        "1".to_owned(),
        "--whisper-bin".to_owned(),
        whisper_bin,
        "--whisper-model".to_owned(),
        whisper_model,
        "--audio-format".to_owned(),
        audio_format,
    ])?;
    run_success(&[
        "status".to_owned(),
        "--data-dir".to_owned(),
        data_dir.clone(),
    ])?;

    let output = run_output(&["list".to_owned(), "--data-dir".to_owned(), data_dir])?;
    let rows: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert!(
        rows.as_array().is_some_and(|rows| !rows.is_empty()),
        "list should emit at least one archived row"
    );

    Ok(())
}

#[test]
fn list_stdout_stays_json_when_ledger_warns() -> Result<(), Box<dyn Error>> {
    let data_dir_guard = tempdir()?;
    create_ledger_with_bad_tags(data_dir_guard.path())?;

    let output = run_output(&[
        "list".to_owned(),
        "--data-dir".to_owned(),
        data_dir_guard.path().to_string_lossy().into_owned(),
    ])?;
    let rows: serde_json::Value = serde_json::from_slice(&output.stdout)?;

    assert_eq!(rows, serde_json::json!([]));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("skipping corrupt ledger row"),
        "expected ledger warning on stderr"
    );
    Ok(())
}

fn create_ledger_with_bad_tags(data_dir: &Path) -> Result<(), Box<dyn Error>> {
    // Materialise the schema directly so this regression test only
    // depends on `list`'s tolerance for corrupt rows, not on
    // `wiki-ingest`'s preflight, lock, or template parser. Keep these
    // columns in sync with `Ledger::init` in src/main.rs.
    let conn = Connection::open(data_dir.join("state.sqlite"))?;
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
            wiki_ingested_at TEXT,
            wiki_ingest_cmd TEXT,
            whisper_model TEXT,
            audio_path TEXT,
            transcript_path TEXT,
            wiki_path TEXT,
            error TEXT
        );
        "#,
    )?;
    conn.execute(
        "INSERT INTO videos (video_id, url, tags) VALUES (?1, ?2, ?3)",
        params![
            "dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "not json"
        ],
    )?;
    Ok(())
}

fn e2e_enabled() -> bool {
    env::var("YTARCH_E2E").is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn run_success(args: &[String]) -> Result<(), Box<dyn Error>> {
    let status = Command::new(env!("CARGO_BIN_EXE_youtube-archiver"))
        .args(args)
        .status()?;
    assert!(
        status.success(),
        "youtube-archiver {} exited with {status}",
        args.join(" ")
    );
    Ok(())
}

fn run_output(args: &[String]) -> Result<std::process::Output, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_youtube-archiver"))
        .args(args)
        .env("RUST_LOG", "warn")
        .output()?;
    assert!(
        output.status.success(),
        "youtube-archiver {} exited with {}\nstdout:\n{}\nstderr:\n{}",
        args.join(" "),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}
