use anyhow::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Real CLI lives in src/cli.rs once rb-lite fills it in; this stub keeps
    // `cargo build` green from commit zero.
    eprintln!("youtube-archiver: not yet implemented — see AGENTS.md");
    Ok(())
}
