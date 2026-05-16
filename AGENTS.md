# youtube-archiver — agent spec

A Rust CLI that archives YouTube content (single videos, full channels, or
playlists), transcribes the audio with OpenAI Whisper, and emits
ingestion-ready markdown for [`nvk/llm-wiki`](https://github.com/nvk/llm-wiki)
so the user can later derive wiki articles from the transcripts.

This file is the source of truth for agents (rb-lite implementer + reviewers).
Keep it short and concrete.

## Non-negotiables

- **Do not reimplement YouTube download.** Shell out to `yt-dlp`.
- **Do not reimplement transcription.** Shell out to `openai-whisper`,
  invoked the way the user already invokes it on this machine:
  ```
  nix run nixpkgs#openai-whisper -- <audio-file> --model large
  ```
  The whisper binary, model, and extra args must be configurable.
- **Idempotent.** Re-running the tool on the same URL must skip work that's
  already done: skip download if the media is on disk, skip transcription
  if the transcript exists, skip markdown emission if the article exists.
- **Three input modes:** single video URL, channel URL, playlist URL.
  All three are normalized to a flat list of video IDs via `yt-dlp
  --flat-playlist --print id` (or equivalent) before processing.

## CLI surface (initial)

```
youtube-archiver ingest <URL> [--data-dir DIR] [--whisper-model MODEL]
                              [--whisper-bin CMD] [--limit N]
                              [--audio-format FMT] [--force]
youtube-archiver status [--data-dir DIR]   # prints per-video state table
youtube-archiver list   [--data-dir DIR]   # lists archived videos as JSON
```

`ingest` is the only command that does work; `status`/`list` are read-only.
`--force` re-runs every step even if outputs exist (useful for changing
whisper models).

## On-disk layout

```
<data-dir>/                       # default: ./data
  state.sqlite                    # ingestion ledger (see schema below)
  media/<video_id>/
    info.json                     # raw yt-dlp metadata
    audio.<ext>                   # downloaded audio (m4a/opus/mp3)
  transcripts/<video_id>/
    transcript.json               # whisper segments + timings
    transcript.txt                # plain text
  wiki/<channel_slug>/<video_id>.md
                                  # llm-wiki ingestion target
```

The `wiki/` tree is what gets fed into `/wiki:ingest` later — markdown with
YAML frontmatter (title, channel, uploader, upload_date, duration, url,
video_id, tags) followed by the transcript body.

## State ledger (`state.sqlite`)

One table, columns roughly: `video_id PK, url, channel_id, channel_title,
title, downloaded_at, transcribed_at, wiki_emitted_at, whisper_model,
audio_path, transcript_path, wiki_path, error`.

`error` is nullable text. A row exists per video as soon as we know about
it; timestamps fill in as stages complete. Skip-if-already-done logic
reads this table first, then verifies the file actually exists on disk
before short-circuiting.

## Pipeline (per video)

1. Resolve URL → list of video IDs (yt-dlp).
2. For each video ID, in order:
   a. **Metadata**: `yt-dlp -j` → store `info.json`, upsert ledger row.
   b. **Audio download**: `yt-dlp -f bestaudio --extract-audio
      --audio-format <fmt>` → `media/<id>/audio.<ext>`. Skip if file
      present and ledger marks `downloaded_at`.
   c. **Transcribe**: invoke whisper command on the audio file, output
      JSON + txt into `transcripts/<id>/`. Skip if both exist and
      `whisper_model` in ledger matches the requested model.
   d. **Emit wiki article**: render markdown with frontmatter into
      `wiki/<channel_slug>/<id>.md`. Skip if file present unless
      `--force`.
3. Update ledger after each stage; never leave half-written files
   (write to temp + rename).

Stages b/c/d must be resumable independently — if whisper crashes
midway, the next run finishes from there.

## Error handling

- Per-video errors get logged to the ledger's `error` column and do not
  abort the batch. The CLI exits non-zero only if every video failed.
- External commands invoked via `tokio::process::Command`; capture
  stderr on failure and surface it in the error message.

## Dependencies (initial)

`clap` (derive), `anyhow`, `tokio` (rt-multi-thread, process, fs),
`serde`/`serde_json`, `rusqlite` (bundled), `regex` for slugifying,
`tracing` + `tracing-subscriber` for logs. Add more only when a stage
needs it.

## Test strategy

- Unit tests for: URL→video-id classification, slugify, frontmatter
  rendering, ledger skip logic (using an in-memory sqlite).
- Integration tests are gated behind an env var (`YTARCH_E2E=1`) since
  they require network + yt-dlp + whisper.
- `cargo test` (no feature flags) must pass offline.

## Out of scope (for now)

- Video file (we keep only audio).
- Subtitles fallback if YouTube already has captions — future
  enhancement, but transcripts via whisper are the primary path because
  the user wants the same whisper invocation they're already using for
  audio notes.
- Calling `/wiki:ingest` automatically. We just produce the files; the
  user runs the slash command themselves.
- A daemon / watch mode.

## Nix

`flake.nix` exposes a devShell with `yt-dlp`, `openai-whisper`, `ffmpeg`,
and rust toolchain pinned via `rust-overlay`. The Rust code does not
depend on `nix run` at build time, but the default whisper invocation
documented in `--help` uses `nix run nixpkgs#openai-whisper --`.
