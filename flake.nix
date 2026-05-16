{
  description = "youtube-archiver — download + whisper-transcribe + llm-wiki ingest";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        rust = pkgs.rust-bin.stable.latest.default;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            rust
            pkgs.pkg-config
            pkgs.yt-dlp
            pkgs.ffmpeg
            pkgs.openai-whisper
            pkgs.sqlite
          ];
        };
      });
}
