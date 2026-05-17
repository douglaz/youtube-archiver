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

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" ];
          targets = [ "x86_64-unknown-linux-musl" ];
        };

        muslCc = "${pkgs.pkgsStatic.stdenv.cc}/bin/${pkgs.pkgsStatic.stdenv.cc.targetPrefix}cc";
      in
      {
        # Default package: fully static musl binary. rusqlite's `bundled` feature
        # compiles SQLite from source, so we only need a C compiler — no system
        # libsqlite needed.
        packages.default = let
          rustPlatformMusl = pkgs.makeRustPlatform {
            cargo = rustToolchain;
            rustc = rustToolchain;
          };
        in rustPlatformMusl.buildRustPackage {
          pname = "youtube-archiver";
          version = "0.1.0";
          src = ./.;

          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = with pkgs; [
            pkg-config
            rustToolchain
            pkgsStatic.stdenv.cc
          ];

          CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER = muslCc;
          CC_x86_64_unknown_linux_musl = muslCc;
          CARGO_BUILD_RUSTFLAGS = "-C target-feature=+crt-static -C link-arg=-static";

          buildPhase = ''
            runHook preBuild
            cargo build \
              --release \
              --target x86_64-unknown-linux-musl \
              --offline \
              -j $NIX_BUILD_CORES
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            mkdir -p $out/bin
            cp target/x86_64-unknown-linux-musl/release/youtube-archiver $out/bin/
            runHook postInstall
          '';

          # Tests want filesystem access for tempdirs; the sandbox handles that,
          # but the gated e2e test (YTARCH_E2E=1) reaches the network, so skip
          # cargo test in the nix build and rely on the GitHub Actions test job.
          doCheck = false;

          postInstall = ''
            echo "Checking if binary is statically linked..."
            file $out/bin/youtube-archiver
            ${pkgs.binutils}/bin/strip $out/bin/youtube-archiver
          '';

          meta = with pkgs.lib; {
            description = "Archive YouTube audio, transcribe with Whisper, emit llm-wiki-ingestible markdown";
            homepage = "https://github.com/douglaz/youtube-archiver";
            license = licenses.mit;
            maintainers = [ ];
            mainProgram = "youtube-archiver";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            bashInteractive
            rustToolchain
            pkg-config
            pkgsStatic.stdenv.cc
            # Runtime dependencies of the CLI itself.
            yt-dlp
            ffmpeg
            openai-whisper
            sqlite
            # Dev / CI tooling.
            gh
            cargo-edit
            cargo-outdated
          ];

          CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER = muslCc;
          CC_x86_64_unknown_linux_musl = muslCc;
        };
      });
}
