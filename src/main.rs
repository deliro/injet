use std::process::exit;

use clap::Parser;

use injet::cli::{Cli, Commands};
use injet::commands::{extract, inject, inspect};

#[cfg(any(
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "macos",
    target_os = "windows",
    target_os = "freebsd",
))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let cli = Cli::parse();
    let result: Result<(), (String, i32)> = match &cli.command {
        Commands::Inject(args) => inject::inject(args).map_err(|e| (e.to_string(), e.exit_code())),
        Commands::Extract(args) => extract::extract(args).map_err(|e| (e.to_string(), 1_i32)),
        Commands::Inspect(args) => inspect::inspect(args).map_err(|e| (e.to_string(), 1_i32)),
    };
    if let Err((msg, code)) = result {
        eprintln!("{msg}");
        exit(code);
    }
}
