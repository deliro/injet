use std::io;
use std::process::exit;

use clap::Parser;

use injet::cli::{Cli, Commands};
use injet::commands::{extract, inject, inspect};

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    if let Err(e) = match cli.command {
        Commands::Inject(args) => inject::inject(args).map_err(|e| e.to_string()),
        Commands::Extract(args) => extract::extract(args).map_err(|e| e.to_string()),
        Commands::Inspect(args) => inspect::inspect(args).map_err(|e| e.to_string()),
    } {
        eprintln!("{e}");
        exit(1);
    }
    Ok(())
}
