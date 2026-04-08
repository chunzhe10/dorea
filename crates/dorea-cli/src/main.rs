use clap::{Parser, Subcommand};
use dorea_cli::config::DoreaConfig;

#[derive(Parser)]
#[command(name = "dorea", about = "Automated underwater video color grading pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build calibration from keyframes + depth maps + RAUNE targets
    Calibrate(dorea_cli::calibrate::CalibrateArgs),
    /// Grade a video end-to-end using calibration (auto-derives if none provided)
    Grade(dorea_cli::grade::GradeArgs),
    /// Generate a before/after contact sheet (5-10 frames)
    Preview(dorea_cli::preview::PreviewArgs),
    /// Detect container, codec, bit depth, and suggest flags
    Probe(dorea_cli::probe::ProbeArgs),
}

fn main() -> anyhow::Result<()> {
    // Parse CLI first so we can inspect the verbose flag before initialising logging.
    let cli = Cli::parse();

    // Determine log level from the verbose flag on any subcommand.
    let verbose = match &cli.command {
        Command::Calibrate(a) => a.verbose,
        Command::Grade(a) => a.verbose,
        Command::Preview(a) => a.verbose,
        Command::Probe(_) => false,
    };

    env_logger::Builder::from_env(env_logger::Env::default())
        .filter_level(if verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    let config = DoreaConfig::load();

    match cli.command {
        Command::Calibrate(args) => dorea_cli::calibrate::run(args, &config),
        Command::Grade(args) => dorea_cli::grade::run(args, &config),
        Command::Preview(args) => dorea_cli::preview::run(args, &config),
        Command::Probe(args) => dorea_cli::probe::run(args),
    }
}
