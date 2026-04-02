use clap::{Parser, Subcommand};

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
    /// Grade a video using a calibration file (Phase 3)
    Grade {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
        #[arg(long)]
        calibration: String,
        #[arg(long, default_value = "1.0")]
        warmth: f32,
        #[arg(long, default_value = "0.8")]
        strength: f32,
    },
    /// Generate a before/after contact sheet (Phase 3)
    Preview {
        #[arg(long)]
        input: String,
        #[arg(long)]
        calibration: String,
        #[arg(long)]
        output: String,
    },
}

fn main() -> anyhow::Result<()> {
    // Parse CLI first so we can inspect the verbose flag before initialising logging.
    let cli = Cli::parse();

    // Determine log level from the verbose flag on the Calibrate subcommand (before init).
    let verbose = matches!(&cli.command, Command::Calibrate(args) if args.verbose);
    env_logger::Builder::from_env(env_logger::Env::default())
        .filter_level(if verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();
    match cli.command {
        Command::Calibrate(args) => dorea_cli::calibrate::run(args),
        Command::Grade { .. } => {
            anyhow::bail!("dorea grade is not yet implemented (Phase 3)")
        }
        Command::Preview { .. } => {
            anyhow::bail!("dorea preview is not yet implemented (Phase 3)")
        }
    }
}
