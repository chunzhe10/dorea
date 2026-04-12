use clap::Parser;
use dorea_cli::{config::DoreaConfig, grade::{self, GradeArgs}};

fn main() -> anyhow::Result<()> {
    let args = GradeArgs::parse();

    env_logger::Builder::from_env(env_logger::Env::default())
        .filter_level(if args.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    let config = DoreaConfig::load();
    grade::run(args, &config)
}
