use anyhow::Result;
use clap::{arg, value_parser, Command};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tig_challenges as challenges;

fn cli() -> Command {
    Command::new("tig-challenges-generator")
        .about("TIG challenge instance generation")
        .arg(
            arg!(<CHALLENGE> "Challenge name: knapsack, vehicle_routing, job_scheduling")
                .value_parser(value_parser!(String)),
        )
        .arg(
            arg!(<TRACK> "Track specification (key=value,key=value format, challenge-specific)")
                .value_parser(value_parser!(String)),
        )
        .arg(
            arg!(--seed [SEED] "Random seed string (hashed for instance generation)")
                .default_value("0")
                .value_parser(value_parser!(String)),
        )
        .arg(
            arg!(-n --n <N> "Number of instances to generate")
                .default_value("1")
                .value_parser(value_parser!(String)),
        )
        .arg(
            arg!(-o --out [OUT] "Output directory for instance files")
                .value_parser(value_parser!(PathBuf)),
        )
}

fn run_generate(
    challenge: &str,
    track_id: &str,
    seed: &str,
    n: usize,
    out: Option<&PathBuf>,
) -> Result<()> {
    let out_dir: PathBuf = out
        .cloned()
        .unwrap_or_else(|| PathBuf::from(format!("{}/{}", challenge, track_id)));
    fs::create_dir_all(&out_dir)?;

    macro_rules! dispatch_generate {
        ($c:ident) => {{
            let track = if track_id.starts_with('"') && track_id.ends_with('"') {
                track_id.to_string()
            } else {
                format!(r#""{}""#, track_id)
            };
            let track = serde_json::from_str::<challenges::$c::Track>(&track).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to parse track '{}' as {}::Track: {}",
                    track,
                    stringify!($c),
                    e
                )
            })?;
            (0..n).into_par_iter().try_for_each(|i| {
                let instance_seed =
                    blake3::hash(format!("{}-{}-{}-{}", challenge, track_id, seed, i).as_bytes());
                let instance =
                    challenges::$c::Challenge::generate_instance(instance_seed.as_bytes(), &track)?;
                let path = Path::new(&out_dir).join(format!("{}.txt", i));
                fs::write(path, instance.to_txt())?;
                Ok::<(), anyhow::Error>(())
            })?;
        }};
    }

    match challenge {
        "knapsack" => dispatch_generate!(knapsack),
        "vehicle_routing" => dispatch_generate!(vehicle_routing),
        "job_scheduling" => dispatch_generate!(job_scheduling),
        _ => anyhow::bail!("Unknown challenge: {}", challenge),
    }
    Ok(())
}

fn main() -> Result<()> {
    let matches = cli().get_matches();
    let challenge = matches.get_one::<String>("CHALLENGE").unwrap();
    let track = matches.get_one::<String>("TRACK").unwrap();
    let seed = matches.get_one::<String>("seed").unwrap();
    let n: usize = matches.get_one::<String>("n").unwrap().parse()?;
    let out = matches.get_one::<PathBuf>("out");
    run_generate(challenge, track, seed, n, out)
}
