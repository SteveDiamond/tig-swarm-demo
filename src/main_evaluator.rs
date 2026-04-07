use anyhow::Result;
use clap::{arg, value_parser, Command};
use std::fs;
use std::path::{Path, PathBuf};
use tig_challenges as challenges;

fn cli() -> Command {
    Command::new("tig-challenges-evaluator")
        .about("TIG challenge evaluation")
        .arg(
            arg!(<CHALLENGE> "Challenge name: knapsack, vehicle_routing, job_scheduling")
                .value_parser(value_parser!(String)),
        )
        .arg(arg!(<INSTANCE_FILE> "Path to the instance file").value_parser(value_parser!(PathBuf)))
        .arg(arg!(<SOLUTION_FILE> "Path to the solution file").value_parser(value_parser!(PathBuf)))
}

fn run_evaluate(challenge: &str, instance_file: &Path, solution_file: &Path) -> Result<()> {
    anyhow::ensure!(
        instance_file.exists(),
        "Instance file does not exist: {}",
        instance_file.display()
    );
    anyhow::ensure!(
        solution_file.exists(),
        "Solution file does not exist: {}",
        solution_file.display()
    );
    let instance_content = fs::read_to_string(instance_file)?;
    let solution_content = fs::read_to_string(solution_file)?;

    macro_rules! dispatch_evaluate {
        ($c:ident) => {{
            let instance = challenges::$c::Challenge::from_txt(&instance_content)?;
            let solution = challenges::$c::Solution::from_txt(&solution_content)?;
            instance.evaluate_solution(&solution)?
        }};
    }

    let out = match challenge {
        "knapsack" => dispatch_evaluate!(knapsack),
        "vehicle_routing" => dispatch_evaluate!(vehicle_routing),
        "job_scheduling" => dispatch_evaluate!(job_scheduling),
        _ => anyhow::bail!("Unknown challenge: {}", challenge),
    };
    println!("Output: {}", out);
    Ok(())
}

fn main() -> Result<()> {
    let matches = cli().get_matches();
    let challenge = matches.get_one::<String>("CHALLENGE").unwrap();
    let instance_file = matches.get_one::<PathBuf>("INSTANCE_FILE").unwrap();
    let solution_file = matches.get_one::<PathBuf>("SOLUTION_FILE").unwrap();
    run_evaluate(challenge, instance_file, solution_file)
}
