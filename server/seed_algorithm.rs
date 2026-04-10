use super::*;
use anyhow::Result;
use serde_json::{Map, Value};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let solution = super::solomon::run(challenge)?;
    save_solution(&solution)?;
    Ok(())
}
