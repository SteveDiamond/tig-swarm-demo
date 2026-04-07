//! Job scheduling challenge (instance); serde (de)serialization uses Brandimarte .fjs format.
//!
//! First line: `<num_jobs> <num_machines> [avg_machines_per_op]`
//! Then for each job: `<num_operations>` then for each operation: `<num_eligible_machines> <machine_1> <time_1> ...`
//! Machine indices in the file are 1-based; stored internally as 0-based. Seed is not serialized.

use anyhow::{anyhow, Result};
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub num_jobs: usize,
    pub num_machines: usize,
    pub num_operations: usize,
    pub jobs_per_product: Vec<usize>,
    pub product_processing_times: Vec<Vec<HashMap<usize, u32>>>,
}

impl Challenge {
    /// Instance generation & solution evaluation is done in the mod.rs file.

    /// Returns the product index for the given job index (0-based), or None if out of range.
    pub fn job_to_product(&self, job_0: usize) -> Option<usize> {
        let mut acc = 0;
        for (p, &count) in self.jobs_per_product.iter().enumerate() {
            if job_0 < acc + count {
                return Some(p);
            }
            acc += count;
        }
        None
    }

    /// Serialize to Brandimarte .fjs format. Machine indices in file are 1-based. Seed not serialized.
    pub fn to_txt(&self) -> String {
        let avg = self
            .product_processing_times
            .iter()
            .flat_map(|ops| ops.iter().map(|m| m.len()))
            .sum::<usize>();
        let total_ops: usize = self
            .product_processing_times
            .iter()
            .map(|ops| ops.len())
            .sum::<usize>();
        let avg_str = if total_ops > 0 {
            format!("{}", avg / total_ops)
        } else {
            "0".to_string()
        };
        let mut out = format!("{} {} {}\n", self.num_jobs, self.num_machines, avg_str);
        for job_0 in 0..self.num_jobs {
            let product = self.job_to_product(job_0).unwrap_or(0);
            let ops = &self.product_processing_times[product];
            let mut line = format!("{}", ops.len());
            for op_map in ops {
                let mut pairs: Vec<(usize, u32)> = op_map.iter().map(|(&m, &t)| (m, t)).collect();
                pairs.sort_by_key(|&(m, _)| m);
                line.push_str(&format!(" {}", pairs.len()));
                for (machine_0, time) in pairs {
                    line.push_str(&format!(" {} {}", machine_0 + 1, time));
                }
            }
            out.push_str(&line);
            out.push('\n');
        }
        out
    }

    /// Deserialize from Brandimarte .fjs format. Infers products by grouping jobs with identical operation structure. Seed set to zeros.
    pub fn from_txt(s: &str) -> Result<Self> {
        let mut lines = s.lines().map(str::trim).filter(|l| !l.is_empty());
        let first = lines
            .next()
            .ok_or_else(|| anyhow!("Missing first line (num_jobs num_machines [avg])"))?;
        let parts: Vec<&str> = first.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(anyhow!(
                "First line must have at least num_jobs and num_machines"
            ));
        }
        let num_jobs: usize = parts[0]
            .parse()
            .map_err(|e| anyhow!("Invalid num_jobs: {}", e))?;
        let num_machines: usize = parts[1]
            .parse()
            .map_err(|e| anyhow!("Invalid num_machines: {}", e))?;

        let mut tokens: Vec<usize> = Vec::new();
        for line in lines {
            for t in line.split_whitespace() {
                let n: u32 = t
                    .parse()
                    .map_err(|e| anyhow!("Invalid number {:?}: {}", t, e))?;
                tokens.push(n as usize);
            }
        }

        let mut pos = 0;
        let mut job_ops: Vec<Vec<HashMap<usize, u32>>> = Vec::with_capacity(num_jobs);
        for _ in 0..num_jobs {
            if pos >= tokens.len() {
                return Err(anyhow!(
                    "Unexpected end of data: expected num_operations for job"
                ));
            }
            let num_ops = tokens[pos];
            pos += 1;
            let mut ops = Vec::with_capacity(num_ops);
            for _ in 0..num_ops {
                if pos >= tokens.len() {
                    return Err(anyhow!(
                        "Unexpected end of data: expected num_machines for operation"
                    ));
                }
                let num_machines_op = tokens[pos];
                pos += 1;
                let mut map = HashMap::new();
                for _ in 0..num_machines_op {
                    if pos + 1 >= tokens.len() {
                        return Err(anyhow!(
                            "Unexpected end of data: expected (machine, time) pair"
                        ));
                    }
                    let machine_1 = tokens[pos];
                    let time = tokens[pos + 1] as u32;
                    pos += 2;
                    if machine_1 == 0 {
                        return Err(anyhow!("Machine index in file must be 1-based (>= 1)"));
                    }
                    map.insert(machine_1 - 1, time);
                }
                ops.push(map);
            }
            job_ops.push(ops);
        }

        if pos != tokens.len() {
            return Err(anyhow!(
                "Extra data after jobs: consumed {} tokens, total {}",
                pos,
                tokens.len()
            ));
        }

        let mut product_processing_times: Vec<Vec<HashMap<usize, u32>>> = Vec::new();
        let mut jobs_per_product: Vec<usize> = Vec::new();
        for job_op in job_ops {
            let found = product_processing_times
                .iter()
                .position(|p: &Vec<HashMap<usize, u32>>| p == &job_op);
            match found {
                Some(idx) => {
                    jobs_per_product[idx] += 1;
                }
                None => {
                    product_processing_times.push(job_op);
                    jobs_per_product.push(1);
                }
            }
        }
        let num_operations = product_processing_times
            .iter()
            .map(|p| p.len())
            .max()
            .unwrap_or(0);

        Ok(Challenge {
            seed: [0u8; 32],
            num_jobs,
            num_machines,
            num_operations,
            jobs_per_product,
            product_processing_times,
        })
    }
}

impl Serialize for Challenge {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_txt())
    }
}

impl<'de> Deserialize<'de> for Challenge {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ChallengeVisitor;
        impl<'de> Visitor<'de> for ChallengeVisitor {
            type Value = Challenge;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str(
                    "Brandimarte .fjs format (num_jobs num_machines, then per-job operations)",
                )
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Challenge::from_txt(v).map_err(de::Error::custom)
            }
        }
        deserializer.deserialize_str(ChallengeVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_txt() {
        let c = Challenge {
            seed: [0u8; 32],
            num_jobs: 4,
            num_machines: 3,
            num_operations: 2, // Brandimarte roundtrip uses max ops per product
            jobs_per_product: vec![2, 2],
            product_processing_times: vec![
                vec![
                    HashMap::from([(0, 3), (1, 4)]),
                    HashMap::from([(0, 2), (1, 1), (2, 3)]),
                ],
                vec![
                    HashMap::from([(0, 2), (1, 1), (2, 3)]),
                    HashMap::from([(2, 4)]),
                ],
            ],
        };
        let txt = c.to_txt();
        let back = Challenge::from_txt(&txt).unwrap();
        assert_eq!(back, c);
    }
}
