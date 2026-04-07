//! Knapsack challenge (instance); serde (de)serialization uses the graph txt format:
//!
//! Line 1: n m type
//! Lines 2..m+1: i j u_ij (edges)
//! Line m+2: q_0 q_1 ... (node weights)
//! Line m+3: B_0 B_1 ... (budgets)

use anyhow::{anyhow, Result};
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub num_items: usize,
    pub weights: Vec<u32>,
    pub values: Vec<u32>,
    pub interaction_values: Vec<Vec<i32>>,
    pub max_weight: u32,
}

impl Challenge {
    /// Instance generation & solution evaluation is done in the mod.rs file.
    ///
    /// Serialize to the graph txt format.
    pub fn to_txt(&self) -> String {
        let n = self.num_items;
        let mut edges: Vec<(usize, usize, i32)> = Vec::new();
        for i in 0..n {
            if self.values[i] != 0 {
                edges.push((i, i, self.values[i] as i32));
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if self.interaction_values[i][j] != 0 {
                    edges.push((i, j, self.interaction_values[i][j]));
                }
            }
        }
        let m = edges.len();
        let type_str = "int";
        let mut out = format!("{} {} {}\n", n, m, type_str);
        for (i, j, u) in edges {
            out.push_str(&format!("{} {} {}\n", i, j, u));
        }
        out.push_str(
            &self
                .weights
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        );
        out.push_str("\n");
        out.push_str(&self.max_weight.to_string());
        out.push_str("\n");
        out
    }

    /// Deserialize from the graph txt format.
    pub fn from_txt(s: &str) -> Result<Self> {
        let mut lines = s.lines().map(str::trim).filter(|l| !l.is_empty());
        let first = lines.next().ok_or_else(|| anyhow!("Missing header line"))?;
        let parts: Vec<&str> = first.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(anyhow!("Header must be 'n m type'"));
        }
        let n: usize = parts[0].parse().map_err(|e| anyhow!("Invalid n: {}", e))?;
        let m: usize = parts[1].parse().map_err(|e| anyhow!("Invalid m: {}", e))?;
        let _type_str = parts[2];

        let mut values = vec![0u32; n];
        let mut interaction_values = vec![vec![0i32; n]; n];

        for _ in 0..m {
            let line = lines.next().ok_or_else(|| anyhow!("Missing edge line"))?;
            let edge_parts: Vec<&str> = line.split_whitespace().collect();
            if edge_parts.len() < 3 {
                return Err(anyhow!("Edge line must be 'i j u_ij'"));
            }
            let i: usize = edge_parts[0]
                .parse()
                .map_err(|e| anyhow!("Invalid i: {}", e))?;
            let j: usize = edge_parts[1]
                .parse()
                .map_err(|e| anyhow!("Invalid j: {}", e))?;
            let u: i32 = edge_parts[2]
                .parse()
                .map_err(|e| anyhow!("Invalid u_ij: {}", e))?;
            if i >= n || j >= n {
                return Err(anyhow!("Edge index out of range: {} {}", i, j));
            }
            if i == j {
                values[i] = u as u32;
            } else {
                interaction_values[i][j] = u;
                interaction_values[j][i] = u;
            }
        }

        let weights_line = lines
            .next()
            .ok_or_else(|| anyhow!("Missing node weights line"))?;
        let weights: Vec<u32> = weights_line
            .split_whitespace()
            .map(|t| {
                t.parse()
                    .map_err(|e| anyhow!("Invalid weight {:?}: {}", t, e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if weights.len() != n {
            return Err(anyhow!(
                "Expected {} node weights, got {}",
                n,
                weights.len()
            ));
        }

        let budgets_line = lines
            .next()
            .ok_or_else(|| anyhow!("Missing budgets line"))?;
        let budgets: Vec<u32> = budgets_line
            .split_whitespace()
            .map(|t| {
                t.parse()
                    .map_err(|e| anyhow!("Invalid budget {:?}: {}", t, e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let max_weight = budgets.first().copied().unwrap_or(0);

        Ok(Challenge {
            seed: [0u8; 32],
            num_items: n,
            weights,
            values,
            interaction_values,
            max_weight,
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
                formatter.write_str("graph txt format (n m type, edges, node weights, budgets)")
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
            num_items: 3,
            weights: vec![40, 5, 4],
            values: vec![35, 2, 100],
            interaction_values: vec![vec![0, 18, 83], vec![18, 0, 12], vec![83, 12, 0]],
            max_weight: 25,
        };
        let txt = c.to_txt();
        let back = Challenge::from_txt(&txt).unwrap();
        assert_eq!(back, c);
    }
}
