//! Vehicle routing challenge (instance); serde (de)serialization uses Solomon-style txt format.
//!
//! VEHICLE section: NUMBER, CAPACITY.
//! CUSTOMER section: CUST NO. XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME (one row per node).

use anyhow::{anyhow, Result};
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub num_nodes: usize,
    pub demands: Vec<i32>,
    pub node_positions: Vec<(i32, i32)>,
    pub distance_matrix: Vec<Vec<i32>>,
    pub max_capacity: i32,
    pub fleet_size: usize,
    pub service_time: i32,
    pub ready_times: Vec<i32>,
    pub due_times: Vec<i32>,
}

fn round_distance(from: (i32, i32), to: (i32, i32)) -> i32 {
    let dx = (from.0 - to.0) as f64;
    let dy = (from.1 - to.1) as f64;
    dx.hypot(dy).round() as i32
}

impl Challenge {
    /// Instance generation & solution evaluation is done in the mod.rs file.

    /// Serialize to Solomon-style txt format (prettified columns). Seed is not serialized.
    pub fn to_txt(&self) -> String {
        let mut out = String::new();
        out.push_str("VEHICLE\n");
        out.push_str("NUMBER     CAPACITY\n");
        out.push_str(&format!(
            "  {:>3}          {}\n\n",
            self.fleet_size, self.max_capacity
        ));
        out.push_str("CUSTOMER\n");
        out.push_str(
            "CUST NO.  XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME\n\n",
        );
        for i in 0..self.num_nodes {
            let (x, y) = self.node_positions[i];
            out.push_str(&format!(
                "{:5} {:10} {:10} {:10} {:12} {:10} {:12}\n",
                i, x, y, self.demands[i], self.ready_times[i], self.due_times[i], self.service_time
            ));
        }
        out
    }

    /// Deserialize from Solomon-style txt format. Seed and greedy_baseline_total_distance set to 0.
    pub fn from_txt(s: &str) -> Result<Self> {
        let mut lines = s.lines().map(str::trim).filter(|l| !l.is_empty());
        // Optionally skip an instance name line if not "VEHICLE"
        let _veh = {
            let l = lines
                .next()
                .ok_or_else(|| anyhow!("Expected VEHICLE header"))?;
            if l.eq_ignore_ascii_case("VEHICLE") {
                Some(l)
            } else {
                // skip this line, expect VEHICLE next
                let l2 = lines
                    .next()
                    .ok_or_else(|| anyhow!("Expected VEHICLE header after instance name line"))?;
                if l2.eq_ignore_ascii_case("VEHICLE") {
                    Some(l2)
                } else {
                    None
                }
            }
        }
        .ok_or_else(|| anyhow!("Expected VEHICLE header"))?;
        let _header = lines
            .next()
            .ok_or_else(|| anyhow!("Expected NUMBER CAPACITY line"))?;
        let num_cap = lines
            .next()
            .ok_or_else(|| anyhow!("Expected fleet size and capacity line"))?;
        let num_cap: Vec<&str> = num_cap.split_whitespace().collect();
        if num_cap.len() < 2 {
            return Err(anyhow!("Expected NUMBER and CAPACITY"));
        }
        let fleet_size: usize = num_cap[0]
            .parse()
            .map_err(|e| anyhow!("Invalid fleet size: {}", e))?;
        let max_capacity: i32 = num_cap[1]
            .parse()
            .map_err(|e| anyhow!("Invalid capacity: {}", e))?;

        let _cust = lines
            .find(|l| l.eq_ignore_ascii_case("CUSTOMER"))
            .ok_or_else(|| anyhow!("Expected CUSTOMER header"))?;
        let _col_header = lines.next(); // skip column header line (blank lines already filtered out)

        let mut node_positions = Vec::new();
        let mut demands = Vec::new();
        let mut ready_times = Vec::new();
        let mut due_times = Vec::new();
        let mut service_time = 0i32;

        for line in lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 7 {
                return Err(anyhow!(
                    "Customer line must have at least 7 columns: {:?}",
                    line
                ));
            }
            let _cust_no: usize = parts[0]
                .parse()
                .map_err(|e| anyhow!("Invalid CUST NO.: {}", e))?;
            let x: i32 = parts[1]
                .parse()
                .map_err(|e| anyhow!("Invalid XCOORD: {}", e))?;
            let y: i32 = parts[2]
                .parse()
                .map_err(|e| anyhow!("Invalid YCOORD: {}", e))?;
            let demand: i32 = parts[3]
                .parse()
                .map_err(|e| anyhow!("Invalid DEMAND: {}", e))?;
            let ready: i32 = parts[4]
                .parse()
                .map_err(|e| anyhow!("Invalid READY TIME: {}", e))?;
            let due: i32 = parts[5]
                .parse()
                .map_err(|e| anyhow!("Invalid DUE DATE: {}", e))?;
            let serv: i32 = parts[6]
                .parse()
                .map_err(|e| anyhow!("Invalid SERVICE TIME: {}", e))?;
            node_positions.push((x, y));
            demands.push(demand);
            ready_times.push(ready);
            due_times.push(due);
            service_time = serv;
        }

        let num_nodes = node_positions.len();
        let distance_matrix: Vec<Vec<i32>> = node_positions
            .iter()
            .map(|&from| {
                node_positions
                    .iter()
                    .map(|&to| round_distance(from, to))
                    .collect()
            })
            .collect();

        Ok(Challenge {
            seed: [0u8; 32],
            num_nodes,
            demands,
            node_positions,
            distance_matrix,
            max_capacity,
            fleet_size,
            service_time,
            ready_times,
            due_times,
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
                formatter.write_str("Solomon-style VEHICLE/CUSTOMER txt")
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
        let node_positions = vec![(70, 70), (10, 86), (16, 62)];
        let distance_matrix: Vec<Vec<i32>> = node_positions
            .iter()
            .map(|&from| {
                node_positions
                    .iter()
                    .map(|&to| round_distance(from, to))
                    .collect()
            })
            .collect();
        let c = Challenge {
            seed: [0u8; 32],
            num_nodes: 3,
            demands: vec![0, 14, 15],
            node_positions: node_positions.clone(),
            distance_matrix,
            max_capacity: 200,
            fleet_size: 50,
            service_time: 10,
            ready_times: vec![0, 411, 396],
            due_times: vec![634, 441, 426],
        };
        let txt = c.to_txt();
        let back = Challenge::from_txt(&txt).unwrap();
        assert_eq!(back, c);
    }
}
