//! Job scheduling solution; serde (de)serialization uses 1-idx txt format:
//!
//! One line per operation: `<Job> <Operation> <Machine> <Start_Time>` (all 1-indexed).
//! When reading, a fifth column (Finish_Time) is ignored if present.

use anyhow::{anyhow, Result};
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Solution {
    pub job_schedule: Vec<Vec<(usize, u32)>>,
}

impl Solution {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize to 1-idx format (4 columns: Job Operation Machine Start).
    pub fn to_txt(&self) -> String {
        let mut lines = Vec::new();
        for (job_0, schedule) in self.job_schedule.iter().enumerate() {
            let job_1 = job_0 + 1;
            for (op_0, &(machine_0, start)) in schedule.iter().enumerate() {
                let op_1 = op_0 + 1;
                let machine_1 = machine_0 + 1;
                lines.push(format!("{} {} {} {}", job_1, op_1, machine_1, start));
            }
        }
        lines.join("\n")
    }

    /// Deserialize from 1-idx format. Accepts 4 or 5 columns per line; Finish_Time (5th column) is ignored if present.
    pub fn from_txt(s: &str) -> Result<Self> {
        let mut rows: Vec<(usize, usize, usize, u32)> = Vec::new();
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(anyhow!("Expected at least 4 columns: Job Operation Machine Start [Finish], got {:?}", line));
            }
            let job_1: usize = parts[0].parse().map_err(|e| anyhow!("Invalid job: {} - {}", parts[0], e))?;
            let op_1: usize = parts[1].parse().map_err(|e| anyhow!("Invalid operation: {} - {}", parts[1], e))?;
            let machine_1: usize = parts[2].parse().map_err(|e| anyhow!("Invalid machine: {} - {}", parts[2], e))?;
            let start: u32 = parts[3].parse().map_err(|e| anyhow!("Invalid start: {} - {}", parts[3], e))?;
            // ignore 5th column (Finish_Time) if present
            if job_1 == 0 || op_1 == 0 || machine_1 == 0 {
                return Err(anyhow!("Job, Operation, and Machine must be 1-indexed (>= 1)"));
            }
            rows.push((job_1, op_1, machine_1, start));
        }
        rows.sort_by_key(|&(j, o, _, _)| (j, o));
        let num_jobs = rows.iter().map(|&(j, _, _, _)| j).max().unwrap_or(0);
        let mut job_schedule: Vec<Vec<(usize, u32)>> = (0..num_jobs).map(|_| Vec::new()).collect();
        for (job_1, _op_1, machine_1, start) in rows {
            let job_0 = job_1 - 1;
            let machine_0 = machine_1 - 1;
            if job_0 < job_schedule.len() {
                job_schedule[job_0].push((machine_0, start));
            }
        }
        Ok(Solution { job_schedule })
    }
}

impl Serialize for Solution {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_txt())
    }
}

impl<'de> Deserialize<'de> for Solution {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SolutionVisitor;
        impl<'de> Visitor<'de> for SolutionVisitor {
            type Value = Solution;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("1-idx solution lines: Job Operation Machine Start [Finish]")
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Solution::from_txt(v).map_err(de::Error::custom)
            }
        }
        deserializer.deserialize_str(SolutionVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_txt() {
        let sol = Solution {
            job_schedule: vec![
                vec![(0, 0), (1, 4)],
                vec![(1, 0), (0, 4)],
            ],
        };
        let txt = sol.to_txt();
        let back = Solution::from_txt(&txt).unwrap();
        assert_eq!(back, sol);
    }

    #[test]
    fn from_txt_ignores_finish_column() {
        // 5-column input (Job Op Machine Start Finish); Finish is ignored
        let txt = "1 1 1 0 3\n1 2 2 3 4\n2 1 2 0 4\n2 2 2 4 8";
        let back = Solution::from_txt(txt).unwrap();
        assert_eq!(back.job_schedule.len(), 2);
        assert_eq!(back.job_schedule[0], vec![(0, 0), (1, 3)]);
        assert_eq!(back.job_schedule[1], vec![(1, 0), (1, 4)]);
    }
}
