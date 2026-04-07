//! Vehicle routing solution; serde (de)serialization uses route list txt format.
//!
//! Format: lines like "Route 1 : 92 195 31 ..." (space-separated node indices per route).
//! Routes are assumed to start and end at depot (node 0); the serialized form omits leading/trailing 0.

use anyhow::{anyhow, Result};
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Solution {
    pub routes: Vec<Vec<usize>>,
}

impl Solution {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize to route list txt format (Route K : n1 n2 n3 ... per line).
    pub fn to_txt(&self) -> String {
        let mut out = String::new();
        for (i, route) in self.routes.iter().enumerate() {
            // Omit leading and trailing depot (0)
            let inner: Vec<usize> = match route.len() {
                0 | 1 => vec![],
                2 => vec![],
                _ => route[1..route.len() - 1].to_vec(),
            };
            let nums: Vec<String> = inner.iter().map(|n| n.to_string()).collect();
            out.push_str(&format!("Route {} : {}\n", i + 1, nums.join(" ")));
        }
        out
    }

    /// Deserialize from route list txt format. Adds depot (0) at start and end of each route.
    pub fn from_txt(s: &str) -> Result<Self> {
        let mut routes = Vec::new();
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let rest = line
                .strip_prefix("Route ")
                .and_then(|r| {
                    let p = r.find(" : ")?;
                    Some(r[p + 3..].trim())
                })
                .ok_or_else(|| anyhow!("Expected 'Route N : ...' line: {:?}", line))?;
            let nodes: Vec<usize> = if rest.is_empty() {
                vec![]
            } else {
                rest.split_whitespace()
                    .map(|t| t.parse().map_err(|e| anyhow!("Invalid node index {:?}: {}", t, e)))
                    .collect::<Result<Vec<_>, _>>()?
            };
            // Wrap with depot at start and end (unless empty route)
            let route = if nodes.is_empty() {
                vec![0, 0]
            } else {
                let mut r = vec![0];
                r.extend(nodes);
                r.push(0);
                r
            };
            routes.push(route);
        }
        Ok(Solution { routes })
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
                formatter.write_str("route list txt (Route N : ...)")
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
            routes: vec![
                vec![0, 92, 195, 31, 0],
                vec![0, 142, 48, 0],
            ],
        };
        let txt = sol.to_txt();
        let back = Solution::from_txt(&txt).unwrap();
        assert_eq!(back, sol);
    }
}
