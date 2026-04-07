//! Knapsack solution; serde (de)serialization uses a single line of space-separated item indices.

use anyhow::Result;
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Solution {
    pub items: Vec<usize>,
}

impl Solution {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize to a single line of space-separated numbers (item indices).
    pub fn to_txt(&self) -> String {
        self.items
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Deserialize from a single line of space-separated numbers.
    pub fn from_txt(s: &str) -> Result<Self> {
        let items: Vec<usize> = s
            .split_whitespace()
            .map(|t| t.parse().map_err(|e| anyhow::anyhow!("Invalid item index {:?}: {}", t, e)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Solution { items })
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
                formatter.write_str("a line of space-separated item indices")
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
            items: vec![0, 2, 4],
        };
        let txt = sol.to_txt();
        let back = Solution::from_txt(&txt).unwrap();
        assert_eq!(back, sol);
    }
}
