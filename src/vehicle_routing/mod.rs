#[cfg(not(feature = "baseline"))]
pub mod algorithm;
#[cfg(feature = "baseline")]
pub mod baseline;
mod challenge;
pub mod solomon;
mod solution;

pub use challenge::*;
pub use solution::*;

use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use statrs::function::erf::{erf, erf_inv};
use std::collections::{HashMap, HashSet};

impl_kv_string_serde! {
    Track {
        n_nodes: usize,
    }
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], track: &Track) -> Result<Self> {
        let mut rng = SmallRng::from_seed(seed.clone());
        let max_capacity = 200;

        let num_clusters = rng.gen_range(3..=8);
        let mut node_positions: Vec<(i32, i32)> = Vec::with_capacity(track.n_nodes);
        let mut node_positions_set: HashSet<(i32, i32)> = HashSet::with_capacity(track.n_nodes);
        node_positions.push((500, 500)); // Depot is node 0, and in the center
        node_positions_set.insert((500, 500));

        let mut cluster_assignments = HashMap::new();
        while node_positions.len() < track.n_nodes {
            let node = node_positions.len();
            if node <= num_clusters || rng.r#gen::<f64>() < 0.5 {
                let pos = (rng.gen_range(0..=1000), rng.gen_range(0..=1000));
                if node_positions_set.contains(&pos) {
                    continue;
                }
                node_positions.push(pos.clone());
                node_positions_set.insert(pos);
            } else {
                let cluster_idx = rng.gen_range(1..=num_clusters);
                let pos = (
                    truncated_normal_sample(
                        &mut rng,
                        node_positions[cluster_idx].0 as f64,
                        60.0,
                        0.0,
                        1000.0,
                    )
                    .round() as i32,
                    truncated_normal_sample(
                        &mut rng,
                        node_positions[cluster_idx].1 as f64,
                        60.0,
                        0.0,
                        1000.0,
                    )
                    .round() as i32,
                );
                if node_positions_set.contains(&pos) {
                    continue;
                }
                node_positions.push(pos.clone());
                node_positions_set.insert(pos);
                cluster_assignments.insert(node, cluster_idx);
            }
        }

        let mut demands: Vec<i32> = (0..track.n_nodes).map(|_| rng.gen_range(1..=35)).collect();
        demands[0] = 0;

        let distance_matrix: Vec<Vec<i32>> = node_positions
            .iter()
            .map(|&from| {
                node_positions
                    .iter()
                    .map(|&to| {
                        let dx = (from.0 - to.0) as f64;
                        let dy = (from.1 - to.1) as f64;
                        dx.hypot(dy).round() as i32
                    })
                    .collect()
            })
            .collect();

        let average_demand = demands.iter().sum::<i32>() as f64 / track.n_nodes as f64;
        let average_route_size = max_capacity as f64 / average_demand;
        let average_distance = (1000.0 / 4.0) * 0.5214;
        let furthest_node = (1..track.n_nodes)
            .max_by_key(|&node| distance_matrix[0][node])
            .unwrap();

        let service_time = 10;
        let mut ready_times = vec![0; track.n_nodes];
        let mut due_times = vec![0; track.n_nodes];

        // time to return to depot
        due_times[0] = distance_matrix[0][furthest_node]
            + ((average_distance + service_time as f64) * average_route_size).ceil() as i32;

        for node in 1..track.n_nodes {
            let min_due_time = distance_matrix[0][node];
            let max_due_time = due_times[0] - distance_matrix[0][node] - service_time;
            due_times[node] = rng.gen_range(min_due_time..=max_due_time);

            if let Some(&closest_cluster) = cluster_assignments.get(&node) {
                due_times[node] = (due_times[node] + due_times[closest_cluster]) / 2;
                due_times[node] = due_times[node].clamp(min_due_time, max_due_time);
            }

            if rng.r#gen::<f64>() < 0.5 {
                ready_times[node] = due_times[node] - rng.gen_range(10..=60);
                ready_times[node] = ready_times[node].max(0);
            }
        }

        let mut c = Challenge {
            seed: seed.clone(),
            num_nodes: track.n_nodes.clone(),
            demands,
            node_positions,
            distance_matrix,
            max_capacity,
            fleet_size: u32::MAX as usize,
            service_time,
            ready_times,
            due_times,
        };

        let solomon_solution = solomon::run(&c)?;
        c.fleet_size = solomon_solution.routes.len() + 2;

        Ok(c)
    }

    pub fn evaluate_total_distance(&self, solution: &Solution) -> Result<i32> {
        if solution.routes.len() > self.fleet_size {
            return Err(anyhow!(
                "Number of routes ({}) exceeds fleet size ({})",
                solution.routes.len(),
                self.fleet_size
            ));
        }
        let mut total_distance = 0;
        let mut visited = vec![false; self.num_nodes];
        visited[0] = true;

        for route in &solution.routes {
            if route.len() <= 2 || route[0] != 0 || route[route.len() - 1] != 0 {
                return Err(anyhow!("Each route must start and end at node 0 (the depot), and visit at least one non-depot node"));
            }

            let mut capacity = self.max_capacity;
            let mut current_node = 0;
            let mut curr_time = 0;
            for &node in &route[1..route.len() - 1] {
                if visited[node] {
                    return Err(anyhow!(
                        "The same non-depot node cannot be visited more than once"
                    ));
                }
                if self.demands[node] > capacity {
                    return Err(anyhow!(
                        "The total demand on each route must not exceed max capacity"
                    ));
                }
                curr_time += self.distance_matrix[current_node][node];
                if curr_time > self.due_times[node] {
                    return Err(anyhow!("Node must be visited before due time"));
                }
                if curr_time < self.ready_times[node] {
                    curr_time = self.ready_times[node];
                }
                curr_time += self.service_time;
                visited[node] = true;
                capacity -= self.demands[node];
                total_distance += self.distance_matrix[current_node][node];
                current_node = node;
            }

            curr_time += self.distance_matrix[current_node][0];
            if curr_time > self.due_times[0] {
                return Err(anyhow!("Must return to depot before due time"));
            }
            total_distance += self.distance_matrix[current_node][0];
        }

        if visited.iter().any(|&v| !v) {
            return Err(anyhow!("All nodes must be visited"));
        }

        Ok(total_distance)
    }

    pub fn evaluate_solution(&self, solution: &Solution) -> Result<f32> {
        Ok(self.evaluate_total_distance(solution)? as f32)
    }
}

fn truncated_normal_sample<T: Rng>(
    rng: &mut T,
    mean: f64,
    std_dev: f64,
    min_val: f64,
    max_val: f64,
) -> f64 {
    let cdf_min = 0.5 * (1.0 + erf((min_val - mean) / (std_dev * (2.0_f64).sqrt())));
    let cdf_max = 0.5 * (1.0 + erf((max_val - mean) / (std_dev * (2.0_f64).sqrt())));
    let sample = rng.r#gen::<f64>() * (cdf_max - cdf_min) + cdf_min;
    mean + std_dev * (2.0_f64).sqrt() * erf_inv(2.0 * sample - 1.0)
}
