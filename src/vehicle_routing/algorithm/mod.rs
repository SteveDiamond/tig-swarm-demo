use super::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_nodes;
    let capacity = challenge.max_capacity;
    let fleet = challenge.fleet_size;
    let dm = &challenge.distance_matrix;

    // Phase 1: Build routes with nearest-neighbor, no fleet limit yet
    let mut visited = vec![false; n];
    visited[0] = true;
    let mut routes: Vec<Vec<usize>> = Vec::new();

    while visited.iter().skip(1).any(|&v| !v) {
        // Seed with farthest unvisited customer from depot
        let mut seed = 0;
        let mut max_dist = -1;
        for i in 1..n {
            if !visited[i] && dm[0][i] > max_dist {
                max_dist = dm[0][i];
                seed = i;
            }
        }
        if seed == 0 { break; }

        visited[seed] = true;
        let mut route = vec![0, seed];
        let mut load = challenge.demands[seed];
        let mut current_time = dm[0][seed].max(challenge.ready_times[seed]) + challenge.service_time;
        let mut current = seed;

        loop {
            let mut best_dist = i32::MAX;
            let mut best_node = None;

            for j in 1..n {
                if visited[j] { continue; }
                if load + challenge.demands[j] > capacity { continue; }
                let travel = dm[current][j];
                let arrival = current_time + travel;
                if arrival > challenge.due_times[j] { continue; }
                let finish = arrival.max(challenge.ready_times[j]) + challenge.service_time;
                if finish + dm[j][0] > challenge.due_times[0] { continue; }
                if travel < best_dist {
                    best_dist = travel;
                    best_node = Some(j);
                }
            }

            match best_node {
                Some(j) => {
                    visited[j] = true;
                    route.push(j);
                    load += challenge.demands[j];
                    let arrival = current_time + dm[current][j];
                    current_time = arrival.max(challenge.ready_times[j]) + challenge.service_time;
                    current = j;
                }
                None => break,
            }
        }

        route.push(0);
        routes.push(route);
    }

    // Phase 2: Merge routes if we exceeded fleet size
    while routes.len() > fleet {
        // Find the two routes whose merge causes the least distance increase
        // and remains feasible
        let mut best_cost = i64::MAX;
        let mut best_i = 0;
        let mut best_j = 0;
        let mut best_merged: Option<Vec<usize>> = None;

        for i in 0..routes.len() {
            for j in (i + 1)..routes.len() {
                // Check combined capacity
                let load_i: i32 = routes[i].iter().filter(|&&x| x != 0).map(|&x| challenge.demands[x]).sum();
                let load_j: i32 = routes[j].iter().filter(|&&x| x != 0).map(|&x| challenge.demands[x]).sum();
                if load_i + load_j > capacity { continue; }

                // Try appending route j's customers after route i's
                let mut merged = routes[i][..routes[i].len() - 1].to_vec(); // drop trailing depot
                merged.extend_from_slice(&routes[j][1..]); // skip leading depot

                if is_route_feasible(&merged, challenge) {
                    let cost = route_distance(&merged, dm);
                    let orig = route_distance(&routes[i], dm) + route_distance(&routes[j], dm);
                    let delta = cost as i64 - orig as i64;
                    if delta < best_cost {
                        best_cost = delta;
                        best_i = i;
                        best_j = j;
                        best_merged = Some(merged);
                    }
                }

                // Also try inserting j's customers one by one into i (cheapest insertion)
                let mut candidate = routes[i].clone();
                let mut feasible = true;
                let j_customers: Vec<usize> = routes[j].iter().filter(|&&x| x != 0).copied().collect();
                for &c in &j_customers {
                    let mut min_cost = i64::MAX;
                    let mut min_pos = 1;
                    let mut found = false;
                    for pos in 1..candidate.len() {
                        let prev = candidate[pos - 1];
                        let next = candidate[pos];
                        let ins_cost = (dm[prev][c] + dm[c][next] - dm[prev][next]) as i64;
                        if ins_cost < min_cost {
                            let mut test = candidate.clone();
                            test.insert(pos, c);
                            if is_route_feasible(&test, challenge) {
                                min_cost = ins_cost;
                                min_pos = pos;
                                found = true;
                            }
                        }
                    }
                    if found {
                        candidate.insert(min_pos, c);
                    } else {
                        feasible = false;
                        break;
                    }
                }
                if feasible {
                    let cost = route_distance(&candidate, dm);
                    let orig = route_distance(&routes[i], dm) + route_distance(&routes[j], dm);
                    let delta = cost as i64 - orig as i64;
                    if delta < best_cost {
                        best_cost = delta;
                        best_i = i;
                        best_j = j;
                        best_merged = Some(candidate);
                    }
                }
            }
        }

        match best_merged {
            Some(merged) => {
                routes[best_i] = merged;
                routes.remove(best_j);
            }
            None => break, // can't merge any more, we're stuck
        }
    }

    let solution = Solution { routes };
    save_solution(&solution)?;
    Ok(())
}

fn is_route_feasible(route: &[usize], ch: &Challenge) -> bool {
    let mut time = 0i32;
    let mut load = 0i32;
    for i in 1..route.len() {
        let prev = route[i - 1];
        let curr = route[i];
        if curr == 0 {
            if time + ch.distance_matrix[prev][0] > ch.due_times[0] { return false; }
            continue;
        }
        load += ch.demands[curr];
        if load > ch.max_capacity { return false; }
        time += ch.distance_matrix[prev][curr];
        if time > ch.due_times[curr] { return false; }
        if time < ch.ready_times[curr] { time = ch.ready_times[curr]; }
        time += ch.service_time;
    }
    true
}

fn route_distance(route: &[usize], dm: &[Vec<i32>]) -> i32 {
    let mut d = 0;
    for i in 1..route.len() {
        d += dm[route[i - 1]][route[i]];
    }
    d
}
