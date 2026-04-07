#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import csv
import glob
import subprocess
import argparse
import time
import json
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger("tig")


def setup_logging(log_level: str) -> None:
    level = getattr(logging, (log_level or "").upper(), None)
    if not isinstance(level, int):
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def require_cargo() -> None:
    try:
        result = subprocess.run(["cargo", "version"], check=True, capture_output=True, text=True)
        logger.debug("Cargo detected: %s", (result.stdout or result.stderr or "").strip())
    except Exception:
        logger.error("Cargo is not installed or not on PATH.")
        logger.error("Install Cargo: https://doc.rust-lang.org/cargo/getting-started/installation.html")
        raise SystemExit(1)


def generate_dataset(challenge: str, config_path: str, out_dir: str):
    if not os.path.exists(f"{ROOT_DIR}/target/release/tig_generator"):
        logger.info("Building `tig_generator` (release)")
        subprocess.run(
            ["cargo", "build", "-r", "--bin", "tig_generator", "--features", "generator"],
            check=True,
            cwd=ROOT_DIR,
        )
    with open(config_path, "r") as f:
        config = json.load(f)
    seed = config.pop("seed")
    if out_dir is None:
        out_dir = os.path.join("datasets", challenge, seed)
    for track_id, n in config.items():
        start = time.time()
        logger.info("Generating %s/%s instances (seed=%s, n=%s)", challenge, track_id, seed, n)
        subprocess.run([
            f"{ROOT_DIR}/target/release/tig_generator",
            challenge,
            track_id,
            "--seed", seed,
            "-n", str(n),
            "-o", os.path.join(out_dir, track_id),
        ], check=True)
        logger.info("Generated in %.2fs", time.time() - start)

def run_algorithm_on_instance(
    challenge: str,
    dataset_dir: str,
    instance_file: str,
    hyperparameters: str = None,
    timeout: int = 60,
    interval: int = None,
    out_dir: str = None,
) -> tuple:
    try:
        time_taken, memory = None, None
        if out_dir:
            os.makedirs(
                os.path.join(out_dir, os.path.dirname(instance_file)),
                exist_ok=True
            )
            solution_path = os.path.join(out_dir, f"{instance_file}.solution")
        else:
            solution_path = os.path.join(dataset_dir, f"{instance_file}.solution")
        logger.info("Solving %s", instance_file)

        cmd = [
            "/usr/bin/time",
            "-f", "Time: %e Memory: %M",
            "timeout", str(timeout),
            "target/release/tig_solver",
            challenge,
            os.path.join(dataset_dir, instance_file),
            solution_path,
        ]
        if hyperparameters:
            cmd += ["--hyperparameters", hyperparameters]
        logger.debug("Solver command: %s", " ".join(cmd))

        snapshot_count = 0
        if interval:
            next_snapshot_at = time.time() + interval
        else:
            next_snapshot_at = None

        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        stderr = None
        while True:
            now = time.time()
            if next_snapshot_at is not None and now >= next_snapshot_at:
                snapshot_count += 1
                snapshot_path = f"{solution_path}.{snapshot_count * interval}"
                if os.path.exists(solution_path):
                    shutil.copy2(solution_path, snapshot_path)
                    logger.debug("Snapshot %s -> %s", solution_path, snapshot_path)
                else:
                    logger.debug("Snapshot skipped; solution does not exist yet: %s", solution_path)
                next_snapshot_at += interval

            try:
                _, stderr = proc.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
        
        for line in (stderr or "").strip().split("\n"):
            if line.startswith("Time:"):
                parts = line.split(" ")
                time_taken = float(parts[1].strip())
                memory = int(parts[3].strip())

        logger.info(
            "Solved %s | time=%.3fs memory=%sKB",
            instance_file,
            time_taken or -1,
            memory,
        )
        return instance_file, time_taken, memory
    except Exception as e:
        logger.exception("Unexpected error: %s (%s)", instance_file, e)
        return instance_file, None, None

def run_algorithm(
    challenge: str,
    dataset_dir: str,
    num_workers: int = 1,
    hyperparameters: str = None,
    timeout: int = 60,
    interval: int = None,
    out_dir: str = None,
    baseline: bool = False,
    csv_path: str = None,
) -> list:
    if baseline:
        logger.info("Building `tig_solver` (release, features=solver,baseline,%s)", challenge)
        subprocess.run(
            ["cargo", "build", "-r", "--bin", "tig_solver", "--features", f"solver,baseline,{challenge}"],
            check=True,
            cwd=ROOT_DIR,
        )
    else:
        logger.info("Building `tig_solver` (release, features=solver,%s)", challenge)
        subprocess.run(
            ["cargo", "build", "-r", "--bin", "tig_solver", "--features", f"solver,{challenge}"],
            check=True,
            cwd=ROOT_DIR,
        )
    
    pool = ThreadPoolExecutor(max_workers=num_workers)

    instances = [
        os.path.relpath(path, dataset_dir)
        for path in glob.glob(f"{dataset_dir}/**/*.txt", recursive=True)
    ]
    logger.info(
        "Running %s instances (challenge=%s workers=%s timeout=%ss interval=%s out=%s)",
        len(instances),
        challenge,
        num_workers,
        timeout,
        interval,
        out_dir,
    )
    logger.debug("Dataset dir: %s", dataset_dir)
    
    results = list(pool.map(
        lambda instance: run_algorithm_on_instance(challenge, dataset_dir, instance, hyperparameters, timeout, interval, out_dir),
        instances
    ))

    if csv_path:
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_file", "time_taken", "memory"])
            for result in results:
                writer.writerow(result)

    return results

def evaluate_solution(
    challenge: str,
    dataset_dir: str,
    solutions_dir: str,
    instance_file: str,
    snapshots: bool = False,
) -> tuple:
    logger.info("Evaluating solutions for %s", instance_file)
    if solutions_dir is None:
        solutions_dir = dataset_dir
    if snapshots:
        solutions = [
            os.path.relpath(path, solutions_dir)
            for path in glob.glob(f"{solutions_dir}/{instance_file}.solution.*")
        ]
    else:
        solutions = [
            os.path.relpath(path, solutions_dir)
            for path in glob.glob(f"{solutions_dir}/{instance_file}.solution")
        ]
    if len(solutions) == 0:
        logger.info("No solutions found for %s", instance_file)
        return [(f"{instance_file}.solution", "not_found")]
    results = []
    for s in solutions:
        solution_file = os.path.join(solutions_dir, s)
        cmd = [
            f"{ROOT_DIR}/target/release/tig_evaluator",
            challenge,
            os.path.join(dataset_dir, instance_file),
            solution_file,
        ]
        logger.debug("Evaluator command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug(
                "Evaluated %s | output=error exit_code=%s stderr=%s",
                result.returncode,
                solution_file,
                (result.stderr or "").strip(),
            )
            output = "error"
        else:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Output:"):
                    output = line.split(":")[1].strip()
                    break
            else:
                output = "unknown"
            logger.info("Evaluated %s | output=%s", solution_file, output)
        results.append((s, output))
    return results

def evaluate_solutions(
    challenge: str,
    dataset_dir: str,
    solutions_dir: str = None,
    snapshots: bool = False,
    num_workers: int = 1,
    csv_path: str = None,
) -> list:
    if not os.path.exists("target/release/tig_evaluator"):
        logger.info("Building `tig_evaluator` (release)")
        subprocess.run(
            ["cargo", "build", "-r", "--bin", "tig_evaluator", "--features", "evaluator"],
            check=True,
            cwd=ROOT_DIR,
        )
    instances = [
        os.path.relpath(path, dataset_dir)
        for path in glob.glob(f"{dataset_dir}/**/*.txt", recursive=True)
    ]
    logger.info(
        "Evaluating solutions for %s instances (challenge=%s workers=%s csv=%s)",
        len(instances),
        challenge,
        num_workers,
        csv_path,
    )
    
    pool = ThreadPoolExecutor(max_workers=num_workers)
    results = [
        x 
        for l in pool.map(
            lambda instance: evaluate_solution(challenge, dataset_dir, solutions_dir, instance, snapshots),
            instances
        ) 
        for x in l
    ]
    
    if csv_path:
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["solution_file", "output"])
            for result in results:
                writer.writerow(result)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Challenges CLI Tool")    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Logging level")
    
    # generate_dataset subcommand
    generate_parser = subparsers.add_parser("generate_dataset", help="Generate datasets")
    generate_parser.add_argument("challenge", choices=["knapsack", "vehicle_routing", "job_scheduling"], help="Challenge name")
    generate_parser.add_argument("config", help="Dataset config file path")
    generate_parser.add_argument("--out", default=None, help="Output directory for dataset (defaults to datasets/<challenge>)")
    # run_algorithm subcommand
    run_parser = subparsers.add_parser("run_algorithm", help="Run the algorithm on datasets")
    run_parser.add_argument("challenge", choices=["knapsack", "vehicle_routing", "job_scheduling"], help="Challenge name")
    run_parser.add_argument("dataset_dir", help="Dataset directory (recursively searches for .txt files)")
    run_parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    run_parser.add_argument("--hyperparameters", help="Hyperparameters string")
    run_parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    run_parser.add_argument("--interval", type=int, default=None, help="Interval (seconds) to snapshot the latest solution")
    run_parser.add_argument("--out", default=None, help="Output directory for saving solutions (defaults to dataset directory)")
    run_parser.add_argument("--baseline", action="store_true", help="Run the baseline algorithm")
    run_parser.add_argument("--csv", default=None, help="CSV file path for saving results")
    
    # evaluate_solutions subcommand
    evaluate_parser = subparsers.add_parser("evaluate_solutions", help="Evaluate solutions")
    evaluate_parser.add_argument("challenge", choices=["knapsack", "vehicle_routing", "job_scheduling"], help="Challenge name")
    evaluate_parser.add_argument("dataset_dir", help="Dataset directory (recursively searches for .txt files)")
    evaluate_parser.add_argument("--solutions", default=None, help="Solutions directory (defaults to dataset directory. Search for <instance>.solution* files.)")
    evaluate_parser.add_argument("--snapshots", action="store_true", help="Evaluate snapshots of the solution")
    evaluate_parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    evaluate_parser.add_argument("--csv", default=None, help="CSV file path for saving results")

    args = parser.parse_args()
    setup_logging(args.log_level)
    require_cargo()
    
    try:
        if args.command == "generate_dataset":
            logger.info("\n\tcommand=generate_dataset\n\tchallenge=%s\n\tconfig=%s\n\tout=%s", args.challenge, args.config, args.out)
            generate_dataset(args.challenge, args.config, args.out)
        elif args.command == "run_algorithm":
            logger.info("\n\tcommand=run_algorithm\n\tchallenge=%s\n\tdataset_dir=%s\n\tworkers=%s\n\thyperparameters=%s\n\ttimeout=%s\n\tinterval=%s\n\tout=%s\n\tbaseline=%s\n\tcsv=%s", args.challenge, args.dataset_dir, args.workers, args.hyperparameters, args.timeout, args.interval, args.out, args.baseline, args.csv)
            run_algorithm(
                args.challenge,
                args.dataset_dir,
                args.workers,
                args.hyperparameters,
                args.timeout,
                args.interval,
                args.out,
                args.baseline,
                args.csv,
            )
        elif args.command == "evaluate_solutions":
            logger.info("\n\tcommand=evaluate_solutions\n\tchallenge=%s\n\tdataset_dir=%s\n\tsolutions=%s\n\tworkers=%s\n\tcsv=%s", args.challenge, args.dataset_dir, args.solutions, args.workers, args.csv)
            evaluate_solutions(
                args.challenge,
                args.dataset_dir,
                args.solutions,
                args.snapshots,
                args.workers,
                args.csv,
            )
        else:
            parser.print_help()
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        raise SystemExit(1)