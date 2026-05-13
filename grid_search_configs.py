"""
Grid search over dataset/ensemble/two-stage/boost-weak combinations.
Runs solution.py with parallel workers and reports results sorted by mAP.
"""

import itertools
import re
import subprocess
import sys
import time

PYTHON = "uv"
SCRIPT = "run"
LOG_FILE = "grid_search_results.txt"

DATASETS = ["knmi", "openmeteo", "all"]
ENSEMBLE = [False, True]
TWO_STAGE = [False, True]
BOOST_WEAK = [0, 3]

CLASS_NAMES = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds",
]


def build_cmd(dataset, ensemble, two_stage, boost_weak):
    cmd = [PYTHON, SCRIPT, "solution.py", "--dataset", dataset]
    if ensemble:
        cmd.append("--ensemble")
    if two_stage:
        cmd.append("--two-stage")
    if boost_weak > 0:
        cmd.extend(["--boost-weak", str(boost_weak)])
    return cmd


def parse_output(stdout):
    result = {"mAP": None, "per_class": {}}

    map_match = re.search(r"OOF Macro-Averaged AP \(mAP\):\s+([\d.]+)", stdout)
    if map_match:
        result["mAP"] = float(map_match.group(1))

    for cls in CLASS_NAMES:
        pattern = rf"{re.escape(cls)}\s*:\s+([\d.]+)"
        match = re.search(pattern, stdout)
        if match:
            result["per_class"][cls] = float(match.group(1))

    return result


def config_label(dataset, ensemble, two_stage, boost_weak):
    parts = [dataset]
    if ensemble:
        parts.append("ens")
    if two_stage:
        parts.append("2stg")
    if boost_weak > 0:
        parts.append(f"bw{boost_weak}")
    return "+".join(parts)


def run_config(args):
    dataset, ensemble, two_stage, boost_weak = args
    label = config_label(dataset, ensemble, two_stage, boost_weak)
    cmd = build_cmd(dataset, ensemble, two_stage, boost_weak)

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            stderr_tail = proc.stderr.strip().split("\n")[-3:]
            return {
                "label": label, "mAP": -1, "per_class": {},
                "time": elapsed, "error": True,
                "error_msg": "\n".join(stderr_tail),
            }

        parsed = parse_output(proc.stdout)
        parsed["label"] = label
        parsed["time"] = elapsed
        parsed["error"] = False
        return parsed

    except subprocess.TimeoutExpired:
        return {
            "label": label, "mAP": -1, "per_class": {},
            "time": time.time() - t0, "error": True,
            "error_msg": "TIMEOUT",
        }


def main():
    configs = list(itertools.product(DATASETS, ENSEMBLE, TWO_STAGE, BOOST_WEAK))
    total = len(configs)

    print(f"Running {total} configurations sequentially...\n", flush=True)

    t_start = time.time()
    results = []
    for i, cfg in enumerate(configs):
        r = run_config(cfg)
        results.append(r)
        status = f"mAP: {r['mAP']:.4f}" if not r["error"] else "FAILED"
        print(f"  [{i+1:2d}/{total}] {r['label']:<35} {status} ({r['time']:.0f}s)", flush=True)

    total_time = time.time() - t_start
    print(f"\nAll done in {total_time:.0f}s")

    # Sort by mAP descending
    results.sort(key=lambda r: r.get("mAP", -1), reverse=True)

    # Build report
    lines = []
    lines.append("=" * 120)
    lines.append("GRID SEARCH RESULTS — sorted by mAP (descending)")
    lines.append("=" * 120)
    lines.append("")

    header = f"{'Rank':<5} {'Config':<35} {'mAP':<8} {'Time':<7} " + " ".join(f"{c[:8]:>8}" for c in CLASS_NAMES)
    lines.append(header)
    lines.append("-" * len(header))

    for rank, r in enumerate(results, 1):
        if r["error"]:
            lines.append(f"{rank:<5} {r['label']:<35} {'FAILED':<8} {r['time']:<7.0f}")
            continue
        per_class_str = " ".join(
            f"{r['per_class'].get(c, 0):<8.4f}" for c in CLASS_NAMES
        )
        lines.append(f"{rank:<5} {r['label']:<35} {r['mAP']:<8.4f} {r['time']:<7.0f} {per_class_str}")

    lines.append("")
    lines.append(f"Best: {results[0]['label']} — mAP {results[0]['mAP']:.4f}")
    lines.append(f"Total time: {total_time:.0f}s")

    report = "\n".join(lines)
    print(f"\n{report}")

    with open(LOG_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {LOG_FILE}")


if __name__ == "__main__":
    main()
