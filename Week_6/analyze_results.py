#!/usr/bin/env python3
"""
analyze_results.py — Week 5, Checklist Phase 5 + Phase 6

Summarizes one or more results CSVs (produced by test_logger_node.py)
into the success-rate + path-deviation numbers your technical notes need.
Run it once on your baseline CSV, then again after tuning on the tuned
CSV, and put both outputs side by side in your before/after table.

Usage:
  python3 analyze_results.py week6_results_baseline.csv
  python3 analyze_results.py week6_results_baseline.csv week6_results_tuned.csv

UNCHANGED from the version you uploaded — no issue was found in this file
during review, so nothing here was rewritten. Included again only so you
have the full matching set of four files together. (Filename in the usage
example above updated to match test_logger_node.py's new default —
week6_results_baseline.csv, not week5_.)
"""

import csv
import sys


def summarize(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f'{path}: no runs logged')
        return

    total = len(rows)
    successes = sum(1 for r in rows if r['result'] == 'SUCCESS')
    max_devs = [float(r['max_deviation_m']) for r in rows]
    mean_devs = [float(r['mean_deviation_m']) for r in rows]

    print(f'\n=== {path} ===')
    print(f'Runs: {total}')
    print(f'Success rate: {successes}/{total} ({100 * successes / total:.0f}%)')
    print(f'Worst max deviation across all runs: {max(max_devs):.3f} m')
    print(f'Average of per-run mean deviation: {sum(mean_devs) / total:.3f} m')
    print('Per-run breakdown:')
    for r in rows:
        print(f"  {r['run_label']:32s} {r['result']:16s} "
              f"max_dev={r['max_deviation_m']:>7s}m mean_dev={r['mean_deviation_m']:>7s}m")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 analyze_results.py <results_csv> [more_results_csv ...]')
        sys.exit(1)
    for csv_path in sys.argv[1:]:
        summarize(csv_path)
