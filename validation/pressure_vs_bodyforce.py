"""Reproduce pressure channel and porous padding cases with saved diagnostics.

A finite pressure-driven sample and a periodic body-force sample are distinct
boundary-value problems; their difference is not a generic solver-error metric.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.run_cases import cases,run
if __name__=='__main__':
    for case in cases():
        if case['id'].startswith('pressure_') and case['geometry']['kind']!='npy':
            run(case,Path('results/pressure-validation'),60)
