"""
Entrypoint for the embodied fly simulation.

Connects the PyTorch FlyWire LIF brain model to a NeuroMechFly v2 (flygym)
physical body, synchronising brain and body every 15 ms of simulated time.

Usage
-----
    # 5 seconds of simulated time, no visual rendering (default)
    python run_embodied.py

    # 60 seconds, with rendering (requires a display)
    python run_embodied.py --duration 60 --render

    # Force CPU if running on a MacBook without MPS/CUDA
    python run_embodied.py --duration 5 --device cpu

    # Save the per-tick log to a JSON file
    python run_embodied.py --duration 5 --log-file data/results/embodied_run.json
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))

from embodied_fly import run_embodied_fly


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Embodied fly: run the connectome LIF brain inside a NMF body.'
    )
    parser.add_argument(
        '--duration', type=float, default=5.0,
        help='Simulated duration in seconds (default: 5.0).',
    )
    parser.add_argument(
        '--render', action='store_true', default=False,
        help='Enable flygym visual rendering (requires a display).',
    )
    parser.add_argument(
        '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'],
        help='Torch device override (default: auto-detect).',
    )
    parser.add_argument(
        '--log-file', type=str, default=None,
        help='Optional path to save the per-tick JSON log.',
    )

    args = parser.parse_args()

    tick_log = run_embodied_fly(
        duration_sec=args.duration,
        render=args.render,
        device=args.device,
    )

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(tick_log, f, indent=2)
        print(f'\nLog saved to: {log_path}')


if __name__ == '__main__':
    main()
