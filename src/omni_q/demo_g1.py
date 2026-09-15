"""Run the actual Unitree G1 TorchScript MuJoCo policy.

    $env:UNITREE_RL_GYM = "$PWD/external/unitree_rl_gym"
    python -m omni_q.demo_g1

This is an interactive viewer demo.  Headless provider/policy validation lives
in ``tests/test_humanoid_g1.py`` and does not open a window.
"""

from __future__ import annotations

import argparse
import time

from .humanoid_g1 import G1Provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the actual Unitree G1 MuJoCo policy")
    parser.add_argument("--unitree-root", default=None,
                        help="unitree_rl_gym checkout; defaults to UNITREE_RL_GYM")
    parser.add_argument("--duration", type=float, default=12.0,
                        help="viewer duration in seconds (default: 12)")
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("--duration must be positive")

    # Keep the viewer dependency out of imports for headless OMNI-Q users.
    import mujoco.viewer

    g1 = G1Provider(unitree_root=args.unitree_root)
    with mujoco.viewer.launch_passive(g1.model, g1.data) as viewer:
        wall_start = time.perf_counter()
        while viewer.is_running():
            elapsed = time.perf_counter() - wall_start
            if elapsed >= args.duration:
                break

            # Stand, walk, stop, then turn toward the camera.
            if elapsed < 2.0:
                g1.stop()
            elif elapsed < min(7.0, args.duration):
                g1.march(0.5)
            elif elapsed < min(9.0, args.duration):
                g1.stop()
            elif elapsed < min(10.5, args.duration):
                g1.turn(0.45)
            else:
                g1.stop()

            before = time.perf_counter()
            g1.step()
            viewer.sync()
            remaining = g1.dt - (time.perf_counter() - before)
            if remaining > 0:
                time.sleep(remaining)

    g1.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
