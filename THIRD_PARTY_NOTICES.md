# Third-party notices

This file records third-party software and research repositories reviewed for
OMNI-Q. It distinguishes material that is actually distributed in this tree
from sources that were consulted only as reference. That distinction is
intentional: a repository's public visibility is not treated as permission to
copy its code or assets.

Review date: 2026-09-14

## Material incorporated in this repository

### Unitree Robotics `unitree_rl_gym` G1 deployment seam

- Source: [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- Pinned source commit: `276801e46c5d433564f24658bac64f254b7d2d4b`
- License: BSD 3-Clause License; copyright (c) 2016-2023 HangZhou YuShu
  TECHNOLOGY CO., LTD. (Unitree Robotics)
- Local source/weights checkout: `external/unitree_rl_gym/` (gitignored and
  not redistributed by this repository)
- Adapted implementation: [`src/omni_q/humanoid_g1.py`](src/omni_q/humanoid_g1.py)
- License text: [`third_party/licenses/unitree_rl_gym-BSD-3-Clause.txt`](third_party/licenses/unitree_rl_gym-BSD-3-Clause.txt)

The G1 provider adapts the official MuJoCo observation, policy-update, and PD
control loop.  It does not copy the vendor checkout into the tracked tree or
grant the policy authority over OMNI-Q objectives, world state, or replanning.

### MuJoCo Menagerie `trs_so_arm100`

- Source: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100)
- Pinned source commit: `8161bba264d7fa7c99ca301e91e7fb44737676ad`
- License: Apache License 2.0
- Distributed copy: [`integrations/intel/assets/menagerie_so_arm100/`](integrations/intel/assets/menagerie_so_arm100/)
- License text: [`integrations/intel/assets/menagerie_so_arm100/LICENSE`](integrations/intel/assets/menagerie_so_arm100/LICENSE)
- Provenance record: [`integrations/intel/assets/menagerie_so_arm100/SOURCE.md`](integrations/intel/assets/menagerie_so_arm100/SOURCE.md)

The vendored asset remains separately identified and retains its upstream
license. OMNI-Q's root MIT license does not replace that Apache-2.0 license.

## Optional runtime dependencies (not vendored)

### Hugging Face LeRobot

- Source: [huggingface/lerobot](https://github.com/huggingface/lerobot)
- License: Apache License 2.0
- OMNI-Q treatment: optional dependency for the SmolVLA controller and the
  LeRobot demonstration recorder; no LeRobot source code or checkpoint is
  copied into this repository.
- Integration points:
  [`src/omni_q/skills/controllers/smolvla.py`](src/omni_q/skills/controllers/smolvla.py)
  and [`scripts/record_smolvla_dataset.py`](scripts/record_smolvla_dataset.py)

## Reference-only sources

The following repositories were inspected for controller, simulator, training,
or architecture ideas. No source code, model weights, meshes, or other
copyrightable assets from them are currently copied or vendored in OMNI-Q.
They are therefore recorded for provenance, not asserted as dependencies or
contributors to the distributed implementation.

| Repository | License observed upstream | Current OMNI-Q treatment |
| --- | --- | --- |
| [ggand0/pick-101](https://github.com/ggand0/pick-101) | MIT; `Copyright (c) 2025 Gota Gando` | Reference only: IK, contact geometry, SAC/DrQ patterns, and evaluation ideas. |
| [johnsutor/so101-nexus](https://github.com/johnsutor/so101-nexus) | Apache-2.0 | Reference only: parallel simulation/training workflow. |
| [lukehollis/lerobot-lewam](https://github.com/lukehollis/lerobot-lewam) | Apache-2.0; `Copyright 2026 LeRobot-WAM contributors` | Reference only: World Action Model design and candidate-action evaluation. Its vendored Menagerie asset is not copied here. |
| [MarkRedeman/physicalai-plugins](https://github.com/MarkRedeman/physicalai-plugins) | Apache-2.0 | Reference only: Physical AI plugin integration patterns. |
| [johannstark/so-101-mujoco](https://github.com/johannstark/so-101-mujoco) | Apache-2.0 | Reference only: SO-101 MuJoCo/Warp setup patterns. |
| [TourEnduro/SO-101-robot-arm](https://github.com/TourEnduro/SO-101-robot-arm) | No license located in the repository | Reference only. Do not copy code, assets, or documentation from this repository without permission or a newly verified license. |

The license texts for the two permissive families reviewed here are retained
under [`third_party/licenses/`](third_party/licenses/) as compliance references.

## Rules for future ingestion

Before copying code, assets, weights, or substantial documentation from a
reference-only repository:

1. Re-verify the exact source commit and license.
2. Move the source into the incorporated-material section of this file.
3. Preserve the upstream copyright, attribution, license, and NOTICE material.
4. Mark modified files when Apache-2.0 requires it.
5. Record the copied paths and source commit in a per-vendor `SOURCE.md`.
6. Keep unlicensed material out unless the copyright holder grants permission.

This ledger is an attribution and provenance record, not legal advice. The
OMNI-Q implementation remains under its root MIT license except for separately
licensed incorporated material identified above.
