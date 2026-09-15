# VLA-first motion for the table task (2026-09-15)

Built after the hosts' note that a VLA must be the dominant robot-control
policy. Additive to the submission stack; nothing in `src/omni_q/intel_sim.py`
is modified. See `docs/submission-readiness-2026-09-15.md` for status.

```text
record_expert_demos.py   the governed expert runs; per arm, at 10 Hz: 3 cameras (overhead, front, wrist),
                         9-d state, 7-d Cartesian-delta action (absolute jaw by default), per-step language
                         -> LeRobot v3 dataset
train_smolvla.sh         lerobot-train, lerobot/smolvla_base fine-tune (action expert), RTX 4050
vla_world.py             offline evaluation adapter: the proposal-only SmolVLA controller drives every
                         single-arm PICK / MOVE; the world's pad-pose IK only realises each Cartesian delta;
                         the world's own verifiers decide; governed primitive = counted fallback
run_vla_eval.py          seeds -> per-op VLA-vs-fallback counts, task outcome, end-of-run physical check; --record clips
```

Reproduce (or skip the first two steps: `bash scripts/fetch_checkpoints.sh`
downloads the fine-tune used for the submission clips):

```bash
.venv/Scripts/python -m pip install -e ".[intel,smolvla]"
.venv/Scripts/python integrations/intel/vla/record_expert_demos.py --seeds 900 901 902 903 904 905 906 907 909 910 911 912 913 914 915 916 917 918 919 920 --root datasets/so101_table_vla_absjaw --repo-id omni-q/so101_table_vla_absjaw --jaw-mode absolute
bash integrations/intel/vla/train_smolvla.sh 3000 4
OMNIQ_VLA_CHECKPOINT=outputs/train/<run>/checkpoints/last/pretrained_model \
  .venv/Scripts/python integrations/intel/vla/run_vla_eval.py --seeds 900 901 902 903 904 --record
```

The plate (two-arm rim pinch) stays on the governed bimanual primitive: the
policy is single-arm. Every receipt carries `vla_attempt` on each PICK/MOVE
and `fallback: governed primitive` when the policy did not finish the step.

Evidence boundary: `VLAWorld` is a simulation/evaluation adapter, not the
production `SkillRuntime`. Its measurements do not promote the checkpoint to
`ACTIVE`; production use still requires the normal artifact digest, manifest,
promotion evidence, and supervisor/actuator path.
