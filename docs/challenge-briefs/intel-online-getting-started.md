# Intel Online Challenge — host "Getting Started" pointers

> Companion to
> [`intel-online-physical-ai-challenge.md`](intel-online-physical-ai-challenge.md)
> (the scored rubric/deliverables brief). This page is reference links the
> hosts posted separately, plus where each one lands in this repo.

The hosts are explicit: **participants build the MuJoCo scene from scratch** —
there is no provided stock scene/asset pack.

| Host pointer | Link | Status in this repo |
| --- | --- | --- |
| Install MuJoCo | [mujoco.readthedocs.io](https://mujoco.readthedocs.io/en/stable/python.html) | done — `mujoco` in the `intel` extra ([`pyproject.toml`](../../pyproject.toml)) |
| SO-101 assets | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | done — vendored via MuJoCo Menagerie's `trs_so_arm100` mirror of this exact source, Apache-2.0, see [`integrations/intel/assets/menagerie_so_arm100/SOURCE.md`](../../integrations/intel/assets/menagerie_so_arm100/SOURCE.md) |
| Free 3D asset sources (Sketchfab / CGTrader / Free3D) | — | not needed yet — tableware in the scene is procedural MuJoCo primitives (boxes/cylinders in `dual_so101_xml()`, [`src/omni_q/intel_sim.py`](../../src/omni_q/intel_sim.py)), not downloaded meshes. If we ever pull a mesh from one of these, re-confirm its license before vendoring — the hosts are explicit that Intel takes no responsibility for licensing here. |
| LeRobot — data format / training | [huggingface/lerobot](https://github.com/huggingface/lerobot) | done — `lerobot` in the `intel` extra; not yet wired to record demonstrations from the MuJoCo scene (open TODO in [`integrations/intel/README.md`](../../integrations/intel/README.md)) |
| LeRobot inference on Intel — PyTorch XPU | [torch_accelerators docs](https://huggingface.co/docs/lerobot/en/torch_accelerators) | documented, not installed — see below |
| Physical AI Studio | [open-edge-platform/physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio) | evaluated, not adopted yet — see below |
| OpenVINO | [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) | done — `openvino` + `optimum-intel[openvino]` + `nncf` in the `intel` extra |

## PyTorch XPU (Intel GPU) backend

LeRobot can run training/inference on `cuda`, `mps`, `xpu`, or plain `cpu` —
selected via `--policy.device` on `lerobot-train`/`lerobot-eval`, or
auto-detected. XPU targets Intel integrated *and* discrete GPUs directly
through PyTorch (a different path from exporting to OpenVINO IR).

```
pip install torch --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; print(torch.xpu.is_available())"
```

**Not installed in the shared `.venv` right now.** This dev machine's torch is
pinned to the CUDA build (`2.11.0+cu126`) so local policy training uses the
RTX 4050 GPU — see [`integrations/intel/README.md`](../../integrations/intel/README.md).
A single `torch` install can only carry one accelerator backend at a time, and
the XPU wheel index tops out around `2.9.x` (older than the CUDA build here),
so swapping would both downgrade torch and lose local GPU training. The
Intel-hardware validation this machine *can* do today is through OpenVINO
directly (`Core().available_devices` already sees `CPU`, Intel `GPU.0`
iGPU, and the NVIDIA `GPU.1`) — that already covers the brief's actual
requirement (OpenVINO inference on Core Ultra Series 2/3). Revisit XPU-backend
`lerobot-eval` specifically if/when we're validating directly against real
Core Ultra hardware and want a non-OpenVINO comparison point.

## Physical AI Studio

`pip install physicalai-train` is a real, installable package (`0.1.0`) — an
imitation-learning framework (LeRobot policy zoo + datasets, PushT/LIBERO/
RoboCasa-style benchmarks, `policy.export(backend="openvino")`).

**Evaluated, not installed.** A dry-run showed it pulls a large, opinionated
stack (Lightning, ONNX/onnxruntime, transformers, opencv, gym-pusht, pygame,
shapely, …) and — more importantly — wants `openvino~=2026.1`, while this repo
pins `openvino>=2026.3` in the `intel` extra; installing it here would
silently downgrade OpenVINO for everyone using the shared extra. We already
have a working, lighter train→export path (`lerobot` + `optimum-intel[openvino]`
+ `nncf`) that covers the same "train and export to OpenVINO" job. Worth a
second look if that path hits a wall LeRobot/optimum-intel can't clear, but
adopt deliberately (probably in its own venv or Docker, matching its own
documented install options) rather than layering it into the shared one.
