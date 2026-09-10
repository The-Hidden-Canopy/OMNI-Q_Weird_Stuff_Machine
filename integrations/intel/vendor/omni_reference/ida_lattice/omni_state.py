# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/omni_state.py`; body unmodified except as recorded in ../SOURCE.md.

"""Packet-causal reference for resident, morphable state-space experts.

Identity routing reads the current representation only. Recall and prediction
condition operations after identity selection. All state is caller-owned: a
forward returns a new snapshot and never mutates an input cache.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .constitutional_router import ConstitutionalRouter
from .local_workspace import LocalAttentionWorkspace
from .multiscale_memory import MultiscaleMemoryBank


@dataclass(frozen=True)
class OmniEvidenceBatch:
    stream_ids: tuple[str, ...]
    step_index: torch.Tensor
    prediction_boundary: torch.Tensor
    attention_mask: torch.Tensor
    correspondence: torch.Tensor | None = None

    def validate(self, batch: int, length: int) -> None:
        if len(self.stream_ids) != batch or len(set(self.stream_ids)) != batch:
            raise ValueError("each batch lane requires a distinct stream identity")
        if self.step_index.shape != (batch,) or self.prediction_boundary.shape != (batch,):
            raise ValueError("step_index and prediction_boundary must have shape [batch]")
        if self.attention_mask.shape != (batch, length):
            raise ValueError("attention mask does not match evidence batch")
        if bool(((self.prediction_boundary < 1) | (self.prediction_boundary > length)).any()):
            raise ValueError("prediction boundary must retain nonempty evidence")
        if bool((self.step_index < 0).any()):
            raise ValueError("negative evidence step")
        for b, end in enumerate(self.prediction_boundary.tolist()):
            if not bool(self.attention_mask[b, :end].bool().all()):
                raise ValueError("evidence prefix must be contiguous and unpadded")


@dataclass(frozen=True)
class OmniLayerState:
    recurrent: torch.Tensor                 # [batch, heads, head_dim, state_dim]
    anchors: torch.Tensor                   # [batch, anchors, hidden]
    anchor_steps: torch.Tensor              # [batch, anchors], -1 = unoccupied
    prior_pss_error: torch.Tensor            # [batch]
    committed_position: torch.Tensor        # [batch]
    stream_ids: tuple[str, ...]
    workspace: torch.Tensor | None = None    # transient language continuation only
    architecture: str = "morphable_state_v1"

    def detach(self) -> "OmniLayerState":
        return replace(self, **{
            key: value.detach() for key, value in vars(self).items()
            if isinstance(value, torch.Tensor)
        })

    def reorder(self, indices: torch.Tensor) -> "OmniLayerState":
        return replace(self, **{
            key: value.index_select(0, indices.to(value.device))
            for key, value in vars(self).items() if isinstance(value, torch.Tensor)
        }, stream_ids=tuple(self.stream_ids[i] for i in indices.tolist()))


@dataclass(frozen=True)
class ExpertOperationPlan:
    expert_ids: torch.Tensor                # [batch, k]
    mixture_weights: torch.Tensor           # [batch, k]
    modes: torch.Tensor                     # [batch, k, 3, 2], hard-forward ST
    delta_coefficients: torch.Tensor        # [batch, k, rank]
    contributions: torch.Tensor            # [batch, k, 3 sources, 6+rank]
    coupling_shape: tuple[int, int, int] | None = None  # groups, state width, rank
    factor_versions: tuple[tuple[int, int, int], ...] = ()
    read_write_offsets: tuple[tuple[int, int, int], ...] = ()

    @property
    def mode_bits(self) -> torch.Tensor:
        return self.modes.detach().argmax(-1)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weight).sum(1) / weight.sum(1).clamp_min(1)


def hard_binary(logits: torch.Tensor) -> torch.Tensor:
    soft = logits.float().softmax(-1).to(logits.dtype)
    hard = F.one_hot(soft.argmax(-1), 2).to(soft.dtype)
    return hard + (soft - soft.detach())


def candidate_scan(
    u: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    decay: torch.Tensor, dt: torch.Tensor, weights: torch.Tensor,
    initial: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """u [B,K,T,H,P], B/C [B,K,T,H,N]; exact sequential oracle.

    The native scan uses the same affine coefficients. Outputs are candidate
    readouts [B,K,T,H,P]; only their weighted combination is persisted.
    """
    # Preserve double precision for numerical verification; production state
    # accumulation is FP32 even when GEMMs use low-precision operands.
    acc = torch.float64 if initial.dtype == torch.float64 else torch.float32
    state = initial.to(acc)
    outputs = []
    for t in range(u.shape[2]):
        candidate = (
            decay[:, :, t].to(acc)[..., None, None] * state[:, None]
            + dt[:, :, t].to(acc)[..., None, None]
            * u[:, :, t].to(acc)[..., None] * b[:, :, t].to(acc)[..., None, :]
        )
        read = (candidate * c[:, :, t].to(acc)[..., None, :]).sum(-1)
        valid = mask[:, t].bool()
        outputs.append(read * valid[:, None, None, None])
        merged = (candidate * weights.to(acc)[..., None, None, None]).sum(1)
        state = torch.where(valid[:, None, None, None], merged, state)
    return torch.stack(outputs, dim=2), state


class StateExpert(nn.Module):
    """Resident A/B/C substrate, shared by every layer view of this identity."""
    def __init__(self, hidden: int, inner: int, heads: int, state_dim: int, groups: int):
        super().__init__()
        if inner % heads or heads % groups:
            raise ValueError("SSM inner width/head/group divisibility is required")
        self.heads, self.head_dim, self.state_dim, self.groups = heads, inner // heads, state_dim, groups
        self.input = nn.Linear(hidden, inner, bias=False)
        self.gate = nn.Linear(hidden, inner, bias=False)
        self.readout = nn.Linear(inner, hidden, bias=False)
        self.b = nn.Linear(hidden, 2 * groups * state_dim, bias=False)
        self.c = nn.Linear(hidden, 2 * groups * state_dim, bias=False)
        self.dt = nn.Linear(hidden, heads)
        self.a_log = nn.Parameter(torch.stack((torch.zeros(heads), torch.ones(heads))))
        self.skip = nn.Parameter(torch.ones(heads))

    def operands(self, x: torch.Tensor, modes: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, length, _ = x.shape
        u = self.input(x).reshape(batch, length, self.heads, self.head_dim)
        shape = (batch, length, 2, self.groups, self.state_dim)
        bv = (self.b(x).reshape(shape) * modes[:, None, 1, :, None, None]).sum(2)
        cv = (self.c(x).reshape(shape) * modes[:, None, 2, :, None, None]).sum(2)
        bv = bv.repeat_interleave(self.heads // self.groups, dim=2)
        cv = cv.repeat_interleave(self.heads // self.groups, dim=2)
        a = ((-self.a_log.float().exp()) * modes[:, 0, :, None]).sum(1)
        dt = F.softplus(self.dt(x).float()).clamp(1e-5, 100.0)
        decay = (dt * a[:, None]).exp()
        return u, bv, cv, dt, decay, self.gate(x)


class OperationControl(nn.Module):
    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.current = nn.Linear(hidden, 6 + rank)
        self.lrss = nn.Linear(hidden, 6 + rank, bias=False)
        self.pss = nn.Linear(hidden, 6 + rank, bias=False)
        self.error = nn.Linear(1, 6, bias=False)
        self.delta_down = nn.Linear(hidden, rank, bias=False)
        self.delta_up = nn.Linear(rank, hidden, bias=False)

    def forward(self, current, lrss, pss, error):
        sources = torch.stack((self.current(current), self.lrss(lrss), self.pss(pss)), dim=1)
        total = sources.sum(1)
        logits = total[:, :6] + self.error(error[:, None].to(current.dtype))
        modes = hard_binary(logits.reshape(-1, 3, 2))
        return modes, total[:, 6:], sources


class PacketPSS(nn.Module):
    """Forecast the next post-layer packet summary before its operation executes."""
    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.current = nn.Linear(hidden, rank, bias=False)
        self.anchor = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)

    def forward(self, current, anchor):
        return anchor + self.up(F.gelu(self.current(current) + self.anchor(anchor)))


class MorphableStateLayer(nn.Module):
    def __init__(self, config, substrate: nn.ModuleList):
        super().__init__()
        self._substrate_ref = weakref.ref(substrate)
        self.hidden = config.hidden_size
        self.top_k = config.omni_top_k
        self.anchor_capacity = config.multiscale_memory_num_anchors
        self.heads = config.omni_ssm_heads
        self.head_dim = config.omni_ssm_inner_size // self.heads
        self.state_dim = config.omni_ssm_state_size
        self.attention_window = config.local_attention_window
        self.scan_backend = config.omni_scan_backend
        self.operation_ablation = config.omni_operation_ablation
        self.architecture = config.omni_architecture
        self.coupled = self.architecture == "state_coupled_v1"
        self.coupling_groups = config.omni_ssm_groups
        self.coupling_rank = config.omni_coupling_rank
        self.pre_norm = nn.LayerNorm(self.hidden, eps=config.layer_norm_eps)
        self.post_norm = nn.LayerNorm(self.hidden, eps=config.layer_norm_eps)
        self.attention = LocalAttentionWorkspace(self.hidden, config.num_attention_heads, config.local_attention_window)
        self.pressure = nn.Linear(self.hidden, config.num_cognitive_routes)
        self.router = ConstitutionalRouter(self.hidden, len(substrate), self.top_k,
                                          pressure_size=config.num_cognitive_routes)
        if self.coupled:
            from .omni_coupling import CoupledOperationControl
            self.controls = nn.ModuleList(CoupledOperationControl(self.hidden,self.coupling_groups,
                self.state_dim,self.coupling_rank,config.omni_program_selection) for _ in substrate)
        else:
            self.controls = nn.ModuleList(OperationControl(self.hidden, config.omni_mode_rank) for _ in substrate)
        self.lrss = MultiscaleMemoryBank(self.hidden, config.multiscale_memory_num_scales,
                                       self.anchor_capacity, config.multiscale_memory_tau_min,
                                       config.multiscale_memory_tau_max, recall_only=True)
        self.pss = PacketPSS(self.hidden, config.state_supersampler_rank)

    @property
    def substrate(self):
        result = self._substrate_ref()
        if result is None:
            raise RuntimeError("resident expert owner no longer exists")
        return result

    def empty_state(self, x, stream_ids):
        b = x.size(0)
        return OmniLayerState(
            torch.zeros(b, self.heads, self.head_dim, self.state_dim, device=x.device, dtype=torch.float32),
            x.new_zeros(b, self.anchor_capacity, self.hidden),
            torch.full((b, self.anchor_capacity), -1, device=x.device, dtype=torch.long),
            x.new_zeros(b), torch.full((b,), -1, device=x.device, dtype=torch.long), tuple(stream_ids),
            architecture=self.architecture,
        )

    def forward(self, hidden, mask, state, step_index, *, commit=True, lrss_override=None, pss_override=None,
                pss_target=None, execution="packed"):
        if execution not in {"packed", "direct"}:
            raise ValueError("unknown reference dispatch")
        if state.architecture != self.architecture:
            raise ValueError("recurrent cache architecture mismatch")
        x = self.pre_norm(hidden)
        current = masked_mean(x, mask)
        # Selection deliberately precedes and does not consume either predictor.
        _, logits = self.router(current[:, None], torch.tanh(self.pressure(current)))
        ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :self.top_k]
        weights = logits.gather(1, ids).float().softmax(-1).to(x.dtype)
        recalled, previous = [], []
        for lane in range(x.size(0)):
            valid = state.anchor_steps[lane] >= 0
            previous.append(state.anchors[lane, -1])
            if bool(valid.any()):
                ages = (step_index[lane] - state.anchor_steps[lane, valid]).clamp_min(1)
                recalled.append(self.lrss(current[lane:lane+1], state.anchors[lane, valid], ages)["recall"])
            else:
                recalled.append(current[lane:lane+1] * 0)
        lrss = torch.cat(recalled) if lrss_override is None else lrss_override
        pss = self.pss(current, torch.stack(previous)) if pss_override is None else pss_override
        recall_control = torch.zeros_like(lrss) if self.operation_ablation == "no_lrss" else lrss
        prediction_control = torch.zeros_like(pss) if self.operation_ablation == "no_pss" else pss
        error_control = (torch.zeros_like(state.prior_pss_error)
                         if self.operation_ablation == "no_pss" else state.prior_pss_error)
        batch, length, _ = x.shape
        matrix_executions=[]
        def record_matrix(name, module, rows):
            matrix_executions.append(dict(name=name,rows=rows,
                forward_flops=2*rows*module.in_features*module.out_features))
        # Dispatch gathers only lanes assigned to an expert. Both B/C bases are
        # evaluated for the straight-through contrast in training, not eight banks.
        slot_operands = [None] * (batch * self.top_k)
        modes_all, gamma_all, sources_all = [None] * (batch*self.top_k), [None] * (batch*self.top_k), [None] * (batch*self.top_k)
        for expert_id in ids.unique(sorted=True).tolist():
            assignments = (ids == expert_id).nonzero()
            lanes, slots = assignments[:, 0], assignments[:, 1]
            control = self.controls[expert_id]
            for name in ("current","lrss","pss","error"):
                record_matrix(f"controls.{expert_id}.{name}",getattr(control,name),len(assignments))
            modes, gamma, sources = control(current[lanes], recall_control[lanes], prediction_control[lanes], error_control[lanes])
            if self.operation_ablation == "fixed":
                modes = torch.zeros_like(modes); modes[...,0] = 1
                gamma = torch.zeros_like(gamma)
            for j, (lane, slot) in enumerate(assignments.tolist()):
                at = lane * self.top_k + slot
                modes_all[at], gamma_all[at], sources_all[at] = modes[j], gamma[j], sources[j]
        groups = {}
        for lane in range(batch):
            for slot in range(self.top_k):
                at=lane*self.top_k+slot
                bits=tuple(modes_all[at].detach().argmax(-1).tolist())
                key=(int(ids[lane,slot]),*bits)
                if execution == "direct": key=(*key,lane,slot)
                groups.setdefault(key,[]).append((lane,slot))
        for key,assignments in sorted(groups.items()):
            lanes=torch.tensor([a[0] for a in assignments],device=x.device)
            modes=torch.stack([modes_all[lane*self.top_k+slot] for lane,slot in assignments])
            for name in ("input","b","c","dt","gate"):
                record_matrix(f"substrate.{key[0]}.{name}",getattr(self.substrate[key[0]],name),len(assignments)*length)
            operands=self.substrate[key[0]].operands(x[lanes],modes)
            for j,(lane,slot) in enumerate(assignments):
                slot_operands[lane*self.top_k+slot]=tuple(o[j] for o in operands)
        u, bv, cv, dt, decay, gates = [torch.stack([o[i] for o in slot_operands]).reshape(batch, self.top_k, length, *slot_operands[0][i].shape[1:]) for i in range(6)]
        scale = None
        if self.coupled:
            from .omni_coupling import coupled_scan
            order=ids.flatten().tolist()
            left=torch.stack([self.controls[e].state_u for e in order]).reshape(batch,self.top_k,
                self.coupling_groups,self.state_dim,self.coupling_rank)
            right=torch.stack([self.controls[e].state_v for e in order]).reshape_as(left)
            gamma=torch.stack(gamma_all).reshape(batch,self.top_k,self.coupling_groups,self.coupling_rank)
            repeats=self.heads//self.coupling_groups
            left,right,gamma=[v.repeat_interleave(repeats,dim=2) for v in (left,right,gamma)]
            mode=("native_cuda_packed" if self.scan_backend=="native_cuda_packed"
                  else "native_cuda" if self.scan_backend=="native_cuda"
                  else "candidate" if execution=="direct" else "packed")
            reads,next_recurrent,scale=coupled_scan(u,bv,cv,decay,dt,weights,state.recurrent,mask,
                                                   left,right,gamma,execution=mode)
        else:
            scan = candidate_scan
            if self.scan_backend == "native_cuda":
                from .omni_native_scan import native_candidate_scan
                scan = native_candidate_scan
            reads, next_recurrent = scan(u, bv, cv, decay, dt, weights, state.recurrent, mask)
        output = torch.zeros_like(x)
        for expert_id in ids.unique(sorted=True).tolist():
            assignments = (ids == expert_id).nonzero()
            lanes, slots = assignments[:, 0], assignments[:, 1]
            expert, control = self.substrate[expert_id], self.controls[expert_id]
            read = reads[lanes, slots].to(x.dtype) + u[lanes, slots] * expert.skip[None, None, :, None]
            read = read.flatten(-2) * F.silu(gates[lanes, slots])
            base = expert.readout(read)
            record_matrix(f"substrate.{expert_id}.readout",expert.readout,len(assignments)*length)
            gamma = torch.stack([gamma_all[lane*self.top_k+slot] for lane, slot in assignments.tolist()])
            delta = 0 if self.coupled else control.delta_up(control.delta_down(x[lanes]) * gamma[:, None])
            if not self.coupled:
                for name in ("delta_down","delta_up"):
                    record_matrix(f"controls.{expert_id}.{name}",getattr(control,name),len(assignments)*length)
            output = output.index_add(0, lanes, (base + delta) * weights[lanes, slots, None, None])
        workspace = x if state.workspace is None else torch.cat((state.workspace.to(x.dtype), x), dim=1)
        # Continuation caches contain only valid, observed prefixes (the model
        # processes individual batch lanes when evidence lengths differ).
        workspace_mask = torch.ones(workspace.shape[:2], device=x.device, dtype=mask.dtype)
        workspace_mask[:, -length:] = mask
        attended = self.attention(workspace, attention_mask=workspace_mask)[:, -length:]
        workspace = workspace[:, -self.attention_window:]
        result = self.post_norm(hidden + output + attended) * mask[..., None].to(x.dtype)
        actual = masked_mean(result, mask)
        score_target = actual if pss_target is None else pss_target
        if score_target.shape != actual.shape:
            raise ValueError("PSS score target must match the post-layer packet summary")
        error = (pss.float() - score_target.detach().float()).square().mean(-1)
        if commit:
            if bool((step_index <= state.committed_position).any()):
                raise ValueError("evidence position was already committed or moved backward")
            anchors = torch.cat((state.anchors[:, 1:], actual.detach()[:, None]), dim=1)
            anchor_steps = torch.cat((state.anchor_steps[:, 1:], step_index[:, None]), dim=1)
            next_state = replace(state, recurrent=next_recurrent, anchors=anchors,
                                 anchor_steps=anchor_steps, prior_pss_error=error.detach(),
                                 committed_position=step_index.clone(), workspace=workspace)
        else:
            next_state = replace(state, recurrent=next_recurrent, workspace=workspace)
        plan = ExpertOperationPlan(ids, weights, torch.stack(modes_all).reshape(batch,self.top_k,3,2),
                                   torch.stack(gamma_all).reshape(batch,self.top_k,-1),
                                   torch.stack(sources_all).reshape(batch,self.top_k,3,-1),
                                   coupling_shape=(self.coupling_groups,self.state_dim,self.coupling_rank) if self.coupled else None,
                                   factor_versions=tuple((e,self.controls[e].state_u._version,self.controls[e].state_v._version)
                                       for e in ids.unique(sorted=True).tolist()) if self.coupled else (),
                                   read_write_offsets=tuple((k,k*self.coupling_rank,self.top_k*self.coupling_rank+k)
                                       for k in range(self.top_k)) if self.coupled else ())
        return result, next_state, plan, {"pss_loss": error.mean(), "pss_prediction": pss,
                                         "lrss_recall": lrss, "pss_error": error,
                                         "assigned_rows": mask.new_tensor(batch * length * self.top_k, dtype=torch.long),
                                         "matrix_executions": matrix_executions,
                                         "matrix_execution_scope": "expert_substrate_and_operation_controls",
                                         "valid_assigned_rows": mask.sum() * self.top_k,
                                         "execution_groups": tuple((key,tuple(assignments)) for key,assignments in sorted(groups.items())),
                                         "substrate_applications": batch*self.top_k,
                                         "basis_contrast_count": 2,
                                         "scan_backend": self.scan_backend,
                                         "gemm_backend": "torch_reference",
                                         "prior_pss_error_control": error_control.detach(),
                                         "coupling_factor_shape": ((self.coupling_groups,self.state_dim,self.coupling_rank)
                                                                    if self.coupled else None),
                                         "coupling_norm_scale": (None if scale is None else scale.detach()),
                                         "operation_control_sources": ("current", "lrss", "pss"),
                                         "history_commit": bool(commit)}
