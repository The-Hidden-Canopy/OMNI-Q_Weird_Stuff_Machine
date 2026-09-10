# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/omni_state_model.py`; body unmodified except as recorded in ../SOURCE.md.

"""Trainable Omni model with evidence-boundary state prediction."""
from __future__ import annotations

from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .config import IDALatticeConfig
from .omni import OmniEvidenceBridge, OmniStateHead, compute_omni_losses
from .omni_state import OmniEvidenceBatch, OmniLayerState, StateExpert, MorphableStateLayer, masked_mean


# Runtime semantics are part of checkpoint identity even when tensor shapes stay unchanged.
OMNI_EXECUTION_REVISION = "availability_horizon_complete_execution_v2"


def forecast_horizon_encoding(target_time, evidence_time, hidden, reference):
    """Parameter-free readout query; a forecast request does not rewrite evidence history."""
    if target_time is None:
        return reference.new_zeros(reference.size(0), hidden)
    if evidence_time is None or target_time.shape != (reference.size(0),) or evidence_time.shape != target_time.shape:
        raise ValueError("forecast target and evidence time must be [batch]")
    if not bool(torch.isfinite(target_time).all() & torch.isfinite(evidence_time).all()) or bool((target_time < evidence_time).any()):
        raise ValueError("forecast target must be finite and not precede evidence")
    horizon = (target_time.to(device=reference.device, dtype=torch.float64)
               - evidence_time.to(device=reference.device, dtype=torch.float64)).log1p()
    frequencies = torch.exp(-torch.arange((hidden+1)//2, device=reference.device, dtype=torch.float64)
                            * (9.210340371976184 / max(1, (hidden+1)//2)))
    angle = horizon[:, None] * frequencies
    return torch.stack((angle.sin(), angle.cos()-1), -1).flatten(1)[:, :hidden].to(reference.dtype)


class OmniPatchProjection(nn.Module):
    """First-party RGB/IR patches with frame time and spatial coordinates."""
    def __init__(self, hidden: int, patch: int):
        super().__init__()
        self.patch = patch
        self.rgb = nn.Linear(3 * patch * patch, hidden, bias=False)
        self.ir = nn.Linear(patch * patch, hidden, bias=False)
        self.coordinates = nn.Linear(3, hidden, bias=False)

    def forward(self, frames: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.size(2) not in (1, 3):
            raise ValueError("frames must be [batch,frames,1|3,height,width]")
        b, f, c, height, width = frames.shape
        if height % self.patch or width % self.patch or times.shape != (b, f):
            raise ValueError("frame size must divide patch size and each frame needs a timestamp")
        if not bool(torch.isfinite(frames).all()) or not bool(torch.isfinite(times).all()):
            raise ValueError("non-finite media observation")
        projection = self.ir if c == 1 else self.rgb
        pixels = frames.to(projection.weight.dtype).reshape(b*f,c,height,width)
        patches = F.unfold(pixels, self.patch, stride=self.patch).transpose(1,2)
        rows, cols = height // self.patch, width // self.patch
        yy, xx = torch.meshgrid(torch.arange(rows, device=frames.device), torch.arange(cols, device=frames.device), indexing="ij")
        coords = torch.stack((xx.flatten()/max(1,cols), yy.flatten()/max(1,rows)), -1)
        coords = coords[None,None].expand(b,f,-1,-1)
        stamp = times[...,None,None].expand(b,f,rows*cols,1)
        position = self.coordinates(torch.cat((stamp,coords),-1).to(projection.weight.dtype))
        return (projection(patches).reshape(b,f,rows*cols,-1) + position).flatten(1,2)


def correspondence_loss(slots, pairs, valid=None):
    """0=unlabelled, 1=matching observation, -1=contradiction; upper triangle."""
    if slots is None or pairs is None:
        return None
    if pairs.shape != slots.shape[:2] + (slots.size(1),):
        raise ValueError("correspondence must be [batch,slots,slots]")
    if bool(((pairs != 0) & (pairs != 1) & (pairs != -1)).any()):
        raise ValueError("unknown correspondence relation")
    if not torch.equal(pairs, pairs.transpose(-1,-2)) or bool(pairs.diagonal(dim1=-2,dim2=-1).any()):
        raise ValueError("correspondence must be symmetric with a zero diagonal")
    norm = F.normalize(slots.float(),dim=-1)
    similarity = norm @ norm.transpose(-1,-2)
    triangle = torch.ones_like(pairs,dtype=torch.bool).triu(1)
    if valid is not None:
        triangle = triangle & valid.bool()[:,:,None] & valid.bool()[:,None,:]
    positive = triangle & (pairs == 1)
    negative = triangle & (pairs == -1)
    terms = (1-similarity)*positive + F.relu(similarity)*negative
    return terms.sum() / (positive.sum()+negative.sum()).clamp_min(1)


class OmniMorphableForCausalLM(PreTrainedModel):
    config_class = IDALatticeConfig
    base_model_prefix = "omni"
    # Keep safe-serialization aware of the deliberate input/output embedding tie.
    # Dict form (target -> source), not the pre-v5 transformers list-of-names
    # form: transformers>=5's PreTrainedModel.get_expanded_tied_weights_keys
    # calls .keys()/.values() on this unconditionally once tie_word_embeddings
    # is set, so a plain list breaks post_init() for every instantiation, not
    # just serialization. This is bookkeeping only -- the actual tie happens
    # explicitly in __init__ (`self.lm_head.weight = self.embed_tokens.weight`);
    # this attribute does not perform the tie itself.
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}
    supports_gradient_checkpointing = True
    _no_split_modules = ["MorphableStateLayer", "StateExpert"]

    def __init__(self, config: IDALatticeConfig):
        super().__init__(config)
        if config.omni_architecture not in {"morphable_state_v1", "state_coupled_v1"}:
            raise ValueError("OmniMorphableForCausalLM requires a morphable or coupled state architecture")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.substrate = nn.ModuleList(StateExpert(config.hidden_size, config.omni_ssm_inner_size,
             config.omni_ssm_heads, config.omni_ssm_state_size, config.omni_ssm_groups)
             for _ in range(config.omni_num_experts))
        self.layers = nn.ModuleList(MorphableStateLayer(config,self.substrate) for _ in range(config.num_hidden_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.state_projection = nn.Linear(config.omni_ssm_inner_size,config.recurrent_state_size,bias=False)
        if config.omni_architecture == "state_coupled_v1":
            self.boundary_state_query = nn.Linear(config.hidden_size,config.omni_ssm_heads*config.omni_ssm_state_size)
        self.omni_state_head = OmniStateHead(config.hidden_size,config.recurrent_state_size,config.omni_state_size,
                        config.omni_response_mode_count,config.omni_ambiguity_count,config.omni_claim_count)
        self.evidence_bridge = OmniEvidenceBridge(config.hidden_size,config.omni_feature_size,config.omni_feature_quality_size)
        self.patch_projection = OmniPatchProjection(config.hidden_size,config.omni_patch_size)
        self.provenance_projection = nn.Linear(32,config.hidden_size,bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module,"bias",None) is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def get_precision_runtime_state(self):
        return {"architecture": self.config.omni_architecture, "backend": "torch_reference",
                "execution_revision": OMNI_EXECUTION_REVISION,
                "scan_backend": self.config.omni_scan_backend,
                "native_nvfp4_executed": False, "state_accumulator": "float32",
                "operation_control": "lrss_pss_separate",
                "coupling_matrix": "factored_signed_low_rank" if self.config.omni_architecture == "state_coupled_v1" else "output_low_rank_delta"}

    def _packet(self, hidden, states, stream_ids, step_index, *, commit):
        plans, stats, updated = [], [], []
        mask = torch.ones(hidden.shape[:2],device=hidden.device,dtype=torch.bool)
        for idx, layer in enumerate(self.layers):
            previous = layer.empty_state(hidden,stream_ids) if states is None else states[idx]
            if previous.stream_ids != tuple(stream_ids):
                raise ValueError("stream cache identity mismatch; explicitly reorder cache")
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                # Non-reentrant checkpoint supports the structured output. The
                # layer is pure, so recompute cannot commit to an external cache.
                hidden, next_state, plan, metrics = checkpoint(
                    lambda x, layer=layer, previous=previous: layer(
                        x, mask, previous, step_index, commit=commit),
                    hidden, use_reentrant=False)
            else:
                hidden,next_state,plan,metrics = layer(hidden,mask,previous,step_index,commit=commit)
            updated.append(next_state); plans.append(plan); stats.append(metrics)
        return self.final_norm(hidden),tuple(updated),plans,stats

    def forward(self,input_ids=None,attention_mask=None,labels=None,
                omni_prediction_boundary=None,omni_stream_ids=None,omni_step_index=None,
                omni_layer_states=None,omni_correspondence=None,
                omni_feature_slots=None,omni_feature_mask=None,omni_feature_type=None,
                omni_feature_time=None,omni_feature_quality=None,
                omni_evidence_time=None,omni_target_time=None,omni_feature_available_at=None,omni_feature_provenance=None,
                omni_frames=None,omni_frame_times=None,omni_frame_quality=None,omni_frame_provenance=None,
                omni_ir_frames=None,omni_ir_times=None,omni_ir_quality=None,omni_ir_provenance=None,
                omni_frame_available_at=None,omni_ir_available_at=None,**targets):
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch,sequence]")
        batch,length = input_ids.shape
        device = input_ids.device
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        if omni_prediction_boundary is None:
            if labels is not None:
                raise ValueError("training requires an explicit evidence prediction boundary")
            omni_prediction_boundary = attention_mask.sum(-1)
        boundary = omni_prediction_boundary.to(device=device,dtype=torch.long)
        steps = torch.zeros(batch,device=device,dtype=torch.long) if omni_step_index is None else omni_step_index.to(device)
        stream_ids = tuple(str(i) for i in range(batch)) if omni_stream_ids is None else tuple(omni_stream_ids)
        if omni_layer_states is not None and omni_stream_ids is None:
            raise ValueError("carried state requires explicit stream identity")
        evidence = OmniEvidenceBatch(stream_ids,steps,boundary,attention_mask,omni_correspondence)
        evidence.validate(batch,length)
        if omni_layer_states is not None and len(omni_layer_states) != len(self.layers):
            raise ValueError("carried state must contain exactly one cache per layer")
        if labels is not None:
            for lane, end in enumerate(boundary.tolist()):
                if bool((labels[lane,:end] != -100).any()):
                    raise ValueError("language labels must exclude the packet evidence prefix")
        if length > self.config.max_position_embeddings:
            raise ValueError("input exceeds configured position capacity")
        embedded = self.embed_tokens(input_ids) + self.position_embeddings(torch.arange(length,device=device))[None]
        horizon_query = forecast_horizon_encoding(omni_target_time, omni_evidence_time,
                                                  self.config.hidden_size, embedded)
        if omni_feature_slots is not None or omni_frames is not None or omni_ir_frames is not None:
            if omni_evidence_time is None or omni_evidence_time.shape!=(batch,) or not bool(torch.isfinite(omni_evidence_time).all()):
                raise ValueError("multimodal packets require a finite evidence time per stream")
        if omni_feature_slots is not None:
            if omni_feature_available_at is None or omni_feature_time is None or omni_feature_provenance is None:
                raise ValueError("feature packets require availability, observation time, and provenance")
            if omni_feature_provenance.shape!=(*omni_feature_slots.shape[:2],32):
                raise ValueError("feature provenance must be a 32-byte normalized digest per slot")
            available=(omni_feature_available_at<=omni_evidence_time[:,None]) & (omni_feature_time<=omni_evidence_time[:,None])
            if omni_feature_mask is None: omni_feature_mask=torch.ones_like(available)
            omni_feature_mask=omni_feature_mask*available
            # Invalid/future feature values cannot reach GEMMs, even as NaN*0.
            valid=omni_feature_mask.bool()
            omni_feature_slots=torch.where(valid[...,None],omni_feature_slots,0)
            omni_feature_time=torch.where(valid,omni_feature_time,0)
            omni_feature_provenance=torch.where(valid[...,None],omni_feature_provenance,0)
            if omni_feature_quality is not None:
                omni_feature_quality=torch.where(valid[...,None],omni_feature_quality,0)
        bridge = self.evidence_bridge(omni_feature_slots,omni_feature_mask,omni_feature_type,omni_feature_time,omni_feature_quality)
        if bridge is not None:
            slots=self.evidence_bridge.context_norm(bridge["slots"]+self.provenance_projection(omni_feature_provenance.to(embedded.dtype)))
            weights=omni_feature_mask.to(slots.dtype)[...,None]
            context=self.evidence_bridge.context_gate((slots*weights).sum(1)/weights.sum(1).clamp_min(1))
            bridge={"slots":slots,"context":context*weights.any(1),"mask":omni_feature_mask}
        media_parts=[]
        for frames,times,quality,provenance,available_at in (
                (omni_frames,omni_frame_times,omni_frame_quality,omni_frame_provenance,omni_frame_available_at),
                (omni_ir_frames,omni_ir_times,omni_ir_quality,omni_ir_provenance,omni_ir_available_at)):
            if frames is None: continue
            if times is None or quality is None or provenance is None or available_at is None:
                raise ValueError("raw frames require availability, timestamps, quality, and provenance")
            if times.shape != frames.shape[:2] or available_at.shape != times.shape:
                raise ValueError("frame availability/timestamp shape mismatch")
            if not bool(torch.isfinite(times).all() & torch.isfinite(available_at).all()) or bool((available_at < times).any()):
                raise ValueError("frame availability must be finite and not precede observation")
            if quality.shape!=(*frames.shape[:2],self.config.omni_feature_quality_size) or provenance.shape!=(*frames.shape[:2],32):
                raise ValueError("frame metadata shape mismatch")
            visible=(times<=omni_evidence_time[:,None]) & (available_at<=omni_evidence_time[:,None])
            if not bool(torch.isfinite(quality[visible]).all() & torch.isfinite(provenance[visible]).all()):
                raise ValueError("non-finite visible frame metadata")
            safe_frames=torch.where(visible[:,:,None,None,None],frames,0)
            safe_times=torch.where(visible,times,0)
            projected=self.patch_projection(safe_frames,safe_times)
            patches=projected.size(1)//frames.size(1)
            metadata=(self.provenance_projection(torch.where(visible[...,None],provenance,0).to(embedded.dtype))
                      +self.evidence_bridge.quality_projection(torch.where(visible[...,None],quality,0).to(embedded.dtype))
                      +self.evidence_bridge.modality_embedding.weight[2])
            projected=projected+metadata.repeat_interleave(patches,1)
            media_parts.append((projected,visible.repeat_interleave(patches,1)))
        logits, predictions, lane_states, lane_plans, pss_losses, layer_metrics = [], [], [], [], [], []
        execution_packets = []
        for lane in range(batch):
            end = int(boundary[lane]); valid_len = int(attention_mask[lane].sum())
            if valid_len < end or not bool(attention_mask[lane,:valid_len].bool().all()):
                raise ValueError("only contiguous right-padded evidence sequences are accepted")
            x = embedded[lane:lane+1,:end]
            if bridge is not None:
                x = x + bridge["context"][lane:lane+1,None]
            for media,visible in media_parts:
                x = torch.cat((media[lane:lane+1,visible[lane]],x),dim=1)
            states = None if omni_layer_states is None else tuple(s.reorder(torch.tensor([lane],device=device)) for s in omni_layer_states)
            hidden,committed,plans,metrics = self._packet(x,states,(stream_ids[lane],),steps[lane:lane+1],commit=True)
            execution_packets.append(dict(phase="evidence",lane=lane,position=end,plans=plans,layers=metrics))
            forecast_hidden = hidden + horizon_query[lane:lane+1,None]
            if self.config.omni_architecture == "state_coupled_v1":
                query=self.boundary_state_query(forecast_hidden.mean(1)).reshape(1,self.config.omni_ssm_heads,self.config.omni_ssm_state_size)
                query=F.normalize(query.float(),dim=-1)
                addressed=(committed[-1].recurrent*query[:,:,None,:]).sum(-1)
            else:
                addressed=committed[-1].recurrent.mean(-1)
            state_vector = self.state_projection(addressed.flatten(1).to(hidden.dtype))
            before = targets.get("omni_state_before")
            predictions.append(self.omni_state_head(forecast_hidden,state_vector,None if before is None else before[lane:lane+1]))
            lane_states.append(tuple(replace(s,workspace=None) for s in committed))
            lane_plans.append(plans)
            layer_metrics.append(metrics)
            pss_losses.extend(m["pss_loss"] for m in metrics)
            parts = [self.lm_head(forecast_hidden[:,-end:])]
            # Teacher-forced tokens advance only a private continuation branch.
            continuation = committed
            for pos in range(end,valid_len):
                h,continuation,continuation_plans,continuation_metrics = self._packet(
                                                embedded[lane:lane+1,pos:pos+1]+horizon_query[lane:lane+1,None],continuation,
                                                (stream_ids[lane],),steps[lane:lane+1],commit=False)
                execution_packets.append(dict(phase="continuation",lane=lane,position=pos,
                                              plans=continuation_plans,layers=continuation_metrics))
                parts.append(self.lm_head(h))
            row = torch.cat(parts,1)
            logits.append(F.pad(row,(0,0,0,length-valid_len)))
        logits = torch.cat(logits)
        predicted = {key:torch.cat([p[key] for p in predictions]) for key in predictions[0]}
        next_states=[]
        for idx in range(len(self.layers)):
            values=[lane_states[b][idx] for b in range(batch)]
            next_states.append(OmniLayerState(
                **{key:torch.cat([getattr(s,key) for s in values]) for key in
                   ("recurrent","anchors","anchor_steps","prior_pss_error","committed_position")},
                stream_ids=stream_ids,architecture=self.config.omni_architecture))
        loss=None; metrics={}
        if labels is not None:
            selected=labels[:,1:] != -100
            loss=F.cross_entropy(logits[:,:-1].reshape(-1,self.config.vocab_size).float(),labels[:,1:].reshape(-1),ignore_index=-100) if bool(selected.any()) else logits.sum()*0
            metrics["omni_language"]=loss.detach()
        if targets.get("omni_state_after") is not None:
            auxiliary,metrics_omni=compute_omni_losses(predicted,targets=targets,feature_slots=None,feature_mask=None,
                feature_type=None,transition_weight=self.config.omni_transition_loss_weight,
                semantic_weight=self.config.omni_semantic_loss_weight,calibration_weight=self.config.omni_calibration_loss_weight,
                alignment_weight=0)
            loss=auxiliary if loss is None else loss+auxiliary
            metrics.update(metrics_omni)
        alignment_mask = omni_feature_mask
        if alignment_mask is not None and targets.get("omni_valid") is not None:
            alignment_mask = alignment_mask * targets["omni_valid"][:,None]
        alignment=correspondence_loss(None if bridge is None else bridge["slots"],omni_correspondence,alignment_mask)
        if alignment is not None:
            loss=alignment*self.config.omni_alignment_loss_weight if loss is None else loss+alignment*self.config.omni_alignment_loss_weight
            metrics["omni_alignment"]=alignment.detach()
        pss_loss=torch.stack(pss_losses).mean()
        if loss is not None:
            loss=loss+self.config.state_supersampler_loss_weight*pss_loss
        metrics["omni_pss"]=pss_loss.detach()
        result=CausalLMOutputWithCrossAttentions(loss=loss,logits=logits)
        result.omni_predictions=predicted
        result.omni_layer_states=tuple(next_states)
        result.omni_operation_plans=lane_plans
        result.omni_loss_metrics=metrics
        result.omni_layer_metrics=layer_metrics
        # Legacy surfaces above remain evidence-only; full-forward consumers use this receipt.
        result.omni_execution_packets=execution_packets
        result.omni_execution_summary=self.execution_summary(execution_packets)
        return result

    def execution_summary(self, packets):
        expert_ids=set(); layer_experts=set(); phases={}
        matrix_work={}
        for packet in packets:
            phase=phases.setdefault(packet["phase"],dict(packets=0,assigned_rows=0,valid_assigned_rows=0))
            phase["packets"]+=1
            for layer,(plan,metrics) in enumerate(zip(packet["plans"],packet["layers"])):
                phase["assigned_rows"]+=int(metrics["assigned_rows"])
                phase["valid_assigned_rows"]+=int(metrics["valid_assigned_rows"])
                for work in metrics["matrix_executions"]:
                    name=work["name"] if work["name"].startswith("substrate.") else f"layers.{layer}."+work["name"]
                    total=matrix_work.setdefault(name,dict(calls=0,rows=0,forward_flops=0))
                    total["calls"]+=1
                    total["rows"]+=work["rows"]
                    total["forward_flops"]+=work["forward_flops"]
                for expert in plan.expert_ids.flatten().tolist():
                    expert_ids.add(expert); layer_experts.add((layer,expert))
        # Conservative dense-inclusive bound, not a claim that all dense matrices ran.
        selected_parameters=0
        for name,param in self.named_parameters():
            bits=name.split(".")
            if bits[0]=="substrate" and int(bits[1]) not in expert_ids: continue
            if bits[0]=="layers" and bits[2]=="controls" and (int(bits[1]),int(bits[3])) not in layer_experts: continue
            selected_parameters+=param.numel()
        return dict(scope="complete_forward_all_lanes",phases=phases,
                    assigned_rows=sum(p["assigned_rows"] for p in phases.values()),
                    unique_expert_ids=sorted(expert_ids),selected_layer_experts=sorted(layer_experts),
                    selected_parameter_union_upper_bound=selected_parameters,
                    matrix_work=matrix_work,matrix_work_scope="expert_substrate_and_operation_controls",
                    gemm_backend="torch_reference",scan_backend=self.config.omni_scan_backend)
