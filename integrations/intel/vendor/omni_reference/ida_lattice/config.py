# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/config.py`; body unmodified except as recorded in ../SOURCE.md.

from __future__ import annotations

from transformers import PretrainedConfig


DEFAULT_COGNITIVE_ROUTES: list[str] = [
    "PERCEPTION",
    "MEMORY",
    "SALIENCE",
    "CAUSAL_INSPECTION",
    "PLANNING",
    "INHIBITION",
    "CREATION",
    "ERROR_CORRECTION",
    "EXPRESSION",
]

# Per-seat initial route bias (route_name → float).
# Positive = stronger activation bias; negative = weaker.
# Unlisted routes default to 0.0.
SEAT_COGNITIVE_BIAS: dict[str, dict[str, float]] = {
    "SENTINEL": {"INHIBITION": 0.4, "SALIENCE": 0.3, "CAUSAL_INSPECTION": 0.1},
    "PRISM":    {"CAUSAL_INSPECTION": 0.4, "SALIENCE": 0.2, "MEMORY": 0.1},
    "VECTOR":   {"PLANNING": 0.4, "CAUSAL_INSPECTION": 0.2, "EXPRESSION": 0.1},
    "ECHO":     {"MEMORY": 0.4, "PERCEPTION": 0.2, "CAUSAL_INSPECTION": 0.1},
    "FORGE":    {"CREATION": 0.4, "PLANNING": 0.2, "INHIBITION": -0.1},
    "ATLAS":    {"CAUSAL_INSPECTION": 0.3, "MEMORY": 0.2, "PLANNING": 0.2},
    "SHADE":    {"SALIENCE": 0.3, "INHIBITION": 0.2, "PERCEPTION": 0.2},
    "JUDGE":    {"INHIBITION": 0.3, "CAUSAL_INSPECTION": 0.3, "ERROR_CORRECTION": 0.2},
    "PULSE":    {"SALIENCE": 0.3, "PERCEPTION": 0.3, "EXPRESSION": 0.1},
    "ORBIT":    {"PLANNING": 0.3, "MEMORY": 0.2, "EXPRESSION": 0.2},
}


class IDALatticeConfig(PretrainedConfig):
    model_type = "ida_lattice"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 1024,
        recurrent_state_size: int | None = None,
        local_attention_window: int = 128,
        num_cognitive_routes: int | None = None,
        top_k_routes: int = 3,
        cognitive_route_names: list[str] | None = None,
        num_personality_experts: int | None = None,
        personality_residual_expert_width: int | None = None,
        top_k_experts: int | None = None,
        use_personality_residual_experts: bool = False,
        layered_expert_routing_enabled: bool = False,
        morphable_expert_enabled: bool = False,
        morphable_expert_rank: int = 0,
        training_loss_only: bool = False,
        training_loss_chunk_size: int = 128,
        seat: str = "IDA",
        family: str = "edge",
        seat_cognitive_bias: dict[str, float] | None = None,
        action_gate_size: int = 6,
        workspace_slot_count: int = 8,
        workspace_slot_size: int | None = None,
        thalamic_route_count: int = 4,
        student_state_size: int | None = None,
        future_prediction_horizon: int = 2,
        reconstruction_enabled: bool = True,
        reconstruction_bottleneck_size: int | None = None,
        reconstruction_max_span: int = 8,
        reconstruction_loss_weight: float = 0.05,
        circuit_reconstruction_enabled: bool = True,
        circuit_reconstruction_rank: int = 64,
        circuit_reconstruction_loss_weight: float = 0.01,
        continuation_enabled: bool = True,
        continuation_loss_weight: float = 0.02,
        omni_enabled: bool = False,
        omni_architecture: str = "legacy",
        omni_num_experts: int = 8,
        omni_top_k: int = 2,
        omni_ssm_inner_size: int | None = None,
        omni_ssm_heads: int = 8,
        omni_ssm_state_size: int = 16,
        omni_ssm_groups: int = 1,
        omni_mode_rank: int = 64,
        omni_patch_size: int = 16,
        omni_scan_backend: str = "reference",
        omni_coupling_rank: int = 8,
        omni_operation_ablation: str = "full",
        omni_program_selection: str = "factorized",
        omni_target_version: str = "disabled",
        omni_target_manifest_hash: str = "",
        omni_state_size: int = 16,
        omni_response_mode_count: int = 8,
        omni_ambiguity_count: int = 15,
        omni_claim_count: int = 8,
        omni_feature_size: int | None = None,
        omni_feature_quality_size: int = 4,
        omni_feature_slot_count: int = 4,
        omni_transition_loss_weight: float = 1.0,
        omni_semantic_loss_weight: float = 0.25,
        omni_calibration_loss_weight: float = 0.10,
        omni_alignment_loss_weight: float = 0.05,
        temporal_execution_mode: str = "full_reality",
        temporal_anchor_tokens: int = 4,
        temporal_refresh_threshold: float = 0.35,
        governed_memory_enabled: bool = True,
        governed_memory_size: int | None = None,
        trace_emitter_enabled: bool = True,
        trace_emitter_capacity: int = 64,
        multiscale_memory_enabled: bool = False,
        multiscale_memory_num_scales: int = 8,
        multiscale_memory_num_anchors: int = 32,
        multiscale_memory_tau_min: float = 1.0,
        multiscale_memory_tau_max: float = 64.0,
        multiscale_memory_loss_weight: float = 0.0,
        multiscale_memory_contract_backend: str = "einsum",
        multiscale_memory_contract_fallback: str = "matmul",
        state_supersampler_enabled: bool = False,
        state_supersampler_rank: int = 128,
        state_supersampler_loss_weight: float = 0.02,
        supersampler_curriculum_enabled: bool = True,
        supersampler_curriculum_crossover: float = 0.3,
        lrss_content_pooling: bool = True,
        telemetry_mode: str = "full",
        telemetry_interval_steps: int = 1,
        control_summary_mode: str | None = None,
        torch_compile_backend: str = "inductor",
        torch_compile_dynamic: bool = False,
        torch_compile_fullgraph: bool = False,
        torch_compile_disable_runtime_telemetry: bool = True,
        fp8_enabled: bool = False,
        fp8_backend: str = "off",
        fp8_scope: list[str] | None = None,
        fp8_recipe: str = "delayed_hybrid",
        fp8_amax_history_len: int = 16,
        fp8_amax_compute_algo: str = "max",
        fp8_warmup_steps: int = 100,
        fp8_weight_cache: bool = False,
        fp8_wgrad_fusion: bool = False,
        fp8_fallback: str = "native_scaled_mm",
        expert_bank_backend: str = "independent",
        expert_bank_backend_fallback: str = "independent",
        developmental_stage: str = "juvenile",
        developmental_plasticity: dict | None = None,
        developmental_exploration: dict | None = None,
        developmental_pruning: dict | None = None,
        developmental_local_reopen: dict | None = None,
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ) -> None:
        # Backward-compatibility shims for pre-rename saved configs
        if "num_student_pathways" in kwargs and num_cognitive_routes is None:
            num_cognitive_routes = int(kwargs.pop("num_student_pathways"))
        else:
            kwargs.pop("num_student_pathways", None)
        if "top_k_pathways" in kwargs:
            top_k_routes = int(kwargs.pop("top_k_pathways"))
        if "pathway_names" in kwargs and cognitive_route_names is None:
            cognitive_route_names = list(kwargs.pop("pathway_names"))
        else:
            kwargs.pop("pathway_names", None)
        if "family_state_size" in kwargs and student_state_size is None:
            student_state_size = int(kwargs.pop("family_state_size"))
        else:
            kwargs.pop("family_state_size", None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.recurrent_state_size = recurrent_state_size or hidden_size
        self.local_attention_window = local_attention_window
        self.seat = str(seat or "IDA").strip().upper() or "IDA"
        self.family = str(family or "edge").strip().lower() or "edge"
        self.cognitive_route_names = list(cognitive_route_names or DEFAULT_COGNITIVE_ROUTES)
        self.num_cognitive_routes = max(
            1, int(num_cognitive_routes or len(self.cognitive_route_names))
        )
        self.top_k_routes = max(1, min(int(top_k_routes), self.num_cognitive_routes))
        self.num_personality_experts = max(
            1,
            int(num_personality_experts or self.num_cognitive_routes),
        )
        self.personality_residual_expert_width = max(
            1,
            int(personality_residual_expert_width or intermediate_size),
        )
        self.top_k_experts = max(
            1,
            min(
                int(top_k_experts or self.top_k_routes),
                self.num_personality_experts,
            ),
        )
        self.use_personality_residual_experts = bool(
            use_personality_residual_experts
        )
        # When enabled, each layer routes individual token representations.
        # The default remains sequence-level routing for legacy bodies and
        # native lineage compatibility.
        self.layered_expert_routing_enabled = bool(layered_expert_routing_enabled)
        self.morphable_expert_enabled = bool(morphable_expert_enabled)
        self.morphable_expert_rank = max(0, int(morphable_expert_rank))
        if self.morphable_expert_enabled and self.morphable_expert_rank <= 0:
            raise ValueError("morphable_expert_rank must be positive when morphable_expert_enabled=True")
        self.training_loss_only = bool(training_loss_only)
        self.training_loss_chunk_size = max(1, int(training_loss_chunk_size))
        self.seat_cognitive_bias = dict(
            seat_cognitive_bias
            if seat_cognitive_bias is not None
            else SEAT_COGNITIVE_BIAS.get(self.seat, {})
        )
        self.action_gate_size = action_gate_size
        self.workspace_slot_count = workspace_slot_count
        self.workspace_slot_size = workspace_slot_size or hidden_size
        self.thalamic_route_count = thalamic_route_count
        self.student_state_size = student_state_size or hidden_size
        self.future_prediction_horizon = future_prediction_horizon
        self.reconstruction_enabled = bool(reconstruction_enabled)
        self.reconstruction_bottleneck_size = reconstruction_bottleneck_size or hidden_size
        self.reconstruction_max_span = max(1, int(reconstruction_max_span))
        self.reconstruction_loss_weight = max(0.0, float(reconstruction_loss_weight))
        self.circuit_reconstruction_enabled = bool(circuit_reconstruction_enabled)
        self.circuit_reconstruction_rank = max(0, int(circuit_reconstruction_rank))
        self.circuit_reconstruction_loss_weight = max(0.0, float(circuit_reconstruction_loss_weight))
        self.continuation_enabled = bool(continuation_enabled)
        self.continuation_loss_weight = max(0.0, float(continuation_loss_weight))
        self.omni_enabled = bool(omni_enabled)
        self.omni_architecture = str(omni_architecture)
        self.omni_num_experts = int(omni_num_experts)
        self.omni_top_k = int(omni_top_k)
        self.omni_ssm_inner_size = int(omni_ssm_inner_size or hidden_size)
        self.omni_ssm_heads = int(omni_ssm_heads)
        self.omni_ssm_state_size = int(omni_ssm_state_size)
        self.omni_ssm_groups = int(omni_ssm_groups)
        self.omni_mode_rank = int(omni_mode_rank)
        self.omni_patch_size = int(omni_patch_size)
        self.omni_scan_backend = str(omni_scan_backend)
        self.omni_coupling_rank = int(omni_coupling_rank)
        if self.omni_scan_backend not in {"reference", "native_cuda", "native_cuda_packed"}:
            raise ValueError("unknown Omni scan backend")
        self.omni_operation_ablation = str(omni_operation_ablation)
        if self.omni_operation_ablation not in {"full", "no_lrss", "no_pss", "fixed"}:
            raise ValueError("unknown Omni operation ablation")
        self.omni_program_selection = str(omni_program_selection)
        if self.omni_program_selection not in {"factorized", "correlated"}:
            raise ValueError("unknown Omni program selection")
        if self.omni_architecture not in {"legacy", "morphable_state_v1", "state_coupled_v1"}:
            raise ValueError("unknown Omni architecture")
        if self.omni_architecture in {"morphable_state_v1", "state_coupled_v1"}:
            if not self.omni_enabled:
                raise ValueError("morphable state architecture requires omni_enabled")
            if min(self.omni_num_experts, self.omni_top_k, self.omni_ssm_inner_size,
                   self.omni_ssm_heads, self.omni_ssm_state_size, self.omni_ssm_groups,
                   self.omni_mode_rank, self.omni_patch_size) <= 0:
                raise ValueError("Omni dimensions and budgets must be positive")
            if self.omni_top_k > self.omni_num_experts:
                raise ValueError("Omni top-k exceeds resident expert count")
            if self.omni_ssm_inner_size % self.omni_ssm_heads or self.omni_ssm_heads % self.omni_ssm_groups:
                raise ValueError("invalid Omni SSM head/group geometry")
            if self.omni_architecture == "state_coupled_v1" and not 0 < self.omni_coupling_rank < self.omni_ssm_state_size:
                raise ValueError("coupling rank must be positive and smaller than state-coordinate width")
        self.omni_target_version = str(omni_target_version or "disabled").strip() or "disabled"
        self.omni_target_manifest_hash = str(omni_target_manifest_hash or "").strip()
        self.omni_state_size = max(1, int(omni_state_size))
        self.omni_response_mode_count = max(1, int(omni_response_mode_count))
        self.omni_ambiguity_count = max(1, int(omni_ambiguity_count))
        self.omni_claim_count = max(1, int(omni_claim_count))
        self.omni_feature_size = max(1, int(omni_feature_size or hidden_size))
        self.omni_feature_quality_size = max(1, int(omni_feature_quality_size))
        self.omni_feature_slot_count = max(1, int(omni_feature_slot_count))
        self.omni_transition_loss_weight = max(0.0, float(omni_transition_loss_weight))
        self.omni_semantic_loss_weight = max(0.0, float(omni_semantic_loss_weight))
        self.omni_calibration_loss_weight = max(0.0, float(omni_calibration_loss_weight))
        self.omni_alignment_loss_weight = max(0.0, float(omni_alignment_loss_weight))
        self.temporal_execution_mode = str(temporal_execution_mode or "full_reality").strip().lower() or "full_reality"
        self.temporal_anchor_tokens = max(1, int(temporal_anchor_tokens))
        self.temporal_refresh_threshold = min(1.0, max(0.0, float(temporal_refresh_threshold)))
        self.governed_memory_enabled = governed_memory_enabled
        self.governed_memory_size = governed_memory_size or hidden_size
        self.trace_emitter_enabled = bool(trace_emitter_enabled)
        self.trace_emitter_capacity = max(1, int(trace_emitter_capacity))
        self.multiscale_memory_enabled = bool(multiscale_memory_enabled)
        self.multiscale_memory_num_scales = max(1, int(multiscale_memory_num_scales))
        self.multiscale_memory_num_anchors = max(1, int(multiscale_memory_num_anchors))
        self.multiscale_memory_tau_min = max(1e-3, float(multiscale_memory_tau_min))
        self.multiscale_memory_tau_max = max(self.multiscale_memory_tau_min + 1.0, float(multiscale_memory_tau_max))
        self.multiscale_memory_loss_weight = max(0.0, float(multiscale_memory_loss_weight))
        self.multiscale_memory_contract_backend = (
            str(multiscale_memory_contract_backend or "einsum").strip().lower() or "einsum"
        )
        self.multiscale_memory_contract_fallback = (
            str(multiscale_memory_contract_fallback or "matmul").strip().lower() or "matmul"
        )
        self.state_supersampler_enabled = bool(state_supersampler_enabled)
        self.state_supersampler_rank = max(1, int(state_supersampler_rank))
        self.state_supersampler_loss_weight = max(0.0, float(state_supersampler_loss_weight))
        self.supersampler_curriculum_enabled = bool(supersampler_curriculum_enabled)
        self.supersampler_curriculum_crossover = min(0.99, max(0.0, float(supersampler_curriculum_crossover)))
        self.lrss_content_pooling = bool(lrss_content_pooling)
        self.telemetry_mode = str(telemetry_mode or "full").strip().lower() or "full"
        self.telemetry_interval_steps = max(1, int(telemetry_interval_steps))
        if control_summary_mode is None:
            control_summary_mode = "off" if self.telemetry_mode in {"off", "haul"} else "full"
        self.control_summary_mode = str(control_summary_mode or "full").strip().lower() or "full"
        self.torch_compile_backend = str(torch_compile_backend or "inductor").strip() or "inductor"
        self.torch_compile_dynamic = bool(torch_compile_dynamic)
        self.torch_compile_fullgraph = bool(torch_compile_fullgraph)
        self.torch_compile_disable_runtime_telemetry = bool(
            torch_compile_disable_runtime_telemetry
        )
        self.fp8_enabled = bool(fp8_enabled)
        self.fp8_backend = str(fp8_backend or "off").strip().lower() or "off"
        self.fp8_scope = [str(item).strip().lower() for item in (fp8_scope or ["experts", "workspace"]) if str(item).strip()]
        self.fp8_recipe = str(fp8_recipe or "delayed_hybrid").strip() or "delayed_hybrid"
        self.fp8_amax_history_len = max(1, int(fp8_amax_history_len))
        self.fp8_amax_compute_algo = str(fp8_amax_compute_algo or "max").strip().lower() or "max"
        self.fp8_warmup_steps = max(0, int(fp8_warmup_steps))
        self.fp8_weight_cache = bool(fp8_weight_cache)
        self.fp8_wgrad_fusion = bool(fp8_wgrad_fusion)
        self.fp8_fallback = str(fp8_fallback or "native_scaled_mm").strip().lower() or "native_scaled_mm"
        self.expert_bank_backend = str(
            expert_bank_backend or "independent"
        ).strip().lower() or "independent"
        self.expert_bank_backend_fallback = str(
            expert_bank_backend_fallback or "independent"
        ).strip().lower() or "independent"
        self.developmental_stage = developmental_stage
        self.developmental_plasticity = dict(developmental_plasticity or {})
        self.developmental_exploration = dict(developmental_exploration or {})
        self.developmental_pruning = dict(developmental_pruning or {})
        self.developmental_local_reopen = dict(developmental_local_reopen or {})
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = bool(tie_word_embeddings)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )
