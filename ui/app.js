const $ = (selector) => document.querySelector(selector);

const state = {
  cursor: 0,
  eventSource: null,
  sessionId: null,
  parents: new Map(),
  events: [],
  graph: null,
  status: "standby",
  startedAt: null,
  objectById: new Map(),
  currentStep: null,
  completedSteps: new Set(),
  stepStates: new Map(),
  verifiedSteps: new Set(),
  hash: "",
  runtime: {},
  observation: null,
  constraints: [],
  latestDecision: null,
  receipt: null,
  reconnectAttempts: 0,
  streamOpen: false,
};

const phaseOrder = ["observe", "plan", "act", "verify"];
const modeLabels = {
  NOMINAL: "NOMINAL",
  DEGRADED: "DEGRADED",
  LOCAL_ONLY: "LOCAL ONLY",
  HOLD: "HOLD",
  PROTECTIVE_STOP: "PROTECTIVE STOP",
};

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = value ?? "—";
}

function setStatus(selector, value, tone = "") {
  const node = $(selector);
  if (!node) return;
  node.textContent = value ?? "—";
  node.className = `status-label ${tone}`.trim();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function shortHash(value) {
  if (!value) return "waiting";
  const raw = String(value);
  return raw.length > 27 ? `${raw.slice(0, 15)}…${raw.slice(-8)}` : raw;
}

function nowLabel(ts = Date.now()) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(ts);
}

function label(value, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value)
    .replaceAll("_", " ")
    .replaceAll(".", " / ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function statusValue(value, fallback = "LIVE") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value).split(".").pop().toUpperCase();
}

function humanMode(value) {
  const raw = statusValue(value, "NOMINAL");
  return modeLabels[raw] || label(raw).toUpperCase();
}

function humanObject(objectId) {
  const item = state.objectById.get(objectId);
  return item ? `${label(item.cls)} / ${objectId}` : objectId || "workspace";
}

function setConnection(text, tone = "") {
  const bar = $("#connection");
  if (!bar) return;
  bar.className = `connection-bar ${tone}`.trim();
  const message = bar.querySelector("span:nth-child(2)");
  if (message) message.textContent = text;
  setText(".connection-seq", state.sessionId
    ? `EVENT STREAM / ${state.cursor.toString().padStart(3, "0")}`
    : "EVENT STREAM / READY");
}

function appendTrace(text, tone = "", kind = "event") {
  const list = $("#trace");
  if (!list) return;
  list.querySelector(".trace-empty")?.remove();
  const item = document.createElement("div");
  item.className = `trace-item ${tone}`.trim();
  item.innerHTML = `<span class="trace-mark"></span><div class="trace-copy"><strong>${escapeHtml(kind.toUpperCase())}</strong> · ${escapeHtml(text)}</div><time class="trace-time">${nowLabel()}</time>`;
  list.prepend(item);
  while (list.children.length > 35) list.lastElementChild.remove();
  setText("#event-count", `${state.events.length} EVENT${state.events.length === 1 ? "" : "S"}`);
}

function setMode(mode, runtime = state.runtime) {
  const badge = $("#mode");
  if (!badge) return;
  const raw = String(mode || runtime?.mode || "mock");
  const isMock = raw.toLowerCase() === "mock" || runtime?.execution === "simulated";
  const text = isMock ? "MOCK / NO HARDWARE" : `${label(raw).toUpperCase()} MODE`;
  badge.innerHTML = `<span class="mode-dot"></span> ${escapeHtml(text)}`;
  badge.className = `mode-badge ${isMock ? "mock" : ""}`.trim();
  badge.title = isMock
    ? "This judge surface is connected to a synthetic session; no hardware is being controlled."
    : `Runtime mode: ${raw}`;
}

function runtimeDefaults() {
  return {
    mode: "mock",
    execution: "simulated",
    capabilities: [
      { id: "perception", kind: "perception", name: "Perception", detail: "Synthetic observer · structured state", available: true, local: true },
      { id: "left", kind: "arm", arm: "left", name: "Left SO-101", detail: "Simulated manipulation", available: true, local: true },
      { id: "right", kind: "arm", arm: "right", name: "Right SO-101", detail: "Simulated manipulation", available: true, local: true },
      { id: "reasoning", kind: "reasoning", name: "OMNI reasoning", detail: "Rule planner · local process", available: true, local: true },
    ],
    placement_devices: [],
    voice: { status: "standby", provider: "Speechmatics adapter", wired: false },
  };
}

function capabilityIcon(kind) {
  return ({ perception: "⌾", arm: "↙", reasoning: "✦", voice: "◖", provider: "◈" })[kind] || "◆";
}

function capabilityTone(node) {
  if (node.available === false || node.online === false) return "standby";
  return node.status === "standby" ? "standby" : "ready";
}

function renderCapabilityRows(nodes, target) {
  if (!target) return;
  target.innerHTML = nodes.length
    ? nodes.map((node) => {
      const tone = capabilityTone(node);
      const detail = [node.detail, node.local === false ? "remote" : "local"]
        .filter(Boolean).join(" · ");
      return `<div class="capability-item ${tone === "standby" ? "muted-node" : ""}">
        <span class="capability-icon ${escapeHtml(node.kind || "provider")}-icon" aria-hidden="true">${capabilityIcon(node.kind)}</span>
        <div><strong>${escapeHtml(node.name || label(node.id))}</strong><small>${escapeHtml(detail || "Capability metadata")}</small></div>
        <span class="tiny-status ${tone}" title="${escapeHtml(tone)}"></span>
      </div>`;
    }).join("")
    : `<div class="capability-placeholder">No capability metadata reported.</div>`;
}

function renderDevices(runtime) {
  const target = $("#devices");
  if (!target) return;
  const nodes = Array.isArray(runtime.capabilities) ? runtime.capabilities : [];
  target.innerHTML = nodes.length
    ? nodes.map((node) => {
      const tone = capabilityTone(node);
      const device = node.device || node.id || "";
      const stateLabel = node.online === false || node.available === false
        ? "OFFLINE" : tone === "standby" ? "STANDBY" : "READY";
      const deviceClass = node.kind === "arm" ? "arm" : node.kind === "reasoning" ? "reasoning" : node.kind === "provider" ? "qualcomm" : "vision";
      return `<div class="device-row ${tone === "standby" ? "offline" : ""}" data-device="${escapeHtml(device)}" data-arm="${escapeHtml(node.arm || "")}" data-role="${escapeHtml(node.kind || "")}">
        <div class="device-icon ${deviceClass}" aria-hidden="true">${capabilityIcon(node.kind)}</div>
        <div class="device-copy"><strong>${escapeHtml(node.name || label(node.id))}</strong><small>${escapeHtml(node.detail || device || "Capability")}</small></div>
        <span class="device-state">${stateLabel}</span>
      </div>`;
    }).join("")
    : `<div class="device-placeholder">Placement metadata will appear when a mission starts.</div>`;

  const placement = Array.isArray(runtime.placement_devices) ? runtime.placement_devices : [];
  const online = placement.filter((device) => device.available !== false && device.online !== false).length;
  const placementSummary = $("#placement-summary");
  if (placementSummary) {
    placementSummary.innerHTML = placement.length
      ? `<span class="placement-label">PLACEMENT OPTIONS</span><strong>${online}/${placement.length} online</strong><span class="placement-devices">${placement.map((device) => `${escapeHtml(device.name || "device")}${device.local === false ? " · remote" : ""}`).join(" · ")}</span>`
      : "";
  }
  setStatus("#device-state-label", nodes.length ? `${nodes.length} NODES · ${runtime.execution === "simulated" ? "SIMULATED" : "AVAILABLE"}` : "METADATA PENDING", runtime.execution === "simulated" ? "warn" : "");
}

function setRuntime(runtime = {}) {
  state.runtime = { ...runtimeDefaults(), ...runtime };
  if (!Array.isArray(state.runtime.capabilities) || !state.runtime.capabilities.length) {
    state.runtime.capabilities = runtimeDefaults().capabilities;
  }
  renderCapabilityRows(state.runtime.capabilities, $("#sidebar-capabilities"));
  setText("#sidebar-node-count", state.runtime.capabilities.length);
  renderDevices(state.runtime);
  const environment = state.runtime.execution === "simulated"
    ? "Mock session · no hardware"
    : state.runtime.environment || "Configured runtime";
  setText("#environment-label", environment);
  setText("#safety-label", state.runtime.safety || "Mission envelope ready");
  setMode(state.runtime.mode, state.runtime);
}

function resetView() {
  state.cursor = 0;
  state.sessionId = null;
  state.parents.clear();
  state.events = [];
  state.graph = null;
  state.status = "running";
  state.startedAt = Date.now();
  state.objectById.clear();
  state.currentStep = null;
  state.completedSteps.clear();
  state.stepStates.clear();
  state.verifiedSteps.clear();
  state.hash = "";
  state.runtime = {};
  state.observation = null;
  state.constraints = [];
  state.latestDecision = null;
  state.receipt = null;
  state.reconnectAttempts = 0;
  state.streamOpen = false;
  $("#trace").innerHTML = `<div class="trace-empty"><span>◌</span><p>Listening for governed execution events…</p></div>`;
  $("#graph").innerHTML = `<div class="graph-empty"><span class="empty-glyph">⌁</span><p>Compiling the first graph…</p><small>Function is separate from placement.</small></div>`;
  $("#scene-objects").innerHTML = `<div class="empty-stage">Waiting for perception</div>`;
  $("#decision-summary").innerHTML = `<span class="decision-empty">Planner decision evidence will appear after observation.</span>`;
  $("#active-constraints").innerHTML = `<span class="constraint-empty">No live constraints</span>`;
  setStatus("#graph-state", "COMPILING");
  setStatus("#scene-status", "CONNECTING");
  setText("#scene-revision", "FRAME —");
  setText("#scene-count", "—");
  setText("#misplaced-count", "—");
  setText("#settled-count", "—");
  setText("#workspace-clear", "—");
  setText("#scene-confidence", "—");
  setText("#scene-source-label", "Connecting to observer");
  setText("#scene-model-label", "Observer metadata pending");
  $(".scene-source")?.classList.remove("live");
  setStatus("#receipt-state", "PENDING");
  setText("#receipt-title", "Mission in progress");
  setText("#receipt-copy", "Waiting for final verification and a chained receipt.");
  $("#receipt-icon").className = "receipt-icon";
  $("#receipt-icon").textContent = "◌";
  setText("#receipt-hash", "waiting");
  $("#copy-hash").disabled = true;
  ["#metric-revisions", "#metric-actions", "#metric-resolved", "#metric-duration", "#metric-rejected"].forEach((selector) => setText(selector, "—"));
  setText("#metric-mode", "NOMINAL");
  setText("#receipt-run-id", "RUN —");
  setText("#receipt-parent", "PARENT GENESIS");
  setText("#receipt-provenance", "PROVENANCE PENDING");
  setText("#run-clock", "00:00");
  $(".live-clock")?.classList.add("running");
  $("#apply-constraint").disabled = false;
  $("#reconnect").hidden = true;
  setRuntime(runtimeDefaults());
  setReason("Waiting for the first observation.");
  updateConstraintControl();
  updatePhases();
  updateProgress();
}

function updateClock() {
  if (!state.startedAt || state.status !== "running") return;
  const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
  setText("#run-clock", `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`);
}

function updatePhases() {
  const activePhase = state.currentStep
    ? "act"
    : state.status === "finished"
      ? "verify"
      : state.graph
        ? "plan"
        : "observe";
  const activeIndex = phaseOrder.indexOf(activePhase);
  phaseOrder.forEach((phase, index) => {
    const row = document.querySelector(`[data-phase="${phase}"]`);
    if (!row) return;
    const stateNode = row.querySelector(".phase-state");
    row.classList.remove("active", "done", "warn");
    if (state.status === "failed" && phase === "verify") {
      row.classList.add("warn");
      stateNode.textContent = "attention";
    } else if (index < activeIndex || (state.status === "finished" && phase !== "verify")) {
      row.classList.add("done");
      stateNode.textContent = "done";
    } else if (phase === activePhase) {
      row.classList.add("active");
      stateNode.textContent = state.status === "finished" ? "done" : "live";
    } else {
      stateNode.textContent = "queued";
    }
  });
}

function updateProgress() {
  const steps = state.graph?.steps || [];
  const total = steps.length;
  const done = steps.filter((step) => state.completedSteps.has(step.id) || state.verifiedSteps.has(step.id)).length;
  const resolved = state.receipt?.metrics?.resolved;
  const percent = state.status === "finished"
    ? resolved === false ? 96 : 100
    : total ? Math.min(96, Math.round((done / total) * 100)) : 0;
  $("#progress-bar").style.width = `${percent}%`;
  setText("#progress-label", `${percent}% COMPLETE`);
}

function objectPosition(detection, index, observation = {}) {
  const center = detection.center || detection.centroid;
  if (Array.isArray(center) && center.length >= 2) {
    const width = Number(observation.width || observation.image_width || 640);
    const height = Number(observation.height || observation.image_height || 480);
    const x = Number(center[0]);
    const y = Number(center[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      return { x: `${Math.max(5, Math.min(95, (x <= 1 ? x : x / width) * 100))}%`, y: `${Math.max(8, Math.min(91, (y <= 1 ? y : y / height) * 100))}%` };
    }
  }
  const positions = {
    A: [18, 28], B: [72, 72], center: [50, 53], table: [50, 53],
    upper_left: [24, 25], upper_right: [76, 25], lower_left: [24, 76], lower_right: [76, 76],
    left: [24, 52], right: [76, 52], tray: [37, 71], tray_plate: [37, 71],
    tray_cup: [69, 28], tray_napkin: [22, 68], bin: [84, 22], setting_1: [84, 68],
    drawer: [50, 17], closed: [50, 17], floor: [72, 86],
  };
  const [x, y] = positions[detection.zone] || [26 + ((index * 23) % 53), 26 + ((index * 37) % 53)];
  return { x: `${x}%`, y: `${y}%` };
}

function objectGlyph(cls) {
  return ({ connector: "◈", sleeve: "○", cable: "⌁", plate: "◉", cup: "◌", fork: "╋", spoon: "◡", knife: "╱", napkin: "▱", drawer: "▤", fixture: "□" })[cls] || "◆";
}

function isMisplaced(detection) {
  return typeof detection.misplaced === "boolean"
    ? detection.misplaced
    : detection.cls !== "fixture" && detection.zone !== detection.target_zone;
}

function isUncertain(detection) {
  const confidence = Number(detection.conf);
  return statusValue(detection.status) !== "LIVE" || !detection.target_zone || (Number.isFinite(confidence) && confidence < 0.5);
}

function updateObjectOptions(detections) {
  const list = $("#observed-object-options");
  if (!list) return;
  list.innerHTML = detections.map((detection) => `<option value="${escapeHtml(detection.object_id)}">${escapeHtml(label(detection.cls))} · ${escapeHtml(detection.object_id)}</option>`).join("");
  updateConstraintControl();
}

function renderObservation(data) {
  const observation = data.observation || {};
  const detections = Array.isArray(observation.detections) ? observation.detections : [];
  state.observation = observation;
  state.objectById = new Map(detections.map((item) => [item.object_id, item]));
  updateObjectOptions(detections);
  const misplaced = detections.filter(isMisplaced);
  const uncertain = detections.filter(isUncertain);
  const settled = detections.length - misplaced.length;
  const hasSignal = detections.length > 0;
  const reportedClear = observation.workspace_clear === true;
  const sceneLabel = !hasSignal ? "NO OBJECT SIGNAL" : reportedClear ? "WORKSPACE CLEAR" : `${misplaced.length} NEED ACTION`;
  const sceneTone = !hasSignal ? "warn" : reportedClear ? "success" : misplaced.length ? "warn" : "";
  setStatus("#scene-status", sceneLabel, sceneTone);
  setText("#scene-revision", `FRAME ${data.frame ?? observation.frame ?? "—"} / REV ${data.state_revision ?? "—"}`);
  setText("#scene-count", detections.length);
  setText("#misplaced-count", misplaced.length);
  setText("#settled-count", settled);
  setText("#workspace-clear", !hasSignal ? "NO SIGNAL" : reportedClear ? "CLEAR" : "ACTIVE");
  const average = detections.length ? detections.reduce((sum, item) => sum + (Number(item.conf) || 0), 0) / detections.length : 0;
  setText("#scene-confidence", detections.length ? `${Math.round(average * 100)}%${uncertain.length ? ` · ${uncertain.length} review` : ""}` : "—");
  setText("#scene-source-label", observation.raw_ref || "structured observation");
  setText("#scene-model-label", [observation.model, observation.detector, data.source || state.runtime.observer].filter(Boolean).map(label).join(" · ") || "Observer metadata unavailable");
  $(".scene-source")?.classList.toggle("live", Boolean(observation.raw_ref));

  const targets = [...new Set(detections.map((detection) => detection.target_zone).filter(Boolean))];
  setText("#target-a", targets[0] ? `TARGET / ${label(targets[0]).toUpperCase()}` : "TARGET / —");
  setText("#target-b", targets[1] ? `TARGET / ${label(targets[1]).toUpperCase()}` : "TARGET / —");
  $("#scene-objects").innerHTML = detections.map((detection, index) => {
    const position = objectPosition(detection, index, observation);
    const misplacedState = isMisplaced(detection);
    const uncertainState = isUncertain(detection);
    const confidence = Number.isFinite(Number(detection.conf)) ? `${Math.round(Number(detection.conf) * 100)}%` : "—";
    const tone = uncertainState ? "uncertain" : misplacedState ? "misplaced" : "settled";
    const target = detection.target_zone ? label(detection.target_zone) : "target unknown";
    return `<div class="scene-object ${tone}" style="--x:${position.x};--y:${position.y}" title="${escapeHtml(detection.object_id)} · ${escapeHtml(detection.zone || "unknown")} → ${escapeHtml(target)}">
      <span class="object-glyph" aria-hidden="true">${objectGlyph(detection.cls)}</span>
      <div><strong>${escapeHtml(detection.object_id)}</strong><small>${escapeHtml(label(detection.cls))} · ${confidence} · ${escapeHtml(statusValue(detection.status))}</small><small>${escapeHtml(label(detection.zone, "unknown"))} → ${escapeHtml(target)}</small></div>
    </div>`;
  }).join("") || `<div class="empty-stage">No object signal · not confirmation of a clear workspace</div>`;
}

function renderDecision(decision) {
  state.latestDecision = decision;
  const target = $("#decision-summary");
  if (!target) return;
  const rejected = decision.rejected && typeof decision.rejected === "object" ? Object.keys(decision.rejected).length : 0;
  const fallback = /fallback|unavailable|no valid/i.test(decision.reason || "");
  const selected = Array.isArray(decision.selected_ops) ? decision.selected_ops : [];
  const constraints = Array.isArray(decision.governing_constraints) ? decision.governing_constraints : [];
  target.innerHTML = `<div class="decision-heading"><span class="mono-label">PLAN DECISION · REV ${escapeHtml(decision.revision ?? "—")}</span><span class="decision-badge ${fallback ? "fallback" : "accepted"}">${fallback ? "FALLBACK USED" : "CORE ACCEPTED"}</span></div>
    <div class="decision-stats"><span><strong>${escapeHtml(decision.candidates_feasible ?? "—")}</strong> feasible / ${escapeHtml(decision.candidates_considered ?? "—")} considered</span><span><strong>${escapeHtml(selected.length)}</strong> selected ops</span><span><strong>${escapeHtml(rejected)}</strong> rejected</span><span>${escapeHtml(humanMode(decision.mode))}</span></div>
    <p class="decision-reason">${escapeHtml(decision.reason || "Selected the feasible graph for the observed world.")}</p>
    <div class="decision-chips">${constraints.map((item) => `<span>${escapeHtml(label(item))}</span>`).join("")}${decision.latency_ms ? `<span>${escapeHtml(Math.round(decision.latency_ms))} ms</span>` : ""}</div>`;
}

function renderGraph(graph) {
  state.graph = graph;
  const steps = graph?.steps || [];
  setText("#graph-revision", `REV ${graph?.revision ?? "—"}`);
  setStatus("#graph-state", steps.length ? "COMPILED" : "EMPTY", steps.length ? "success" : "warn");
  setText("#graph-step-count", `${steps.length} STEP${steps.length === 1 ? "" : "S"}`);
  $("#graph").innerHTML = steps.length ? steps.map((step, index) => {
    const stateValue = state.stepStates.get(step.id) || step.state || "pending";
    const isActive = state.currentStep === step.id || stateValue === "running";
    const target = step.args?.object ? humanObject(step.args.object) : step.args?.to ? `to ${label(step.args.to)}` : step.rationale || step.id;
    const contract = step.contract === "manipulate" ? "ACT" : label(step.contract).toUpperCase();
    const arm = step.arm ? `${label(step.arm)} arm` : step.device || "system";
    const deps = Array.isArray(step.deps) && step.deps.length ? `after ${step.deps.join(", ")}` : "root step";
    const classes = ["graph-step", isActive ? "active" : "", stateValue === "done" ? "done" : "", stateValue === "failed" || stateValue === "denied" ? "failed" : "", step.contract !== "manipulate" ? "system" : ""].filter(Boolean).join(" ");
    return `<div class="${classes}" data-step-id="${escapeHtml(step.id)}" title="${escapeHtml(step.rationale || deps)}">
      <div class="step-topline"><span class="step-number">${String(index + 1).padStart(2, "0")}</span><span class="step-contract">${escapeHtml(contract)}</span></div>
      <h3>${escapeHtml(step.op)}</h3><div class="step-target">${escapeHtml(target)}</div><div class="step-deps">${escapeHtml(deps)}</div>
      <div class="step-footer"><span>${escapeHtml(arm)}</span><span class="step-state">${escapeHtml(stateValue)}</span></div>
    </div>`;
  }).join("") : `<div class="graph-empty"><span class="empty-glyph">⌁</span><p>No executable steps were returned.</p><small>Review the planner decision and constraints.</small></div>`;
  updateProgress();
}

function deviceKeyFor(step) {
  if (step?.device) return step.device;
  if (step?.arm) return step.arm;
  if (step?.contract === "observe" || step?.contract === "verify") return "perception";
  return "reasoning";
}

function updateDevice(step, status) {
  const key = deviceKeyFor(step);
  const nodes = [...document.querySelectorAll("[data-device]")];
  const node = nodes.find((candidate) => candidate.dataset.device === key)
    || nodes.find((candidate) => candidate.dataset.arm === step?.arm)
    || nodes.find((candidate) => candidate.dataset.role === (step?.contract === "manipulate" ? "arm" : "reasoning"));
  if (!node) return;
  node.classList.toggle("active", status === "running");
  const stateNode = node.querySelector(".device-state");
  if (stateNode) stateNode.textContent = status === "ready" ? "READY" : status.toUpperCase();
}

function handleStepEvent(event) {
  const step = event.data || {};
  if (!step.id) return;
  if (event.kind === "step.started") {
    state.currentStep = step.id;
    state.stepStates.set(step.id, "running");
  } else if (event.kind === "step.finished") {
    state.currentStep = null;
    state.stepStates.set(step.id, step.state || "done");
    if (step.state === "done") state.completedSteps.add(step.id);
  } else if (event.kind === "step.denied") {
    state.currentStep = null;
    state.stepStates.set(step.id, "denied");
  }
  updateDevice(step, event.kind === "step.started" ? "running" : "ready");
  if (state.graph) renderGraph(state.graph);
  updatePhases();
  updateProgress();
}

function setReason(text) {
  setText("#current-reason", text);
}

function eventTone(kind, data = {}) {
  if (["run.finished", "step.finished"].includes(kind) || (kind === "verified" && data.ok)) return "ok";
  if (["graph.recompiled", "constraint.queued", "constraint.added", "mode.changed", "plan.decision", "world.perturbed"].includes(kind)) return "warn";
  if (["step.denied", "capability.lost", "placement.failed", "authorization.failed", "execution.interrupt.requested"].includes(kind) || (kind === "verified" && !data.ok)) return "danger";
  return "";
}

function constraintText(data) {
  const value = data.value && typeof data.value === "object" ? data.value.object_id || data.value.value : data.value;
  return `${label(data.kind)}${value ? ` = ${value}` : ""}`;
}

function renderConstraints() {
  const target = $("#active-constraints");
  if (!target) return;
  target.innerHTML = state.constraints.length
    ? state.constraints.map((constraint) => `<span class="constraint-chip"><span class="constraint-dot"></span>${escapeHtml(constraintText(constraint))}${constraint.justification ? `<small>${escapeHtml(constraint.justification)}</small>` : ""}</span>`).join("")
    : `<span class="constraint-empty">No live constraints</span>`;
}

function rememberConstraint(data) {
  const key = JSON.stringify([data.kind, data.value, data.justification]);
  if (!state.constraints.some((item) => JSON.stringify([item.kind, item.value, item.justification]) === key)) state.constraints.push(data);
  renderConstraints();
}

function updateConstraintControl() {
  const kind = $("#constraint-kind")?.value;
  $("#constraint-object-field")?.classList.toggle("visible", kind === "forbid_object");
}

function handleAuxiliaryEvent(event) {
  const data = event.data || {};
  if (event.kind.startsWith("voice.")) {
    const speaker = data.speaker_id || data.source || "voice";
    appendTrace(`${label(event.kind)} · ${speaker}${data.text ? `: “${data.text}”` : ""}.`, eventTone(event.kind, data), "voice");
    if (event.kind === "voice.intent.committed") setReason("A governed voice intent was committed to the same mission boundary.");
    return true;
  }
  if (event.kind.startsWith("residency.")) {
    const tier = data.tier || data.to || data.mode || "CORE_ONLY";
    setReason(`Reasoner residency changed to ${label(tier)}; the governed core remains in control.`);
    appendTrace(`Residency ${label(tier)}.`, "warn", "residency");
    return true;
  }
  if (event.kind === "world.perturbed") {
    const objectId = data.object_id || data.object;
    const zone = data.zone || data.to;
    setReason("The world changed independently; the next observation will determine whether a replan is required.");
    appendTrace(objectId ? `${humanObject(objectId)} moved to ${label(zone)}.` : "The world changed independently.", "warn", "world changed");
    return true;
  }
  return false;
}

function handleEvent(event) {
  if (!validEvent(event)) return;
  const data = event.data || {};
  state.events.push(event);
  const tone = eventTone(event.kind, data);
  if (event.kind === "run.started") {
    setConnection(`Session ${state.sessionId} is executing a governed mission.`, "running");
    setReason("Mission envelope accepted. OMNI-Q is observing before it commits a plan.");
    appendTrace(`Mission committed: “${data.goal || "operator objective"}”.`, "ok", "run started");
  } else if (event.kind === "observed") {
    renderObservation(data);
    const count = Array.isArray(data.misplaced) ? data.misplaced.length : (data.observation?.detections || []).filter(isMisplaced).length;
    appendTrace(`Observed ${count} object${count === 1 ? "" : "s"} requiring attention.`, tone, "observed");
  } else if (event.kind === "graph.compiled" || event.kind === "graph.recompiled") {
    renderGraph(data.graph);
    setReason(event.kind === "graph.recompiled" ? `The graph was rebuilt because ${data.reason || "the live state changed"}.` : "The objective is now a capability graph with a route for every step.");
    appendTrace(`${event.kind === "graph.recompiled" ? "Recompiled" : "Compiled"} ${data.graph?.steps?.length || 0}-step graph · ${data.reason || "initial objective decomposition"}.`, tone, event.kind.replace(".", " "));
  } else if (event.kind === "plan.decision") {
    renderDecision(data);
    const constraints = (data.governing_constraints || []).join(", ");
    setReason(data.reason || (constraints ? `Plan constrained by ${constraints}.` : "Selected the feasible graph for the live world state."));
    appendTrace(data.reason || `Selected ${data.selected_ops?.length || 0} executable capabilities.`, tone, "plan decision");
  } else if (event.kind === "constraint.queued") {
    rememberConstraint(data);
    appendTrace(`Operator queued ${constraintText(data)}.`, tone, "constraint queued");
  } else if (event.kind === "constraint.added") {
    rememberConstraint(data);
    setReason(`Constraint applied: ${constraintText(data)}.`);
    appendTrace(`Applied ${constraintText(data)} · ${data.justification || "operator instruction"}.`, tone, "constraint added");
  } else if (event.kind === "step.authorized") {
    const authorization = data.authorization || {};
    appendTrace(`${authorization.verdict || "ALLOW"} · ${authorization.op || "step"} · ${authorization.reason || "mission envelope"}.`, authorization.verdict === "ALLOW" ? "ok" : "danger", "authorization");
  } else if (["step.started", "step.finished", "step.denied"].includes(event.kind)) {
    handleStepEvent(event);
    const target = data.args?.object ? humanObject(data.args.object) : data.args?.to ? `to ${label(data.args.to)}` : data.id;
    if (event.kind === "step.started") setReason(`${data.op} is executing on ${data.device || data.arm || "the active capability"} · ${target}.`);
    if (event.kind === "step.finished") appendTrace(`${data.op} ${data.state === "done" ? "completed" : `returned ${data.state}`} · ${target}.`, data.state === "done" ? "ok" : "danger", "step finished");
  } else if (event.kind === "verified") {
    if (data.ok && data.step) state.verifiedSteps.add(data.step);
    setReason(data.ok ? "The observed state matches the expected postcondition." : `Verification found ${data.mismatch?.join(", ") || "a mismatch"}; replanning.`);
    appendTrace(data.ok ? `Verification passed for ${data.step}.` : `Verification mismatch for ${data.step} · ${(data.mismatch || []).join(", ")}.`, data.ok ? "ok" : "danger", "verified");
  } else if (event.kind === "mode.changed") {
    setText("#metric-mode", humanMode(data.to));
    setReason(`Autonomy tightened from ${humanMode(data.frm)} to ${humanMode(data.to)}; the run stays inside a narrower envelope.`);
    appendTrace(`Autonomy mode tightened ${humanMode(data.frm)} → ${humanMode(data.to)}.`, "warn", "mode changed");
  } else if (["capability.lost", "placement.failed", "authorization.failed"].includes(event.kind)) {
    setReason(event.kind === "capability.lost" ? `${data.op || "A capability"} is unavailable, so OMNI-Q is looking for another route.` : data.error || "The current route could not be authorized.");
    appendTrace(data.error || `${event.kind.replace(".", " ")} · replanning from the current state.`, "danger", event.kind.replace(".", " "));
  } else if (event.kind === "run.finished") {
    finishRun(data);
  } else if (!handleAuxiliaryEvent(event)) {
    appendTrace(`Received ${event.kind}.`, tone, "event");
  }
  updatePhases();
  updateProgress();
}

function renderReceipt(receipt) {
  if (!receipt) return;
  state.receipt = receipt;
  const metrics = receipt.metrics || {};
  state.hash = receipt.content_hash || receipt.hashes?.content || state.hash;
  const resolved = metrics.resolved === true;
  const review = metrics.resolved === false;
  setStatus("#receipt-state", resolved ? "FINALIZED" : review ? "REVIEW" : "PENDING", resolved ? "success" : review ? "danger" : "warn");
  setText("#receipt-title", resolved ? "Mission verified" : review ? "Mission needs review" : "Mission receipt available");
  setText("#receipt-copy", resolved ? "The final state passed verification and the receipt was sealed." : "The run ended without a fully resolved workspace; inspect the evidence.");
  $("#receipt-icon").className = `receipt-icon ${resolved ? "success" : review ? "failed" : ""}`.trim();
  $("#receipt-icon").textContent = resolved ? "✓" : review ? "!" : "◌";
  setText("#receipt-hash", shortHash(state.hash));
  $("#copy-hash").disabled = !state.hash;
  setText("#metric-revisions", metrics.revisions ?? "—");
  setText("#metric-actions", metrics.steps_executed ?? "—");
  setText("#metric-resolved", metrics.resolved === undefined ? "—" : metrics.resolved ? "YES" : "NO");
  setText("#metric-duration", metrics.wall_seconds === undefined ? "—" : `${metrics.wall_seconds}s`);
  setText("#metric-rejected", Array.isArray(receipt.rejected) ? receipt.rejected.length : "—");
  setText("#metric-mode", humanMode(metrics.mode));
  setText("#receipt-run-id", `RUN ${receipt.run_id || "—"}`);
  setText("#receipt-parent", `PARENT ${shortHash(receipt.parent_hash || "GENESIS")}`);
  const provenance = receipt.provenance || {};
  setText("#receipt-provenance", `PROVENANCE ${provenance.component || provenance.source || "RECORDED"}`);
  updateProgress();
}

function finishRun(data) {
  state.status = "finished";
  const metrics = data.metrics || {};
  state.hash = data.content_hash || state.hash;
  setConnection(metrics.resolved ? "Mission finished. The workspace is resolved." : "Mission finished with unresolved state; review the audit trail.", metrics.resolved ? "running" : "failed");
  setText("#run-clock", "COMPLETE");
  $(".live-clock")?.classList.remove("running");
  setReason(metrics.resolved ? "Closed-loop verification passed. This mission is complete." : "The loop closed, but the final state needs operator review.");
  appendTrace(metrics.resolved ? "Receipt finalized · mission resolved." : "Receipt finalized · unresolved state recorded.", metrics.resolved ? "ok" : "danger", "run finished");
  renderReceipt({ metrics, content_hash: state.hash, run_id: data.run_id, parent_hash: data.parent_hash, provenance: data.provenance });
  void loadReceipt();
}

function validEvent(event) {
  if (!Number.isInteger(event.seq) || event.seq <= state.cursor) return false;
  const runKey = event.run_id || "global";
  const expectedParent = state.parents.get(runKey) || null;
  if (event.parent_id !== expectedParent) {
    appendTrace(`Rejected event ${event.event_id || event.seq}: causal parent mismatch.`, "danger", "stream guard");
    return false;
  }
  state.cursor = event.seq;
  state.parents.set(runKey, event.event_id);
  setText(".connection-seq", `EVENT STREAM / ${state.cursor.toString().padStart(3, "0")}`);
  return true;
}

function constraintPayload() {
  const kind = $("#constraint-kind")?.value;
  if (!kind) return null;
  const value = kind === "forbid_object" ? $("#constraint-object")?.value.trim() : kind === "prefer_arm" ? "left" : null;
  if (kind === "forbid_object" && !value) throw new Error("Choose an observed object before protecting it.");
  const justification = $("#justification")?.value.trim();
  if (!justification) throw new Error("Every constraint needs a short justification.");
  return { kind, value, justification };
}

function attachStreamHandlers(source) {
  source.onopen = () => {
    state.streamOpen = true;
    state.reconnectAttempts = 0;
    $("#reconnect").hidden = true;
    if (state.status === "running") setConnection(`Session ${state.sessionId} is connected. Listening for governed execution events.`, "running");
  };
  source.onmessage = (message) => {
    try {
      handleEvent(JSON.parse(message.data));
    } catch {
      appendTrace("Malformed event payload rejected by the stream guard.", "danger", "stream guard");
    }
  };
  source.addEventListener("terminal", (message) => {
    const terminal = JSON.parse(message.data);
    if (terminal.status === "failed") {
      state.status = "failed";
      setConnection(`Mission failed: ${terminal.error || "unknown session error"}.`, "failed");
      appendTrace(terminal.error || "Session failed before a receipt could be finalized.", "danger", "session");
    } else if (state.status === "running") {
      state.status = "finished";
      setConnection("Event stream closed after the terminal receipt.", "running");
      $(".live-clock")?.classList.remove("running");
    }
    source.close();
    state.eventSource = null;
    state.streamOpen = false;
    $("#start").disabled = false;
    $("#apply-constraint").disabled = true;
    $("#reconnect").hidden = true;
    updatePhases();
    void loadReceipt();
  });
  source.onerror = () => {
    if (state.status !== "running") return;
    state.streamOpen = false;
    state.reconnectAttempts += 1;
    $("#reconnect").hidden = false;
    setConnection(`Event stream interrupted; reconnect attempt ${state.reconnectAttempts}.`, "reconnecting");
  };
}

function connectStream(cursor = state.cursor) {
  if (!state.sessionId) return;
  if (state.eventSource) state.eventSource.close();
  const source = new EventSource(`/sessions/${encodeURIComponent(state.sessionId)}/events?cursor=${cursor}`);
  state.eventSource = source;
  attachStreamHandlers(source);
}

async function loadReceipt() {
  if (!state.sessionId) return;
  try {
    const response = await fetch(`/sessions/${encodeURIComponent(state.sessionId)}/receipt`);
    if (!response.ok) return;
    renderReceipt(await response.json());
  } catch {
    appendTrace("Receipt details could not be refreshed; the event summary remains available.", "warn", "receipt");
  }
}

async function start(event) {
  event?.preventDefault();
  const button = $("#start");
  button.disabled = true;
  let constraint;
  try {
    constraint = constraintPayload();
  } catch (error) {
    appendTrace(error.message, "danger", "operator");
    button.disabled = false;
    return;
  }
  if (state.eventSource) state.eventSource.close();
  resetView();
  try {
    const response = await fetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal: $("#goal").value.trim(), constraints: constraint ? [constraint] : [] }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "session failed to start");
    state.sessionId = payload.session_id;
    setText("#session-label", state.sessionId);
    setRuntime(payload.runtime || {});
    setMode(payload.mode || state.runtime.mode, state.runtime);
    setConnection(`Session ${state.sessionId} started. Listening for governed execution events.`, "running");
    connectStream(0);
  } catch (error) {
    state.status = "failed";
    setConnection(`Mission was not started: ${error.message}`, "failed");
    appendTrace(`Start rejected: ${error.message}`, "danger", "start");
    button.disabled = false;
    $("#apply-constraint").disabled = true;
  }
}

async function applyConstraint() {
  if (!state.sessionId || state.status !== "running") return;
  let constraint;
  try {
    constraint = constraintPayload();
  } catch (error) {
    appendTrace(error.message, "danger", "operator");
    return;
  }
  const button = $("#apply-constraint");
  button.disabled = true;
  try {
    const response = await fetch(`/sessions/${encodeURIComponent(state.sessionId)}/constraints`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(constraint),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "constraint rejected");
    rememberConstraint(constraint);
    if (payload.runtime) setRuntime(payload.runtime);
    appendTrace(`Live constraint accepted: ${constraintText(constraint)}. The next loop will recompile.`, "warn", "operator");
  } catch (error) {
    appendTrace(error.message, "danger", "operator");
  } finally {
    button.disabled = state.status !== "running";
  }
}

function reconnect() {
  if (!state.sessionId || state.status !== "running") return;
  setConnection(`Replaying events after sequence ${state.cursor}.`, "reconnecting");
  connectStream(state.cursor);
}

async function copyHash() {
  if (!state.hash) return;
  try {
    await navigator.clipboard.writeText(state.hash);
    appendTrace("Receipt content hash copied to clipboard.", "ok", "receipt");
  } catch {
    appendTrace("Clipboard access was unavailable; the hash remains visible above.", "warn", "receipt");
  }
}

$("#mission-form").addEventListener("submit", start);
$("#apply-constraint").addEventListener("click", applyConstraint);
$("#reconnect").addEventListener("click", reconnect);
$("#copy-hash").addEventListener("click", copyHash);
$("#constraint-kind").addEventListener("change", updateConstraintControl);
setRuntime(runtimeDefaults());
setInterval(updateClock, 1000);
