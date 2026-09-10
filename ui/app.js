const $ = (selector) => document.querySelector(selector);

const state = {
  cursor: 0, eventSource: null, sessionId: null, parents: new Map(), events: [], graph: null,
  status: "standby", startedAt: null, objectById: new Map(), currentStep: null,
  completedSteps: new Set(), stepStates: new Map(), hash: "",
};
const phaseOrder = ["observe", "plan", "act", "verify"];

function setText(selector, value) { const node = $(selector); if (node) node.textContent = value ?? "—"; }
function escapeHtml(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
function shortHash(value) { if (!value) return "waiting"; const raw = String(value); return raw.length > 27 ? `${raw.slice(0, 15)}…${raw.slice(-8)}` : raw; }
function nowLabel(ts = Date.now()) { return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(ts); }
function humanObject(objectId) { const item = state.objectById.get(objectId); return item ? `${item.cls} / ${objectId}` : objectId || "workspace"; }

function setConnection(text, tone = "") {
  const bar = $("#connection"); bar.className = `connection-bar ${tone}`.trim();
  bar.querySelector("span:nth-child(2)").textContent = text;
  setText(".connection-seq", state.sessionId ? `EVENT STREAM / ${state.cursor.toString().padStart(3, "0")}` : "EVENT STREAM / READY");
}

function appendTrace(text, tone = "", kind = "event") {
  const list = $("#trace"); list.querySelector(".trace-empty")?.remove();
  const item = document.createElement("div"); item.className = `trace-item ${tone}`.trim();
  item.innerHTML = `<span class="trace-mark"></span><div class="trace-copy"><strong>${escapeHtml(kind.toUpperCase())}</strong> · ${escapeHtml(text)}</div><time class="trace-time">${nowLabel()}</time>`;
  list.prepend(item); while (list.children.length > 35) list.lastElementChild.remove();
  setText("#event-count", `${state.events.length} EVENT${state.events.length === 1 ? "" : "S"}`);
}

function setMode(mode) {
  const badge = $("#mode"); const value = String(mode || "mock").replaceAll("_", " ").toUpperCase();
  badge.innerHTML = `<span class="mode-dot"></span> ${escapeHtml(value === "MOCK" ? "MOCK MODE — NOT HARDWARE" : `${value} MODE`)}`;
  badge.className = `mode-badge ${value === "MOCK" ? "mock" : ""}`;
}

function resetView() {
  state.cursor = 0; state.parents.clear(); state.events = []; state.graph = null; state.status = "running"; state.startedAt = Date.now();
  state.objectById.clear(); state.currentStep = null; state.completedSteps.clear(); state.stepStates.clear(); state.hash = "";
  $("#trace").innerHTML = `<div class="trace-empty"><span>◌</span><p>Listening for governed execution events…</p></div>`;
  $("#graph").innerHTML = `<div class="graph-empty"><span class="empty-glyph">⌁</span><p>Compiling the first graph…</p><small>Function is separate from placement.</small></div>`;
  $("#scene-objects").innerHTML = `<div class="empty-stage">Waiting for perception</div>`;
  setText("#graph-state", "COMPILING"); setText("#scene-status", "CONNECTING"); setText("#scene-revision", "FRAME —");
  setText("#scene-count", "—"); setText("#misplaced-count", "—"); setText("#settled-count", "—"); setText("#workspace-clear", "—"); setText("#scene-source-label", "Connecting to observer");
  $(".scene-source").classList.remove("live"); setText("#receipt-state", "PENDING"); setText("#receipt-title", "Mission in progress"); setText("#receipt-copy", "Waiting for final verification and a chained receipt.");
  $("#receipt-icon").className = "receipt-icon"; $("#receipt-icon").textContent = "◌"; setText("#receipt-hash", "waiting"); $("#copy-hash").disabled = true;
  setText("#metric-revisions", "—"); setText("#metric-actions", "—"); setText("#metric-mode", "NOMINAL"); setText("#run-clock", "00:00"); $(".live-clock").classList.add("running"); $("#apply-constraint").disabled = false;
  setReason("Waiting for the first observation."); updatePhases(); updateProgress();
}

function updateClock() {
  if (!state.startedAt || state.status !== "running") return;
  const seconds = Math.floor((Date.now() - state.startedAt) / 1000); setText("#run-clock", `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`);
}

function updatePhases() {
  const activePhase = state.currentStep ? "act" : state.status === "finished" ? "verify" : state.graph ? "plan" : "observe";
  const activeIndex = phaseOrder.indexOf(activePhase);
  phaseOrder.forEach((phase, index) => {
    const row = document.querySelector(`[data-phase="${phase}"]`); const stateNode = row.querySelector(".phase-state"); row.classList.remove("active", "done", "warn");
    if (state.status === "failed" && phase === "verify") { row.classList.add("warn"); stateNode.textContent = "attention"; }
    else if (index < activeIndex || (state.status === "finished" && phase !== "verify")) { row.classList.add("done"); stateNode.textContent = "done"; }
    else if (phase === activePhase) { row.classList.add("active"); stateNode.textContent = state.status === "finished" ? "done" : "live"; }
    else stateNode.textContent = "queued";
  });
}

function updateProgress() {
  const total = state.graph?.steps?.length || 0; const done = state.completedSteps.size; const percent = state.status === "finished" ? 100 : total ? Math.min(96, Math.round((done / total) * 100)) : 0;
  $("#progress-bar").style.width = `${percent}%`; setText("#progress-label", `${percent}% COMPLETE`);
}

function objectPosition(detection, index) {
  const positions = { A: [18, 28], B: [72, 72], tray: [37, 71], tray_plate: [37, 71], tray_cup: [69, 28], tray_napkin: [22, 68], bin: [84, 22], setting_1: [84, 68], drawer: [50, 19], closed: [50, 19], floor: [72, 86] };
  const [x, y] = positions[detection.zone] || [26 + ((index * 23) % 53), 26 + ((index * 37) % 53)]; return { x: `${x}%`, y: `${y}%` };
}
function objectGlyph(cls) { return ({ connector: "◈", sleeve: "○", cable: "⌁", plate: "◉", cup: "◌", fork: "╋", spoon: "◡", napkin: "▱", fixture: "□" })[cls] || "◆"; }

function renderObservation(data) {
  const observation = data.observation || {}; const detections = observation.detections || []; state.objectById = new Map(detections.map((item) => [item.object_id, item]));
  const misplaced = detections.filter((item) => item.zone !== item.target_zone && item.cls !== "fixture"); const settled = detections.length - misplaced.length;
  setText("#scene-status", observation.workspace_clear ? "WORKSPACE CLEAR" : `${misplaced.length} NEED ACTION`); setText("#scene-revision", `FRAME ${data.frame ?? observation.frame ?? "—"} / REV ${data.state_revision ?? "—"}`);
  setText("#scene-count", detections.length); setText("#misplaced-count", misplaced.length); setText("#settled-count", settled); setText("#workspace-clear", observation.workspace_clear ? "CLEAR" : "ACTIVE"); setText("#scene-source-label", observation.raw_ref || "live structured observation"); $(".scene-source").classList.add("live");
  $("#scene-objects").innerHTML = detections.map((detection, index) => {
    const position = objectPosition(detection, index); const isMisplaced = detection.zone !== detection.target_zone && detection.cls !== "fixture";
    return `<div class="scene-object ${isMisplaced ? "misplaced" : "settled"}" style="--x:${position.x};--y:${position.y}" title="${escapeHtml(detection.object_id)} · ${escapeHtml(detection.zone)} → ${escapeHtml(detection.target_zone)}"><span class="object-glyph">${objectGlyph(detection.cls)}</span><div><strong>${escapeHtml(detection.object_id)}</strong><small>${escapeHtml(detection.zone)} → ${escapeHtml(detection.target_zone)}</small></div></div>`;
  }).join("") || `<div class="empty-stage">No detections in frame</div>`;
}

function renderGraph(graph) {
  state.graph = graph; const steps = graph?.steps || []; setText("#graph-revision", `REV ${graph?.revision ?? "—"}`); setText("#graph-state", steps.length ? "COMPILED" : "EMPTY"); setText("#graph-step-count", `${steps.length} STEP${steps.length === 1 ? "" : "S"}`);
  $("#graph").innerHTML = steps.length ? steps.map((step, index) => {
    const stateValue = state.stepStates.get(step.id) || step.state || "pending"; const isActive = state.currentStep === step.id || stateValue === "running"; const target = step.args?.object ? humanObject(step.args.object) : step.args?.to ? `to ${step.args.to}` : step.rationale || step.id; const contract = step.contract === "manipulate" ? "ACT" : step.contract.toUpperCase(); const arm = step.arm ? `${step.arm} arm` : step.device || "system";
    return `<div class="graph-step ${isActive ? "active" : ""} ${stateValue === "done" ? "done" : ""} ${stateValue === "failed" || stateValue === "denied" ? "failed" : ""} ${step.contract !== "manipulate" ? "system" : ""}" data-step-id="${escapeHtml(step.id)}"><div class="step-topline"><span class="step-number">${String(index + 1).padStart(2, "0")}</span><span class="step-contract">${contract}</span></div><h3>${escapeHtml(step.op)}</h3><div class="step-target">${escapeHtml(target)}</div><div class="step-footer"><span>${escapeHtml(arm)}</span><span class="step-state">${escapeHtml(stateValue)}</span></div></div>`;
  }).join("") : `<div class="graph-empty"><span class="empty-glyph">⌁</span><p>No executable steps were returned.</p></div>`;
  updateProgress();
}

function deviceKeyFor(step) { if (step?.device?.includes("perception")) return "perception"; if (step?.device?.includes("left") || step?.arm === "left") return "left"; if (step?.device?.includes("right") || step?.arm === "right") return "right"; return "reasoning"; }
function updateDevice(step, status) { const node = document.querySelector(`[data-device="${deviceKeyFor(step)}"]`); if (!node) return; node.classList.toggle("active", status === "running"); const stateNode = node.querySelector(".device-state"); if (stateNode && status !== "ready") stateNode.textContent = status.toUpperCase(); }

function handleStepEvent(event) {
  const step = event.data || {}; if (!step.id) return;
  if (event.kind === "step.started") { state.currentStep = step.id; state.stepStates.set(step.id, "running"); }
  else if (event.kind === "step.finished") { state.currentStep = null; state.stepStates.set(step.id, step.state || "done"); if (step.state === "done") state.completedSteps.add(step.id); }
  else if (event.kind === "step.denied") { state.currentStep = null; state.stepStates.set(step.id, "denied"); }
  updateDevice(step, event.kind === "step.started" ? "running" : "ready"); if (state.graph) renderGraph(state.graph); updatePhases(); updateProgress();
}

function setReason(text) { setText("#current-reason", text); }
function eventTone(kind) { if (["verified", "run.finished", "step.finished"].includes(kind)) return "ok"; if (["graph.recompiled", "constraint.queued", "constraint.added", "mode.changed", "plan.decision"].includes(kind)) return "warn"; if (["step.denied", "capability.lost", "placement.failed", "authorization.failed"].includes(kind)) return "danger"; return ""; }

function handleEvent(event) {
  if (!validEvent(event)) return; const data = event.data || {}; state.events.push(event); const tone = eventTone(event.kind);
  if (event.kind === "run.started") { setConnection(`Session ${state.sessionId} is executing a governed mission.`, "running"); setReason("Mission envelope accepted. Omni Q is observing before it commits a plan."); appendTrace(`Mission committed: “${data.goal || "operator objective"}”.`, "ok", "run started"); }
  else if (event.kind === "observed") { renderObservation(data); appendTrace(`Observed ${data.misplaced?.length || 0} object${data.misplaced?.length === 1 ? "" : "s"} requiring attention.`, tone, "observed"); }
  else if (event.kind === "graph.compiled" || event.kind === "graph.recompiled") { renderGraph(data.graph); setReason(event.kind === "graph.recompiled" ? `The graph was rebuilt because ${data.reason || "the live state changed"}.` : "The objective is now a capability graph with a route for every step."); appendTrace(`${event.kind === "graph.recompiled" ? "Recompiled" : "Compiled"} ${data.graph?.steps?.length || 0}-step graph · ${data.reason || "initial objective decomposition"}.`, tone, event.kind.replace(".", " ")); }
  else if (event.kind === "plan.decision") { const constraints = (data.governing_constraints || []).join(", "); setReason(data.reason || (constraints ? `Plan constrained by ${constraints}.` : "Selected the feasible graph for the live world state.")); appendTrace(data.reason || `Selected ${data.selected_ops?.length || 0} executable capabilities.`, tone, "plan decision"); }
  else if (event.kind === "constraint.queued") appendTrace(`Operator queued ${data.kind}${data.value ? ` = ${data.value}` : ""}.`, tone, "constraint queued");
  else if (event.kind === "constraint.added") { setReason(`Constraint applied: ${data.kind}${data.value ? ` = ${data.value}` : ""}.`); appendTrace(`Applied ${data.kind}${data.value ? ` = ${data.value}` : ""} · ${data.justification || "operator instruction"}.`, tone, "constraint added"); }
  else if (event.kind === "step.authorized") { const authorization = data.authorization || {}; appendTrace(`${authorization.verdict || "ALLOW"} · ${authorization.op || "step"} · ${authorization.reason || "mission envelope"}.`, authorization.verdict === "ALLOW" ? "ok" : "danger", "authorization"); }
  else if (["step.started", "step.finished", "step.denied"].includes(event.kind)) { handleStepEvent(event); const target = data.args?.object ? humanObject(data.args.object) : data.args?.to ? `to ${data.args.to}` : data.id; if (event.kind === "step.started") setReason(`${data.op} is executing on ${data.device || data.arm || "the active capability"} · ${target}.`); if (event.kind === "step.finished") appendTrace(`${data.op} ${data.state === "done" ? "completed" : `returned ${data.state}`} · ${target}.`, data.state === "done" ? "ok" : "danger", "step finished"); }
  else if (event.kind === "verified") { setReason(data.ok ? "The observed state matches the expected postcondition." : `Verification found ${data.mismatch?.join(", ") || "a mismatch"}; replanning.`); appendTrace(data.ok ? `Verification passed for ${data.step}.` : `Verification mismatch for ${data.step} · ${(data.mismatch || []).join(", ")}.`, data.ok ? "ok" : "danger", "verified"); }
  else if (event.kind === "mode.changed") { setText("#metric-mode", data.to || "DEGRADED"); setReason(`Autonomy tightened from ${data.frm} to ${data.to}; the run stays inside a narrower envelope.`); appendTrace(`Autonomy mode tightened ${data.frm} → ${data.to}.`, "warn", "mode changed"); }
  else if (["capability.lost", "placement.failed", "authorization.failed"].includes(event.kind)) { setReason(event.kind === "capability.lost" ? `${data.op} is unavailable, so Omni Q is looking for another route.` : data.error || "The current route could not be authorized."); appendTrace(data.error || `${event.kind.replace(".", " ")} · replanning from the current state.`, "danger", event.kind.replace(".", " ")); }
  else if (event.kind === "run.finished") finishRun(data);
  updatePhases(); updateProgress();
}

function finishRun(data) {
  state.status = "finished"; const metrics = data.metrics || {}; state.hash = data.content_hash || ""; setConnection(metrics.resolved ? "Mission finished. The workspace is resolved." : "Mission finished with unresolved state; review the audit trail.", metrics.resolved ? "running" : "failed"); setText("#run-clock", "COMPLETE"); $(".live-clock").classList.remove("running");
  setText("#receipt-state", metrics.resolved ? "FINALIZED" : "REVIEW"); setText("#receipt-title", metrics.resolved ? "Mission verified" : "Mission needs review"); setText("#receipt-copy", metrics.resolved ? "The final state passed verification and the receipt was sealed." : "The run ended without a fully resolved workspace."); $("#receipt-icon").className = `receipt-icon ${metrics.resolved ? "success" : "failed"}`; $("#receipt-icon").textContent = metrics.resolved ? "✓" : "!";
  setText("#receipt-hash", shortHash(state.hash)); $("#copy-hash").disabled = !state.hash; setText("#metric-revisions", metrics.revisions ?? "—"); setText("#metric-actions", metrics.steps_executed ?? "—"); setText("#metric-mode", metrics.mode || "NOMINAL"); setReason(metrics.resolved ? "Closed-loop verification passed. This mission is complete." : "The loop closed, but the final state needs operator review."); appendTrace(metrics.resolved ? "Receipt finalized · mission resolved." : "Receipt finalized · unresolved state recorded.", metrics.resolved ? "ok" : "danger", "run finished");
}

function validEvent(event) {
  if (!Number.isInteger(event.seq) || event.seq <= state.cursor) return false; const runKey = event.run_id || "global"; const expectedParent = state.parents.get(runKey) || null;
  if (event.parent_id !== expectedParent) { appendTrace(`Rejected event ${event.event_id || event.seq}: causal parent mismatch.`, "danger", "stream guard"); return false; }
  state.cursor = event.seq; state.parents.set(runKey, event.event_id); return true;
}

function constraintPayload() { const kind = $("#constraint-kind").value; if (!kind) return null; return { kind, value: { forbid_object: "connector_2", prefer_arm: "left", keep_local: null }[kind], justification: $("#justification").value.trim() }; }

async function start(event) {
  event?.preventDefault(); const button = $("#start"); button.disabled = true; if (state.eventSource) state.eventSource.close(); resetView(); const constraint = constraintPayload();
  try {
    const response = await fetch("/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ goal: $("#goal").value, constraints: constraint ? [constraint] : [] }) }); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "session failed to start");
    state.sessionId = payload.session_id; setText("#session-label", state.sessionId); setMode(payload.mode || "mock"); setConnection(`Session ${state.sessionId} started. Listening for governed execution events.`, "running");
    const source = new EventSource(`/sessions/${state.sessionId}/events?cursor=0`); state.eventSource = source;
    source.onmessage = (message) => { try { handleEvent(JSON.parse(message.data)); } catch { appendTrace("Malformed event payload rejected by the stream guard.", "danger", "stream guard"); } };
    source.addEventListener("terminal", (message) => { const terminal = JSON.parse(message.data); if (terminal.status === "failed") { state.status = "failed"; setConnection(`Mission failed: ${terminal.error || "unknown session error"}.`, "failed"); appendTrace(terminal.error || "Session failed before a receipt could be finalized.", "danger", "session"); } source.close(); button.disabled = false; $("#apply-constraint").disabled = true; updatePhases(); });
    source.onerror = () => { if (state.status === "running") setConnection("Event stream interrupted; the session remains available for reconnect.", "failed"); };
  } catch (error) { state.status = "failed"; setConnection(`Mission was not started: ${error.message}`, "failed"); appendTrace(`Start rejected: ${error.message}`, "danger", "start"); button.disabled = false; $("#apply-constraint").disabled = true; }
}

async function applyConstraint() {
  if (!state.sessionId) return; const constraint = constraintPayload(); if (!constraint) { appendTrace("Choose a live constraint before applying it.", "warn", "operator"); return; }
  const button = $("#apply-constraint"); button.disabled = true;
  try { const response = await fetch(`/sessions/${state.sessionId}/constraints`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(constraint) }); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "constraint rejected"); appendTrace(`Live constraint accepted: ${constraint.kind}. The next loop will recompile.`, "warn", "operator"); }
  catch (error) { appendTrace(error.message, "danger", "operator"); } finally { button.disabled = state.status !== "running"; }
}
async function copyHash() { if (!state.hash) return; try { await navigator.clipboard.writeText(state.hash); appendTrace("Receipt content hash copied to clipboard.", "ok", "receipt"); } catch { appendTrace("Clipboard access was unavailable; the hash remains visible above.", "warn", "receipt"); } }

$("#mission-form").addEventListener("submit", start); $("#apply-constraint").addEventListener("click", applyConstraint); $("#copy-hash").addEventListener("click", copyHash); setInterval(updateClock, 1000);
