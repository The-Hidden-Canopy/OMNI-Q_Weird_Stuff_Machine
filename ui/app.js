const $ = (selector) => document.querySelector(selector);
const state = { cursor: 0, eventSource: null, sessionId: null, parents: new Map() };

function setText(selector, text) { $(selector).textContent = text; }
function fact(label, value) {
  const target = [...$("#verification").querySelectorAll("div")]
    .find((node) => node.querySelector("dt").textContent === label);
  if (target) target.querySelector("dd").textContent = value;
}
function deviceFact(label, value) {
  const target = [...$("#devices").querySelectorAll("div")]
    .find((node) => node.querySelector("dt").textContent === label);
  if (target) target.querySelector("dd").textContent = value;
}
function appendTrace(text, tone = "") {
  const list = $("#trace");
  if (list.children.length === 1 && list.firstElementChild.textContent.startsWith("Every decision")) list.innerHTML = "";
  const item = document.createElement("li");
  item.textContent = text;
  item.className = tone;
  list.prepend(item);
}
function renderList(selector, rows, render) {
  const list = $(selector);
  list.innerHTML = "";
  rows.forEach((row) => { const item = document.createElement("li"); item.textContent = render(row); list.append(item); });
}

function validEvent(event) {
  if (!Number.isInteger(event.seq) || event.seq <= state.cursor) return false;
  const expectedParent = state.parents.get(event.run_id || "global") || null;
  if (event.parent_id !== expectedParent) {
    appendTrace(`Rejected event ${event.event_id}: causal parent mismatch.`, "denied");
    return false;
  }
  state.cursor = event.seq;
  state.parents.set(event.run_id || "global", event.event_id);
  return true;
}

function handleEvent(event) {
  if (!validEvent(event)) return;
  const data = event.data || {};
  if (event.kind === "run.started") {
    setText("#connection", `Session ${state.sessionId}: executing governed ${event.run_id || "mock"} run.`);
  } else if (event.kind === "observed") {
    setText("#scene-revision", `frame ${data.frame} / state ${event.state_revision ?? "—"}`);
    appendTrace(`Observed ${data.misplaced?.length || 0} misplaced object(s).`);
  } else if (event.kind === "graph.compiled" || event.kind === "graph.recompiled") {
    const graph = data.graph;
    setText("#graph-revision", `rev ${graph.revision}`);
    renderList("#graph", graph.steps, (step) => `${step.op} — ${step.rationale || step.id}`);
    appendTrace(`${event.kind === "graph.recompiled" ? "Recompiled" : "Compiled"} graph: ${data.reason || "initial plan"}.`, "warn");
  } else if (event.kind === "step.authorized") {
    const auth = data.authorization;
    appendTrace(`${auth.verdict}: ${auth.op} (${auth.step_id}) — ${auth.reason}`, auth.verdict === "ALLOW" ? "ok" : "denied");
  } else if (event.kind === "step.started" || event.kind === "step.finished") {
    const step = data;
    deviceFact("CURRENT ACTION", `${step.op} @ ${step.device || "unplaced"} (${step.state})`);
    if (step.device?.includes("left")) deviceFact("LEFT ARM", step.state);
    if (step.device?.includes("right")) deviceFact("RIGHT ARM", step.state);
  } else if (event.kind === "verified") {
    fact("VERDICT", data.ok ? "verified" : `mismatch: ${(data.mismatch || []).join(", ")}`);
    appendTrace(`Verification ${data.ok ? "passed" : "failed"} for ${data.step}.`, data.ok ? "ok" : "denied");
  } else if (event.kind === "mode.changed") {
    fact("AUTONOMY MODE", data.to);
    appendTrace(`Autonomy mode tightened: ${data.frm} → ${data.to}.`, "warn");
  } else if (event.kind === "run.finished") {
    fact("RECEIPT", data.content_hash || "finalized");
    fact("AUTONOMY MODE", data.metrics?.mode || "NOMINAL");
    setText("#connection", `Run finished. ${data.metrics?.resolved ? "Layout resolved." : "Not resolved; see audit trace."}`);
  }
}

async function start() {
  const button = $("#start");
  button.disabled = true;
  state.cursor = 0; state.parents.clear();
  if (state.eventSource) state.eventSource.close();
  const kind = $("#constraint-kind").value;
  const constraints = kind ? [{
    kind,
    value: kind === "forbid_object" ? "connector_2" : null,
    justification: $("#justification").value,
  }] : [];
  try {
    const response = await fetch("/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ goal: $("#goal").value, constraints }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "session failed to start");
    state.sessionId = payload.session_id;
    setText("#connection", `Session ${state.sessionId} started in explicitly labelled mock mode.`);
    const source = new EventSource(`/sessions/${state.sessionId}/events?cursor=0`);
    state.eventSource = source;
    source.onmessage = (message) => {
      try { handleEvent(JSON.parse(message.data)); } catch { appendTrace("Rejected malformed event payload.", "denied"); }
    };
    source.addEventListener("terminal", (message) => { const terminal = JSON.parse(message.data); appendTrace(`Session ${terminal.status}.`, terminal.status === "finished" ? "ok" : "denied"); source.close(); button.disabled = false; });
    source.onerror = () => { setText("#connection", "Event stream interrupted; reconnect with the shown session id and cursor."); };
  } catch (error) {
    setText("#connection", `Mission was not started: ${error.message}`);
    appendTrace(`Start rejected: ${error.message}`, "denied");
    button.disabled = false;
  }
}

$("#start").addEventListener("click", start);
