/* Dashboard logic. No external libraries: it works with the network cable unplugged. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const SVGNS = "http://www.w3.org/2000/svg";

  const state = { mode: "idle", viewingId: null, timer: null, lastData: null };

  // ---------- tiny helpers ----------
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") e.className = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) e.setAttribute(k, v);
    }
    for (const kid of kids.flat()) e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    return e;
  }
  function s(tag, attrs, ...kids) {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    for (const kid of kids) e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    return e;
  }
  const mmss = (sec) => {
    sec = Math.max(0, Math.round(sec || 0));
    return String(Math.floor(sec / 60)).padStart(2, "0") + ":" + String(sec % 60).padStart(2, "0");
  };
  const clock = (iso) => (iso || "").slice(11, 19);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || r.statusText);
    return body;
  };
  const scoreColor = (v) => `hsl(${Math.round(Math.max(0, Math.min(100, v)) * 1.3)}, 62%, 48%)`; // red -> green

  // ---------- chart ----------
  function renderChart(d) {
    const box = $("chart");
    box.replaceChildren();
    const wins = d.windows || [];
    const W = 760, H = 270, m = { l: 38, r: 14, t: 12, b: 28 };
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" });
    if (!wins.length) {
      svg.append(s("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "empty" },
        "Engagement appears after the first 30-second window."));
      box.append(svg);
      return;
    }
    const step = wins[0].end_s - wins[0].start_s;
    const tMax = Math.max(wins[wins.length - 1].end_s, d.elapsed_s || d.duration_s || 0);
    const X = (t) => m.l + (t / tMax) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - v / 100) * (H - m.t - m.b);

    const axis = s("g", { class: "axis" });
    for (const v of [0, 25, 50, 75, 100]) {
      axis.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid-line" }));
      axis.append(s("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, v));
    }
    const tickEvery = tMax > 1800 ? 600 : tMax > 600 ? 120 : 60;
    for (let t = 0; t <= tMax; t += tickEvery) {
      axis.append(s("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, mmss(t)));
    }
    svg.append(axis);

    // audio context bands
    for (const w of wins) {
      const cls = w.interaction === "discussion" ? "band-discussion"
        : w.interaction === "silent" ? "band-silent" : null;
      if (cls) svg.append(s("rect", { x: X(w.start_s), y: m.t, width: X(w.end_s) - X(w.start_s),
        height: H - m.t - m.b, class: cls }));
    }
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(50), y2: Y(50), class: "thr" }));

    const pts = wins.map((w) => [X(w.start_s + step / 2), Y(w.score)]);
    const line = pts.map((p) => p.join(",")).join(" ");
    svg.append(s("polygon", { points: `${pts[0][0]},${Y(0)} ${line} ${pts[pts.length - 1][0]},${Y(0)}`, class: "area" }));
    svg.append(s("polyline", { points: line, class: "line" }));
    wins.forEach((w, i) => {
      const dot = s("circle", { cx: pts[i][0], cy: pts[i][1], r: 4.5, class: "dot" + (w.score < 50 ? " low" : "") });
      dot.append(s("title", {}, `${mmss(w.start_s)}-${mmss(w.end_s)}  score ${w.score}  | attentive ${w.attentive_pct}%  | ${w.interaction}`));
      svg.append(dot);
    });
    box.append(svg);

    $("chartLegend").replaceChildren(
      h("span", {}, h("i", { style: "background:var(--series)" }), "Class engagement (0-100)"),
      h("span", {}, h("i", { style: "background:#14b8a6;opacity:.4" }), "Discussion / interaction"),
      h("span", {}, h("i", { style: "background:#f59e0b;opacity:.45" }), "Silent"),
      h("span", {}, h("i", { style: "background:var(--bad)" }), "Below 50 (low)")
    );
  }

  // ---------- panels ----------
  function renderKpis(d) {
    const live = d.present_now !== undefined;
    const present = live ? d.present_now : (d.attendance || []).filter((a) => a.status === "present").length;
    $("kPresent").textContent = present;
    $("kAttendees").textContent = (d.attendance || []).length;
    const t = d.trends || {};
    $("kEng").textContent = t.mean != null ? Math.round(t.mean) + "/100" : "–";
    $("kTrend").textContent = t.direction && t.direction !== "n/a"
      ? ({ improving: "↗ up", declining: "↘ down", steady: "→ steady" })[t.direction] : "–";
    $("kFiltered").textContent = d.filtered_transients ?? "–";
    $("kElapsed").textContent = mmss(d.elapsed_s ?? d.duration_s);
  }

  function renderAttendance(d) {
    const tbody = $("attTable").querySelector("tbody");
    tbody.replaceChildren();
    const eng = Object.fromEntries((d.participants || []).map((p) => [p.id, p]));
    const saved = state.mode === "saved";
    for (const a of d.attendance || []) {
      const p = eng[a.id] || {};
      const label = h("td", {}, a.label);
      if (saved) {
        label.append(" ", h("button", { class: "ghost", title: "Rename", "aria-label": "Rename " + a.label,
          style: "border:none;background:none;cursor:pointer;color:var(--muted)",
          onclick: () => renameParticipant(a.label) }, "✎"));
      }
      tbody.append(h("tr", {},
        label,
        h("td", {}, h("span", { class: "pill " + a.status }, a.status)),
        h("td", { class: "num" }, clock(a.check_in)),
        h("td", { class: "num" }, h("span", { class: "bar" }, h("b", { style: `width:${a.presence_pct}%` })), a.presence_pct + "%"),
        h("td", { class: "num" }, p.engagement != null ? p.engagement : "–"),
        h("td", { class: "num" }, p.attention_pct != null ? p.attention_pct + "%" : "–"),
        h("td", { class: "num" }, p.inactive_s != null ? Math.round(p.inactive_s) : "–")));
    }
    if (!(d.attendance || []).length) {
      tbody.append(h("tr", {}, h("td", { colspan: 7, class: "empty-note" }, "No one has been confirmed yet. People are marked present after sustained presence.")));
    }
  }

  function renderSuggestions(d) {
    const ul = $("suggestions");
    ul.replaceChildren();
    const note = $("coachNote");
    if (d.coach_note) { note.textContent = d.coach_note; note.classList.remove("hidden"); }
    else note.classList.add("hidden");
    const items = d.suggestions || [];
    if (!items.length) ul.append(h("li", { class: "empty-note" }, "No suggestions yet. They appear as engagement patterns develop."));
    for (const sg of items) {
      ul.append(h("li", { class: sg.priority },
        h("div", { class: "t" }, sg.title, h("span", { class: "when" }, mmss(sg.t))),
        h("p", {}, sg.rationale)));
    }
  }

  function renderEvents(d) {
    const ul = $("events");
    ul.replaceChildren();
    const ev = (d.events || []).slice().sort((a, b) => b.offset_s - a.offset_s);
    if (!ev.length) ul.append(h("li", { class: "empty-note" }, "No events yet."));
    for (const e of ev) {
      ul.append(h("li", {}, h("span", { class: "tm" }, clock(e.time)), h("span", { class: "k " + e.kind }, e.kind.replace("_", " ")), h("span", {}, e.participant)));
    }
  }

  function renderHeat(d) {
    const box = $("heat");
    box.replaceChildren();
    const wins = d.windows || [];
    if (!wins.length || !(d.participants || []).length) {
      box.append(h("p", { class: "empty-note" }, "Waiting for data."));
      return;
    }
    const axis = h("div", { class: "heat-axis" });
    wins.forEach((w, i) => axis.append(h("span", {}, i % 4 === 0 ? mmss(w.start_s) : "")));
    box.append(axis);
    for (const p of d.participants) {
      const byStart = Object.fromEntries((p.timeline || []).map((x) => [x.start_s, x.score]));
      const row = h("div", { class: "heat-row" }, h("div", { class: "heat-label", title: p.label }, p.label));
      for (const w of wins) {
        const v = byStart[w.start_s];
        row.append(h("div", { class: "heat-cell", title: v == null ? `${mmss(w.start_s)}: not present` : `${mmss(w.start_s)}: ${Math.round(v)}`,
          style: v == null ? null : `background:${scoreColor(v)}` }));
      }
      box.append(row);
    }
  }

  function render(d) {
    state.lastData = d;
    renderKpis(d); renderChart(d); renderAttendance(d); renderSuggestions(d); renderEvents(d); renderHeat(d);
  }

  // ---------- live polling ----------
  const STATUS = {
    idle: "Idle. Run the demo to see the full pipeline without a camera.",
    running: "Session running…", saving: "Finishing and saving…",
    done: "Session saved. See it under Saved sessions.", error: "Session stopped with an error.",
  };
  function setControls(status) {
    const busy = status === "running" || status === "saving";
    for (const id of ["btnSim", "btnLive", "btnUploadToggle"]) $(id).disabled = busy;
    $("btnStop").disabled = status !== "running";
    $("uploadForm").querySelectorAll("button,input").forEach((e) => (e.disabled = busy));
  }
  async function poll() {
    try {
      const d = await api("/api/live");
      const status = d.status || "idle";
      setControls(status);
      let text = STATUS[status] || status;
      if (status === "error") text += " " + (d.error || "");
      if (d.audio_note) text += " Audio: " + d.audio_note;
      $("statusText").textContent = text;
      $("progressBar").style.width = (d.progress != null ? Math.round(d.progress * 100) : status === "running" ? 100 : 0) + "%";
      if (status !== "idle" && state.mode !== "saved") { state.mode = "live"; render(d); }
      if (status === "done" || status === "error" || status === "idle") {
        stopPolling();
        if (status === "done") {
          loadSaved();
          // swap the live view for the saved report: it carries the full end-of-session review
          if (d.saved_id && state.mode !== "saved") await openSaved(d.saved_id, false);
        }
      }
    } catch (e) {
      $("statusText").textContent = "Cannot reach the local server: " + e.message;
      stopPolling();
    }
  }
  function startPolling() {
    state.mode = "live";
    stopPolling();
    poll();
    state.timer = setInterval(poll, 1500);
  }
  function stopPolling() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

  async function start(url, body) {
    try {
      await api(url, body);
      startPolling();
    } catch (e) {
      $("statusText").textContent = "Could not start: " + e.message;
    }
  }

  // ---------- saved sessions ----------
  async function loadSaved() {
    const list = await api("/api/sessions").catch(() => []);
    const ul = $("saved");
    ul.replaceChildren();
    if (!list.length) ul.append(h("li", { class: "empty-note" }, "Nothing saved yet."));
    for (const x of list) {
      ul.append(h("li", {},
        h("div", { class: "nm" }, x.name),
        h("div", { class: "meta" }, `${x.started_at.replace("T", " ")} · ${mmss(x.duration_s)} · ${x.attendees} attendees · avg ${x.mean_engagement ?? "–"}`),
        h("div", { class: "acts" },
          h("button", { onclick: () => openSaved(x.id) }, "Open"),
          h("a", { href: `/api/sessions/${x.id}/export/attendance.csv` }, "Attendance CSV"),
          h("a", { href: `/api/sessions/${x.id}/export/engagement.csv` }, "Engagement CSV"),
          h("a", { href: `/api/sessions/${x.id}/export/events.csv` }, "Events CSV"),
          h("button", { onclick: () => deleteSaved(x.id, x.name) }, "Delete"))));
    }
  }
  async function openSaved(id, scroll = true) {
    stopPolling();
    state.mode = "saved"; state.viewingId = id;
    const d = await api(`/api/sessions/${id}`);
    $("statusText").textContent = `Session saved. Viewing the final report: ${d.name}`;
    render(d);
    if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
  }
  async function deleteSaved(id, name) {
    if (!confirm(`Delete "${name}" permanently? This removes its stored metrics.`)) return;
    await api(`/api/sessions/${id}`, { method: "DELETE" });
    loadSaved();
  }
  async function renameParticipant(oldLabel) {
    const name = prompt(`Name for "${oldLabel}":`);
    if (!name || !name.trim()) return;
    try {
      const d = await api(`/api/sessions/${state.viewingId}/labels`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [oldLabel]: name.trim() }) });
      render(d);
    } catch (e) { alert(e.message); }
  }

  // ---------- wiring ----------
  $("btnSim").addEventListener("click", () => {
    state.mode = "live";
    start("/api/sessions/simulate", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("sessName").value || "Demo class", duration_s: 900, speed: 60 }) });
  });
  $("btnLive").addEventListener("click", () => {
    state.mode = "live";
    start("/api/sessions/live", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("sessName").value || "Live class", camera: 0, mic: true }) });
  });
  $("btnUploadToggle").addEventListener("click", () => $("uploadForm").classList.toggle("hidden"));
  $("uploadForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const fd = new FormData($("uploadForm"));
    fd.append("name", $("sessName").value || "Uploaded recording");
    $("statusText").textContent = "Uploading…";
    state.mode = "live";
    start("/api/sessions/upload", { method: "POST", body: fd });
  });
  $("btnStop").addEventListener("click", () => api("/api/sessions/stop", { method: "POST" }).catch(() => {}));

  api("/api/health").then((hc) => {
    $("detectorBadge").textContent = hc.detector_ok ? `face detector: ${hc.detector}` : "face detector: not installed (demo still works)";
    if (!hc.detector_ok && hc.detector_note) $("detectorBadge").title = hc.detector_note;
  }).catch(() => {});
  loadSaved();
  poll();
})();
