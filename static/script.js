const tabs = document.querySelectorAll(".tab");
const tabsContainer = document.querySelector(".tabs");
const tabIndicator = document.querySelector(".tab-indicator");
const blocks = document.querySelectorAll(".mode-block");
const modeTipNode = document.getElementById("mode-tip");
const controlsPanel = document.querySelector(".controls");
const form = document.getElementById("analyzer-form");
const runBtn = document.getElementById("run-btn");
const cancelBtn = document.getElementById("cancel-btn");
const statusNode = document.getElementById("status");
const resultsNode = document.getElementById("results");
const claimWarningNode = document.getElementById("claim-warning");
const processPanel = document.getElementById("process-panel");
const previewNode = document.getElementById("process-preview");
const stepsNode = document.getElementById("process-steps");

const appMainNode = document.getElementById("app-main");
const welcomeNode = document.getElementById("welcome-screen");
const enterAppBtn = document.getElementById("enter-app-btn");

let mode = "claim";
let currentController = null;
let progressTimer = null;
let progressStartedAt = null;
let basePreview = "";

const modeTips = {
  claim: "Tip: include a concrete subject, timeframe, and measurable fact for better evidence retrieval.",
  image: "Tip: clearer text and tighter crops improve OCR precision and evidence ranking quality.",
};

const defaultWorkflowStages = {
  claim: ["Input", "Checkability", "Context", "Domain Routing", "Evidence Gathering", "Relevance", "Stance", "Verdict"],
  image: ["Input", "OCR", "Checkability", "Context", "Domain Routing", "Evidence Gathering", "Relevance", "Stance", "Verdict"],
};

function moveTabIndicatorToActive() {
  if (!tabsContainer || !tabIndicator) return;
  const activeTab = tabsContainer.querySelector(".tab.active");
  if (!activeTab) return;

  const containerRect = tabsContainer.getBoundingClientRect();
  const activeRect = activeTab.getBoundingClientRect();
  const left = activeRect.left - containerRect.left;
  const width = activeRect.width;

  tabIndicator.style.width = `${width}px`;
  tabIndicator.style.transform = `translateX(${left}px)`;
}

function setMode(nextMode) {
  mode = nextMode;
  tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  blocks.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  if (modeTipNode) modeTipNode.textContent = modeTips[mode] || modeTips.claim;
  moveTabIndicatorToActive();

  if (controlsPanel) {
    controlsPanel.classList.add("is-switching");
    setTimeout(() => {
      controlsPanel.classList.remove("is-switching");
    }, 220);
  }
}

function enterMainApp() {
  document.body.classList.remove("home-mode");
  document.body.classList.add("app-mode");

  if (welcomeNode) {
    welcomeNode.style.pointerEvents = "none";
    welcomeNode.classList.add("is-leaving");
  }

  if (appMainNode) {
    appMainNode.hidden = false;
    appMainNode.classList.add("app-preenter");
    requestAnimationFrame(() => {
      appMainNode.classList.remove("app-preenter");
      appMainNode.classList.add("app-enter");
    });
    appMainNode.scrollTop = 0;
  }

  window.scrollTo({ top: 0, left: 0, behavior: "auto" });

  const finalizeHideWelcome = () => {
    if (!welcomeNode) return;
    welcomeNode.hidden = true;
    welcomeNode.classList.remove("is-leaving");
    welcomeNode.style.pointerEvents = "";
  };

  if (welcomeNode) {
    welcomeNode.addEventListener("transitionend", finalizeHideWelcome, { once: true });
  }
  setTimeout(finalizeHideWelcome, 320);

  moveTabIndicatorToActive();
}

function initWelcomeFlow() {
  if (!appMainNode) return;
  document.body.classList.add("home-mode");
  appMainNode.hidden = true;
  if (welcomeNode) welcomeNode.hidden = false;

  if (enterAppBtn) {
    enterAppBtn.addEventListener("click", () => {
      enterMainApp();
    });
  }
}

for (const tab of tabs) {
  tab.addEventListener("click", () => {
    setMode(tab.dataset.mode);
    resultsNode.innerHTML = "";
    statusNode.textContent = "";
    resetProgressPanel();
  });
}

window.addEventListener("resize", moveTabIndicatorToActive);

initWelcomeFlow();
setMode(mode);

const claimTextNode = document.getElementById("claim-text");
if (claimTextNode) {
  claimTextNode.addEventListener("input", () => setClaimWarning(""));
}

cancelBtn.addEventListener("click", () => {
  if (currentController) {
    currentController.abort();
    statusNode.textContent = "Cancelled.";
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  runBtn.disabled = true;
  cancelBtn.hidden = false;
  statusNode.textContent = "Analyzing...";
  resultsNode.innerHTML = "";

  if (mode === "claim") {
    const claim = document.getElementById("claim-text").value.trim();
    if (!claim) {
      statusNode.textContent = "Please enter a claim.";
      runBtn.disabled = false;
      cancelBtn.hidden = true;
      return;
    }
    const wordCount = countWords(claim);
    if (wordCount < 6) {
      const warningText = "Claim must be at least 6 words. Analysis blocked to save compute.";
      setClaimWarning(warningText);
      statusNode.textContent = warningText;
      runBtn.disabled = false;
      cancelBtn.hidden = true;
      return;
    }
    setClaimWarning("");
  }

  if (mode === "image") {
    const file = document.getElementById("image-file").files[0];
    if (!file) {
      statusNode.textContent = "Please choose an image.";
      runBtn.disabled = false;
      cancelBtn.hidden = true;
      return;
    }
  }

  const progressId = createProgressId();
  startProgressForCurrentInput(progressId);
  currentController = new AbortController();

  try {
    const data = await callApiForMode(currentController.signal, progressId);
    enrichProgressWithResponse(data);
    completeProgress();
    renderResult(data);
    statusNode.textContent = "Done.";
  } catch (error) {
    if (error.name === "AbortError") {
      statusNode.textContent = "Cancelled.";
      renderWorkflow({
        status: "cancelled",
        stages: [{
          id: "cancelled",
          label: "Cancelled",
          status: "cancelled",
          detail: "Process cancelled by user.",
          substeps: [],
        }],
      });
    } else {
      statusNode.textContent = "Failed.";
      renderWorkflow({
        status: "error",
        stages: [{
          id: "error",
          label: "Error",
          status: "error",
          detail: error.message,
          substeps: [],
        }],
      });
      resultsNode.innerHTML = cardHtml("Error", `<p>${escapeHtml(error.message)}</p>`);
    }
  } finally {
    stopProgressTimers();
    currentController = null;
    runBtn.disabled = false;
    cancelBtn.hidden = true;
  }
});

function createProgressId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `progress-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildPlaceholderWorkflow() {
  return {
    status: "running",
    stages: (defaultWorkflowStages[mode] || defaultWorkflowStages.claim).map((label, index) => ({
      id: `${mode}-${index}`,
      label,
      status: index === 0 ? "active" : "pending",
      detail: index === 0 ? "Queued for analysis" : "",
      substeps: [],
    })),
  };
}

function startProgressForCurrentInput(progressId) {
  processPanel.hidden = false;
  progressStartedAt = Date.now();
  basePreview = mode === "claim" ? "Claim analysis in progress" : "Image analysis in progress";
  previewNode.textContent = basePreview;
  renderWorkflow(buildPlaceholderWorkflow());
}

function enrichProgressWithResponse(data) {
  const verdict = extractVerdict(data);
  previewNode.textContent = `${basePreview} | Final verdict: ${verdict}`;
}

function completeProgress() {
  stopProgressTimers();
}

function resetProgressPanel() {
  stopProgressTimers();
  processPanel.hidden = true;
  previewNode.textContent = "Submit a claim or image to start.";
  stepsNode.innerHTML = "";
  basePreview = "";
}

function stopProgressTimers() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function renderWorkflow(payload) {
  const allStages = Array.isArray(payload?.stages) ? payload.stages : [];
  const stages = allStages.filter((stage) => {
    const status = String(stage?.status || "pending");
    return status !== "pending";
  });
  if (!stages.length) {
    stepsNode.innerHTML = "";
    return;
  }
  stepsNode.innerHTML = stages.map((stage, index) => {
    const stageStatus = escapeAttr(stage.status || "pending");
    const detail = stage.detail ? `<div class="step-detail">${escapeHtml(stage.detail)}</div>` : "";
    const connector = index < stages.length - 1 ? '<span class="step-connector" aria-hidden="true"></span>' : "";
    return `
      <li class="workflow-step ${stageStatus}">
        <div class="step-node-wrap">
          <span class="step-node" aria-hidden="true"></span>
          ${connector}
        </div>
        <div class="step-body">
          <span class="step-title">${escapeHtml(stage.label || stage.id || "Stage")}</span>
          ${detail}
        </div>
      </li>
    `;
  }).join("");
}

function extractVerdict(data) {
  if (data.verdict) return data.verdict;
  if (data.final_verdict) return data.final_verdict;
  return "Completed";
}

function renderResult(data) {
  const rawVerdict = String(data.verdict || "unknown").toLowerCase();
  const verdictLabel = escapeHtml(data.verdict || "Unknown");
  const confidence = data.confidence || 0;
  const confidenceLabel = formatPct(confidence);
  const evidence = data.evidence || [];
  const reasoning = escapeHtml(data.reasoning || "No reasoning available");
  
  const details = data.details || {};
  const evidenceCount = details.evidence_count || evidence.length;
  const llm = details.llm_verifier || {};
  const llmStatus = llm.enabled
    ? (llm.triggered ? `Triggered (${escapeHtml(llm.provider || "provider")} / ${escapeHtml(llm.model || "model")})` : "Enabled (not triggered)")
    : "Disabled";

  const evidenceInsights = summarizeEvidence(evidence);
  const avgRel = formatPct(evidenceInsights.avgRelevance);
  const avgCred = formatPct(evidenceInsights.avgCredibility);
  const stanceMix = `S:${evidenceInsights.stance.support} R:${evidenceInsights.stance.refute} N:${evidenceInsights.stance.neutral}`;

  let html = `
    ${cardHtml("Final Verdict", `
      <div class="verdict-hero ${escapeAttr(rawVerdict)}">
        <div class="verdict-main">
          <div class="meta">Final Decision</div>
          <div class="verdict-row">
            <span class="pill ${escapeAttr(rawVerdict)}">${verdictLabel}</span>
            <strong>${confidenceLabel}</strong>
          </div>
          <div class="confidence-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round((Number(confidence) || 0) * 100)}">
            <span style="width:${Math.max(0, Math.min(100, (Number(confidence) || 0) * 100)).toFixed(1)}%"></span>
          </div>
        </div>
        <div class="verdict-side">
          <p><strong>LLM verifier:</strong> ${llmStatus}</p>
          <p><strong>Evidence mix:</strong> ${stanceMix}</p>
        </div>
      </div>
      <p style="margin-top:12px">${reasoning}</p>
    `)}

    ${cardHtml("Key Metrics", `
      <div class="kpi">
        <div class="tile">
          <div class="meta">Verdict</div>
          <strong><span class="pill ${escapeAttr(verdictLabel.toLowerCase())}">${verdictLabel}</span></strong>
        </div>
        <div class="tile">
          <div class="meta">Confidence</div>
          <strong>${confidenceLabel}</strong>
        </div>
        <div class="tile">
          <div class="meta">Evidence</div>
          <strong>${num(evidenceCount)} items</strong>
        </div>
      </div>
    `)}

    ${cardHtml("Evidence Quality Summary", `
      <div class="kpi">
        <div class="tile">
          <div class="meta">Avg Relevance</div>
          <strong>${avgRel}</strong>
        </div>
        <div class="tile">
          <div class="meta">Avg Credibility</div>
          <strong>${avgCred}</strong>
        </div>
        <div class="tile">
          <div class="meta">Stance Balance</div>
          <strong>${stanceMix}</strong>
        </div>
      </div>
    `)}
  `;

  if (evidence.length > 0) {
    const evidenceHtml = evidence.map((ev, idx) => {
      const sourceRaw = ev.source || "Unknown Source";
      const source = escapeHtml(cleanSourceName(sourceRaw));
      const cleanedText = cleanEvidenceText(ev.text || "");
      const previewText = escapeHtml(truncateText(cleanedText, 260));
      const fullText = escapeHtml(cleanedText);
      const relevance = formatPct(ev.relevance || 0);
      const credibility = formatPct(ev.credibility || 0);
      const stance = escapeHtml(ev.stance || "N/A");
      const sourceUrl = getEvidenceSourceUrl(ev);
      const sourceLink = sourceUrl
        ? `<a class="evidence-source-link" href="${escapeAttr(sourceUrl)}" target="_blank" rel="noopener noreferrer">${source}</a>`
        : `<span>${source}</span>`;
      
      return `
        <details class="evidence-item" ${idx === 0 ? "open" : ""}>
          <summary>
            <div class="evidence-summary-head">
              <strong>${sourceLink}</strong>
              <span class="pill ${escapeAttr(String(stance).toLowerCase())}">${stance}</span>
            </div>
            <div class="evidence-summary-meta">Relevance: ${relevance} | Credibility: ${credibility}</div>
          </summary>
          <div class="evidence-body">
            <p class="evidence-preview">${previewText}</p>
            ${cleanedText.length > 260 ? `<p class="meta">Full excerpt:</p><p>${fullText}</p>` : ""}
          </div>
        </details>
      `;
    }).join("");

    html += cardHtml("Evidence Found", `
      <div class="evidence-list">
        ${evidenceHtml}
      </div>
    `);
  }

  resultsNode.innerHTML = html;
}

function cardHtml(title, content) {
  return `
    <div class="card">
      <h3>${title}</h3>
      ${content}
    </div>
  `;
}

function summarizeEvidence(evidence) {
  if (!Array.isArray(evidence) || !evidence.length) {
    return {
      avgRelevance: 0,
      avgCredibility: 0,
      stance: { support: 0, refute: 0, neutral: 0 },
    };
  }
  let rel = 0;
  let cred = 0;
  const stance = { support: 0, refute: 0, neutral: 0 };
  for (const ev of evidence) {
    rel += Number(ev.relevance || 0);
    cred += Number(ev.credibility || 0);
    const s = String(ev.stance || "neutral").toLowerCase();
    if (s.includes("support")) stance.support += 1;
    else if (s.includes("refute")) stance.refute += 1;
    else stance.neutral += 1;
  }
  return {
    avgRelevance: rel / evidence.length,
    avgCredibility: cred / evidence.length,
    stance,
  };
}

function formatPct(value) {
  const numValue = typeof value === "number" ? value : parseFloat(value) || 0;
  return `${(numValue * 100).toFixed(1)}%`;
}

function num(value) {
  const numValue = typeof value === "number" ? value : parseInt(value) || 0;
  return numValue.toString();
}

function escapeHtml(unsafe) {
  if (typeof unsafe !== "string") return String(unsafe);
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeAttr(value) {
  return String(value).replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function setClaimWarning(message) {
  if (!claimWarningNode) return;
  const text = String(message || "").trim();
  claimWarningNode.textContent = text;
  claimWarningNode.hidden = !text;
}

function countWords(text) {
  return String(text || "").trim().split(/\s+/).filter(Boolean).length;
}

function cleanEvidenceText(text) {
  return String(text || "")
    .replace(/\[\.{3}\]/g, " ")
    .replace(/\s+/g, " ")
    .replace(/#+\s*/g, "")
    .trim();
}

function cleanSourceName(source) {
  return String(source || "Unknown Source")
    .replace(/\s+/g, " ")
    .trim();
}

function truncateText(text, maxLen) {
  const value = String(text || "");
  if (value.length <= maxLen) return value;
  return `${value.slice(0, maxLen).trim()}...`;
}

function getEvidenceSourceUrl(ev) {
  if (!ev || typeof ev !== "object") return "";
  return String(ev.url || ev.source_url || ev.link || "").trim();
}

async function callApiForMode(signal, progressId = null) {
  if (mode === "claim") {
    const claim = document.getElementById("claim-text").value.trim();
    if (!claim) throw new Error("Please enter a claim.");
    return postJson("/api/analyze", { claim }, signal);
  }
  
  // Image mode
  const file = document.getElementById("image-file").files[0];
  if (!file) throw new Error("Please choose an image.");
  return postFile("/api/analyze-image", file, signal);
}

async function postJson(url, body, signal = null, headers = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
    signal,
  });
  return handleResponse(response);
}

async function postFile(url, file, signal = null, headers = {}, fields = {}) {
  const formData = new FormData();
  formData.append("image", file);
  Object.entries(fields || {}).forEach(([key, value]) => {
    if (value != null && String(value).trim()) formData.append(key, String(value).trim());
  });
  const response = await fetch(url, {
    method: "POST",
    body: formData,
    headers,
    signal,
  });
  return handleResponse(response);
}

async function handleResponse(response) {
  let payload = {};
  try {
    payload = await response.json();
  } catch (err) {
    throw new Error("Server returned an invalid response.");
  }
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}
