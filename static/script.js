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
const evidenceFilterNode = document.getElementById("evidence-filter");
const topEvidenceCountNode = document.getElementById("top-evidence-count");
const translateBtn = document.getElementById("translate-btn");
const originalBtn = document.getElementById("original-btn");
const analysisTimerNode = document.getElementById("analysis-timer");
const claimPreviewTextNode = document.getElementById("claim-preview-text");
const analysisWarningNode = document.getElementById("run-warning");

const appMainNode = document.getElementById("app-main");
const welcomeNode = document.getElementById("welcome-screen");
const enterAppBtn = document.getElementById("enter-app-btn");

let mode = "claim";
let currentController = null;
let progressTimer = null;
let progressStartedAt = null;
let basePreview = "";
let analysisTimer = null;
let analysisStartedAt = null;
let latestResultPayload = null;
let claimPreviewOriginal = "Submit a claim, image, or PDF to preview extracted claim text.";
let claimPreviewTranslated = "";
let claimPreviewShowingTranslated = false;
let imageClaimLocked = false;

const modeTips = {
  claim: "Tip: include a concrete subject, timeframe, and measurable fact for better evidence retrieval.",
  image: "Tip: clearer text and tighter crops improve OCR precision and evidence ranking quality.",
  pdf: "Tip: text-based PDFs work best. Scanned PDFs may need OCR-enabled extraction.",
};

function getApiBase() {
  const cfg = String(window.__FACTLENS_API_BASE__ || "").trim();
  if (cfg) return cfg.replace(/\/+$/, "");
  const param = new URLSearchParams(window.location.search).get("api_base");
  if (param && String(param).trim()) return String(param).trim().replace(/\/+$/, "");
  const stored = String(window.localStorage?.getItem("FACTLENS_API_BASE") || "").trim();
  if (stored) return stored.replace(/\/+$/, "");
  return "";
}

const API_BASE = getApiBase();

function apiUrl(path) {
  if (!path) return path;
  if (/^https?:\/\//i.test(path)) return path;
  return API_BASE ? `${API_BASE}${path}` : path;
}

const defaultWorkflowStages = {
  claim: ["Input", "Checkability", "Context", "Domain Routing", "Evidence Gathering", "Relevance", "Stance", "Verdict"],
  image: ["Input", "OCR", "Checkability", "Context", "Domain Routing", "Evidence Gathering", "Relevance", "Stance", "Verdict"],
  pdf: ["Input", "PDF Extract", "Checkability", "Context", "Domain Routing", "Evidence Gathering", "Relevance", "Stance", "Verdict"],
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
  if (mode === "image") {
    setClaimPreviewText("Submit image to preview OCR-extracted claim text.");
    imageClaimLocked = false;
  } else if (mode === "pdf") {
    setClaimPreviewText("Submit PDF to preview extracted claim text.");
    imageClaimLocked = false;
  } else {
    const text = document.getElementById("claim-text")?.value?.trim() || "";
    setClaimPreviewText(text || "Submit a claim, image, or PDF to preview extracted claim text.");
    imageClaimLocked = false;
  }
  if (originalBtn) originalBtn.hidden = true;
  setAnalysisWarning("");
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
  claimTextNode.addEventListener("input", () => {
    const text = claimTextNode.value.trim();
    setClaimPreviewText(text || "Submit a claim, image, or PDF to preview extracted claim text.");
  });
}

if (evidenceFilterNode) {
  evidenceFilterNode.addEventListener("change", () => {
    if (latestResultPayload) renderResult(latestResultPayload);
  });
}
if (topEvidenceCountNode) {
  topEvidenceCountNode.addEventListener("change", () => {
    if (latestResultPayload) renderResult(latestResultPayload);
  });
}
if (translateBtn) {
  translateBtn.addEventListener("click", async () => {
    const text = (claimPreviewOriginal || "").trim();
    if (!text || text === "Submit a claim, image, or PDF to preview extracted claim text.") return;
    if (claimPreviewTextNode) claimPreviewTextNode.textContent = "Translating...";
    try {
      const out = await postJson("/api/translate-preview", { text, target_language: "en" });
      const translated = String(out?.translated_text || "").trim();
      if (out?.translated || out?.source_language === "en") {
        claimPreviewTranslated = translated || "";
      } else {
        claimPreviewTranslated = text;
      }
      if (claimPreviewTranslated) {
        claimPreviewShowingTranslated = true;
        if (claimPreviewTextNode) claimPreviewTextNode.textContent = claimPreviewTranslated;
        if (originalBtn) originalBtn.hidden = false;
      }
    } catch (err) {
      if (claimPreviewTextNode) claimPreviewTextNode.textContent = claimPreviewOriginal;
    }
  });
}

if (originalBtn) {
  originalBtn.addEventListener("click", () => {
    claimPreviewShowingTranslated = false;
    if (claimPreviewTextNode) claimPreviewTextNode.textContent = claimPreviewOriginal;
    originalBtn.hidden = true;
  });
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
  latestResultPayload = null;
  claimPreviewTranslated = "";
  claimPreviewShowingTranslated = false;
  if (originalBtn) originalBtn.hidden = true;
  setAnalysisWarning("");
  startAnalysisTimer();

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
    setClaimPreviewText("OCR in progress...");
    imageClaimLocked = false;
  } else if (mode === "pdf") {
    const file = document.getElementById("pdf-file").files[0];
    if (!file) {
      statusNode.textContent = "Please choose a PDF.";
      runBtn.disabled = false;
      cancelBtn.hidden = true;
      return;
    }
    const pageSpec = String(document.getElementById("pdf-page-spec")?.value || "").trim();
    const parsed = parsePdfPageSpec(pageSpec);
    if (!parsed.ok) {
      const msg = "HIGH: Invalid page selector. Use '1' or '1-2'.";
      setAnalysisWarning(msg, "high");
      statusNode.textContent = "";
      runBtn.disabled = false;
      cancelBtn.hidden = true;
      return;
    }
    if ((parsed.pages || []).some((p) => p > 5 || p < 1)) {
      const msg = "HIGH: Page selector out of allowed range (1-5).";
      setAnalysisWarning(msg, "high");
      statusNode.textContent = "";
      runBtn.disabled = false;
      cancelBtn.hidden = true;
      return;
    }
    if (parsed.count === 4) {
      setAnalysisWarning("WARN: 4 pages selected. This run may be time-extensive.", "warn");
    } else if (parsed.count >= 5) {
      setAnalysisWarning("WARN: 5 pages selected. Runtime may be high.", "warn");
    }
    setClaimPreviewText("PDF extraction in progress...");
    imageClaimLocked = false;
  }

  const progressId = createProgressId();
  startProgressForCurrentInput(progressId);
  currentController = new AbortController();

  try {
    const data = await callApiForMode(currentController.signal, progressId);
    latestResultPayload = data;
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
    stopAnalysisTimer();
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
  if (processPanel) processPanel.hidden = false;
  progressStartedAt = Date.now();
  basePreview = mode === "claim"
    ? "Claim analysis in progress"
    : (mode === "image" ? "Image analysis in progress" : "PDF analysis in progress");
  if (previewNode) previewNode.textContent = basePreview;
  renderWorkflow(buildPlaceholderWorkflow());
}

function enrichProgressWithResponse(data) {
  const verdict = extractVerdict(data);
  if (previewNode) previewNode.textContent = `${basePreview} | Final verdict: ${verdict}`;
  if (mode === "image") {
    // Keep preview stable after OCR lock so analyzed claim does not appear to "jump".
    if (!imageClaimLocked) {
      const previewText = data.claim || data.summary_claim || data.ocr_text || "";
      if (previewText) {
        setClaimPreviewText(previewText);
        imageClaimLocked = true;
      }
    }
  } else if (mode === "pdf") {
    if (!imageClaimLocked) {
      const previewText = data.claim || data.summary_claim || data.pdf_text || "";
      if (previewText) {
        setClaimPreviewText(previewText);
        imageClaimLocked = true;
      }
    }
  } else {
    const previewText = document.getElementById("claim-text")?.value || "";
    if (previewText) setClaimPreviewText(previewText);
  }
}

function completeProgress() {
  stopProgressTimers();
}

function resetProgressPanel() {
  stopProgressTimers();
  if (processPanel) processPanel.hidden = true;
  if (previewNode) previewNode.textContent = "Submit a claim, image, or PDF to start.";
  if (stepsNode) stepsNode.innerHTML = "";
  basePreview = "";
}

function stopProgressTimers() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function renderWorkflow(payload) {
  if (!stepsNode) return;
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
  const evidenceRaw = Array.isArray(data.evidence) ? data.evidence : [];
  const evidenceFiltered = applyEvidenceControls(evidenceRaw);
  const evidence = evidenceFiltered;
  const reasoning = escapeHtml(data.reasoning || "No reasoning available");
  
  const details = data.details || {};
  const evidenceCount = details.evidence_count || evidence.length;
  const llm = details.llm_verifier || {};
  const llmStatus = llm.enabled
    ? (llm.triggered ? `Triggered (${escapeHtml(llm.provider || "provider")} / ${escapeHtml(llm.model || "model")})` : "Enabled (not triggered)")
    : "Disabled";

  const evidenceInsights = summarizeEvidence(evidenceRaw);
  const avgRel = formatPct(evidenceInsights.avgRelevance);
  const avgCred = formatPct(evidenceInsights.avgCredibility);
  const stanceMix = `S:${evidenceInsights.stance.support} R:${evidenceInsights.stance.refute} N:${evidenceInsights.stance.neutral}`;
  const warnings = Array.isArray(data.warnings) ? data.warnings.filter(Boolean) : [];

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
          <strong>${num(evidence.length || evidenceCount)} items</strong>
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

  if (Number(data.page_count || 0) > 0 || data.extraction_engine) {
    const selectedSpec = String(data.selected_page_spec || "").trim();
    const processedCount = Array.isArray(data.selected_pages) ? data.selected_pages.length : 0;
    const selectedClaim = String(data.selected_claim || data.summary_claim || "").trim();
    const contextPreview = truncateText(String(data.pdf_text || "").replace(/\s+/g, " ").trim(), 260);
    html += cardHtml("Document Metadata", `
      <div class="kpi">
        <div class="tile">
          <div class="meta">Pages Processed</div>
          <strong>${num(processedCount || 0)}</strong>
        </div>
        <div class="tile">
          <div class="meta">Total Pages</div>
          <strong>${num(data.page_count || 0)}</strong>
        </div>
        <div class="tile">
          <div class="meta">Pages Selected</div>
          <strong>${escapeHtml(selectedSpec || "default")}</strong>
        </div>
        <div class="tile">
          <div class="meta">Extractor</div>
          <strong>${escapeHtml(String(data.extraction_engine || "N/A"))}</strong>
        </div>
      </div>
      ${selectedClaim ? `<p class="meta" style="margin-top:10px;">Claim Used</p><p>${escapeHtml(selectedClaim)}</p>` : ""}
      ${contextPreview ? `<p class="meta" style="margin-top:8px;">Page Context</p><p>${escapeHtml(contextPreview)}</p>` : ""}
    `);
  }
  if (warnings.length) {
    const high = warnings.find((w) => String(w).startsWith("HIGH:"));
    const warn = warnings.find((w) => String(w).startsWith("WARN:"));
    const msg = String(high || warn || warnings[0] || "");
    const level = msg.startsWith("HIGH:") ? "high" : (msg.startsWith("WARN:") ? "warn" : "info");
    setAnalysisWarning(msg, level);
  } else {
    setAnalysisWarning("");
  }

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
      const llmAdjusted = Boolean(ev.llm_adjusted);
      const llmAdjustedTag = llmAdjusted ? `<span class="pill neutral">LLM-adjusted</span>` : "";
      
      return `
        <details class="evidence-item" ${idx === 0 ? "open" : ""}>
          <summary>
            <div class="evidence-summary-head">
              <strong>${sourceLink}</strong>
              <span>
                <span class="pill ${escapeAttr(String(stance).toLowerCase())}">${stance}</span>
                ${llmAdjustedTag}
              </span>
            </div>
            <div class="evidence-summary-meta">Relevance: ${relevance} | Credibility: ${credibility}</div>
          </summary>
          <div class="evidence-body">
            <p class="evidence-preview" data-original="${escapeAttr(cleanedText)}">${previewText}</p>
            ${cleanedText.length > 260 ? `<p class="meta">Full excerpt:</p><p class="evidence-full-text" data-original="${escapeAttr(cleanedText)}">${fullText}</p>` : ""}
          </div>
        </details>
      `;
    }).join("");

    html += cardHtml("Evidence Found", `
      <div class="evidence-tools">
        <button id="evidence-translate-btn" type="button" class="tool-btn tool-btn-inline">Translate Evidence</button>
        <button id="evidence-original-btn" type="button" class="tool-btn tool-btn-inline" hidden>Original</button>
      </div>
      <div class="evidence-list">
        ${evidenceHtml}
      </div>
    `);
  }

  resultsNode.innerHTML = html;
  bindEvidenceTranslateHandlers();
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

function setAnalysisWarning(message, level = "info") {
  if (!analysisWarningNode) return;
  const text = String(message || "").trim();
  analysisWarningNode.textContent = text;
  analysisWarningNode.hidden = !text;
  analysisWarningNode.classList.remove("warn", "high");
  if (text) {
    if (level === "high") analysisWarningNode.classList.add("high");
    else if (level === "warn") analysisWarningNode.classList.add("warn");
  }
}

function parsePdfPageSpec(spec) {
  const raw = String(spec || "").trim();
  if (!raw) return { ok: true, pages: [], count: 0, mode: "default" };
  if (/^\d+$/.test(raw)) {
    const n = parseInt(raw, 10);
    return { ok: true, pages: [n], count: 1, mode: "single" };
  }
  const m = raw.match(/^(\d+)\s*-\s*(\d+)$/);
  if (m) {
    let a = parseInt(m[1], 10);
    let b = parseInt(m[2], 10);
    if (a > b) [a, b] = [b, a];
    const pages = [];
    for (let i = a; i <= b; i += 1) pages.push(i);
    return { ok: true, pages, count: pages.length, mode: "range" };
  }
  return { ok: false, pages: [], count: 0, mode: "invalid" };
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

function applyEvidenceControls(evidence) {
  if (!Array.isArray(evidence)) return [];
  let rows = dedupeEvidenceRows(evidence);

  const stanceFilter = String(evidenceFilterNode?.value || "all").toLowerCase();
  if (stanceFilter !== "all") {
    rows = rows.filter((ev) => String(ev?.stance || "").toLowerCase().includes(stanceFilter));
  }

  const topN = String(topEvidenceCountNode?.value || "all").toLowerCase();
  if (topN !== "all") {
    const n = Math.max(1, parseInt(topN, 10) || 5);
    rows = rows.slice(0, n);
  }

  return rows;
}

function dedupeEvidenceRows(rows) {
  const out = [];
  const seen = new Set();
  for (const ev of (rows || [])) {
    const urlKey = String(ev?.url || "").trim().toLowerCase();
    const textKey = String(ev?.text || "").replace(/\s+/g, " ").trim().toLowerCase().slice(0, 220);
    const key = urlKey || textKey;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(ev);
  }
  return out;
}

function setClaimPreviewText(text) {
  if (!claimPreviewTextNode) return;
  const value = truncateText(String(text || "").replace(/\s+/g, " ").trim(), 400);
  claimPreviewOriginal = value || "Submit a claim, image, or PDF to preview extracted claim text.";
  claimPreviewTranslated = "";
  claimPreviewShowingTranslated = false;
  claimPreviewTextNode.textContent = claimPreviewOriginal;
  if (originalBtn) originalBtn.hidden = true;
}

function startAnalysisTimer() {
  stopAnalysisTimer();
  analysisStartedAt = Date.now();
  if (analysisTimerNode) analysisTimerNode.textContent = "Timer: 0.0s";
  analysisTimer = setInterval(() => {
    if (!analysisTimerNode || !analysisStartedAt) return;
    const sec = (Date.now() - analysisStartedAt) / 1000;
    analysisTimerNode.textContent = `Timer: ${sec.toFixed(1)}s`;
  }, 100);
}

function stopAnalysisTimer() {
  if (analysisTimer) {
    clearInterval(analysisTimer);
    analysisTimer = null;
  }
}

async function callApiForMode(signal, progressId = null) {
  if (mode === "claim") {
    const claim = document.getElementById("claim-text").value.trim();
    if (!claim) throw new Error("Please enter a claim.");
    return postJson("/api/analyze", { claim }, signal);
  }

  if (mode === "image") {
    const file = document.getElementById("image-file").files[0];
    if (!file) throw new Error("Please choose an image.");

    statusNode.textContent = "OCR extracting...";
    const ocrPreview = await postFile(
      "/api/extract-ocr-preview",
      file,
      signal,
      {},
      { language: "auto" },
      "image",
    );
    const ocrClaim = String(ocrPreview?.claim_text || ocrPreview?.ocr_text || "").trim();
    if (ocrClaim) {
      setClaimPreviewText(ocrClaim);
      imageClaimLocked = true;
    }

    statusNode.textContent = "Analyzing...";
    return postFile(
      "/api/analyze-image",
      file,
      signal,
      {},
      {
        language: "auto",
        claim: ocrClaim,
      },
      "image",
    );
  }

  // PDF mode
  const pdfFile = document.getElementById("pdf-file").files[0];
  if (!pdfFile) throw new Error("Please choose a PDF.");
  const pageSpec = String(document.getElementById("pdf-page-spec")?.value || "").trim();

  statusNode.textContent = "Extracting PDF text...";
  const pdfPreview = await postFile(
    "/api/extract-pdf-preview",
    pdfFile,
    signal,
    {},
    { language: "auto", page_spec: pageSpec },
    "pdf",
  );
  const pdfClaim = String(pdfPreview?.claim_text || pdfPreview?.pdf_text || "").trim();
  if (pdfClaim) {
    setClaimPreviewText(pdfClaim);
    imageClaimLocked = true;
  }

  statusNode.textContent = "Analyzing...";
  return postFile(
    "/api/analyze-pdf",
    pdfFile,
    signal,
    {},
    {
      language: "auto",
      claim: pdfClaim,
      page_spec: pageSpec,
    },
    "pdf",
  );
}

async function postJson(url, body, signal = null, headers = {}) {
  const response = await fetch(apiUrl(url), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
    signal,
  });
  return handleResponse(response);
}

async function postFile(url, file, signal = null, headers = {}, fields = {}, fileField = "image") {
  const formData = new FormData();
  formData.append(fileField, file);
  Object.entries(fields || {}).forEach(([key, value]) => {
    if (value != null && String(value).trim()) formData.append(key, String(value).trim());
  });
  const response = await fetch(apiUrl(url), {
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

function bindEvidenceTranslateHandlers() {
  const tBtn = document.getElementById("evidence-translate-btn");
  const oBtn = document.getElementById("evidence-original-btn");
  if (!tBtn) return;

  const items = Array.from(resultsNode.querySelectorAll(".evidence-item")).map((item) => {
    const previewNode = item.querySelector(".evidence-preview");
    const fullNode = item.querySelector(".evidence-full-text");
    const original = String(
      (fullNode?.getAttribute("data-original"))
      || (previewNode?.getAttribute("data-original"))
      || ""
    );
    return { previewNode, fullNode, original };
  }).filter((x) => x.previewNode && x.original);

  const restoreOriginal = () => {
    items.forEach((row) => {
      const base = row.original || "";
      if (row.previewNode) row.previewNode.textContent = truncateText(base, 260);
      if (row.fullNode) row.fullNode.textContent = base;
    });
    if (oBtn) oBtn.hidden = true;
  };

  tBtn.addEventListener("click", async () => {
    if (!items.length) return;
    const texts = items.map((row) => String(row.original || "").trim());
    tBtn.disabled = true;
    try {
      const out = await postJson("/api/translate-batch", { texts, target_language: "en" });
      const translated = Array.isArray(out?.translated_texts) ? out.translated_texts : [];
      items.forEach((row, i) => {
        const v = String(translated[i] || "").trim();
        if (!v) return;
        if (row.previewNode) row.previewNode.textContent = truncateText(v, 260);
        if (row.fullNode) row.fullNode.textContent = v;
      });
      if (oBtn) oBtn.hidden = false;
    } catch (_err) {
      restoreOriginal();
    } finally {
      tBtn.disabled = false;
    }
  });

  if (oBtn) {
    oBtn.addEventListener("click", () => {
      restoreOriginal();
    });
  }
}
