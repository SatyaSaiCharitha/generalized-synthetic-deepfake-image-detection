/**
 * GenImage Detector — Client Application Controller
 * Handles UI interactions, file drag-and-drop, API requests, and forensics visualization.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM References
  const tabButtons = document.querySelectorAll(".nav-tab");
  const tabPanels = document.querySelectorAll(".tab-panel");

  const presetContainer = document.getElementById("preset-sample-container");
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const dropPrompt = document.getElementById("drop-zone-prompt");
  const dropPreview = document.getElementById("drop-preview-container");
  const previewImage = document.getElementById("preview-image");
  const btnRemoveImage = document.getElementById("btn-remove-image");
  const previewMetaFilename = document.getElementById("preview-meta-filename");
  const previewMetaSpecs = document.getElementById("preview-meta-specs");
  const btnAnalyze = document.getElementById("btn-analyze");

  // Results State Elements
  const idleState = document.getElementById("idle-results-state");
  const loadingState = document.getElementById("loading-results-state");
  const resultsContainer = document.getElementById("active-results-container");

  // Results Details
  const verdictBanner = document.getElementById("verdict-banner");
  const verdictText = document.getElementById("verdict-text");
  const confidencePercentage = document.getElementById("confidence-percentage");
  const riskDot = document.getElementById("risk-dot");
  const riskLevelTitle = document.getElementById("risk-level-title");
  const riskLevelDesc = document.getElementById("risk-level-desc");

  const valProbAi = document.getElementById("val-prob-ai");
  const valProbReal = document.getElementById("val-prob-real");
  const meterBarAi = document.getElementById("meter-bar-ai");
  const meterBarReal = document.getElementById("meter-bar-real");
  const latencyIndicator = document.getElementById("latency-indicator");

  const stageOriginalImg = document.getElementById("stage-original-img");
  const stageHeatmapImg = document.getElementById("stage-heatmap-img");
  const stageCaption = document.getElementById("stage-caption");
  const btnToggleOriginal = document.getElementById("btn-toggle-original");
  const btnToggleHeatmap = document.getElementById("btn-toggle-heatmap");

  const metaResolution = document.getElementById("meta-resolution");
  const metaFilesize = document.getElementById("meta-filesize");
  const metaDevice = document.getElementById("meta-device");

  const generatorCardsContainer = document.getElementById("generator-cards-container");
  const systemStatusText = document.getElementById("system-status-text");

  // Application State
  let currentFile = null;
  let currentPresetId = null;
  let currentAnalysis = null;

  /* ==========================================================================
     1. Tab Navigation
     ========================================================================== */
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const targetPanel = document.getElementById(btn.getAttribute("data-target"));
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });

  /* ==========================================================================
     2. System Health & Benchmark Data Initialization
     ========================================================================== */
  async function checkSystemHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        if (data.model_loaded) {
          const archDisplay = data.architecture === "convnext_tiny" ? "ConvNeXt-Tiny" : (data.architecture || "ConvNeXt-Tiny");
          systemStatusText.textContent = `Engine Active: ${archDisplay} (${data.device.toUpperCase()})`;
        } else {
          systemStatusText.textContent = "Engine Offline: Model not loaded";
        }
      }
    } catch (e) {
      systemStatusText.textContent = "Connecting to Local Inference Server...";
    }
  }

  async function loadBenchmarkData() {
    try {
      const res = await fetch("/api/model-info");
      if (!res.ok) return;
      const data = await res.json();

      if (data.generator_breakdown && generatorCardsContainer) {
        generatorCardsContainer.innerHTML = "";
        data.generator_breakdown.forEach(gen => {
          const item = document.createElement("div");
          item.className = "generator-item";
          item.innerHTML = `
            <div class="gen-header">
              <span class="gen-name">${gen.generator}</span>
              <span class="gen-type">${gen.category}</span>
            </div>
            <div class="gen-acc">${gen.accuracy.toFixed(1)}%</div>
            <div class="gen-meter">
              <div class="gen-fill" style="width: ${gen.accuracy}%;"></div>
            </div>
            <div class="gen-samples">${gen.samples} test images</div>
          `;
          generatorCardsContainer.appendChild(item);
        });
      }
    } catch (e) {
      console.warn("Could not load benchmark data", e);
    }
  }

  /* ==========================================================================
     3. Preset Samples Loading & Selection
     ========================================================================== */
  async function loadPresetSamples() {
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) return;
      const samples = await res.json();
      
      presetContainer.innerHTML = "";
      samples.forEach(sample => {
        const item = document.createElement("div");
        item.className = "preset-card-item";
        item.setAttribute("data-id", sample.id);
        
        const isAi = sample.expected === "AI-GENERATED";
        const badgeClass = isAi ? "ai" : "real";
        const badgeText = isAi ? "AI Gen" : "Real";

        item.innerHTML = `
          <div class="preset-thumb-wrap">
            <img class="preset-thumb" src="${sample.preview_url}" alt="${sample.title}">
            <span class="preset-badge-tag ${badgeClass}">${badgeText}</span>
          </div>
          <div class="preset-info-name" title="${sample.title}">${sample.title}</div>
          <div class="preset-info-gen">${sample.generator}</div>
        `;

        item.addEventListener("click", () => {
          selectPresetSample(sample, item);
        });

        presetContainer.appendChild(item);
      });
    } catch (e) {
      presetContainer.innerHTML = `<div class="preset-loading-skeleton">Unable to load demonstration presets.</div>`;
    }
  }

  function selectPresetSample(sample, cardElement) {
    // Deselect other cards
    document.querySelectorAll(".preset-card-item").forEach(c => c.classList.remove("selected"));
    cardElement.classList.add("selected");

    currentPresetId = sample.id;
    currentFile = null;

    // Update dropzone preview
    previewImage.src = sample.preview_url;
    previewMetaFilename.textContent = sample.title;
    previewMetaSpecs.textContent = sample.generator;

    dropPrompt.classList.add("hidden");
    dropPreview.classList.remove("hidden");
    btnAnalyze.disabled = false;

    // Instantly trigger detection analysis for smooth interactive testing
    triggerDetection();
  }

  /* ==========================================================================
     4. File Upload & Drag-and-Drop
     ========================================================================== */
  dropZone.addEventListener("click", (e) => {
    if (e.target.closest("#btn-remove-image")) return;
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUserFile(e.target.files[0]);
    }
  });

  // Drag and drop event handling
  ["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    }, false);
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    }, false);
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files[0]) {
      handleUserFile(dt.files[0]);
    }
  });

  // Global Clipboard paste support (Ctrl+V)
  window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.type.indexOf("image") === 0) {
        const blob = item.getAsFile();
        handleUserFile(blob);
        break;
      }
    }
  });

  function handleUserFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPEG, PNG, WEBP).");
      return;
    }

    currentFile = file;
    currentPresetId = null;
    document.querySelectorAll(".preset-card-item").forEach(c => c.classList.remove("selected"));

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewMetaFilename.textContent = file.name;
      previewMetaSpecs.textContent = `${(file.size / 1024).toFixed(1)} KB`;

      dropPrompt.classList.add("hidden");
      dropPreview.classList.remove("hidden");
      btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  btnRemoveImage.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUploadState();
  });

  function resetUploadState() {
    currentFile = null;
    currentPresetId = null;
    fileInput.value = "";
    previewImage.src = "";
    dropPreview.classList.add("hidden");
    dropPrompt.classList.remove("hidden");
    btnAnalyze.disabled = true;

    document.querySelectorAll(".preset-card-item").forEach(c => c.classList.remove("selected"));

    // Reset results to idle
    idleState.classList.remove("hidden");
    loadingState.classList.add("hidden");
    resultsContainer.classList.add("hidden");
  }

  // Mode & Sensitivity Controls
  const selectMode = document.getElementById("select-mode");
  const selectSensitivity = document.getElementById("select-sensitivity");
  const domainShiftText = document.getElementById("domain-shift-text");
  const domainShiftContainer = document.getElementById("domain-shift-container");

  if (selectMode) {
    selectMode.addEventListener("change", () => {
      if (currentFile || currentPresetId) triggerDetection();
    });
  }
  if (selectSensitivity) {
    selectSensitivity.addEventListener("change", () => {
      if (currentFile || currentPresetId) triggerDetection();
    });
  }

  /* ==========================================================================
     5. Inference Trigger & Analysis
     ========================================================================== */
  btnAnalyze.addEventListener("click", () => {
    triggerDetection();
  });

  async function triggerDetection() {
    if (!currentFile && !currentPresetId) return;

    // Show loading state
    idleState.classList.add("hidden");
    resultsContainer.classList.add("hidden");
    loadingState.classList.remove("hidden");
    btnAnalyze.disabled = true;

    try {
      const formData = new FormData();
      if (currentPresetId) {
        formData.append("preset_sample", currentPresetId);
      } else if (currentFile) {
        formData.append("file", currentFile);
      }

      const mode = selectMode ? selectMode.value : "ensemble";
      const threshold = selectSensitivity ? selectSensitivity.value : "0.50";
      formData.append("mode", mode);
      formData.append("threshold", threshold);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Inference failed");
      }

      const result = await response.json();
      currentAnalysis = result;
      displayResults(result);

    } catch (error) {
      console.error("Error during detection:", error);
      alert(`Detection Error: ${error.message}`);
      idleState.classList.remove("hidden");
      loadingState.classList.add("hidden");
    } finally {
      btnAnalyze.disabled = false;
    }
  }

  /* ==========================================================================
     6. Display Results
     ========================================================================== */
  function displayResults(data) {
    loadingState.classList.add("hidden");
    resultsContainer.classList.remove("hidden");

    const isAi = data.verdict === "AI-GENERATED";

    // 1. Verdict Banner
    verdictBanner.className = "card verdict-card " + (isAi ? "verdict-ai" : "verdict-real");
    verdictText.textContent = isAi ? "AI-GENERATED IMAGE" : "AUTHENTIC REAL IMAGE";
    confidencePercentage.textContent = `${data.confidence}%`;

    // 2. Risk Assessment
    riskDot.className = "risk-indicator-dot " + (data.risk_assessment.color === "emerald" ? "green" : data.risk_assessment.color);
    riskLevelTitle.textContent = data.risk_assessment.level;
    riskLevelDesc.textContent = data.risk_assessment.description;

    // 2b. Domain Consistency
    if (domainShiftText && domainShiftContainer) {
      domainShiftText.textContent = `Domain Reliability: ${data.domain_confidence} (Threshold: ${Math.round(data.threshold_used * 100)}%)`;
      if (data.domain_confidence && data.domain_confidence.includes("Low")) {
        domainShiftContainer.className = "domain-shift-pill borderline";
      } else {
        domainShiftContainer.className = "domain-shift-pill";
      }
    }

    // 3. Dual Probability Meter
    valProbAi.textContent = `${data.probabilities.AI_GENERATED}%`;
    valProbReal.textContent = `${data.probabilities.REAL}%`;
    latencyIndicator.textContent = `Inference Latency: ${data.metadata.inference_time_ms}ms (${data.mode_used})`;

    // Animate bars
    setTimeout(() => {
      meterBarAi.style.width = `${data.probabilities.AI_GENERATED}%`;
      meterBarReal.style.width = `${data.probabilities.REAL}%`;
    }, 50);

    // 4. Artifact & Spectral Inspection Stage
    stageOriginalImg.src = previewImage.src;
    stageHeatmapImg.src = data.artifact_heatmap || previewImage.src;
    
    // Reset view toggle to original
    btnToggleOriginal.classList.add("active");
    btnToggleHeatmap.classList.remove("active");
    stageOriginalImg.classList.remove("hidden");
    stageHeatmapImg.classList.add("hidden");
    stageCaption.textContent = "Displaying original target input image.";

    // 5. Diagnostics Metadata
    metaResolution.textContent = data.metadata.resolution;
    metaFilesize.textContent = `${data.metadata.file_size_kb} KB`;
    metaDevice.textContent = `${data.metadata.device.toUpperCase()} (${data.metadata.architecture})`;
  }

  /* ==========================================================================
     7. Forensic Artifact View Toggle
     ========================================================================== */
  btnToggleOriginal.addEventListener("click", () => {
    btnToggleOriginal.classList.add("active");
    btnToggleHeatmap.classList.remove("active");
    stageOriginalImg.classList.remove("hidden");
    stageHeatmapImg.classList.add("hidden");
    stageCaption.textContent = "Displaying original target input image.";
  });

  btnToggleHeatmap.addEventListener("click", () => {
    btnToggleHeatmap.classList.add("active");
    btnToggleOriginal.classList.remove("active");
    stageOriginalImg.classList.add("hidden");
    stageHeatmapImg.classList.remove("hidden");
    stageCaption.textContent = "Spatial frequency residual (Laplacian filter): highlights generative smoothing & boundary discrepancies.";
  });

  /* ==========================================================================
     Initial Boot
     ========================================================================== */
  checkSystemHealth();
  loadPresetSamples();
  loadBenchmarkData();
});
