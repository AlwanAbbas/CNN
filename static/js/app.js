/* ==========================================================================
   app.js — Logic utama: Webcam, Upload, Prediksi, UI Update
   ========================================================================== */

"use strict";

/* ── Referensi DOM ──────────────────────────────────────────────────────── */
const video           = document.getElementById("video");
const canvas          = document.getElementById("canvas");
const videoPlaceholder= document.getElementById("video-placeholder");
const capturePreview  = document.getElementById("capture-preview");
const capturedImg     = document.getElementById("captured-img");
const camStatus       = document.getElementById("cam-status");
const fileInput       = document.getElementById("file-input");
const dropzone        = document.getElementById("dropzone");
const dropIdle        = document.getElementById("drop-idle");
const dropPreview     = document.getElementById("drop-preview");
const dropImg         = document.getElementById("drop-img");
const dropFilename    = document.getElementById("drop-filename");
const dropFilesize    = document.getElementById("drop-filesize");
const toast           = document.getElementById("toast");
const modelStatus     = document.getElementById("model-status");
const modelStatusText = modelStatus.querySelector(".badge-text");

// Buttons
const btnStartCam     = document.getElementById("btn-start-cam");
const btnCapture      = document.getElementById("btn-capture");
const btnAnalyzeCam   = document.getElementById("btn-analyze-cam");
const btnAnalyzeUpload= document.getElementById("btn-analyze-upload");
const btnDiscard      = document.getElementById("btn-discard");
const btnRemoveFile   = document.getElementById("btn-remove-file");
const themeToggle     = document.getElementById("theme-toggle");
const navCamera       = document.getElementById("nav-camera");
const navUpload       = document.getElementById("nav-upload");

// Result panel
const resultPanel     = document.getElementById("result-panel");
const resultLoading   = document.getElementById("result-loading");
const resultError     = document.getElementById("result-error");
const resultSuccess   = document.getElementById("result-success");
const resultClass     = document.getElementById("result-class");
const resultConfText  = document.getElementById("result-conf-text");
const resultConfBar   = document.getElementById("result-conf-bar");
const resultImg       = document.getElementById("result-img");
const metaTotalClass  = document.getElementById("meta-total-class");
const predictionsList = document.getElementById("predictions-list");
const errorMsg        = document.getElementById("error-msg");
const classCountBadge = document.getElementById("class-count-badge");
const classCountText  = document.getElementById("class-count-text");

/* ── State ──────────────────────────────────────────────────────────────── */
let stream      = null;   // MediaStream dari webcam
let capturedB64 = null;   // Gambar webcam (base64)
let uploadFile  = null;   // File dari input upload
let currentTheme= document.documentElement.getAttribute("data-theme") || "dark";

/* ─────────────────────────────────────────────────────────────────────────
   TEMA DARK / LIGHT
───────────────────────────────────────────────────────────────────────── */
themeToggle.addEventListener("click", () => {
  currentTheme = currentTheme === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", currentTheme);
  localStorage.setItem("batik-theme", currentTheme);
});

// Terapkan tema tersimpan
const savedTheme = localStorage.getItem("batik-theme");
if (savedTheme) {
  currentTheme = savedTheme;
  document.documentElement.setAttribute("data-theme", currentTheme);
}

/* ─────────────────────────────────────────────────────────────────────────
   NAVIGASI TAB
───────────────────────────────────────────────────────────────────────── */
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.getElementById(`tab-${tabId}`).classList.add("active");
  document.getElementById(`nav-${tabId}`).classList.add("active");
}

navCamera.addEventListener("click", () => switchTab("camera"));
navUpload.addEventListener("click", () => switchTab("upload"));

/* ─────────────────────────────────────────────────────────────────────────
   TOAST NOTIFIKASI
───────────────────────────────────────────────────────────────────────── */
let toastTimer = null;
function showToast(msg, type = "info", durasi = 3000) {
  toast.textContent = msg;
  toast.className   = `toast ${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), durasi);
}

/* ─────────────────────────────────────────────────────────────────────────
   WEBCAM
───────────────────────────────────────────────────────────────────────── */
btnStartCam.addEventListener("click", async () => {
  if (stream) {
    // Matikan kamera jika sudah aktif
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    video.style.display = "none";
    videoPlaceholder.style.display = "flex";
    btnStartCam.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
      Aktifkan Kamera`;
    camStatus.className = "status-dot red";
    btnCapture.disabled = true;
    btnAnalyzeCam.disabled = true;
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    video.style.display = "block";
    videoPlaceholder.style.display = "none";
    capturePreview.style.display = "none";

    btnStartCam.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><rect x="9" y="9" width="6" height="6"/></svg>
      Matikan Kamera`;
    camStatus.className    = "status-dot green";
    btnCapture.disabled    = false;
    btnAnalyzeCam.disabled = true;
    capturedB64            = null;
    showToast("Kamera aktif!", "success");
  } catch (err) {
    console.error(err);
    showToast("Gagal mengakses kamera: " + err.message, "error", 4000);
  }
});

btnCapture.addEventListener("click", () => {
  if (!stream) return;

  const ctx = canvas.getContext("2d");
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  capturedB64 = canvas.toDataURL("image/jpeg", 0.92);
  capturedImg.src = capturedB64;

  capturePreview.style.display = "flex";
  btnAnalyzeCam.disabled       = false;
  showToast("Foto berhasil diambil!", "success");
});

btnDiscard.addEventListener("click", () => {
  capturePreview.style.display = "none";
  btnAnalyzeCam.disabled       = true;
  capturedB64                  = null;
});

/* ─────────────────────────────────────────────────────────────────────────
   UPLOAD & DRAG-DROP
───────────────────────────────────────────────────────────────────────── */
dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setUploadFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) setUploadFile(f);
});

function setUploadFile(file) {
  const allowed = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
  if (!allowed.includes(file.type)) {
    showToast("Format tidak didukung! Gunakan JPG, PNG, BMP, atau WebP.", "error");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showToast("Ukuran file melebihi 16 MB!", "error");
    return;
  }

  uploadFile = file;
  const url  = URL.createObjectURL(file);
  dropImg.src        = url;
  dropFilename.textContent = file.name;
  dropFilesize.textContent = formatBytes(file.size);
  dropIdle.style.display    = "none";
  dropPreview.style.display = "flex";
  btnAnalyzeUpload.disabled = false;
  showToast("Gambar siap dianalisis.", "info");
}

btnRemoveFile.addEventListener("click", e => {
  e.stopPropagation();
  uploadFile = null;
  fileInput.value = "";
  dropIdle.style.display    = "flex";
  dropPreview.style.display = "none";
  btnAnalyzeUpload.disabled = true;
});

function formatBytes(b) {
  if (b < 1024)       return b + " B";
  if (b < 1048576)    return (b / 1024).toFixed(1)    + " KB";
  return (b / 1048576).toFixed(2)   + " MB";
}

/* ─────────────────────────────────────────────────────────────────────────
   PREDIKSI — WEBCAM
───────────────────────────────────────────────────────────────────────── */
btnAnalyzeCam.addEventListener("click", async () => {
  if (!capturedB64) return;
  tampilkanLoading(capturedB64);

  try {
    const res  = await fetch("/predict", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ image: capturedB64 }),
    });
    const data = await res.json();
    tampilkanHasil(data, capturedB64);
  } catch (err) {
    tampilkanError("Gagal menghubungi server: " + err.message);
  }
});

/* ─────────────────────────────────────────────────────────────────────────
   PREDIKSI — UPLOAD
───────────────────────────────────────────────────────────────────────── */
btnAnalyzeUpload.addEventListener("click", async () => {
  if (!uploadFile) return;
  const previewUrl = URL.createObjectURL(uploadFile);
  tampilkanLoading(previewUrl);

  const form = new FormData();
  form.append("file", uploadFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();
    tampilkanHasil(data, previewUrl);
  } catch (err) {
    tampilkanError("Gagal menghubungi server: " + err.message);
  }
});

/* ─────────────────────────────────────────────────────────────────────────
   UI HELPERS — PANEL HASIL
───────────────────────────────────────────────────────────────────────── */
function tampilkanLoading(imgSrc) {
  resultPanel.style.display  = "block";
  resultLoading.style.display= "flex";
  resultError.style.display  = "none";
  resultSuccess.style.display= "none";
  resultImg.src              = imgSrc;
  resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function tampilkanError(msg) {
  resultLoading.style.display= "none";
  resultError.style.display  = "flex";
  resultSuccess.style.display= "none";
  errorMsg.textContent        = msg;
  showToast(msg, "error", 5000);
}

function tampilkanHasil(data, imgSrc) {
  resultLoading.style.display= "none";

  if (!data.sukses) {
    tampilkanError(data.error || "Terjadi kesalahan tidak diketahui.");
    return;
  }

  resultError.style.display  = "none";
  resultSuccess.style.display= "block";
  resultImg.src               = imgSrc;
  resultClass.textContent     = data.kelas;
  resultConfText.textContent  = data.confidence.toFixed(2) + "%";
  metaTotalClass.textContent  = data.semua_kelas.length;

  // Animasi confidence bar
  requestAnimationFrame(() => {
    resultConfBar.style.width = "0%";
    setTimeout(() => {
      resultConfBar.style.width = data.confidence + "%";
    }, 50);
  });

  // Render top-10 prediksi
  const top = data.semua_kelas.slice(0, 10);
  predictionsList.innerHTML = top.map((item, i) => `
    <div class="pred-item">
      <span class="pred-rank">${i + 1}</span>
      <span class="pred-name" title="${item.kelas}">${item.kelas}</span>
      <span class="pred-pct">${item.confidence.toFixed(2)}%</span>
      <div class="pred-bar-wrap">
        <div class="pred-bar" style="width:${item.confidence}%"></div>
      </div>
    </div>
  `).join("");

  showToast(`Terdeteksi: ${data.kelas} (${data.confidence.toFixed(1)}%)`, "success");
}

/* ─────────────────────────────────────────────────────────────────────────
   INFO MODEL — saat halaman pertama dimuat
───────────────────────────────────────────────────────────────────────── */
async function muatInfoModel() {
  try {
    const res  = await fetch("/model-info");
    const data = await res.json();

    if (data.sukses) {
      modelStatus.classList.remove("loading", "error");
      modelStatus.classList.add("ready");
      modelStatusText.textContent = `${data.jumlah_kelas} kelas`;
      classCountBadge.style.display = "flex";
      classCountText.textContent    = `${data.jumlah_kelas} kelas batik`;
    } else {
      modelStatus.classList.remove("loading", "ready");
      modelStatus.classList.add("error");
      modelStatusText.textContent = "Model belum ada";
    }
  } catch {
    modelStatus.classList.remove("loading", "ready");
    modelStatus.classList.add("error");
    modelStatusText.textContent = "Server error";
  }
}

// Muat info model setelah halaman siap
window.addEventListener("load", muatInfoModel);
