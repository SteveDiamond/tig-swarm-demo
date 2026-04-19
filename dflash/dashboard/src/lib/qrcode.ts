const REPO_URL = "https://demo.discoveryatscale.com";

let overlayEl: HTMLElement | null = null;
let visible = false;

export function initQROverlay() {
  overlayEl = document.createElement("div");
  overlayEl.className = "qr-overlay";
  overlayEl.innerHTML = `
    <div class="qr-card">
      <div class="qr-title">Join the Swarm</div>
      <div class="qr-canvas-wrap">
        <canvas id="qr-canvas" width="200" height="200"></canvas>
      </div>
      <div class="qr-url">${REPO_URL.replace("https://", "")}</div>
      <div class="qr-instructions">Clone this repo, read CLAUDE.md, start contributing</div>
      <div class="qr-hint">Press Q to close</div>
    </div>
  `;
  overlayEl.style.display = "none";
  document.body.appendChild(overlayEl);

  overlayEl.addEventListener("click", () => toggleQR());

  // Draw QR code
  drawQR();
}

async function drawQR() {
  try {
    // Dynamic import of qrcode library
    const QRCode = await import("qrcode");
    const canvas = document.getElementById("qr-canvas") as HTMLCanvasElement;
    if (canvas) {
      await QRCode.toCanvas(canvas, REPO_URL, {
        width: 200,
        margin: 1,
        color: { dark: "#e8edf5", light: "#0a0e16" },
      });
    }
  } catch {
    // Fallback: just show the URL prominently
    const wrap = overlayEl?.querySelector(".qr-canvas-wrap");
    if (wrap) wrap.innerHTML = `<div class="qr-fallback">${REPO_URL}</div>`;
  }
}

export function toggleQR() {
  visible = !visible;
  if (overlayEl) {
    overlayEl.style.display = visible ? "flex" : "none";
  }
}
