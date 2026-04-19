/**
 * Animate a number from current to target with easing.
 */
export function counterTween(
  el: HTMLElement,
  target: number,
  duration = 400,
  decimals = 0,
): void {
  const start = parseFloat(el.textContent || "0") || 0;
  const delta = target - start;
  if (Math.abs(delta) < 0.001) return;

  const startTime = performance.now();

  function tick(now: number) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + delta * eased;
    el.textContent = decimals > 0 ? current.toFixed(decimals) : Math.round(current).toString();
    if (progress < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

/**
 * Pulse glow effect on an element.
 */
export function pulseGlow(el: HTMLElement, color = "#00e5ff", duration = 600): void {
  el.style.transition = `box-shadow ${duration / 2}ms ease-out`;
  el.style.boxShadow = `0 0 20px ${color}66, inset 0 0 10px ${color}22`;
  setTimeout(() => {
    el.style.boxShadow = "";
    setTimeout(() => {
      el.style.transition = "";
    }, duration / 2);
  }, duration / 2);
}

/**
 * Flash the entire viewport with a subtle color overlay.
 */
export function viewportFlash(color = "rgba(0, 229, 255, 0.03)", duration = 150): void {
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: ${color}; z-index: 9999; pointer-events: none;
    opacity: 1; transition: opacity ${duration}ms ease-out;
  `;
  document.body.appendChild(overlay);
  requestAnimationFrame(() => {
    overlay.style.opacity = "0";
    setTimeout(() => overlay.remove(), duration);
  });
}

/**
 * Format a timestamp to HH:MM:SS.
 */
export function formatTime(timestamp: string): string {
  const d = new Date(timestamp);
  return d.toLocaleTimeString("en-US", { hour12: false });
}
