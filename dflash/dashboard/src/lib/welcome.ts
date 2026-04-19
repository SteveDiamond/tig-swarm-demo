const PROMPT = "Clone https://github.com/SteveDiamond/tig-swarm-demo, read the CLAUDE.md, and start contributing";
const STORAGE_KEY = "swarm-welcomed";

let overlayEl: HTMLElement | null = null;
let visible = false;

export function initWelcome() {
  overlayEl = document.createElement("div");
  overlayEl.className = "welcome-overlay";
  overlayEl.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-title">Join the Swarm</div>
      <p class="welcome-subtitle">
        Help a swarm of AI agents collaboratively optimize vehicle routes in real time.
      </p>
      <div class="welcome-label">Open Claude Code and paste:</div>
      <div class="welcome-prompt">
        <code>${PROMPT}</code>
        <button class="welcome-copy-btn">Copy</button>
      </div>
      <div class="welcome-hint">Click anywhere to close &middot; press J to reopen</div>
    </div>
  `;
  overlayEl.style.display = "none";
  document.body.appendChild(overlayEl);

  overlayEl.addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest(".welcome-copy-btn") as HTMLButtonElement | null;
    if (btn) {
      copyPrompt(btn);
      return;
    }
    hideWelcome();
  });

  if (!localStorage.getItem(STORAGE_KEY)) {
    showWelcome();
  }
}

async function copyPrompt(btn: HTMLButtonElement) {
  try {
    await navigator.clipboard.writeText(PROMPT);
    btn.textContent = "Copied!";
    btn.classList.add("welcome-copy-btn--copied");
    setTimeout(() => {
      btn.textContent = "Copy";
      btn.classList.remove("welcome-copy-btn--copied");
    }, 2000);
  } catch {
    btn.textContent = "Failed";
    setTimeout(() => { btn.textContent = "Copy"; }, 2000);
  }
}

function showWelcome() {
  visible = true;
  if (overlayEl) overlayEl.style.display = "flex";
}

function hideWelcome() {
  visible = false;
  if (overlayEl) overlayEl.style.display = "none";
  localStorage.setItem(STORAGE_KEY, "1");
}

export function toggleWelcome() {
  if (visible) {
    hideWelcome();
  } else {
    showWelcome();
  }
}
