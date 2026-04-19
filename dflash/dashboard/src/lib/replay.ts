import type { WSMessage } from "../types";

let playing = false;

export async function startReplay(
  apiUrl: string,
  handleMessage: (msg: WSMessage) => void,
) {
  if (playing) return;
  playing = true;

  // Fetch history + current state (for num_instances)
  let history: any[];
  let numInstances = 1;
  try {
    const [historyRes, stateRes] = await Promise.all([
      fetch(`${apiUrl}/api/replay`),
      fetch(`${apiUrl}/api/state`),
    ]);
    history = await historyRes.json();
    if (stateRes.ok) {
      const state = await stateRes.json();
      numInstances = state.num_instances || 1;
    }
  } catch {
    playing = false;
    return;
  }

  if (!history.length) {
    playing = false;
    return;
  }

  // Show replay overlay
  const overlay = document.createElement("div");
  overlay.className = "replay-overlay";
  overlay.innerHTML = `
    <div class="replay-banner">EVOLUTION REPLAY</div>
    <div class="replay-progress">
      <span class="replay-step" id="replay-step">0 / ${history.length}</span>
      <span class="replay-score" id="replay-score"></span>
    </div>
  `;
  document.body.appendChild(overlay);

  const stepEl = document.getElementById("replay-step")!;
  const scoreEl = document.getElementById("replay-score")!;
  const firstScore = history[0].score;

  // Play through each best
  for (let i = 0; i < history.length; i++) {
    const entry = history[i];
    stepEl.textContent = `${i + 1} / ${history.length}`;
    scoreEl.textContent = `Score: ${entry.score.toFixed(4)}`;

    if (entry.route_data) {
      const prevScore = i > 0 ? history[i - 1].score : null;
      const incremental =
        prevScore != null && prevScore > 0
          ? ((entry.score - prevScore) / prevScore) * 100
          : null;
      handleMessage({
        type: "new_global_best",
        experiment_id: entry.experiment_id,
        agent_name: entry.agent_name,
        agent_id: "",
        score: entry.score,
        improvement_pct: firstScore > 0 ? ((entry.score - firstScore) / firstScore) * 100 : 0,
        incremental_improvement_pct: incremental,
        num_instances: numInstances,
        route_data: entry.route_data,
        timestamp: entry.created_at,
      });
    }

    // Wait between steps (faster for early ones, slower for later)
    const delay = i < 3 ? 2000 : 1500;
    await new Promise((r) => setTimeout(r, delay));
  }

  // Show final result
  const lastScore = history[history.length - 1].score;
  const totalImprovement = firstScore > 0 ? ((lastScore - firstScore) / firstScore) * 100 : 0;

  overlay.innerHTML = `
    <div class="replay-banner">EVOLUTION COMPLETE</div>
    <div class="replay-final">
      <div class="replay-final-score">${lastScore.toFixed(4)}</div>
      <div class="replay-final-improvement">${totalImprovement.toFixed(1)}% improvement</div>
      <div class="replay-final-steps">${history.length} breakthroughs</div>
    </div>
  `;

  // Dismiss after 5s or on click
  await new Promise<void>((resolve) => {
    const dismiss = () => { resolve(); };
    overlay.addEventListener("click", dismiss);
    setTimeout(dismiss, 8000);
  });

  overlay.remove();
  playing = false;
}
