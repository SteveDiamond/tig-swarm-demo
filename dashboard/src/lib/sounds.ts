let ctx: AudioContext | null = null;
let muted = false;

function getCtx(): AudioContext {
  if (!ctx) ctx = new AudioContext();
  return ctx;
}

function ensureResumed() {
  if (ctx?.state === "suspended") ctx.resume();
}

export function isMuted(): boolean {
  return muted;
}

export function toggleMute(): boolean {
  muted = !muted;
  return muted;
}

function playTone(
  freq: number,
  duration: number,
  type: OscillatorType = "sine",
  volume = 0.15,
  freqEnd?: number,
) {
  if (muted) return;
  ensureResumed();
  const c = getCtx();
  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, c.currentTime);
  if (freqEnd) {
    osc.frequency.exponentialRampToValueAtTime(freqEnd, c.currentTime + duration);
  }
  gain.gain.setValueAtTime(volume, c.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + duration);
  osc.connect(gain).connect(c.destination);
  osc.start();
  osc.stop(c.currentTime + duration);
}

function playNoise(duration: number, volume = 0.08) {
  if (muted) return;
  ensureResumed();
  const c = getCtx();
  const bufferSize = c.sampleRate * duration;
  const buffer = c.createBuffer(1, bufferSize, c.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) {
    data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / bufferSize, 3);
  }
  const source = c.createBufferSource();
  source.buffer = buffer;
  const gain = c.createGain();
  gain.gain.setValueAtTime(volume, c.currentTime);
  const filter = c.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.setValueAtTime(2000, c.currentTime);
  filter.frequency.exponentialRampToValueAtTime(200, c.currentTime + duration);
  filter.Q.value = 1;
  source.connect(filter).connect(gain).connect(c.destination);
  source.start();
}

// ── Sound events ──

export function soundAgentJoined() {
  playTone(880, 0.25, "sine", 0.08);
  setTimeout(() => playTone(1100, 0.15, "sine", 0.05), 80);
}

const STRATEGY_FREQ: Record<string, number> = {
  construction: 440,
  local_search: 494,
  metaheuristic: 523,
  constraint_relaxation: 587,
  decomposition: 659,
  hybrid: 698,
  data_structure: 784,
  other: 392,
};

export function soundHypothesisProposed(strategyTag: string) {
  const freq = STRATEGY_FREQ[strategyTag] || 440;
  playTone(freq, 0.35, "triangle", 0.1);
}

export function soundExperimentPublished() {
  playTone(600, 0.06, "sine", 0.04);
}

export function soundNewGlobalBest() {
  // Whoosh (filtered noise sweep)
  playNoise(1.5, 0.12);
  // Rising tone
  setTimeout(() => {
    playTone(220, 1.2, "sine", 0.15, 880);
  }, 200);
  // Resolve chord
  setTimeout(() => {
    playTone(440, 0.8, "sine", 0.08);
    playTone(554, 0.8, "sine", 0.06);
    playTone(659, 0.8, "sine", 0.05);
  }, 800);
}

// ── Ambient heartbeat ──

let heartbeatInterval: number | null = null;

export function startHeartbeat(agentCount: number) {
  stopHeartbeat();
  if (muted || agentCount === 0) return;

  const bpm = Math.min(30 + agentCount * 4, 80);
  const intervalMs = (60 / bpm) * 1000;

  heartbeatInterval = window.setInterval(() => {
    playTone(55, 0.15, "sine", 0.03);
  }, intervalMs);
}

export function stopHeartbeat() {
  if (heartbeatInterval !== null) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
  }
}
