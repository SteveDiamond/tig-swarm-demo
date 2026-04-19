import type { WSMessage } from "../types";

type MessageHandler = (msg: WSMessage) => void;

export class SwarmWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: MessageHandler[] = [];
  private reconnectDelay = 1000;
  private maxDelay = 10000;

  constructor(url: string) {
    this.url = url;
  }

  onMessage(handler: MessageHandler) {
    this.handlers.push(handler);
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log("[WS] Connected");
        this.reconnectDelay = 1000;
        this.updateStatus(true);
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as WSMessage;
          this.handlers.forEach((h) => h(msg));
        } catch (e) {
          console.warn("[WS] Parse error:", e);
        }
      };

      this.ws.onclose = () => {
        console.log("[WS] Disconnected, reconnecting...");
        this.updateStatus(false);
        setTimeout(() => this.connect(), this.reconnectDelay);
        this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxDelay);
      };

      this.ws.onerror = () => {
        this.ws?.close();
      };
    } catch {
      setTimeout(() => this.connect(), this.reconnectDelay);
    }
  }

  private updateStatus(connected: boolean) {
    const el = document.getElementById("ws-status");
    if (el) {
      el.textContent = connected ? "LIVE" : "RECONNECTING...";
      el.className = connected ? "ws-status connected" : "ws-status disconnected";
    }
  }
}
