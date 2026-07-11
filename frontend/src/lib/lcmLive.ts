import { writable } from 'svelte/store';


export enum LCMLiveStatus {
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
    WAIT = "wait",
    SEND_FRAME = "send_frame",
    TIMEOUT = "timeout",
}

const initStatus: LCMLiveStatus = LCMLiveStatus.DISCONNECTED;

export const lcmLiveStatus = writable<LCMLiveStatus>(initStatus);
export const streamId = writable<string | null>(null);

let websocket: WebSocket | null = null;
export const lcmLiveActions = {
    async start(getSreamdata: () => any[]) {
        return new Promise((resolve, reject) => {
            let settled = false;
            const settle = (fn: () => void) => { if (!settled) { settled = true; fn(); } };

            try {
                const userId = crypto.randomUUID();
                const websocketURL = `${window.location.protocol === "https:" ? "wss" : "ws"
                    }:${window.location.host}/api/ws/${userId}`;

                // Keep a local reference so stale onclose handlers can detect
                // they belong to a superseded connection and avoid clobbering
                // the status set by the newer one.
                const ws = new WebSocket(websocketURL);
                websocket = ws;

                ws.onopen = () => {
                    console.log("Connected to websocket");
                };
                ws.onclose = () => {
                    // Only update global state when this is still the active socket.
                    // If stop() or a newer start() has already replaced websocket,
                    // leave the new connection's state alone.
                    if (websocket === ws) {
                        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                        streamId.set(null);
                        websocket = null;
                    }
                    console.log("Disconnected from websocket");
                    settle(() => resolve({ status: "disconnected" }));
                };
                ws.onerror = (err) => {
                    console.error(err);
                };
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    switch (data.status) {
                        case "connected":
                            lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                            streamId.set(userId);
                            settle(() => resolve({ status: "connected", userId }));
                            break;
                        case "send_frame":
                            lcmLiveStatus.set(LCMLiveStatus.SEND_FRAME);
                            const streamData = getSreamdata();
                            ws.send(JSON.stringify({ status: "next_frame" }));
                            for (const d of streamData) {
                                this.send(d);
                            }
                            break;
                        case "wait":
                            lcmLiveStatus.set(LCMLiveStatus.WAIT);
                            break;
                        case "timeout":
                            console.log("timeout");
                            lcmLiveStatus.set(LCMLiveStatus.TIMEOUT);
                            streamId.set(null);
                            settle(() => reject(new Error("timeout")));
                            break;
                        case "error":
                            console.log(data.message);
                            lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                            streamId.set(null);
                            settle(() => reject(new Error(data.message)));
                            break;
                    }
                };

            } catch (err) {
                console.error(err);
                lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                streamId.set(null);
                settle(() => reject(err as Error));
            }
        });
    },
    send(data: Blob | { [key: string]: any }) {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (data instanceof Blob) {
                websocket.send(data);
            } else {
                websocket.send(JSON.stringify(data));
            }
        } else {
            console.log("WebSocket not connected");
        }
    },
    async stop() {
        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
        streamId.set(null);
        const ws = websocket;
        websocket = null;   // clear first so onclose sees websocket !== ws
        if (ws) {
            ws.close();
        }
    },
};
