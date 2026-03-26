export function createSocketService({ state, onPayload, onOpen, onStatus }) {
  function connect() {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${proto}://${window.location.host}/ws/demo`;
    const ws = new WebSocket(wsUrl);
    state.ws = ws;
    onStatus("connecting");

    ws.onopen = () => {
      onStatus("connected");
      if (state.reconnectHandle) {
        clearTimeout(state.reconnectHandle);
        state.reconnectHandle = null;
      }
      onOpen();
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        onPayload(msg);
      } catch {
        // ignore malformed packet
      }
    };

    ws.onclose = () => {
      onStatus("disconnected");
      if (!state.reconnectHandle) {
        state.reconnectHandle = setTimeout(connect, 1200);
      }
    };

    ws.onerror = () => {
      onStatus("disconnected");
      ws.close();
    };
  }

  return {
    connect,
  };
}
