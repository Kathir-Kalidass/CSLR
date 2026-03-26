# Demo UI Components Map

This folder contains modular UI building blocks for the CSLR demo frontend.

- `layout.js`
  - Builds the full page shell and static sections.
  - Renders controls, flow stages, module cards, gauges, wave bars, and timeline chips.
- `popup.js`
  - Handles reusable popup modal open/close behavior.
- `camera.js`
  - Owns webcam permission flow, stream start/stop, and overlay simulation drawing.
- `animations.js`
  - Handles flow-dot motion, particle background, boot sequence animation, and wave bar animation.
- `pipeline-ui.js`
  - Central dynamic UI updater for live websocket payloads:
    - status chips
    - control state
    - narration
    - flow progress
    - module cards
    - timeline
    - analytics gauges
    - transcript history
    - parser console

## Wiring

`/static/app.js` is the entrypoint that composes all components and services:

1. Render shell (`layout.js`)
2. Initialize popup/camera/animations
3. Connect websocket (`services/socket.js`)
4. Apply backend payload through `pipeline-ui.js`
5. Run optional automation (Auto Demo / Presenter Mode)
