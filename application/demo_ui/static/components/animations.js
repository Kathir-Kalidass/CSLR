import { escapeHtml } from "../core/utils.js";

export function initFlowMotion(state, refs) {
  const dot = refs.flowDot;
  const track = refs.flowTrack;
  if (!dot || !track || !window.gsap) {
    return;
  }

  if (state.flowTween) {
    state.flowTween.kill();
    state.flowTween = null;
  }

  const width = Math.max(10, track.clientWidth - 18);
  state.flowTween = window.gsap.to(dot, {
    x: width,
    duration: state.grandMode ? 1.35 : 1.85,
    ease: "none",
    repeat: -1,
    yoyo: true,
  });
}

export function runBootSequence(state, refs, sequence) {
  if (state.bootPlayed) {
    return;
  }

  const overlay = refs.bootOverlay;
  const list = refs.bootList;
  if (!overlay || !list) {
    state.bootPlayed = true;
    return;
  }

  state.bootPlayed = true;
  overlay.classList.add("show");
  list.innerHTML = sequence.map((item) => `<li><span class="boot-dot"></span><span>${escapeHtml(item)}</span></li>`).join("");

  const rows = list.querySelectorAll("li");
  rows.forEach((row, idx) => {
    setTimeout(() => {
      row.classList.add("done");
    }, 320 + idx * 420);
  });

  setTimeout(() => {
    overlay.classList.remove("show");
  }, 320 + sequence.length * 420 + 600);
}

export function animateWave(arr, state) {
  const bars = document.querySelectorAll(".wave-bar");
  bars.forEach((bar, idx) => {
    const raw = Number(arr[idx] || 0);
    const h = Math.max(8, Math.round(raw * 48));
    bar.style.height = `${h}px`;
    bar.style.opacity = state.ttsEnabled ? "1" : "0.25";
  });
}

export function initParticleBackground(state, particleCanvas) {
  const ctx = particleCanvas.getContext("2d");
  const nodes = [];
  const nodeCount = 64;
  const pointer = { x: 0, y: 0 };

  function resize() {
    particleCanvas.width = window.innerWidth;
    particleCanvas.height = window.innerHeight;
  }

  function seed() {
    nodes.length = 0;
    for (let i = 0; i < nodeCount; i += 1) {
      nodes.push({
        x: Math.random() * particleCanvas.width,
        y: Math.random() * particleCanvas.height,
        vx: (Math.random() - 0.5) * 0.24,
        vy: (Math.random() - 0.5) * 0.24,
        r: Math.random() * 1.7 + 0.8,
      });
    }
  }

  function render() {
    ctx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
    ctx.fillStyle = "rgba(9, 15, 31, 0.22)";
    ctx.fillRect(0, 0, particleCanvas.width, particleCanvas.height);

    nodes.forEach((node, i) => {
      const speedBoost = state.grandMode ? 1.7 : 1;
      node.x += node.vx * speedBoost;
      node.y += node.vy * speedBoost;

      if (node.x < -20) node.x = particleCanvas.width + 20;
      if (node.x > particleCanvas.width + 20) node.x = -20;
      if (node.y < -20) node.y = particleCanvas.height + 20;
      if (node.y > particleCanvas.height + 20) node.y = -20;

      const dxp = pointer.x - node.x;
      const dyp = pointer.y - node.y;
      const dp = Math.sqrt(dxp * dxp + dyp * dyp);
      if (dp < 150) {
        node.x -= dxp * 0.0007;
        node.y -= dyp * 0.0007;
      }

      ctx.beginPath();
      ctx.fillStyle = "rgba(94, 234, 212, 0.75)";
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
      ctx.fill();

      for (let j = i + 1; j < nodes.length; j += 1) {
        const other = nodes[j];
        const dx = node.x - other.x;
        const dy = node.y - other.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 140) {
          const alpha = (1 - d / 140) * (state.grandMode ? 0.3 : 0.16);
          ctx.strokeStyle = `rgba(45, 212, 191, ${alpha.toFixed(3)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(other.x, other.y);
          ctx.stroke();
        }
      }
    });

    state.particleRaf = requestAnimationFrame(render);
  }

  window.addEventListener("mousemove", (event) => {
    pointer.x = event.clientX;
    pointer.y = event.clientY;
  });

  window.addEventListener("resize", () => {
    resize();
    seed();
  });

  resize();
  seed();
  state.particleRaf = requestAnimationFrame(render);
}
