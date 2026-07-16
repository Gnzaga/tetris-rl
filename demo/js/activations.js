// MarI/O-style activation visualisation (PLAN2.md §8, Phase F).
//
// Canvas-drawn views of one PolicyNet inference: conv1/conv2/conv3 feature-map
// grids (tiny heatmaps), the FC-256 activation strip (16×16), and a node-and-wire
// graph for the FC→5 action head (edge brightness = |weight × activation|, the
// winning action node glows). All pure-canvas, no deps; throttleable by the
// caller to every Nth inference if the frame budget demands.

const ACCENT = [124, 92, 255];   // violet (matches the demo theme)
const HOT = [64, 224, 208];      // teal/cyan for high activation
const BG = "#0b0e15";

// value v in [0,1] -> "rgb(...)" ramp: dark -> violet -> teal.
function ramp(v) {
  v = v < 0 ? 0 : v > 1 ? 1 : v;
  let r, g, b;
  if (v < 0.5) {
    const t = v / 0.5;
    r = Math.round(18 + (ACCENT[0] - 18) * t);
    g = Math.round(20 + (ACCENT[1] - 20) * t);
    b = Math.round(30 + (ACCENT[2] - 30) * t);
  } else {
    const t = (v - 0.5) / 0.5;
    r = Math.round(ACCENT[0] + (HOT[0] - ACCENT[0]) * t);
    g = Math.round(ACCENT[1] + (HOT[1] - ACCENT[1]) * t);
    b = Math.round(ACCENT[2] + (HOT[2] - ACCENT[2]) * t);
  }
  return `rgb(${r},${g},${b})`;
}

function normRange(data, len) {
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = data[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const span = mx - mn || 1;
  return { mn, span };
}

// Draw a C×(H×W) feature-map grid. act = { data, C, H, W, len }. Fits the grid
// into the canvas width; each map is a small heatmap (nearest-neighbour).
export function drawFeatureMaps(ctx, canvas, act, cols) {
  const { data, C, H, W, len } = act;
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!C) return;
  const rows = Math.ceil(C / cols);
  const gap = 3;
  const mapW = Math.floor((canvas.width - gap * (cols + 1)) / cols);
  const scale = Math.max(1, Math.floor(mapW / W));
  const cw = W * scale;
  const ch = H * scale;
  const { mn, span } = normRange(data, len);
  for (let m = 0; m < C; m++) {
    const gx = gap + (m % cols) * (cw + gap);
    const gy = gap + Math.floor(m / cols) * (ch + gap);
    const off = m * H * W;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        ctx.fillStyle = ramp((data[off + y * W + x] - mn) / span);
        ctx.fillRect(gx + x * scale, gy + y * scale, scale, scale);
      }
    }
  }
}

// FC-256 activation strip as a 16×16 heatmap grid.
export function drawFcStrip(ctx, canvas, fc, len) {
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const side = 16;
  const scale = Math.max(1, Math.floor(Math.min(canvas.width, canvas.height) / side));
  const { mn, span } = normRange(fc, len);
  for (let i = 0; i < len && i < side * side; i++) {
    const x = i % side;
    const y = Math.floor(i / side);
    ctx.fillStyle = ramp((fc[i] - mn) / span);
    ctx.fillRect(x * scale, y * scale, scale - 1, scale - 1);
  }
}

// Node-and-wire graph: 64 sampled FC nodes (left) -> 5 action nodes (right).
// Edge brightness = |weight × activation| (normalized); the winning action node
// (argmax logits) glows. weight is the 5×256 pi matrix; fc the 256 activations.
export function drawWireGraph(ctx, canvas, fc, weight, logits, chosen, legend, sample = 64) {
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);
  if (!weight || !fc) return;
  const n = weight[0].length; // 256
  const step = Math.max(1, Math.floor(n / sample));
  const idx = [];
  for (let j = 0; j < n && idx.length < sample; j += step) idx.push(j);

  const lx = 26;
  const rx = W - 46;
  const topPad = 14;
  const lgap = (H - 2 * topPad) / (idx.length - 1 || 1);
  const nA = weight.length; // 5
  const rgap = (H - 2 * topPad) / (nA - 1 || 1);
  const { mn: fmn, span: fspan } = normRange(fc, fc.length);

  // Precompute max |w*a| for normalization.
  let emax = 1e-9;
  for (const j of idx) {
    for (let a = 0; a < nA; a++) {
      const s = Math.abs(weight[a][j] * fc[j]);
      if (s > emax) emax = s;
    }
  }

  // Edges.
  for (let k = 0; k < idx.length; k++) {
    const j = idx[k];
    const y1 = topPad + k * lgap;
    for (let a = 0; a < nA; a++) {
      const y2 = topPad + a * rgap;
      const strength = Math.abs(weight[a][j] * fc[j]) / emax;
      if (strength < 0.04) continue;
      const alpha = Math.min(1, strength);
      const col = weight[a][j] >= 0 ? HOT : ACCENT;
      ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},${alpha.toFixed(3)})`;
      ctx.lineWidth = 0.4 + strength * 1.6;
      ctx.beginPath();
      ctx.moveTo(lx, y1);
      ctx.lineTo(rx, y2);
      ctx.stroke();
    }
  }

  // Left nodes (FC), coloured by activation.
  for (let k = 0; k < idx.length; k++) {
    const j = idx[k];
    const y1 = topPad + k * lgap;
    ctx.fillStyle = ramp((fc[j] - fmn) / fspan);
    ctx.beginPath();
    ctx.arc(lx, y1, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Right nodes (actions); winner glows.
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  for (let a = 0; a < nA; a++) {
    const y2 = topPad + a * rgap;
    const win = a === chosen;
    if (win) {
      ctx.shadowColor = "rgba(64,224,208,0.95)";
      ctx.shadowBlur = 14;
    }
    ctx.fillStyle = win ? "rgb(64,224,208)" : "rgba(150,150,170,0.85)";
    ctx.beginPath();
    ctx.arc(rx, y2, win ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = win ? "#d8fff9" : "rgba(180,180,200,0.9)";
    ctx.fillText(legend[a] ?? String(a), rx + 10, y2);
  }
}

// Blit a 96×96 uint8 obs to a canvas, nearest-neighbour upscaled to fill it.
export function drawObsInset(ctx, canvas, obs, obsSize = 96) {
  const scale = Math.max(1, Math.floor(canvas.width / obsSize));
  const img = ctx.createImageData(obsSize * scale, obsSize * scale);
  const d = img.data;
  const rowStride = obsSize * scale * 4;
  for (let y = 0; y < obsSize; y++) {
    for (let x = 0; x < obsSize; x++) {
      const v = obs[y * obsSize + x];
      for (let dy = 0; dy < scale; dy++) {
        let p = (y * scale + dy) * rowStride + x * scale * 4;
        for (let dx = 0; dx < scale; dx++) {
          d[p] = v; d[p + 1] = v; d[p + 2] = v; d[p + 3] = 255;
          p += 4;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
