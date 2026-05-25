// ─────────────────────────────────────────────
// WEBSOCKET CONNECTION (Backend Integration)
// ─────────────────────────────────────────────
let backendData = null;
let wsConnected = false;
let wsReconnectTimer = null;
let socket = null;

function connectWebSocket() {
  if (socket && socket.readyState === WebSocket.OPEN) return;

  socket = new WebSocket("ws://localhost:8765");

  socket.onopen = () => {
    wsConnected = true;
    console.log("[WS] Connected to backend");
  };

  socket.onmessage = (event) => {
    try {
      backendData = JSON.parse(event.data);
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };

  socket.onclose = () => {
    wsConnected = false;
    console.log("[WS] Disconnected. Reconnecting...");

    if (!wsReconnectTimer) {
      wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        connectWebSocket();
      }, 3000);
    }
  };

  socket.onerror = () => {
    console.warn("[WS] Error");
  };
}

// Start WebSocket connection
connectWebSocket();

// --- SIMULATION CONSTANTS ---
const W = 800;
const H = 800;
const CENTER = 400;
const ROAD_WIDTH = 120;
const HALF_ROAD = ROAD_WIDTH / 2;
const LANE_WIDTH = 30;
const STOP_LINE = CENTER - HALF_ROAD; // 340
const EXIT_LINE = CENTER + HALF_ROAD; // 460
const SAFE_GAP = 40;
const NORMAL_SPEED = 0.9;
const EMERGENCY_SPEED = 2.5;

const VEHICLE_MODELS = [
  { type: 'sedan', length: 28, width: 14, radius: 4 },
  { type: 'suv', length: 32, width: 16, radius: 3 },
  { type: 'hatchback', length: 24, width: 14, radius: 5 },
  { type: 'van', length: 36, width: 16, radius: 2 }
];

// --- PATH DEFINITIONS ---
class LineSegment {
  constructor(x1, y1, x2, y2) {
    this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2;
    this.dx = x2 - x1; this.dy = y2 - y1;
    this.length = Math.hypot(this.dx, this.dy);
    this.angle = Math.atan2(this.dy, this.dx);
  }
  getPoint(d) {
    const t = d / this.length;
    return { x: this.x1 + this.dx * t, y: this.y1 + this.dy * t };
  }
  getAngle(_d) {
    return this.angle;
  }
}

class ArcSegment {
  constructor(cx, cy, r, startAngle, endAngle, clockwise) {
    this.cx = cx; this.cy = cy; this.r = r;
    this.startAngle = startAngle; this.endAngle = endAngle;
    this.clockwise = clockwise;
    let dAngle = endAngle - startAngle;
    if (clockwise) {
      while (dAngle < 0) dAngle += Math.PI * 2;
    } else {
      while (dAngle > 0) dAngle -= Math.PI * 2;
    }
    this.dAngle = dAngle;
    this.length = Math.abs(dAngle) * r;
  }
  getPoint(d) {
    const t = d / this.length;
    const angle = this.startAngle + this.dAngle * t;
    return { x: this.cx + this.r * Math.cos(angle), y: this.cy + this.r * Math.sin(angle) };
  }
  getAngle(d) {
    const t = d / this.length;
    const angle = this.startAngle + this.dAngle * t;
    return angle + (this.clockwise ? Math.PI / 2 : -Math.PI / 2);
  }
}

class Path {
  constructor(segments, exitDistance) {
    this.segments = segments;
    this.totalLength = segments.reduce((sum, seg) => sum + seg.length, 0);
    this.exitDistance = exitDistance || (segments[0].length + (segments.length > 1 ? segments[1].length : 0));
  }
  getPoint(d) {
    let currentD = d;
    for (let i = 0; i < this.segments.length; i++) {
      const seg = this.segments[i];
      if (currentD <= seg.length || i === this.segments.length - 1) {
        return seg.getPoint(currentD);
      }
      currentD -= seg.length;
    }
    return null;
  }
  getAngle(d) {
    let currentD = d;
    for (let i = 0; i < this.segments.length; i++) {
      const seg = this.segments[i];
      if (currentD <= seg.length || i === this.segments.length - 1) {
        return seg.getAngle(currentD);
      }
      currentD -= seg.length;
    }
    return 0;
  }
}

const PATHS = {
  N: {
    STRAIGHT: new Path([new LineSegment(415, 0, 415, 800)], 460),
    LEFT: new Path([
      new LineSegment(445, 0, 445, 340),
      new ArcSegment(460, 340, 15, Math.PI, Math.PI/2, false),
      new LineSegment(460, 355, 800, 355)
    ], 340 + 15 * Math.PI / 2),
    RIGHT: new Path([
      new LineSegment(415, 0, 415, 340),
      new ArcSegment(340, 340, 75, 0, Math.PI/2, true),
      new LineSegment(340, 415, 0, 415)
    ], 340 + 75 * Math.PI / 2)
  },
  S: {
    STRAIGHT: new Path([new LineSegment(385, 800, 385, 0)], 460),
    LEFT: new Path([
      new LineSegment(355, 800, 355, 460),
      new ArcSegment(340, 460, 15, 0, -Math.PI/2, false),
      new LineSegment(340, 445, 0, 445)
    ], 340 + 15 * Math.PI / 2),
    RIGHT: new Path([
      new LineSegment(385, 800, 385, 460),
      new ArcSegment(460, 460, 75, Math.PI, 3*Math.PI/2, true),
      new LineSegment(460, 385, 800, 385)
    ], 340 + 75 * Math.PI / 2)
  },
  E: {
    STRAIGHT: new Path([new LineSegment(800, 415, 0, 415)], 460),
    LEFT: new Path([
      new LineSegment(800, 445, 460, 445),
      new ArcSegment(460, 460, 15, -Math.PI/2, -Math.PI, false),
      new LineSegment(445, 460, 445, 800)
    ], 340 + 15 * Math.PI / 2),
    RIGHT: new Path([
      new LineSegment(800, 415, 460, 415),
      new ArcSegment(460, 340, 75, Math.PI/2, Math.PI, true),
      new LineSegment(385, 340, 385, 0)
    ], 340 + 75 * Math.PI / 2)
  },
  W: {
    STRAIGHT: new Path([new LineSegment(0, 385, 800, 385)], 460),
    LEFT: new Path([
      new LineSegment(0, 355, 340, 355),
      new ArcSegment(340, 340, 15, Math.PI/2, 0, false),
      new LineSegment(355, 340, 355, 0)
    ], 340 + 15 * Math.PI / 2),
    RIGHT: new Path([
      new LineSegment(0, 385, 340, 385),
      new ArcSegment(340, 460, 75, -Math.PI/2, 0, true),
      new LineSegment(415, 460, 415, 800)
    ], 340 + 75 * Math.PI / 2)
  }
};

// --- STATE ---
let vehicles = [];
let vehicleIdCounter = 0;

let rewardHistory = [];
let waitHistory = [];

const PHASES = [
  'N_GREEN', 'N_YELLOW', 'ALL_RED', 
  'E_GREEN', 'E_YELLOW', 'ALL_RED', 
  'S_GREEN', 'S_YELLOW', 'ALL_RED', 
  'W_GREEN', 'W_YELLOW', 'ALL_RED'
];
let currentPhaseIndex = 0;
let phaseTimer = 0;
const GREEN_TIME = 300; // frames
const YELLOW_TIME = 100; // frames
const RED_TIME = 100; // frames

function getSignalState(dir) {
  if (backendData && backendData.phase) {
    const phase = backendData.phase;

    if (phase === "YELLOW") return "YELLOW";
    if (phase === "ALL_RED") return "ALL_RED";

    if (phase === `${dir}_GREEN`) return "GREEN";

    return "RED";
  }

  // fallback
  const phase = PHASES[currentPhaseIndex];
  if (phase === 'ALL_RED') return 'ALL_RED';
  if (phase === `${dir}_GREEN`) return 'GREEN';
  if (phase === `${dir}_YELLOW`) return 'YELLOW';
  return 'RED';
}

// --- LOGIC ---
function spawnVehicle() {
  const dirs = ['N', 'E', 'S', 'W'];
  const turns = ['LEFT', 'STRAIGHT', 'RIGHT'];
  
  const dir = dirs[Math.floor(Math.random() * dirs.length)];
  const turn = turns[Math.floor(Math.random() * turns.length)];
  const isEmergency = Math.random() < 0.02;
  
  const path = PATHS[dir][turn];
  
  const spawnDist = 0;
  let clear = true;
  
  const getLane = (t) => t === 'LEFT' ? 1 : 2;
  
  for (const v of vehicles) {
    if (v.direction === dir && getLane(v.turn) === getLane(turn) && v.distance < 60) {
      clear = false;
      break;
    }
  }
  
  if (clear) {
    const pt = path.getPoint(spawnDist);
    if (!pt) return;
    
    const model = isEmergency ? { type: 'emergency', length: 30, width: 14, radius: 4 } : VEHICLE_MODELS[Math.floor(Math.random() * VEHICLE_MODELS.length)];
    
    vehicles.push({
      id: vehicleIdCounter++,
      direction: dir,
      turn: turn,
      path: path,
      distance: spawnDist,
      speed: isEmergency ? EMERGENCY_SPEED : NORMAL_SPEED,
      maxSpeed: isEmergency ? EMERGENCY_SPEED : NORMAL_SPEED,
      x: pt.x,
      y: pt.y,
      angle: path.getAngle(spawnDist),
      mustExit: false,
      exited: false,
      color: isEmergency ? '#ef4444' : `hsl(${Math.random() * 360}, 70%, 50%)`,
      isEmergency,
      modelType: model.type,
      length: model.length,
      width: model.width,
      radius: model.radius
    });
  }
}

function updateSignalsUI() {
  const dirs = ['N', 'E', 'S', 'W'];
  
  dirs.forEach(dir => {
    const isGreen = getSignalState(dir) === 'GREEN';
    const isYellow = getSignalState(dir) === 'YELLOW';
    const signalEl = document.getElementById(`signal-${dir}`);
    if (!signalEl) return;
    
    const red = signalEl.querySelector('.red');
    const yellow = signalEl.querySelector('.yellow');
    const green = signalEl.querySelector('.green');
    
    if (isGreen) {
      red.classList.remove("active");
      yellow.classList.remove("active");
      green.classList.add("active");
    } else if (isYellow) {
      red.classList.remove("active");
      yellow.classList.add("active");
      green.classList.remove("active");
    } else {
      green.classList.remove("active");
      yellow.classList.remove("active");
      red.classList.add("active");
    }
  });
}

function updateStatsUI() {
  const counts = { N: 0, E: 0, S: 0, W: 0 };
  const emergencyCounts = { N: 0, E: 0, S: 0, W: 0 };
  
  vehicles.forEach(v => {
    counts[v.direction]++;
    if (v.isEmergency) emergencyCounts[v.direction]++;
  });

  ['N', 'E', 'S', 'W'].forEach(dir => {
    const el = document.getElementById(`stat-${dir}`);
    if (el) {
      // Show both frontend vehicle count and backend queue count
      const backendCount = backendData?.vehicles?.[dir] ?? "–";
      el.innerText = `${dir}: ${counts[dir]} vis / ${backendCount} queue (E: ${emergencyCounts[dir]})`;
    }
  });
}

function updateRLStatsUI() {
  // ─── BACKEND DATA (replaces ALL dummy Math.random() values) ───
  if (backendData) {
    document.getElementById("metric-algo").innerText = "Algorithm: " + (backendData.algorithm || "–");
    document.getElementById("metric-epsilon").innerText = "Epsilon: " + (backendData.epsilon ?? "–");
    document.getElementById("metric-reward").innerText = "Reward: " + (backendData.reward ?? "–");

    document.getElementById("metric-wait").innerText = "Avg Wait Time: " + (backendData.waiting_time ?? "–") + "s";
    document.getElementById("metric-queue").innerText = "Queue Length: " + (backendData.queue_length ?? "–");
    document.getElementById("metric-throughput").innerText = "Throughput: " + (backendData.throughput ?? "–") + " veh/ep";

    document.getElementById("metric-phase").innerText = "Current Phase: " + (backendData.phase || "–");
    document.getElementById("metric-timer").innerText = "Phase Time: " + (backendData.time_in_phase ?? "–") + "s";

    // Episode progression
    const epEl = document.getElementById("metric-episode");
    if (epEl) {
      const ep = backendData.episode || 0;
      const maxEp = backendData.max_episodes || "–";
      const mode = backendData.mode ? backendData.mode.toUpperCase() : "–";
      epEl.innerText = `Episode: ${ep} / ${maxEp} [${mode}]`;
    }

    // Connection status
    const connEl = document.getElementById("metric-connection");
    if (connEl) {
      connEl.innerText = "⚡ Backend Connected";
      connEl.style.color = "#22c55e";
    }
  } else {
    // No backend — show waiting message
    document.getElementById("metric-algo").innerText = "Algorithm: Waiting...";
    document.getElementById("metric-epsilon").innerText = "Epsilon: –";
    document.getElementById("metric-reward").innerText = "Reward: –";
    document.getElementById("metric-wait").innerText = "Avg Wait Time: –";
    document.getElementById("metric-queue").innerText = "Queue Length: –";
    document.getElementById("metric-throughput").innerText = "Throughput: –";
    document.getElementById("metric-phase").innerText = "Current Phase: " + PHASES[currentPhaseIndex];
    document.getElementById("metric-timer").innerText = "Phase Time: " + Math.floor(phaseTimer / 60) + "s";

    const connEl = document.getElementById("metric-connection");
    if (connEl) {
      connEl.innerText = "⏳ Waiting for backend...";
      connEl.style.color = "#eab308";
    }
  }
}

// ─── GRAPH FUNCTIONS (using backend data when available) ───

function drawGraph(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  if (data.length < 2) return;

  const max = Math.max(...data);
  const min = Math.min(...data);

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  data.forEach((val, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((val - min) / (max - min + 0.0001)) * h;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

function updateGraphs() {
  // ─── USE BACKEND GRAPH DATA (replaces generateDummyTrainingData) ───
  if (backendData && backendData.graph) {
    rewardHistory = backendData.graph.reward || [];
    waitHistory = backendData.graph.wait || [];
  }

  drawGraph("rewardChart", rewardHistory, "#22c55e"); // green
  drawGraph("waitChart", waitHistory, "#ef4444");     // red
}

// ─── COMPARISON TABLE (sent from backend after --mode compare) ───

function updateComparisonTable() {
  const container = document.getElementById("comparison-table");
  if (!container) return;

  if (!backendData || !backendData.comparison) {
    container.innerHTML = '<div style="color:#888; font-size:11px;">Run <code>--mode compare</code> to see results</div>';
    return;
  }

  const rows = backendData.comparison;
  let html = `
    <table style="width:100%; border-collapse:collapse; font-size:11px;">
      <thead>
        <tr style="border-bottom:1px solid #555;">
          <th style="text-align:left; padding:4px; color:#ccc;">Algorithm</th>
          <th style="text-align:right; padding:4px; color:#ccc;">Avg Wait</th>
          <th style="text-align:right; padding:4px; color:#ccc;">Throughput</th>
          <th style="text-align:right; padding:4px; color:#ccc;">Reward</th>
        </tr>
      </thead>
      <tbody>`;

  for (const r of rows) {
    const algo = r.algorithm || r.Algorithm || "–";
    const wait = r.avg_wait || r["Avg Wait Time (s)"] || "–";
    const tput = r.throughput || r["Throughput (veh/min)"] || "–";
    const reward = r.reward || r["Total Reward"] || "–";

    html += `
      <tr style="border-bottom:1px solid #333;">
        <td style="padding:4px; color:#60a5fa;">${algo}</td>
        <td style="text-align:right; padding:4px;">${wait}s</td>
        <td style="text-align:right; padding:4px;">${tput}</td>
        <td style="text-align:right; padding:4px;">${reward}</td>
      </tr>`;
  }

  html += '</tbody></table>';
  container.innerHTML = html;
}

function render(ctx) {
  ctx.fillStyle = '#262626';
  ctx.fillRect(0, 0, W, H);

  // Draw roads
  ctx.fillStyle = '#404040';
  ctx.fillRect(CENTER - HALF_ROAD, 0, ROAD_WIDTH, H); // N-S
  ctx.fillRect(0, CENTER - HALF_ROAD, W, ROAD_WIDTH); // E-W
  ctx.fillRect(CENTER - HALF_ROAD, CENTER - HALF_ROAD, ROAD_WIDTH, ROAD_WIDTH); // Intersection

  // Draw markings
  ctx.strokeStyle = '#a3a3a3';
  ctx.lineWidth = 2;
  ctx.setLineDash([15, 15]);

  // N-S dashed lines
  ctx.beginPath();
  ctx.moveTo(CENTER, 0); ctx.lineTo(CENTER, STOP_LINE);
  ctx.moveTo(CENTER, EXIT_LINE); ctx.lineTo(CENTER, H);
  ctx.stroke();

  // E-W dashed lines
  ctx.beginPath();
  ctx.moveTo(0, CENTER); ctx.lineTo(STOP_LINE, CENTER);
  ctx.moveTo(EXIT_LINE, CENTER); ctx.lineTo(W, CENTER);
  ctx.stroke();

  // Lane dividers (solid)
  ctx.setLineDash([]);
  ctx.strokeStyle = '#737373';
  ctx.beginPath();
  ctx.moveTo(CENTER - LANE_WIDTH, 0); ctx.lineTo(CENTER - LANE_WIDTH, STOP_LINE);
  ctx.moveTo(CENTER + LANE_WIDTH, 0); ctx.lineTo(CENTER + LANE_WIDTH, STOP_LINE);
  ctx.moveTo(CENTER - LANE_WIDTH, EXIT_LINE); ctx.lineTo(CENTER - LANE_WIDTH, H);
  ctx.moveTo(CENTER + LANE_WIDTH, EXIT_LINE); ctx.lineTo(CENTER + LANE_WIDTH, H);

  ctx.moveTo(0, CENTER - LANE_WIDTH); ctx.lineTo(STOP_LINE, CENTER - LANE_WIDTH);
  ctx.moveTo(0, CENTER + LANE_WIDTH); ctx.lineTo(STOP_LINE, CENTER + LANE_WIDTH);
  ctx.moveTo(EXIT_LINE, CENTER - LANE_WIDTH); ctx.lineTo(W, CENTER - LANE_WIDTH);
  ctx.moveTo(EXIT_LINE, CENTER + LANE_WIDTH); ctx.lineTo(W, CENTER + LANE_WIDTH);
  ctx.stroke();

  // Stop lines
  ctx.lineWidth = 4;
  const drawStopLine = (dir, color) => {
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();

  if (dir === 'N') {
    // Coming from top → use RIGHT half
    ctx.moveTo(CENTER, STOP_LINE);
    ctx.lineTo(CENTER + HALF_ROAD, STOP_LINE);
  }

  if (dir === 'S') {
    // Coming from bottom → use LEFT half
    ctx.moveTo(CENTER - HALF_ROAD, EXIT_LINE);
    ctx.lineTo(CENTER, EXIT_LINE);
  }

  if (dir === 'E') {
    // Coming from right → use BOTTOM half
    ctx.moveTo(EXIT_LINE, CENTER);
    ctx.lineTo(EXIT_LINE, CENTER + HALF_ROAD);
  }

  if (dir === 'W') {
    // Coming from left → use TOP half
    ctx.moveTo(STOP_LINE, CENTER - HALF_ROAD);
    ctx.lineTo(STOP_LINE, CENTER);
  }

  ctx.stroke();
};

  ['N', 'E', 'S', 'W'].forEach(dir => {
    const state = getSignalState(dir);
    let color = '#ef4444'; // RED
    if (state === 'GREEN') color = '#22c55e';
    else if (state === 'YELLOW') color = '#eab308';
    
    drawStopLine(dir, color);
  });

  // Free Left Lane Indicators
  ctx.globalAlpha = 1.0;
  function drawCurvedLeftArrow(x, y, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Arrow shaft
    ctx.beginPath();
    ctx.moveTo(-16, 0);
    ctx.lineTo(4, 0);
    ctx.quadraticCurveTo(16, 0, 16, -12);
    ctx.stroke();

    // Arrow head (left turn)
    ctx.beginPath();
    ctx.moveTo(16, -18);
    ctx.lineTo(10, -8);
    ctx.moveTo(16, -18);
    ctx.lineTo(22, -8);
    ctx.stroke();

    ctx.restore();
  }

  // NORTH (top → down, free-left = right lane)
  drawCurvedLeftArrow(CENTER + LANE_WIDTH + (LANE_WIDTH / 2), STOP_LINE, Math.PI / 2);
  // SOUTH (bottom → up, free-left = left lane)
  drawCurvedLeftArrow(CENTER - LANE_WIDTH - (LANE_WIDTH / 2), EXIT_LINE, -Math.PI / 2);
  // EAST (right → left, free-left = bottom lane)
  drawCurvedLeftArrow(EXIT_LINE, CENTER + LANE_WIDTH + (LANE_WIDTH / 2), Math.PI);
  // WEST (left → right, free-left = top lane)
  drawCurvedLeftArrow(STOP_LINE, CENTER - LANE_WIDTH - (LANE_WIDTH / 2), 0);

  function drawStraightArrow(x, y, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Arrow shaft
    ctx.beginPath();
    ctx.moveTo(-16, 0);
    ctx.lineTo(22, 0);
    ctx.stroke();

    // Arrow head (straight)
    ctx.beginPath();
    ctx.moveTo(22, 0);
    ctx.lineTo(12, -8);
    ctx.moveTo(22, 0);
    ctx.lineTo(12, 8);
    ctx.stroke();

    ctx.restore();
  }

  // NORTH (top → down, straight/right = inner lane)
  drawStraightArrow(CENTER + (LANE_WIDTH / 2), STOP_LINE - 80, Math.PI / 2);
  // SOUTH (bottom → up, straight/right = inner lane)
  drawStraightArrow(CENTER - (LANE_WIDTH / 2), EXIT_LINE + 80, -Math.PI / 2);
  // EAST (right → left, straight/right = inner lane)
  drawStraightArrow(EXIT_LINE + 80, CENTER + (LANE_WIDTH / 2), Math.PI);
  // WEST (left → right, straight/right = inner lane)
  drawStraightArrow(STOP_LINE - 80, CENTER - (LANE_WIDTH / 2), 0);
  
  ctx.globalAlpha = 1.0;

  // Draw vehicles
  for (const v of vehicles) {
    ctx.save();
    ctx.translate(v.x, v.y);
    ctx.rotate(v.angle);

    if (v.isEmergency) {
      const blink = Math.floor(Date.now() / 200) % 2 === 0;

      // Body
      ctx.fillStyle = '#ff3b3b';
      ctx.beginPath();
      ctx.roundRect(-v.length/2, -v.width/2, v.length, v.width, v.radius);
      ctx.fill();

      // Blinking border (white-blue police effect)
      ctx.strokeStyle = blink ? '#ffffff' : '#3b82f6';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.roundRect(-v.length/2, -v.width/2, v.length, v.width, v.radius);
      ctx.stroke();

      // Top light (centered on roof)
      if (blink) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(-3, -5, 6, 10);
      }
      
      // Windows for emergency
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.beginPath();
      ctx.roundRect(v.length/2 - 12, -v.width/2 + 2, 6, v.width - 4, 2); // Windshield
      ctx.roundRect(-v.length/2 + 4, -v.width/2 + 2, 4, v.width - 4, 2); // Rear window
      ctx.fill();
    } else {
      // Normal vehicle body
      ctx.fillStyle = v.color;
      ctx.beginPath();
      ctx.roundRect(-v.length/2, -v.width/2, v.length, v.width, v.radius);
      ctx.fill();
      
      // Windows
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.beginPath();
      if (v.modelType === 'sedan') {
        ctx.roundRect(v.length/2 - 12, -v.width/2 + 2, 6, v.width - 4, 2); // Windshield
        ctx.roundRect(-v.length/2 + 4, -v.width/2 + 2, 4, v.width - 4, 2); // Rear window
      } else if (v.modelType === 'suv') {
        ctx.roundRect(v.length/2 - 14, -v.width/2 + 2, 6, v.width - 4, 1); // Windshield
        ctx.roundRect(-v.length/2 + 10, -v.width/2 + 3, 8, v.width - 6, 1); // Sunroof
        ctx.roundRect(-v.length/2 + 2, -v.width/2 + 2, 4, v.width - 4, 1); // Rear window
      } else if (v.modelType === 'hatchback') {
        ctx.roundRect(v.length/2 - 12, -v.width/2 + 2, 6, v.width - 4, 2); // Windshield
        ctx.roundRect(-v.length/2 + 2, -v.width/2 + 2, 3, v.width - 4, 1); // Rear window
      } else if (v.modelType === 'van') {
        ctx.roundRect(v.length/2 - 9, -v.width/2 + 2, 5, v.width - 4, 1); // Windshield
      }
      ctx.fill();
    }

    // Headlights
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(v.length/2 - 2, -v.width/2 + 3, 2, 0, Math.PI * 2);
    ctx.arc(v.length/2 - 2, v.width/2 - 3, 2, 0, Math.PI * 2);
    ctx.fill();

    // Taillights
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(-v.length/2 + 2, -v.width/2 + 3, 2, 0, Math.PI * 2);
    ctx.arc(-v.length/2 + 2, v.width/2 - 3, 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }
}

function update() {
  // ─── SIGNAL PHASING ───
  // When backend is connected, signals are driven by backend data.
  // When disconnected, use local timer-based cycling as fallback.
  if (!backendData) {
    phaseTimer++;
    const currentPhase = PHASES[currentPhaseIndex];
    let phaseDuration = RED_TIME;
    if (currentPhase.endsWith('_GREEN')) phaseDuration = GREEN_TIME;
    else if (currentPhase.endsWith('_YELLOW')) phaseDuration = YELLOW_TIME;
    
    if (phaseTimer >= phaseDuration) {
      phaseTimer = 0;
      currentPhaseIndex = (currentPhaseIndex + 1) % PHASES.length;
    }
  }

  if (Math.random() < 0.04) spawnVehicle();

  updateSignalsUI();
  updateStatsUI();
  updateRLStatsUI();
  updateGraphs();
  updateComparisonTable();

  const getLane = (t) => t === 'LEFT' ? 1 : 2;

  // ═══════════════════════════════════════════════════════════════
  // SAFE BRAKING DISTANCE — vehicles approaching stop line during
  // YELLOW must begin braking this many pixels before the line
  // ═══════════════════════════════════════════════════════════════
  const SAFE_BRAKE_DISTANCE = 60;

  for (const v of vehicles) {
    // ─── Queue & front-vehicle detection ───
    const sameLaneVehicles = vehicles
      .filter(o => o.direction === v.direction && getLane(o.turn) === getLane(v.turn))
      .sort((a, b) => b.distance - a.distance);
    
    const queueIndex = sameLaneVehicles.indexOf(v);
    const queueOffset = SAFE_GAP * queueIndex;

    let frontDist = Infinity;
    let frontVehicle = null;

    if (queueIndex > 0) {
      frontVehicle = sameLaneVehicles[queueIndex - 1];
      frontDist = frontVehicle.distance - v.distance;
    }

    // ===============================
    // 🚨 FREE LEFT HARD OVERRIDE
    // ===============================
    

    // Free-left flag
    const isFreeLeft = v.turn === 'LEFT';
    // Free-left is allowed on RED but NOT during ALL_RED transition
    // (ALL_RED is for intersection clearance — even free-left waits)
    const isAllRed = backendData?.phase === "ALL_RED";

    if (isFreeLeft) {

      // ALWAYS MOVE (ignore signals completely)
      let targetSpeed = v.maxSpeed;

      // Only collision handling
      if (frontDist < SAFE_GAP) {
        v.speed = 0;
        if (frontVehicle) {
          v.distance = Math.min(v.distance, frontVehicle.distance - SAFE_GAP);
        }
      } else if (frontDist < SAFE_GAP + 30) {
        targetSpeed = NORMAL_SPEED * ((frontDist - SAFE_GAP) / 30);
        v.speed += (targetSpeed - v.speed) * 0.2;
      } else {
        v.speed += (targetSpeed - v.speed) * 0.2;
    }

    // Move vehicle
    v.distance += v.speed;

    const pos = v.path.getPoint(v.distance);
    if (pos) {
      v.x = pos.x;
      v.y = pos.y;
      v.angle = v.path.getAngle(v.distance);
    }

    // Exit condition
    const EXIT_BOUNDARY = { top: -50, bottom: H + 50, left: -50, right: W + 50 };
    if (
      v.x < EXIT_BOUNDARY.left ||
      v.x > EXIT_BOUNDARY.right ||
      v.y < EXIT_BOUNDARY.top ||
      v.y > EXIT_BOUNDARY.bottom
    ) {
      v.exited = true;
    }

    continue; // 🚨 CRITICAL — stops ALL signal logic below
    }

    // ─── Get EXACT signal state for this vehicle's direction ───
    const signalState = getSignalState(v.direction);  // 'GREEN', 'YELLOW', or 'RED'
    const isGreen  = signalState === 'GREEN';
    const isYellow = signalState === 'YELLOW';
    const isRed    = signalState === 'RED';

    // ─── Position checks ───
    const stopLinePos = STOP_LINE - 14 - queueOffset;   // where THIS vehicle should stop
    const isBeforeStopLine = v.distance + 14 <= STOP_LINE;
    const isInside = v.distance + 14 > STOP_LINE && v.distance - 14 < v.path.exitDistance;
    const isPastIntersection = v.distance - 14 >= v.path.exitDistance;

    // ─── Track legal entry: vehicle only gets "entered" flag when it crosses
    //     the stop line during a GREEN phase. This prevents YELLOW/RED sneaking.
    if (!v.enteredOnGreen && isInside && isGreen) {
      v.enteredOnGreen = true;
    }

    // Base target speed
    let targetSpeed = v.isEmergency ? EMERGENCY_SPEED : NORMAL_SPEED;

    // ═══════════════════════════════════════════════════════════════
    //  RULE 1: VEHICLES INSIDE INTERSECTION — EXIT ONLY IF LEGAL
    // ═══════════════════════════════════════════════════════════════
    if (isInside || isPastIntersection) {
      if (v.enteredOnGreen || (isFreeLeft && !isAllRed)) {
        v.mustExit = true;
        targetSpeed = Math.max(v.speed, NORMAL_SPEED);
      } else {
        v.mustExit = true;
        targetSpeed = NORMAL_SPEED;
      }
    }
    // ═══════════════════════════════════════════════════════════════
    //  🔥 THE FIX: RULE 2, 3 & 4: HARD STOP ON YELLOW & RED
    // ═══════════════════════════════════════════════════════════════
    else if ((isAllRed || isRed || isYellow) && !isFreeLeft) {
      const distToStop = stopLinePos - v.distance;
      
      if (distToStop < SAFE_BRAKE_DISTANCE && distToStop > 0) {
        // Smoothly brake as they approach the line
        targetSpeed = NORMAL_SPEED * (distToStop / SAFE_BRAKE_DISTANCE);
      }
      
      // Absolute Physics Lock: Hit the brakes to exactly 0, do not cross stop line
      if (v.distance + 14 >= STOP_LINE - queueOffset) {
        v.distance = stopLinePos;
        targetSpeed = 0;
        v.speed = 0;
      }
    }
    // ═══════════════════════════════════════════════════════════════
    //  RULE 5: GREEN — Vehicle may proceed safely
    // ═══════════════════════════════════════════════════════════════
    else if (isGreen && !isFreeLeft) {
      const intersectionOccupied = vehicles.some(other => {
        if (other.id === v.id) return false;
        const otherInside = other.distance + 14 > STOP_LINE && other.distance - 14 < other.path.exitDistance;
        return otherInside && other.direction !== v.direction && other.turn !== 'LEFT';
      });
      
      // If someone is stuck in the middle, wait for them to clear
      if (intersectionOccupied) {
        const distToStop = stopLinePos - v.distance;
        if (distToStop < 40 && distToStop > 0) targetSpeed = NORMAL_SPEED * (distToStop / 40);
        
        if (v.distance + 14 >= STOP_LINE - queueOffset) {
          v.distance = stopLinePos;
          targetSpeed = 0;
          v.speed = 0;
        }
      }
    }

    // ═══════════════════════════════════════════════════════════════
    //  RULE 6: SAFE DISTANCE — Never overlap with front vehicle
    // ═══════════════════════════════════════════════════════════════
    if (frontDist < SAFE_GAP) {
      v.speed = 0;
      if (frontVehicle) {
        v.distance = Math.min(v.distance, frontVehicle.distance - SAFE_GAP);
      }
    } else if (frontDist < SAFE_GAP + 30) {
      targetSpeed = Math.min(targetSpeed, NORMAL_SPEED * ((frontDist - SAFE_GAP) / 30));
      v.speed += (targetSpeed - v.speed) * 0.2;
      if (v.speed < 0.01) v.speed = 0;
    } else {
      // Apply speed smoothing
      v.speed += (targetSpeed - v.speed) * 0.2;
      if (v.speed < 0.01) v.speed = 0;
    }

    // ═══════════════════════════════════════════════════════════════
    //  RULE 7: FINAL GUARD — Prevent ANY stop-line crossing when
    //          signal is not GREEN (catches floating-point overshoots)
    // ═══════════════════════════════════════════════════════════════
    if (!isGreen && !v.mustExit && !v.enteredOnGreen && v.turn !== 'LEFT') {
      // If vehicle would overshoot past stop line, clamp it
      if (v.distance + v.speed + 14 > STOP_LINE - queueOffset) {
        v.distance = stopLinePos;
        v.speed = 0;
      }
    }

    // ─── Move vehicle ───
    v.distance += v.speed;

    // ─── Debug: detect illegal crossing ───
    if (!isGreen && !v.mustExit && v.distance + 14 > STOP_LINE && isBeforeStopLine) {
      console.error(`[VIOLATION] Vehicle ${v.id} (${v.direction}, ${v.turn}) crossed stop line during ${signalState}!`);
    }

    const pos = v.path.getPoint(v.distance);
    if (pos) {
      v.x = pos.x;
      v.y = pos.y;
      v.angle = v.path.getAngle(v.distance);
    }

    const EXIT_BOUNDARY = { top: -50, bottom: H + 50, left: -50, right: W + 50 };
    if (v.x < EXIT_BOUNDARY.left || v.x > EXIT_BOUNDARY.right || v.y < EXIT_BOUNDARY.top || v.y > EXIT_BOUNDARY.bottom) {
      v.exited = true;
    }
  }

  vehicles = vehicles.filter(v => !v.exited);

  const canvas = document.getElementById('simCanvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    render(ctx);
  }

  requestAnimationFrame(update);
}

// Start simulation
window.onload = () => {
  requestAnimationFrame(update);
};
