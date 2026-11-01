// Interactive front end: point editing, rendering, and worker management.
// Coordinates are kept in [-1, 1]^2 (y up); the canvas maps them to pixels.

const RESOLUTION = 121;
const COLORS = { 1: [31, 95, 191], [-1]: [212, 82, 28] };
const NAMES = {
    adaboost: 'AdaBoost',
    adaboostv: 'AdaBoostV',
    madaboost: 'MadaBoost',
    smoothboost: 'SmoothBoost',
    corrective_erlpboost: 'Corrective ERLPBoost',
    lpboost: 'LPBoost',
    erlpboost: 'ERLPBoost',
    softboost: 'SoftBoost',
    totalboost: 'TotalBoost',
    mlpboost: 'MLPBoost',
    graph_separation: 'Graph Separation Boosting',
    sparsiboost: 'SparsiBoost',
    brownboost: 'BrownBoost',
};
const LEARNER_NAMES = {
    decision_tree: 'decision trees',
    logistic_regression: 'logistic regression',
    bounded_linear: 'bounded linear functions',
    naive_bayes: 'Gaussian naive Bayes',
    rbf: 'RBF bumps',
};
// Which optional parameter groups each algorithm uses.
const USES = {
    adaboost: ['tolerance'],
    adaboostv: ['tolerance'],
    madaboost: ['tolerance'],
    smoothboost: ['smooth'],
    corrective_erlpboost: ['tolerance', 'nu'],
    lpboost: ['tolerance', 'nu'],
    erlpboost: ['tolerance', 'nu'],
    softboost: ['tolerance', 'nu'],
    totalboost: ['tolerance'],
    mlpboost: ['tolerance', 'nu'],
    graph_separation: [],
    sparsiboost: ['sparsi'],
    brownboost: ['brown'],
};

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const statusBox = $('status');

let points = []; // { x, y, label }
let history = [];
let field = null; // { grid, resolution } of the last successful run
let stale = false;

// ---------------------------------------------------------------- drawing

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const size = Math.round(canvas.getBoundingClientRect().width * dpr);
    if (size > 0 && canvas.width !== size) {
        canvas.width = canvas.height = size;
    }
    draw();
}

const toWorld = (px, py) => [(px / canvas.width) * 2 - 1, 1 - (py / canvas.height) * 2];

function draw() {
    const { width: w, height: h } = canvas;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    if (field) {
        drawField(field);
    }
    drawAxes();
    for (const p of points) {
        drawMarker(ctx, w, h, p, 5 * (window.devicePixelRatio || 1));
    }
}

function drawField({ grid, resolution: n }) {
    const off = new OffscreenCanvas(n, n);
    const octx = off.getContext('2d');
    const image = octx.createImageData(n, n);
    // Shade relative to the largest |confidence| so that ensembles with
    // small margins remain visible.
    const scale = Math.max(1e-12, ...grid.map(Math.abs));
    for (let i = 0; i < n * n; i++) {
        const c = grid[i];
        const [r, g, b] = COLORS[c >= 0 ? 1 : -1];
        const t = 0.12 + 0.5 * Math.abs(c) / scale;
        image.data[4 * i] = Math.round(255 + (r - 255) * t);
        image.data[4 * i + 1] = Math.round(255 + (g - 255) * t);
        image.data[4 * i + 2] = Math.round(255 + (b - 255) * t);
        image.data[4 * i + 3] = 255;
    }
    octx.putImageData(image, 0, 0);
    ctx.save();
    ctx.globalAlpha = stale ? 0.45 : 1;
    ctx.imageSmoothingEnabled = true;
    // Grid sample (r, c) sits at pixel ((c / (n - 1)) w, (r / (n - 1)) h);
    // stretch by half a cell on each side so the samples land there.
    const cw = canvas.width / (n - 1);
    const ch = canvas.height / (n - 1);
    ctx.drawImage(off, -cw / 2, -ch / 2, canvas.width + cw, canvas.height + ch);
    drawBoundary(grid, n);
    ctx.restore();
}

// Zero level set of the confidence by marching squares with linear
// interpolation along cell edges.
function drawBoundary(grid, n) {
    const sx = canvas.width / (n - 1);
    const sy = canvas.height / (n - 1);
    const at = (r, c) => grid[r * n + c];
    const lerp = (a, b) => a / (a - b);
    ctx.beginPath();
    for (let r = 0; r < n - 1; r++) {
        for (let c = 0; c < n - 1; c++) {
            const v = [at(r, c), at(r, c + 1), at(r + 1, c + 1), at(r + 1, c)];
            // Corners: top-left, top-right, bottom-right, bottom-left.
            const cross = [];
            const corners = [[0, 0], [1, 0], [1, 1], [0, 1]];
            for (let k = 0; k < 4; k++) {
                const a = v[k];
                const b = v[(k + 1) % 4];
                if ((a >= 0) !== (b >= 0)) {
                    const t = lerp(a, b);
                    const [x0, y0] = corners[k];
                    const [x1, y1] = corners[(k + 1) % 4];
                    cross.push([(c + x0 + (x1 - x0) * t) * sx, (r + y0 + (y1 - y0) * t) * sy]);
                }
            }
            for (let k = 0; k + 1 < cross.length; k += 2) {
                ctx.moveTo(...cross[k]);
                ctx.lineTo(...cross[k + 1]);
            }
        }
    }
    ctx.strokeStyle = 'rgba(20, 20, 40, 0.85)';
    ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1);
    ctx.stroke();
}

function drawAxes() {
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
    ctx.restore();
}

// Draws a +1 point as a circle and a -1 point as a square of half-size
// `radius` on a `w × h` context.
function drawMarker(target, w, h, { x, y, label }, radius) {
    const px = ((x + 1) / 2) * w;
    const py = ((1 - y) / 2) * h;
    const [r, g, b] = COLORS[label];
    target.beginPath();
    if (label === 1) {
        target.arc(px, py, 1.1 * radius, 0, 2 * Math.PI);
    } else {
        target.rect(px - radius, py - radius, 2 * radius, 2 * radius);
    }
    target.fillStyle = `rgb(${r}, ${g}, ${b})`;
    target.fill();
    target.lineWidth = 0.3 * radius;
    target.strokeStyle = '#fff';
    target.stroke();
}

// ---------------------------------------------------------------- editing

function setPoints(next) {
    history.push(points);
    if (history.length > 200) history.shift();
    points = next;
    pointsChanged();
}

function pointsChanged() {
    stale = field !== null;
    const pos = points.filter((p) => p.label === 1).length;
    $('count').textContent = `${points.length} points (+1: ${pos}, −1: ${points.length - pos})`;
    $('nu').max = Math.max(1, points.length);
    draw();
    if ($('auto').checked) {
        run();
    } else if (stale) {
        setStatus('Points changed. Press Run to retrain.');
    }
}

function selectedLabel() {
    return Number(document.querySelector('input[name="cls"]:checked').value);
}

function addOrRemove(event, flip) {
    const rect = canvas.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const py = ((event.clientY - rect.top) / rect.height) * canvas.height;
    const [x, y] = toWorld(px, py);
    if (event.altKey) {
        if (points.length === 0) return;
        let best = 0;
        let bestDist = Infinity;
        points.forEach((p, i) => {
            const d = (p.x - x) ** 2 + (p.y - y) ** 2;
            if (d < bestDist) [best, bestDist] = [i, d];
        });
        setPoints(points.filter((_, i) => i !== best));
        return;
    }
    const label = flip ? -selectedLabel() : selectedLabel();
    setPoints([...points, { x, y, label }]);
}

canvas.addEventListener('click', (e) => addOrRemove(e, e.shiftKey));
canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    addOrRemove(e, true);
});

$('undo').addEventListener('click', () => {
    if (history.length === 0) return;
    points = history.pop();
    pointsChanged();
});
$('clear').addEventListener('click', () => {
    field = null;
    setPoints([]);
    setStatus('Add points of both classes, then press Run.');
});

// ---------------------------------------------------------------- presets

// Small seeded generator so the example data are reproducible.
function mulberry32(seed) {
    return () => {
        seed |= 0;
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function gaussian(rand) {
    const u = 1 - rand();
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const clip = (v) => Math.max(-0.97, Math.min(0.97, v));
const uniform = (rand) => rand() * 1.8 - 0.9;

// Each preset maps a seeded generator to a list of points. The seed is
// fixed, so the thumbnail shows exactly the points that a click places.
const PRESETS = [
    ['linear', 'Linear', (rand) => {
        const out = [];
        while (out.length < 50) {
            const x = uniform(rand);
            const y = uniform(rand);
            const s = y - 0.5 * x - 0.1;
            if (Math.abs(s) > 0.12) out.push({ x, y, label: s > 0 ? 1 : -1 });
        }
        return out;
    }],
    ['noisy', 'Noisy linear', (rand) => {
        const out = [];
        for (let i = 0; i < 60; i++) {
            const x = uniform(rand);
            const y = uniform(rand);
            let label = y > 0.6 * x ? 1 : -1;
            if (rand() < 0.1) label = -label;
            out.push({ x, y, label });
        }
        return out;
    }],
    ['blobs', 'Blobs', (rand) => {
        const out = [];
        for (let i = 0; i < 50; i++) {
            const label = i % 2 ? 1 : -1;
            out.push({ x: 0.4 * label + 0.22 * gaussian(rand), y: 0.25 * label + 0.22 * gaussian(rand), label });
        }
        return out;
    }],
    ['xor', 'XOR', (rand) => {
        const out = [];
        for (let i = 0; i < 80; i++) {
            const x = uniform(rand);
            const y = uniform(rand);
            out.push({ x, y, label: x * y >= 0 ? 1 : -1 });
        }
        return out;
    }],
    ['circles', 'Circles', (rand) => {
        const out = [];
        for (let i = 0; i < 80; i++) {
            const label = i % 2 ? 1 : -1;
            const radius = (label === 1 ? 0.3 : 0.75) + 0.07 * gaussian(rand);
            const theta = 2 * Math.PI * rand();
            out.push({ x: radius * Math.cos(theta), y: radius * Math.sin(theta), label });
        }
        return out;
    }],
    ['moons', 'Moons', (rand) => {
        const out = [];
        for (let i = 0; i < 80; i++) {
            const label = i % 2 ? 1 : -1;
            const t = Math.PI * rand();
            const [x, y] = label === 1
                ? [Math.cos(t) - 0.5, Math.sin(t) - 0.25]
                : [0.5 - Math.cos(t), 0.25 - Math.sin(t)];
            out.push({ x: 0.6 * x + 0.05 * gaussian(rand), y: 0.6 * y + 0.05 * gaussian(rand), label });
        }
        return out;
    }],
    ['spirals', 'Spirals', (rand) => {
        const out = [];
        for (let i = 0; i < 100; i++) {
            const label = i % 2 ? 1 : -1;
            const t = 0.25 + 2.5 * Math.PI * rand();
            const r = (0.85 * t) / (2.75 * Math.PI);
            const phase = label === 1 ? 0 : Math.PI;
            out.push({
                x: r * Math.cos(t + phase) + 0.03 * gaussian(rand),
                y: r * Math.sin(t + phase) + 0.03 * gaussian(rand),
                label,
            });
        }
        return out;
    }],
    ['checker', 'Checkerboard', (rand) => {
        const out = [];
        for (let i = 0; i < 120; i++) {
            const x = uniform(rand);
            const y = uniform(rand);
            const cell = Math.floor((x + 0.9) / 0.6) + Math.floor((y + 0.9) / 0.6);
            out.push({ x, y, label: cell % 2 ? 1 : -1 });
        }
        return out;
    }],
];

const PRESET_SEED = 20261101;
const generate = (make, seed) =>
    make(mulberry32(seed)).map((p) => ({ ...p, x: clip(p.x), y: clip(p.y) }));

function drawThumbnail(target, data) {
    const dpr = window.devicePixelRatio || 1;
    const size = Math.round(target.getBoundingClientRect().width * dpr) || 80;
    target.width = target.height = size;
    const tctx = target.getContext('2d');
    for (const p of data) {
        drawMarker(tctx, size, size, p, size / 45);
    }
}

for (const [key, title, make] of PRESETS) {
    const button = document.createElement('button');
    button.type = 'button';
    button.title = `Load the "${title}" example`;
    const thumb = document.createElement('canvas');
    const caption = document.createElement('span');
    caption.textContent = title;
    button.append(thumb, caption);
    $('presets').append(button);
    const data = generate(make, PRESET_SEED);
    requestAnimationFrame(() => drawThumbnail(thumb, data));
    button.addEventListener('click', () => {
        field = null;
        setPoints(data.map((p) => ({ ...p })));
        loadedMessage(`Loaded “${title}”.`);
    });
}

// Random Gaussian clusters, each with a random label, plus optional label
// noise. The seed is reported so that a configuration can be recreated.
function randomData(rand) {
    const k = 2 + Math.floor(rand() * 5);
    const labels = Array.from({ length: k }, (_, i) => (i % 2 ? 1 : -1));
    for (let i = k - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [labels[i], labels[j]] = [labels[j], labels[i]];
    }
    const noise = rand() < 0.5 ? 0 : 0.05;
    const out = [];
    for (const label of labels) {
        const cx = rand() * 1.4 - 0.7;
        const cy = rand() * 1.4 - 0.7;
        const sx = 0.06 + 0.2 * rand();
        const sy = 0.06 + 0.2 * rand();
        const count = 10 + Math.floor(rand() * 21);
        for (let i = 0; i < count; i++) {
            const flip = rand() < noise ? -1 : 1;
            out.push({ x: cx + sx * gaussian(rand), y: cy + sy * gaussian(rand), label: flip * label });
        }
    }
    return out;
}

$('random').addEventListener('click', () => {
    const seed = 1 + Math.floor(Math.random() * 999999);
    field = null;
    setPoints(generate(randomData, seed));
    loadedMessage(`Random data (seed ${seed}).`);
});

function loadedMessage(text) {
    if (!$('auto').checked) setStatus(`${text} Press Run to train.`);
}

// ---------------------------------------------------------------- parameters

function updateParamVisibility() {
    const uses = USES[$('algorithm').value];
    for (const el of document.querySelectorAll('.param')) {
        const group = el.dataset.for;
        el.hidden = group !== 'all' && !uses.includes(group);
    }
    const learner = $('learner').value;
    for (const el of document.querySelectorAll('.lparam')) {
        el.hidden = el.dataset.learner !== learner;
    }
}
for (const id of ['algorithm', 'learner']) {
    $(id).addEventListener('change', () => {
        updateParamVisibility();
        if ($('auto').checked) run();
    });
}
for (const id of [
    'maxIterations', 'tolerance', 'nu', 'gamma', 'kappa', 'nHypotheses', 'gap',
    'brownTolerance', 'edgeThreshold',
    'maxDepth', 'splitBy', 'regularization', 'logisticOutput', 'varSmoothing', 'bandwidth',
]) {
    $(id).addEventListener('change', () => {
        if ($('auto').checked) run();
    });
}

// Booster and weak-learner parameters; the wasm side reads only those of
// the chosen algorithm and learner.
function readParams() {
    const num = (id) => Number($(id).value);
    return {
        booster: {
            maxIterations: Math.round(num('maxIterations')),
            tolerance: num('tolerance'),
            nu: num('nu'),
            gamma: num('gamma'),
            kappa: num('kappa'),
            // Empty SparsiBoost targets mean "not set": T = 0, v = NaN.
            nHypotheses: $('nHypotheses').value === '' ? 0 : Math.round(num('nHypotheses')),
            gap: $('gap').value === '' ? NaN : num('gap'),
            brownTolerance: num('brownTolerance'),
            edgeThreshold: num('edgeThreshold'),
        },
        weak: {
            maxDepth: Math.round(num('maxDepth')),
            splitBy: $('splitBy').value,
            regularization: num('regularization'),
            output: $('logisticOutput').value,
            varSmoothing: num('varSmoothing'),
            bandwidth: num('bandwidth'),
        },
    };
}

// ---------------------------------------------------------------- training

let worker = null;
let nextId = 0;
let running = null; // { id, algorithm }
let rerun = false;

function startWorker() {
    worker?.terminate();
    worker = new Worker('worker.js', { type: 'module' });
    worker.onmessage = onResult;
    worker.onerror = (e) => {
        e.preventDefault();
        finish();
        setStatus(`Could not load the WebAssembly module: ${e.message || 'unknown error'}`, 'error');
    };
}

function setStatus(html, kind = '') {
    statusBox.className = `status ${kind}`;
    statusBox.innerHTML = html;
}

function setBusy(busy) {
    $('run').disabled = busy;
    $('stop').disabled = !busy;
}

function run() {
    if (running) {
        rerun = true;
        return;
    }
    const algorithm = $('algorithm').value;
    const learner = $('learner').value;
    const { booster: params, weak } = readParams();
    const labels = new Set(points.map((p) => p.label));
    if (!labels.has(1) || !labels.has(-1)) {
        setStatus('Place at least one point of each class.', 'error');
        return;
    }
    if (USES[algorithm].includes('nu') && params.nu > points.length) {
        setStatus(`&nu; must lie between 1 and the number of points (${points.length}).`, 'error');
        return;
    }
    if (algorithm === 'sparsiboost' && params.nHypotheses === 0 && Number.isNaN(params.gap)) {
        setStatus('SparsiBoost needs a target size T, a target gap v, or both.', 'error');
        return;
    }
    running = { id: ++nextId, algorithm, learner, params };
    setBusy(true);
    setStatus(`Running ${NAMES[algorithm]} with ${LEARNER_NAMES[learner]}&hellip;`);
    worker.postMessage({
        id: running.id,
        algorithm,
        learner,
        xs: points.map((p) => p.x),
        ys: points.map((p) => p.y),
        labels: points.map((p) => p.label),
        booster: params,
        weak,
        resolution: RESOLUTION,
    });
}

function finish() {
    running = null;
    setBusy(false);
}

function onResult({ data }) {
    if (!running || data.id !== running.id) return;
    const { algorithm, learner, params } = running;
    finish();
    if (data.ok) {
        field = { grid: data.grid, resolution: RESOLUTION };
        stale = false;
        draw();
        const how = {
            algorithm_stop: 'stopped by the algorithm',
            max_iterations: 'reached max iterations',
            round_budget: 'reached the round bound without the stopping test firing',
            numerical_stop: 'stopped early',
        }[data.stopReason] ?? data.stopReason;
        setStatus(
            `<b>${NAMES[algorithm]}</b> with ${LEARNER_NAMES[learner]}: ` +
            `${data.rounds} rounds (${how})<br>` +
            `Training error ${(100 * data.trainError).toFixed(1)}% &middot; ${data.millis.toFixed(0)} ms` +
            smoothBoostNote(algorithm, params, data.stopReason) +
            (data.note ? `<br><span class="muted">${escapeHtml(data.note)}</span>` : ''),
            'ok',
        );
    } else {
        // A panic leaves the wasm instance unusable; start over.
        startWorker();
        const message = /unreachable/i.test(data.message)
            ? 'MiniBoosts panicked (details in the browser console).'
            : data.message;
        setStatus(`<b>${NAMES[algorithm]}</b> failed: ${escapeHtml(message)}`, 'error');
    }
    if (rerun) {
        rerun = false;
        run();
    }
}

// SmoothBoost's margin guarantee (Servedio 2003, Theorem 2) holds only when
// its stopping test fires; Theorem 3 bounds the rounds this takes when the
// weak learner has advantage gamma.
function smoothBoostNote(algorithm, params, stopReason) {
    if (algorithm !== 'smoothboost' || stopReason === 'algorithm_stop') return '';
    const { kappa, gamma } = params;
    const bound = Math.ceil(2 / (kappa * gamma * gamma * Math.sqrt(1 - gamma)));
    const why = stopReason === 'round_budget'
        ? 'The weak learner did not reach the assumed advantage &gamma;; try a stronger weak learner (e.g. a deeper tree) or a smaller &gamma;.'
        : `Max iterations is below the round bound ${bound}; increase it to let the stopping test decide.`;
    return `<br><span class="muted">Margin guarantee does not apply. ${why}</span>`;
}

function escapeHtml(s) {
    return s.replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);
}

$('run').addEventListener('click', run);
$('stop').addEventListener('click', () => {
    rerun = false;
    startWorker();
    finish();
    setStatus('Stopped.');
});

// ---------------------------------------------------------------- start

startWorker();
updateParamVisibility();
pointsChanged();
setStatus('Add points of both classes, then press Run.');
new ResizeObserver(resizeCanvas).observe(canvas);
