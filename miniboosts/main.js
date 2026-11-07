// Interactive front end: point editing, rendering, and worker management.
// Coordinates are kept in [-1, 1]^2 (y up); the canvas maps them to pixels.

const RESOLUTION = 121;
// Color themes. `classes` are the class colors (RGB); the decision field
// mixes `background` toward the class color by `shade[0] + shade[1] ×
// |confidence| / max |confidence|`.
const THEMES = {
    classic: {
        background: [255, 255, 255],
        classes: { 1: [31, 95, 191], [-1]: [212, 82, 28] },
        shade: [0.12, 0.5],
        boundary: 'rgba(20, 20, 40, 0.85)',
        glow: null,
        axes: 'rgba(0, 0, 0, 0.08)',
        ring: '#fff',
        testFill: 'rgba(255, 255, 255, 0.55)',
        chart: {
            background: null,
            grid: '#e6e6ee',
            text: '#666',
            label: '#444',
            labelBackground: 'rgba(255, 255, 255, 0.85)',
            cursor: '#9aa0b4',
            train: '#000080',
            test: '#0f7d6b',
            reference: '#c0392b',
        },
    },
    // Neon accents on the page's light background.
    cyberpunk: {
        background: [255, 255, 255],
        classes: { 1: [0, 174, 230], [-1]: [255, 0, 128] },
        shade: [0.05, 0.3],
        boundary: '#7a00ff',
        glow: 'rgba(122, 0, 255, 0.45)',
        axes: 'rgba(122, 0, 255, 0.1)',
        ring: '#fff',
        testFill: 'rgba(255, 255, 255, 0.6)',
        chart: {
            background: null,
            grid: 'rgba(122, 0, 255, 0.1)',
            text: '#666',
            label: '#333',
            labelBackground: 'rgba(255, 255, 255, 0.85)',
            cursor: 'rgba(122, 0, 255, 0.35)',
            train: '#7a00ff',
            test: '#ff8a00',
            reference: '#ff0080',
        },
    },
    // The same idea on a dark panel, for dark surroundings.
    'cyberpunk-dark': {
        background: [10, 11, 26],
        classes: { 1: [0, 240, 255], [-1]: [255, 42, 109] },
        // Darker regions so that the neon points and boundary stand out.
        shade: [0.06, 0.3],
        boundary: '#f9f871',
        glow: 'rgba(249, 248, 113, 0.9)',
        axes: 'rgba(0, 240, 255, 0.14)',
        ring: '#0a0b1a',
        testFill: 'rgba(10, 11, 26, 0.55)',
        chart: {
            background: '#0a0b1a',
            grid: 'rgba(0, 240, 255, 0.12)',
            text: '#8b90b8',
            label: '#d6d9f5',
            labelBackground: 'rgba(10, 11, 26, 0.85)',
            cursor: 'rgba(214, 217, 245, 0.45)',
            train: '#f9f871',
            test: '#b967ff',
            reference: '#ff2a6d',
        },
    },
};
const storedTheme = (() => {
    try {
        return localStorage.getItem('miniboosts-theme');
    } catch {
        return null;
    }
})();
let theme = THEMES[storedTheme] ?? THEMES.cyberpunk;
const rgb = (c) => `rgb(${c.join(', ')})`;
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
    arcgv: 'Arc-gv',
    bbm: 'Boost-by-Majority',
    robustboost: 'RobustBoost',
    gradient_boosting: 'Gradient Boosting',
    logitboost: 'LogitBoost',
    majority_of_five: 'Majority-of-5',
    larsen_ritzert: 'Larsen\u2013Ritzert',
};
const LEARNER_NAMES = {
    decision_tree: 'decision trees',
    logistic_regression: 'logistic regression',
    bounded_linear: 'bounded linear functions',
    naive_bayes: 'Gaussian naive Bayes',
    rbf: 'RBF bumps',
    regression_tree: 'regression trees',
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
    arcgv: [],
    bbm: ['bbm'],
    robustboost: ['robust'],
    gradient_boosting: ['gbm'],
    logitboost: ['logit'],
    majority_of_five: ['tolerance', 'subsample', 'mof'],
    larsen_ritzert: ['tolerance', 'subsample', 'lr'],
};
// Majorities of AdaBoost runs on sub-samples: no rounds, so no timeline.
const SUBSAMPLE = new Set(['majority_of_five', 'larsen_ritzert']);
// Boosters that fit real-valued responses and need regression trees.
const REGRESSION = new Set(['gradient_boosting', 'logitboost']);

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const statusBox = $('status');

let points = []; // { x, y, label }
let history = [];
let field = null; // { grid, resolution } of the last successful run
let stale = false;
// Timeline of the last run: frames (coarse grid and example weights after
// selected rounds) and the training confidences behind the loss curve.
let timeline = null;
let frameIndex = 0;
let playTimer = null; // requestAnimationFrame id while playing
// While playing: the (fractional) round up to which the loss lines are
// drawn; otherwise null, and they end at the round of the shown frame.
let chartRound = null;

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
    ctx.fillStyle = rgb(theme.background);
    ctx.fillRect(0, 0, w, h);
    const view = currentView();
    if (view) {
        drawField(view);
    }
    drawAxes();
    const dpr = window.devicePixelRatio || 1;
    const dist = currentDist();
    // Weights belong to the training points, in order; test points have none.
    let k = 0;
    const showTest = $('showTest').checked;
    for (const p of points) {
        if (p.test && !showTest) continue;
        let r = 5;
        if (!p.test && dist) {
            // Area proportional to the weight; 5 px is the uniform weight 1/n.
            r = Math.min(14, Math.max(2, 5 * Math.sqrt(dist.length * dist[k])));
        }
        if (!p.test) k++;
        drawMarker(ctx, w, h, p, r * dpr);
    }
}

// The decision field to show: a timeline frame while the slider is before
// the last round, the full-resolution final model otherwise.
function currentView() {
    if (!field) return null;
    if (timeline && !stale && frameIndex < timeline.frames.length - 1) {
        return { grid: timeline.frames[frameIndex].grid, resolution: timeline.frameResolution };
    }
    return field;
}

function currentDist() {
    // Runs without rounds (the sub-sample majorities) have no frames.
    if (!timeline || stale || !$('showWeights').checked || !timeline.frames.length) return null;
    return timeline.frames[frameIndex].dist;
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
        const color = theme.classes[c >= 0 ? 1 : -1];
        const t = theme.shade[0] + theme.shade[1] * Math.abs(c) / scale;
        for (let ch = 0; ch < 3; ch++) {
            const base = theme.background[ch];
            image.data[4 * i + ch] = Math.round(base + (color[ch] - base) * t);
        }
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
    const dpr = window.devicePixelRatio || 1;
    ctx.strokeStyle = theme.boundary;
    ctx.lineWidth = 1.5 * dpr;
    if (theme.glow) {
        ctx.shadowColor = theme.glow;
        ctx.shadowBlur = 6 * dpr;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
}

function drawAxes() {
    ctx.save();
    ctx.strokeStyle = theme.axes;
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
// Class is shown by color and shape; a training point is filled, a test
// point hollow and smaller, so that the held-out points stay in the
// background (a thin outline in the class color over a translucent fill).
function drawMarker(target, w, h, { x, y, label, test }, radius) {
    const px = ((x + 1) / 2) * w;
    const py = ((1 - y) / 2) * h;
    const color = rgb(theme.classes[label]);
    const path = (r) => {
        target.beginPath();
        if (label === 1) target.arc(px, py, 1.1 * r, 0, 2 * Math.PI);
        else target.rect(px - r, py - r, 2 * r, 2 * r);
    };
    if (test) {
        path(0.8 * radius);
        target.fillStyle = theme.testFill;
        target.fill();
        target.lineWidth = 0.28 * radius;
        target.strokeStyle = color;
        target.stroke();
        return;
    }
    path(radius);
    target.fillStyle = color;
    target.fill();
    target.lineWidth = 0.3 * radius;
    target.strokeStyle = theme.ring;
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
    selectPreset(null);
    // The timeline (snapshots and weights) belongs to the old points.
    hideTimeline();
    timeline = null;
    const train = points.filter((p) => !p.test);
    const pos = train.filter((p) => p.label === 1).length;
    const nTest = points.length - train.length;
    $('count').textContent =
        `${train.length} training points (+1: ${pos}, −1: ${train.length - pos})` +
        (nTest ? ` · ${nTest} test points` : '');
    $('nu').max = Math.max(1, train.length);
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
    const test = document.querySelector('input[name="split"]:checked').value === 'test';
    setPoints([...points, { x, y, label, test }]);
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
    // The soft hyphen lets the label break as "Checker-/board" in a narrow button.
    ['checker', 'Checker\u00adboard', (rand) => {
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

const thumbnails = [];

function drawThumbnail(target, data) {
    const dpr = window.devicePixelRatio || 1;
    const size = Math.round(target.getBoundingClientRect().width * dpr) || 80;
    target.width = target.height = size;
    const tctx = target.getContext('2d');
    tctx.fillStyle = rgb(theme.background);
    tctx.fillRect(0, 0, size, size);
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
    thumbnails.push([thumb, data]);
    requestAnimationFrame(() => drawThumbnail(thumb, data));
    button.addEventListener('click', () => {
        field = null;
        // Test points: an independent draw from the same generator.
        const test = generate(make, PRESET_SEED + 1).map((p) => ({ ...p, test: true }));
        setPoints([...data.map((p) => ({ ...p })), ...test]);
        selectPreset(button);
        loadedMessage(`Loaded “${title}”.`);
    });
}

// Marks the preset whose points are shown; any other change of the points
// (editing, Random, Clear, Undo) clears the mark in `pointsChanged`.
function selectPreset(selected) {
    for (const button of $('presets').querySelectorAll('button')) {
        const on = button === selected;
        button.classList.toggle('selected', on);
        button.setAttribute('aria-pressed', String(on));
    }
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
        // Every other point of each cluster is held out as a test point,
        // so the test set follows the same distribution.
        const count = 10 + Math.floor(rand() * 21);
        for (let i = 0; i < 2 * count; i++) {
            const flip = rand() < noise ? -1 : 1;
            out.push({
                x: cx + sx * gaussian(rand),
                y: cy + sy * gaussian(rand),
                label: flip * label,
                test: i % 2 === 1,
            });
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
    // Regression trees go with gradient boosting and LogitBoost only.
    const regression = REGRESSION.has($('algorithm').value);
    for (const option of $('learner').options) {
        option.disabled = (option.value === 'regression_tree') !== regression;
    }
    if ($('learner').selectedOptions[0].disabled) {
        $('learner').value = regression ? 'regression_tree' : 'decision_tree';
    }
    const learner = $('learner').value;
    for (const el of document.querySelectorAll('.lparam')) {
        el.hidden = !el.dataset.learner.split(' ').includes(learner);
    }
}
// After a change of algorithm, learner, or parameters, the shown result no
// longer matches the settings: fade it and hide its timeline (slider,
// playback, loss chart) until the next run, as after a change of points.
function settingsChanged() {
    if (!field || stale) return;
    stale = true;
    hideTimeline();
    timeline = null;
    draw();
    setStatus('Settings changed. Press Run to retrain.');
}

for (const id of ['algorithm', 'learner']) {
    $(id).addEventListener('change', () => {
        updateParamVisibility();
        settingsChanged();
        if ($('auto').checked) run();
    });
}
for (const id of [
    'maxIterations', 'tolerance', 'nu', 'gamma', 'kappa', 'nHypotheses', 'gap',
    'brownTolerance', 'edgeThreshold', 'bbmGamma', 'bbmTolerance', 'robustTolerance', 'robustTheta',
    'robustSigma', 'gbmLoss', 'shrinkage', 'maxResponse',
    'maxDepth', 'splitBy', 'regularization', 'logisticOutput', 'varSmoothing', 'bandwidth',
    'nParts', 'maxSubsamples', 'subsampleSeed', 'exactSplits', 'leafConfidence', 'rbfSelection',
    'rbfGamma',
]) {
    // `input` fires while typing, before the field loses focus.
    $(id).addEventListener('input', settingsChanged);
    $(id).addEventListener('change', () => {
        settingsChanged();
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
            bbmGamma: num('bbmGamma'),
            bbmTolerance: num('bbmTolerance'),
            robustTolerance: num('robustTolerance'),
            robustTheta: num('robustTheta'),
            robustSigma: num('robustSigma'),
            gbmLoss: $('gbmLoss').value,
            shrinkage: num('shrinkage'),
            maxResponse: num('maxResponse'),
            nParts: Math.round(num('nParts')),
            // Empty means all sub-samples.
            maxSubsamples: $('maxSubsamples').value === '' ? 0 : Math.round(num('maxSubsamples')),
            subsampleSeed: Math.round(num('subsampleSeed')),
        },
        weak: {
            maxDepth: Math.round(num('maxDepth')),
            splitBy: $('splitBy').value,
            regularization: num('regularization'),
            output: $('logisticOutput').value,
            varSmoothing: num('varSmoothing'),
            bandwidth: num('bandwidth'),
            exactSplits: $('exactSplits').checked,
            leafConfidence: $('leafConfidence').value,
            selection: $('rbfSelection').value,
            gamma: num('rbfGamma'),
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
    const train = points.filter((p) => !p.test);
    const test = points.filter((p) => p.test);
    const labels = new Set(train.map((p) => p.label));
    if (!labels.has(1) || !labels.has(-1)) {
        setStatus('Place at least one training point of each class.', 'error');
        return;
    }
    if (USES[algorithm].includes('nu') && params.nu > train.length) {
        setStatus(`&nu; must lie between 1 and the number of training points (${train.length}).`, 'error');
        return;
    }
    if (algorithm === 'sparsiboost' && params.nHypotheses === 0 && Number.isNaN(params.gap)) {
        setStatus('SparsiBoost needs a target size T, a target gap v, or both.', 'error');
        return;
    }
    running = {
        id: ++nextId,
        algorithm,
        learner,
        params,
        labels: train.map((p) => p.label),
        testLabels: test.map((p) => p.label),
    };
    setBusy(true);
    setStatus(`Running ${NAMES[algorithm]} with ${LEARNER_NAMES[learner]}&hellip;`);
    worker.postMessage({
        id: running.id,
        algorithm,
        learner,
        xs: train.map((p) => p.x),
        ys: train.map((p) => p.y),
        labels: train.map((p) => p.label),
        test: { xs: test.map((p) => p.x), ys: test.map((p) => p.y), labels: test.map((p) => p.label) },
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
    const { algorithm, learner, params, labels, testLabels } = running;
    finish();
    if (data.ok) {
        field = { grid: data.grid, resolution: RESOLUTION };
        stale = false;
        timeline = buildTimeline(data, labels, testLabels);
        showTimeline(algorithm, params);
        draw();
        const how = {
            algorithm_stop: 'stopped by the algorithm',
            max_iterations: 'reached max iterations',
            round_budget: 'reached the round bound without the stopping test firing',
            numerical_stop: 'stopped early',
        }[data.stopReason] ?? data.stopReason;
        const progress = SUBSAMPLE.has(algorithm)
            ? `${data.rounds} AdaBoost runs`
            : `${data.rounds} rounds (${how})`;
        setStatus(
            `<b>${NAMES[algorithm]}</b> with ${LEARNER_NAMES[learner]}: ` +
            `${progress}<br>` +
            `Training error ${(100 * data.trainError).toFixed(1)}%` +
            (testLabels.length ? ` &middot; Test error ${(100 * data.testError).toFixed(1)}%` : '') +
            ` &middot; ${data.millis.toFixed(0)} ms` +
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

// ---------------------------------------------------------------- view options

function applyTheme(name) {
    const key = name in THEMES ? name : 'cyberpunk';
    theme = THEMES[key];
    $('theme').value = key;
    $('playground').dataset.theme = key;
    $('playground').style.setProperty('--train-color', theme.chart.train);
    $('playground').style.setProperty('--test-color', theme.chart.test);
    $('playground').style.setProperty('--accent', theme.chart.train);
    try {
        localStorage.setItem('miniboosts-theme', key);
    } catch {
        // Storage may be unavailable (private mode); the theme still applies.
    }
    for (const [thumb, data] of thumbnails) drawThumbnail(thumb, data);
    draw();
    drawLossChart();
}
$('theme').addEventListener('change', (e) => applyTheme(e.target.value));

// Test points exist (and count toward the test error) whether shown or not.
$('showTest').addEventListener('change', draw);
for (const radio of document.querySelectorAll('input[name="split"]')) {
    radio.addEventListener('change', () => {
        // Points being added as test points should be visible.
        if (radio.value === 'test' && radio.checked && !$('showTest').checked) {
            $('showTest').checked = true;
            draw();
        }
    });
}

// ---------------------------------------------------------------- timeline

function buildTimeline(data, labels, testLabels) {
    const n = labels.length;
    const nt = testLabels.length;
    const g = data.frameResolution ** 2;
    const frames = Array.from(data.frameRounds, (round, k) => ({
        round,
        grid: data.frameGrids.subarray(k * g, (k + 1) * g),
        dist: data.frameDists.subarray(k * n, (k + 1) * n),
    }));
    const curve = Array.from(data.curveRounds, (round, k) => ({
        round,
        conf: data.curveConfidences.subarray(k * n, (k + 1) * n),
        testConf: data.curveTestConfidences.subarray(k * nt, (k + 1) * nt),
        objective: data.curveObjective[k],
    }));
    return {
        frames,
        curve,
        labels,
        testLabels,
        frameResolution: data.frameResolution,
        finalStep: data.finalStep,
    };
}

function showTimeline(algorithm, params) {
    stopPlaying();
    if (!timeline.frames.length) {
        hideTimeline();
        return;
    }
    const last = timeline.frames.length - 1;
    frameIndex = last;
    $('round').max = last;
    $('round').value = last;
    $('round').disabled = last === 0;
    $('play').disabled = last === 0;
    $('saveGif').disabled = last === 0;
    $('timeline').hidden = false;
    timeline.objective = objectiveInfo(algorithm, params);
    const option = $('objectiveOption');
    option.disabled = !timeline.objective;
    option.textContent = timeline.objective ? `Objective: ${timeline.objective.short}` : 'Objective (not available)';
    if (!timeline.objective) $('lossKind').value = 'error';
    // SparsiBoost: the rounds are its AdaBoostV stage; one more frame (at
    // the same round) shows the ensemble after sparsification.
    timeline.sparsiNote = timeline.finalStep
        ? `Rounds are SparsiBoost\u2019s AdaBoostV stage; the last frame is after ${timeline.finalStep}. `
        : '';
    updateRoundLabel();
    drawLossChart();
}

function hideTimeline() {
    stopPlaying();
    $('timeline').hidden = true;
}

function updateRoundLabel() {
    const frames = timeline.frames;
    const last = frames.length - 1;
    const step = timeline.finalStep && frameIndex === last ? ` \u00b7 after ${timeline.finalStep}` : '';
    $('roundLabel').textContent = `round ${frames[frameIndex].round} / ${frames[last].round}${step}`;
}

function setFrame(index) {
    frameIndex = index;
    $('round').value = index;
    updateRoundLabel();
    draw();
    drawLossChart();
}

$('round').addEventListener('input', (e) => {
    stopPlaying();
    setFrame(Number(e.target.value));
});
$('showWeights').addEventListener('change', draw);
$('lossKind').addEventListener('change', () => drawLossChart());

function stopPlaying() {
    cancelAnimationFrame(playTimer);
    playTimer = null;
    chartRound = null;
    $('play').innerHTML = '&#9654;';
    $('play').setAttribute('aria-label', 'Play');
}

// Plays the rounds at a constant speed (the whole run in about 5 seconds):
// the loss lines grow continuously, and the plot switches to each frame as
// its round is reached.
const PLAY_MILLIS = 5000;
$('play').addEventListener('click', () => {
    if (playTimer) {
        stopPlaying();
        drawLossChart();
        return;
    }
    const frames = timeline.frames;
    const last = frames.length - 1;
    // An extra final frame (e.g. after sparsification) is shown at the end.
    const lastRoundFrame = timeline.finalStep ? last - 1 : last;
    const first = frames[0].round;
    const end = frames[lastRoundFrame].round;
    if (frameIndex >= lastRoundFrame) setFrame(0);
    const start = frames[frameIndex].round;
    const speed = (end - first) / PLAY_MILLIS || Infinity;
    const t0 = performance.now();
    $('play').innerHTML = '&#10074;&#10074;';
    $('play').setAttribute('aria-label', 'Pause');
    const step = (now) => {
        const round = Math.min(end, start + (now - t0) * speed);
        chartRound = round;
        let index = frameIndex;
        while (index < lastRoundFrame && frames[index + 1].round <= round) index++;
        if (index !== frameIndex) {
            frameIndex = index;
            $('round').value = index;
            updateRoundLabel();
            draw();
        }
        if (round >= end) {
            stopPlaying();
            setFrame(last);
            return;
        }
        drawLossChart();
        playTimer = requestAnimationFrame(step);
    };
    playTimer = requestAnimationFrame(step);
});

// Training 0-1 error of confidences `conf`, with the classifiers' rule
// that confidence 0 predicts +1.
function zeroOneError(conf, labels) {
    let wrong = 0;
    for (let i = 0; i < labels.length; i++) {
        if ((conf[i] >= 0 ? 1 : -1) !== labels[i]) wrong++;
    }
    return wrong / labels.length;
}

// What each algorithm optimizes, as computed in the wasm module (see
// `Objective` there). F = Σ α_t h_t is the unnormalized combination; f is
// the normalized combined hypothesis.
function objectiveInfo(algorithm, params) {
    const softMargin = (nu) => ({
        short: nu === 1 ? 'minimum margin' : `soft margin (ν = ${nu})`,
        title: nu === 1
            ? 'Minimum margin  minᵢ yᵢ f(xᵢ)'
            : `Soft margin  min over d ∈ Δ(m, ν) of Σ dᵢ yᵢ f(xᵢ),  ν = ${nu}`,
        better: 'higher',
    });
    const deviance = {
        short: 'deviance',
        title: 'Binomial deviance  (1/m) Σ ln(1 + exp(−2y F(x))),  F = half the log-odds',
        better: 'lower',
    };
    switch (algorithm) {
        case 'adaboost':
            return {
                short: 'exponential loss',
                title: 'Exponential loss  (1/m) Σ exp(−y F(x)),  F = Σ αₜ hₜ',
                better: 'lower',
            };
        case 'madaboost':
            return {
                short: 'MadaBoost loss',
                title: 'MadaBoost loss  (1/m) Σ φ(y F(x)),  φ(z) = e⁻ᶻ (z ≥ 0), 1 − z (z < 0)',
                better: 'lower',
            };
        case 'adaboostv':
        case 'arcgv':
        case 'totalboost':
        case 'sparsiboost':
            return softMargin(1);
        case 'logitboost':
            return deviance;
        case 'gradient_boosting':
            return {
                logistic: deviance,
                exponential: {
                    short: 'exponential loss',
                    title: 'Exponential loss  (1/m) Σ exp(−y F(x)),  F = c₀ + Σ cₜ fₜ',
                    better: 'lower',
                },
                squared: {
                    short: 'squared loss',
                    title: 'Squared loss  (1/m) Σ (y − F(x))² / 2,  F = c₀ + Σ cₜ fₜ',
                    better: 'lower',
                },
            }[params.gbmLoss];
        case 'lpboost':
        case 'erlpboost':
        case 'corrective_erlpboost':
        case 'mlpboost':
        case 'softboost':
            return {
                ...softMargin(params.nu),
                extra: algorithm === 'erlpboost' || algorithm === 'corrective_erlpboost'
                    ? 'ERLPBoost-type methods maximize an entropy-regularized version of it. '
                    : '',
            };
        case 'smoothboost':
            return {
                short: 'SmoothBoost potential',
                title: '(1/m) Σ M(j),  M(j) = min(1, (1 − γ)^(N/2)),  N = t (y f(x) − θ)',
                better: 'lower',
                reference: { value: params.kappa, label: `κ = ${params.kappa}` },
            };
        default:
            return null;
    }
}

// A nice step for about four intervals over `span`.
function niceStep(span) {
    const raw = span / 4;
    const mag = 10 ** Math.floor(Math.log10(raw));
    return [1, 2, 2.5, 5, 10].map((f) => f * mag).find((v) => v >= raw);
}

// Axis range [bottom, top] covering `values` (and `extra`), rounded to a
// nice step; it starts at 0 whenever all values are nonnegative.
function niceRange(values) {
    let lo = Math.min(...values);
    let hi = Math.max(...values);
    if (lo >= 0) lo = 0;
    if (hi <= lo) hi = lo + (lo === 0 ? 1 : Math.abs(lo));
    const step = niceStep(hi - lo);
    return { bottom: Math.floor(lo / step + 1e-9) * step, top: Math.ceil(hi / step - 1e-9) * step, step };
}

const chart = $('lossChart');
const PAD = { left: 52, right: 14, top: 10, bottom: 30 };
// The plotted series: [{ name, values, color }].
let chartSeries = [];

function chartGeometry(rect = chart.getBoundingClientRect()) {
    const rounds = timeline.curve.map((p) => p.round);
    const x0 = Math.min(...rounds);
    const x1 = Math.max(...rounds);
    const width = rect.width - PAD.left - PAD.right;
    const xOf = (t) => PAD.left + (x1 === x0 ? 0.5 : (t - x0) / (x1 - x0)) * width;
    return { rect, x0, x1, xOf };
}

// Draws on the page's chart, or on `target` at `size` ({ width, height }) for
// the GIF export.
function drawLossChart(target = chart, size = null) {
    if (!timeline || $('timeline').hidden) return;
    const dpr = size ? 1 : window.devicePixelRatio || 1;
    const { rect, x0, x1, xOf } = chartGeometry(size ?? undefined);
    if (rect.width === 0) return;
    target.width = Math.round(rect.width * dpr);
    target.height = Math.round(rect.height * dpr);
    const c = target.getContext('2d');
    const colors = theme.chart;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, rect.width, rect.height);
    if (colors.background) {
        c.fillStyle = colors.background;
        c.fillRect(0, 0, rect.width, rect.height);
    }

    const info = $('lossKind').value === 'objective' ? timeline.objective : null;
    const hasTest = timeline.testLabels.length > 0;
    $('lossTitle').textContent = info ? info.title : hasTest ? '0-1 error' : 'Training 0-1 error';
    $('timelineNote').textContent = timeline.sparsiNote + (info
        ? `${info.better === 'higher' ? 'Higher' : 'Lower'} is better. ${info.extra ?? ''}` +
          'Computed on the training points after each round.'
        : 'Fraction of points the combined hypothesis misclassifies after each round' +
          (hasTest ? '; test points are never used for training.' : '.'));
    // The objective is defined on the training points only; the 0-1 error
    // is shown for the training and (if any) the test points.
    chartSeries = info
        ? [{ name: 'objective', values: timeline.curve.map((p) => p.objective), color: colors.train }]
        : [
              {
                  name: 'train',
                  values: timeline.curve.map((p) => zeroOneError(p.conf, timeline.labels)),
                  color: colors.train,
              },
              ...(hasTest
                  ? [{
                        name: 'test',
                        values: timeline.curve.map((p) => zeroOneError(p.testConf, timeline.testLabels)),
                        color: colors.test,
                    }]
                  : []),
          ];
    $('lossLegend').hidden = chartSeries.length < 2;
    // The axes cover the whole run, so they stay fixed while the lines grow.
    const allValues = chartSeries.flatMap((series) => series.values);
    const ref = info?.reference;
    const { bottom, top, step } = niceRange(ref ? [...allValues, ref.value] : allValues);
    const plotH = rect.height - PAD.top - PAD.bottom;
    const yOf = (v) => PAD.top + plotH * (1 - (v - bottom) / (top - bottom));

    // Recessive grid and axis labels.
    c.font = '12px Play, sans-serif';
    c.fillStyle = colors.text;
    c.strokeStyle = colors.grid;
    c.lineWidth = 1;
    c.textAlign = 'right';
    c.textBaseline = 'middle';
    // Enough decimals to print every multiple of `step` exactly.
    const digits = [0, 1, 2, 3, 4].find((d) => Math.abs(step * 10 ** d - Math.round(step * 10 ** d)) < 1e-9) ?? 4;
    for (let v = bottom; v <= top + step / 2; v += step) {
        const y = Math.round(yOf(v)) + 0.5;
        c.beginPath();
        c.moveTo(PAD.left, y);
        c.lineTo(rect.width - PAD.right, y);
        c.stroke();
        c.fillText(v.toFixed(digits), PAD.left - 6, y);
    }
    c.textAlign = 'center';
    c.textBaseline = 'top';
    const xStep = Math.max(1, Math.round(niceStep(x1 - x0 || 1)));
    for (let t = Math.ceil(x0 / xStep) * xStep; t <= x1; t += xStep) {
        c.fillText(String(t), xOf(t), rect.height - PAD.bottom + 6);
    }

    // A reference level, e.g. SmoothBoost's stopping threshold κ.
    if (ref) {
        const y = Math.round(yOf(ref.value)) + 0.5;
        c.strokeStyle = colors.reference;
        c.setLineDash([6, 4]);
        c.beginPath();
        c.moveTo(PAD.left, y);
        c.lineTo(rect.width - PAD.right, y);
        c.stroke();
        c.setLineDash([]);
        c.fillStyle = colors.reference;
        // Labeled at the left, away from the curve's end.
        c.textAlign = 'left';
        c.textBaseline = 'bottom';
        c.fillText(ref.label, PAD.left + 4, y - 2);
    }

    // The lines run up to `upTo`: the round of the shown frame, or the
    // fractional round reached during playback. The extra final frame
    // (e.g. after sparsification) adds the last point, which shares its
    // round with the point before it.
    const atStep = timeline.finalStep && frameIndex === timeline.frames.length - 1;
    const upTo = chartRound ?? timeline.frames[frameIndex].round;
    const usable = timeline.finalStep && !atStep ? timeline.curve.length - 1 : timeline.curve.length;
    // Polyline vertices: every point up to `upTo`, plus an interpolated end.
    const vertices = (values) => {
        const out = [];
        for (let j = 0; j < usable; j++) {
            const round = timeline.curve[j].round;
            if (round <= upTo) {
                out.push([round, values[j]]);
            } else {
                if (j > 0) {
                    const r0 = timeline.curve[j - 1].round;
                    const f = (upTo - r0) / (round - r0);
                    out.push([upTo, values[j - 1] + f * (values[j] - values[j - 1])]);
                }
                break;
            }
        }
        return out;
    };

    const cx = Math.round(xOf(upTo)) + 0.5;
    c.strokeStyle = colors.cursor;
    c.setLineDash([4, 3]);
    c.beginPath();
    c.moveTo(cx, PAD.top);
    c.lineTo(cx, PAD.top + plotH);
    c.stroke();
    c.setLineDash([]);

    // Each series: a 2 px line, plus a marker at its current end.
    const ends = [];
    for (const series of chartSeries) {
        const vs = vertices(series.values);
        if (!vs.length) continue;
        c.strokeStyle = series.color;
        c.lineWidth = 2;
        c.lineJoin = 'round';
        if (theme.glow) {
            c.shadowColor = series.color;
            c.shadowBlur = 6;
        }
        c.beginPath();
        vs.forEach(([round, v], j) => {
            if (j === 0) c.moveTo(xOf(round), yOf(v));
            else c.lineTo(xOf(round), yOf(v));
        });
        c.stroke();
        c.shadowBlur = 0;
        const [round, v] = vs[vs.length - 1];
        ends.push([xOf(round), yOf(v)]);
        c.fillStyle = series.color;
        c.strokeStyle = colors.background ?? '#fff';
        c.beginPath();
        c.arc(xOf(round), yOf(v), 5, 0, 2 * Math.PI);
        c.fill();
        c.stroke();
    }
    // With two series, label each line at its current end so that identity
    // does not rest on color alone; nudge the labels apart if they collide.
    if (chartSeries.length > 1 && ends.length === chartSeries.length) {
        const ys = ends.map(([, y]) => y);
        if (Math.abs(ys[0] - ys[1]) < 13) {
            const mid = (ys[0] + ys[1]) / 2;
            const sign = ys[0] <= ys[1] ? -1 : 1;
            ys[0] = mid + sign * 6.5;
            ys[1] = mid - sign * 6.5;
        }
        c.font = '11px Play, sans-serif';
        c.textAlign = 'right';
        c.textBaseline = 'middle';
        chartSeries.forEach((series, j) => {
            const width = c.measureText(series.name).width;
            // Left of the line's end, clamped into the plot area.
            const x = Math.max(PAD.left + width + 20, Math.min(rect.width - PAD.right - 2, ends[j][0] - 10));
            const y = Math.min(PAD.top + plotH - 6, Math.max(PAD.top + 6, ys[j] - 9));
            // Ink-colored text with a short colored line as the key, on a
            // backing that keeps it legible over the lines.
            c.fillStyle = colors.labelBackground;
            c.fillRect(x - width - 18, y - 7, width + 20, 14);
            c.strokeStyle = series.color;
            c.lineWidth = 2;
            c.beginPath();
            c.moveTo(x - width - 15, y);
            c.lineTo(x - width - 5, y);
            c.stroke();
            c.fillStyle = colors.label;
            c.fillText(series.name, x, y);
        });
    }
}

function nearestIndex(items, round) {
    let best = 0;
    items.forEach((p, k) => {
        if (Math.abs(p.round - round) < Math.abs(items[best].round - round)) best = k;
    });
    return best;
}
const nearestCurveIndex = (round) => nearestIndex(timeline.curve, round);

function roundAtPointer(e) {
    const { rect, x0, x1 } = chartGeometry();
    const frac = (e.clientX - rect.left - PAD.left) / (rect.width - PAD.left - PAD.right);
    return x0 + Math.min(1, Math.max(0, frac)) * (x1 - x0);
}

chart.addEventListener('pointermove', (e) => {
    if (!timeline || !chartSeries.length) return;
    const k = nearestCurveIndex(roundAtPointer(e));
    const { rect, xOf } = chartGeometry();
    const tip = $('lossTip');
    const values = chartSeries.map((series) =>
        chartSeries.length > 1 ? `${series.name} ${series.values[k].toFixed(4)}` : series.values[k].toFixed(4));
    tip.textContent = `round ${timeline.curve[k].round} · ${values.join(' · ')}`;
    tip.hidden = false;
    const x = xOf(timeline.curve[k].round);
    tip.style.left = `${Math.min(rect.width - tip.offsetWidth, Math.max(0, x - tip.offsetWidth / 2))}px`;
    tip.style.top = '-30px';
});
chart.addEventListener('pointerleave', () => {
    $('lossTip').hidden = true;
});
chart.addEventListener('click', (e) => {
    if (!timeline) return;
    stopPlaying();
    setFrame(nearestIndex(timeline.frames, roundAtPointer(e)));
});
new ResizeObserver(() => drawLossChart()).observe(chart);

// ---------------------------------------------------------------- GIF export

const GIF_LAYOUT = { pad: 12, header: 34, board: 440, chartTitle: 26, chartHeight: 130 };
const GIF_HOLD_MILLIS = 2500; // pause on the last frame before looping

// Draws the decision boundary of the shown frame with the loss chart below it,
// as on the page. The plot is copied from the page's canvas, so it honors the
// theme and the page's options; the chart is drawn anew at the GIF's size.
function composeGifFrame(context, chartCanvas) {
    const { pad, header, board, chartTitle, chartHeight } = GIF_LAYOUT;
    const { width, height } = context.canvas;
    const colors = theme.chart;
    context.fillStyle = colors.background ?? '#fff';
    context.fillRect(0, 0, width, height);
    context.imageSmoothingQuality = 'high';
    context.drawImage(canvas, pad, header, board, board);

    const selected = (id) => $(id).selectedOptions[0].textContent.trim();
    const titleY = header + board + chartTitle / 2 + 2;
    context.fillStyle = colors.label;
    context.textBaseline = 'middle';
    context.textAlign = 'left';
    context.font = 'bold 15px Play, sans-serif';
    context.fillText(`${selected('algorithm')} + ${selected('learner')}`, pad, header / 2);
    context.font = '13px Play, sans-serif';
    context.fillText($('lossTitle').textContent, pad, titleY);
    context.textAlign = 'right';
    context.fillText($('roundLabel').textContent, width - pad, titleY);

    drawLossChart(chartCanvas, { width: board, height: chartHeight });
    context.drawImage(chartCanvas, pad, header + board + chartTitle);
}

async function saveGif() {
    const run = timeline;
    const button = $('saveGif');
    const shownFrame = frameIndex;
    const last = run.frames.length - 1;
    const stillCurrent = () => timeline === run;
    stopPlaying();
    drawLossChart();
    for (const id of ['saveGif', 'play', 'round']) $(id).disabled = true;
    try {
        const { GIFEncoder, quantize, applyPalette } = await import('./gifenc/gifenc.esm.js');
        const { pad, header, board, chartTitle, chartHeight } = GIF_LAYOUT;
        const out = document.createElement('canvas');
        out.width = pad + board + pad;
        out.height = header + board + chartTitle + chartHeight + pad;
        const context = out.getContext('2d', { willReadFrequently: true });
        const chartCanvas = document.createElement('canvas');
        const render = (index) => {
            setFrame(index);
            composeGifFrame(context, chartCanvas);
            return context.getImageData(0, 0, out.width, out.height).data;
        };

        // One palette for all frames, from frames across the run.
        const samples = [0, last >> 1, last].map(render);
        const pixels = new Uint8ClampedArray(samples.reduce((n, s) => n + s.length, 0));
        samples.reduce((offset, s) => (pixels.set(s, offset), offset + s.length), 0);
        const palette = quantize(pixels, 256);

        // Same pace as the Play button: time is proportional to rounds.
        const span = run.frames[last].round - run.frames[0].round;
        const encoder = GIFEncoder();
        for (let k = 0; k <= last; k++) {
            if (!stillCurrent()) return;
            const rounds = k < last ? run.frames[k + 1].round - run.frames[k].round : 0;
            const delay = k < last
                ? Math.max(20, Math.round((PLAY_MILLIS * rounds) / span / 10) * 10)
                : GIF_HOLD_MILLIS;
            const rgba = render(k);
            encoder.writeFrame(applyPalette(rgba, palette), out.width, out.height, {
                palette: k === 0 ? palette : undefined,
                delay,
            });
            button.textContent = `${Math.round((100 * (k + 1)) / (last + 1))}%`;
            // Keep the page responsive and the progress visible.
            await new Promise((resolve) => setTimeout(resolve));
        }
        encoder.finish();

        const url = URL.createObjectURL(new Blob([encoder.bytes()], { type: 'image/gif' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = `miniboosts-${$('algorithm').value}.gif`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (error) {
        setStatus(`Could not save the GIF: ${escapeHtml(String(error))}`, 'error');
    } finally {
        button.textContent = 'GIF';
        // A new run or an edit replaces the timeline and sets these itself.
        if (stillCurrent()) {
            setFrame(shownFrame);
            for (const id of ['saveGif', 'play', 'round']) $(id).disabled = false;
        }
    }
}
$('saveGif').addEventListener('click', saveGif);

startWorker();
applyTheme(storedTheme);
updateParamVisibility();
pointsChanged();
setStatus('Add points of both classes, then press Run.');
new ResizeObserver(resizeCanvas).observe(canvas);
