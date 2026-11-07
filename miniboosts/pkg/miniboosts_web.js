/**
 * Result of one training run.
 */
export class Fit {
    static __wrap(ptr) {
        const obj = Object.create(Fit.prototype);
        obj.__wbg_ptr = ptr;
        FitFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FitFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fit_free(ptr, 0);
    }
    /**
     * Confidences `f_t(x_i)` of the snapshot after each round in
     * `curveRounds`, one row of `n` values per round. Snapshots are the
     * boosters' combined hypotheses, with weights normalized to `Σ|w| = 1`.
     * @returns {Float64Array}
     */
    get curveConfidences() {
        const ret = wasm.fit_curveConfidences(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * The algorithm's own objective (see `Objective`) after each round in
     * `curveRounds`, or NaN for algorithms without one.
     * @returns {Float64Array}
     */
    get curveObjective() {
        const ret = wasm.fit_curveObjective(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Rounds `t` (number of weak hypotheses produced) whose snapshot
     * confidences on the training points are in `curveConfidences`.
     * @returns {Uint32Array}
     */
    get curveRounds() {
        const ret = wasm.fit_curveRounds(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Confidences of the snapshot after each round in `curveRounds` on the
     * test points, one row per round (empty without test points). The
     * test points are never used for training.
     * @returns {Float64Array}
     */
    get curveTestConfidences() {
        const ret = wasm.fit_curveTestConfidences(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Name of a step after the last round shown as the final frame, such
     * as `"sparsification"`, or empty.
     * @returns {string}
     */
    get finalStep() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.fit_finalStep(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * For each frame, the distribution over the training points that the
     * booster passed to the weak learner in that round.
     * @returns {Float64Array}
     */
    get frameDists() {
        const ret = wasm.fit_frameDists(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Snapshot confidences on the `frameResolution²` grid for each frame,
     * in the same layout as `grid`.
     * @returns {Float32Array}
     */
    get frameGrids() {
        const ret = wasm.fit_frameGrids(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Side length of the frame grids.
     * @returns {number}
     */
    get frameResolution() {
        const ret = wasm.fit_frameResolution(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Rounds kept as timeline frames.
     * @returns {Uint32Array}
     */
    get frameRounds() {
        const ret = wasm.fit_frameRounds(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Ensemble confidence on the grid, row-major from `(x, y) = (-1, 1)`
     * (top-left) to `(1, -1)` (bottom-right).
     * @returns {Float64Array}
     */
    get grid() {
        const ret = wasm.fit_grid(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Algorithm-specific summary (empty for most algorithms).
     * @returns {string}
     */
    get note() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.fit_note(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Number of weak hypotheses produced. A round that stops before calling
     * the weak learner is not counted.
     * @returns {number}
     */
    get rounds() {
        const ret = wasm.fit_rounds(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * `"algorithm_stop"`, `"max_iterations"`, for SmoothBoost
     * `"round_budget"` (the paper's round bound was reached without the
     * stopping test firing, so its margin guarantee does not apply), or for
     * BrownBoost `"numerical_stop"` (stopped before the remaining time ran
     * out; see the note).
     * @returns {string}
     */
    get stopReason() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.fit_stopReason(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Fraction of misclassified test points, or NaN without test points.
     * @returns {number}
     */
    get testError() {
        const ret = wasm.fit_testError(this.__wbg_ptr);
        return ret;
    }
    /**
     * Fraction of misclassified training points.
     * @returns {number}
     */
    get trainError() {
        const ret = wasm.fit_trainError(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Fit.prototype[Symbol.dispose] = Fit.prototype.free;

/**
 * Installs a panic hook so that Rust panics show their message in the
 * browser console. A panic aborts the wasm instance; the caller should
 * discard it (e.g. restart the worker).
 */
export function start() {
    wasm.start();
}

/**
 * Trains `algorithm` with the weak learner `learner` on the points
 * `(xs[i], ys[i])` with labels `labels[i] ∈ {-1, +1}` and evaluates the
 * result on a `resolution × resolution` grid and on the held-out test
 * points `(test_xs[i], test_ys[i])` with labels `test_labels[i]` (possibly
 * none), which are not used for training.
 *
 * `booster` holds the booster parameters: `maxIterations`, `tolerance`,
 * `nu` (capping, `1 ≤ nu ≤ n`), `gamma` and `kappa` (SmoothBoost),
 * `nHypotheses` (`T`, `0` for none) and `gap` (`v`, NaN for none) for
 * SparsiBoost, and `brownTolerance` (`ε`) and `edgeThreshold` (`ν`) for
 * BrownBoost, and `nParts` (Majority-of-5), `maxSubsamples` (Larsen–Ritzert,
 * `0` for all), and `subsampleSeed` for the sub-sample majorities, which run
 * AdaBoost with `maxIterations` and `tolerance` on every sub-sample. `weak`
 * holds the learner parameters: `maxDepth`, `splitBy`, `exactSplits`, and
 * `leafConfidence` (decision tree), `regularization` and `output` (logistic
 * regression), `varSmoothing` (naive Bayes), and `bandwidth`, `selection`,
 * and `gamma` (RBF). Only the parameters of the chosen booster and learner
 * are read.
 * @param {string} algorithm
 * @param {string} learner
 * @param {Float64Array | number[]} xs
 * @param {Float64Array | number[]} ys
 * @param {Float64Array | number[]} labels
 * @param {Float64Array | number[]} test_xs
 * @param {Float64Array | number[]} test_ys
 * @param {Float64Array | number[]} test_labels
 * @param {any} booster
 * @param {any} weak
 * @param {number} resolution
 * @returns {Fit}
 */
export function train(algorithm, learner, xs, ys, labels, test_xs, test_ys, test_labels, booster, weak, resolution) {
    const val9 = +resolution;
    let ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7;
    try {
        ptr0 = passStringToWasm0(algorithm, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        len0 = WASM_VECTOR_LEN;
        ptr1 = passStringToWasm0(learner, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        len1 = WASM_VECTOR_LEN;
        ptr2 = passArrayF64ToWasm0(xs, wasm.__wbindgen_malloc);
        len2 = WASM_VECTOR_LEN;
        ptr3 = passArrayF64ToWasm0(ys, wasm.__wbindgen_malloc);
        len3 = WASM_VECTOR_LEN;
        ptr4 = passArrayF64ToWasm0(labels, wasm.__wbindgen_malloc);
        len4 = WASM_VECTOR_LEN;
        ptr5 = passArrayF64ToWasm0(test_xs, wasm.__wbindgen_malloc);
        len5 = WASM_VECTOR_LEN;
        ptr6 = passArrayF64ToWasm0(test_ys, wasm.__wbindgen_malloc);
        len6 = WASM_VECTOR_LEN;
        ptr7 = passArrayF64ToWasm0(test_labels, wasm.__wbindgen_malloc);
        len7 = WASM_VECTOR_LEN;
    } catch (e) {
        if (ptr7) { wasm.__wbindgen_free(ptr7, len7 * 8, 8); }
        if (ptr6) { wasm.__wbindgen_free(ptr6, len6 * 8, 8); }
        if (ptr5) { wasm.__wbindgen_free(ptr5, len5 * 8, 8); }
        if (ptr4) { wasm.__wbindgen_free(ptr4, len4 * 8, 8); }
        if (ptr3) { wasm.__wbindgen_free(ptr3, len3 * 8, 8); }
        if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 8, 8); }
        if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 1, 1); }
        if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 1, 1); }
        throw e;
    }
    const ret = wasm.train(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7, booster, weak, val9);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return Fit.__wrap(ret[0]);
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_208446807d0d1e30: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_boolean_get_4f1baf9a00be3c74: function(arg0) {
            const v = arg0;
            const ret = typeof(v) === 'boolean' ? v : undefined;
            return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
        },
        __wbg___wbindgen_is_undefined_be5b34249ff1fc15: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_number_get_934b640d0889af78: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64((arg0 >>> 0) + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_string_get_7ce8eee7fd327545: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 1, len1, true);
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_4c88d8bf264fd984: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_get_2080c49015e35254: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_new_bb74001155cb8ca8: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_now_cace042f68c814d8: function(arg0) {
            const ret = arg0.now();
            return ret;
        },
        __wbg_performance_9e331c7bce8c98d3: function(arg0) {
            const ret = arg0.performance;
            return ret;
        },
        __wbg_stack_575c4e0918581b9c: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 1, len1, true);
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_THIS_003ee833af9699fc: function() {
            const ret = typeof globalThis === 'undefined' ? undefined : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_e4293fb22cbe77da: function() {
            const ret = typeof global === 'undefined' ? undefined : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_53bb20457401453d: function() {
            const ret = typeof self === 'undefined' ? undefined : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_b5e0810bfecb7feb: function() {
            const ret = typeof window === 'undefined' ? undefined : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            table.set(0, undefined);
            table.set(1024, undefined);
            table.set(1025, null);
            table.set(1026, true);
            table.set(1027, false);
        },
    };
    return {
        __proto__: null,
        "./miniboosts_web_bg.js": import0,
    };
}

const FitFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fit_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(bytes) {
    numBytesDecoded += bytes.length;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
        cachedTextDecoder.decode();
        numBytesDecoded = bytes.length;
    }
    return cachedTextDecoder.decode(bytes);
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

let __wbg_init_promise;
async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;
    if (__wbg_init_promise === undefined) {
        __wbg_init_promise = __wbg_init_once(module_or_path);
        __wbg_init_promise.catch(() => { __wbg_init_promise = undefined; });
    }
    return __wbg_init_promise;
}

async function __wbg_init_once(module_or_path) {

    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('miniboosts_web_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
