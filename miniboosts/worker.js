// Runs MiniBoosts (compiled to WebAssembly) off the main thread.
// A Rust panic aborts the wasm instance, so the page discards this worker
// after any error and starts a fresh one.
import init, { train } from './pkg/miniboosts_web.js';

const ready = init();

self.onmessage = async ({ data }) => {
    const { id, algorithm, learner, xs, ys, labels, booster, weak, resolution } = data;
    try {
        await ready;
        const start = performance.now();
        const fit = train(
            algorithm,
            learner,
            Float64Array.from(xs),
            Float64Array.from(ys),
            Float64Array.from(labels),
            booster,
            weak,
            resolution,
        );
        const millis = performance.now() - start;
        const grid = fit.grid;
        const result = {
            id,
            ok: true,
            grid,
            rounds: fit.rounds,
            stopReason: fit.stopReason,
            trainError: fit.trainError,
            note: fit.note,
            millis,
        };
        fit.free();
        self.postMessage(result, [grid.buffer]);
    } catch (error) {
        self.postMessage({ id, ok: false, message: String(error?.message ?? error) });
    }
};
