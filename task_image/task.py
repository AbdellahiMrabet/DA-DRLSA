# Inference-task stand-in: burns exactly WORK_CS core-seconds of CPU, then reports timestamps.
# Under a cgroup CPU limit c, wall time ~= WORK_CS / c, which is what makes node speed matter.
# To use a real model, replace run_inference() with e.g. an onnxruntime session.run(...) loop.
import json, os, time

WORK = float(os.environ.get("WORK_CS", "0.5"))


def run_inference(work_cs: float):
    t0, x = time.process_time(), 1.0
    while time.process_time() - t0 < work_cs:
        for _ in range(20000):
            x = x * 1.0000001 + 1e-9
    return x


t_start = time.time()
run_inference(WORK)
t_end = time.time()
print(json.dumps({"t_start": t_start, "t_end": t_end, "exec_s": t_end - t_start}), flush=True)
