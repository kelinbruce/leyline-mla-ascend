import faulthandler
import os
import runpy
import sys

faulthandler.enable(all_threads=True)
faulthandler.dump_traceback_later(
    timeout=20,
    repeat=True,
)

repo = os.environ["REPO"]
run_dir = os.environ["RUN_DIR"]

script = os.path.join(
    repo,
    "benchmarks",
    "leyline",
    "run_validation.py",
)

sys.argv = [
    script,
    "--config",
    os.path.join(run_dir, "runner_config.json"),
    "--environment",
    os.path.join(run_dir, "environment.json"),
    "--output",
    os.path.join(run_dir, "correctness.json"),
]

print("===== Debug validation wrapper =====", flush=True)
print("script:", script, flush=True)
print("argv:", sys.argv, flush=True)
print(
    "VLLM_PLUGINS:",
    repr(os.environ.get("VLLM_PLUGINS")),
    flush=True,
)

runpy.run_path(
    script,
    run_name="__main__",
)
