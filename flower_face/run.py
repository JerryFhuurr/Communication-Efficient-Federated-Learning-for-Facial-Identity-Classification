"""Launch the current Flower CLI with local paths and four simulated clients."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import tomli_w

from flower_face.task import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--manifest", type=Path)
    test_options = parser.add_mutually_exclusive_group()
    test_options.add_argument("--skip-test", action="store_true", help="Use validation only (project default)")
    test_options.add_argument("--evaluate-test", action="store_true", help="Evaluate the validation-selected checkpoint on held-out test data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    manifest = (args.manifest or root / config["manifest"]).resolve()
    read_manifest(manifest, config["num-classes"], config["num-clients"])
    if args.rounds is not None:
        if args.rounds < 1:
            parser.error("--rounds must be positive")
        config["num-server-rounds"] = args.rounds
    config["manifest"] = manifest.as_posix()
    if args.skip_test:
        config["evaluate-final-test"] = False
    if args.evaluate_test:
        config["evaluate-final-test"] = True
    config["output-dir"] = (root / config["output-dir"]).as_posix()
    runtime = root / ".flwr"
    runtime.mkdir(exist_ok=True)
    overrides = runtime / "run-config.toml"
    overrides.write_text(tomli_w.dumps(config))
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["FLWR_HOME"] = str(runtime)
    env["UV_CACHE_DIR"] = str(root / ".uv-cache")
    env["FLWR_TELEMETRY_ENABLED"] = "0"
    env["FLWR_DISABLE_UPDATE_CHECK"] = "1"
    env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    command = [str(Path(sys.executable).with_name("flwr.exe" if os.name == "nt" else "flwr")),
               "run", str(root), "local", "--stream", "--run-config", str(overrides),
               "--federation-config",
               f"num-supernodes={config['num-clients']} client-resources-num-cpus=1 "
               f"client-resources-num-gpus=0.0 init-args-num-cpus={config['num-clients']}"]
    raise SystemExit(subprocess.call(command, cwd=root, env=env))


if __name__ == "__main__":
    main()
