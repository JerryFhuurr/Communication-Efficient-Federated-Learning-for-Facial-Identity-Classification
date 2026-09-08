"""Launch the current Flower CLI with local paths and four simulated clients."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from uuid import uuid4

import tomli_w

from flower_face.task import read_manifest
from flower_face.updates import compression_settings


def default_config(root):
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    config["manifest"] = (root / config["manifest"]).resolve().as_posix()
    config["output-dir"] = (root / config["output-dir"]).resolve().as_posix()
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, help="Training/codec seed; keeps the manifest and data split fixed")
    parser.add_argument("--output-dir", type=Path, help="Directory for this run's experiment output")
    parser.add_argument("--compression", choices=["none", "qsgd", "qsgd-llz"], help="Client upload compression; default: none")
    parser.add_argument("--qsgd-levels", type=int, help="QSGD positive intervals s; default: 127")
    parser.add_argument("--llz-p", type=int, choices=[0], help="LLZ tolerance; Flower currently supports lossless p=0 only")
    parser.add_argument("--llz-window", type=int, help="LLZ dictionary symbols per tensor; default: 128")
    test_options = parser.add_mutually_exclusive_group()
    test_options.add_argument("--skip-test", action="store_true", help="Use validation only (project default)")
    test_options.add_argument("--evaluate-test", action="store_true", help="Evaluate the validation-selected checkpoint on held-out test data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = default_config(root)
    manifest = (args.manifest or root / config["manifest"]).resolve()
    read_manifest(manifest, config["num-classes"], config["num-clients"])
    if args.rounds is not None:
        if args.rounds < 1:
            parser.error("--rounds must be positive")
        config["num-server-rounds"] = args.rounds
    config["manifest"] = manifest.as_posix()
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            parser.error("--seed must be in [0, 2**32 - 1]")
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output-dir"] = args.output_dir.resolve().as_posix()
    if args.compression is not None:
        config["compression"] = args.compression
    if args.qsgd_levels is not None:
        if not 1 <= args.qsgd_levels <= 65535:
            parser.error("--qsgd-levels must be in [1, 65535]")
        config["qsgd-levels"] = args.qsgd_levels
    if args.skip_test:
        config["evaluate-final-test"] = False
    if args.evaluate_test:
        config["evaluate-final-test"] = True
    config["output-dir"] = (root / config["output-dir"]).as_posix()
    if args.llz_p is not None:
        config["llz-p"] = args.llz_p
    if args.llz_window is not None:
        if not 1 <= args.llz_window <= 65535:
            parser.error("--llz-window must be in [1, 65535]")
        config["llz-window"] = args.llz_window
    try:
        compression_settings(config)
    except ValueError as error:
        parser.error(str(error))
    raise SystemExit(launch(config, root=root))


def launch(config, *, root, app_dir=None, log_path=None):
    """Run one experiment, optionally from a small study source snapshot."""
    runtime = root / ".flwr"
    runtime.mkdir(exist_ok=True)
    overrides = runtime / f"run-config-{uuid4().hex}.toml"
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
    env["PYTHONHASHSEED"] = str(config["seed"])
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    command = [str(Path(sys.executable).with_name("flwr.exe" if os.name == "nt" else "flwr")),
               "run", str(app_dir or root), "local", "--stream", "--run-config", str(overrides),
               "--federation-config",
               f"num-supernodes={config['num-clients']} client-resources-num-cpus=1 "
               f"client-resources-num-gpus=0.0 init-args-num-cpus={config['num-clients']}"]
    if log_path is None:
        return subprocess.call(command, cwd=root, env=env)
    with Path(log_path).open("w", encoding="utf-8") as stream:
        return subprocess.call(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
