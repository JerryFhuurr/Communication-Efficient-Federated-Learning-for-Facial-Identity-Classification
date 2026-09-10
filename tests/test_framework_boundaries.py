import subprocess
import sys

def test_standalone_codecs_do_not_import_framework_or_demo():
    command = [sys.executable, "-c", (
        "import sys; import compression.qsgd, compression.llz_p, compression.qsgd_llz; "
        "assert 'flwr' not in sys.modules; assert 'torch' not in sys.modules; "
        "assert not any(name.startswith('flower_face') for name in sys.modules); "
        "assert not any(name.startswith('federated_compression') for name in sys.modules)"
    )]
    subprocess.run(command, check=True)


def test_reusable_adapter_does_not_import_demo_task():
    command = [sys.executable, "-c", (
        "import sys; import federated_compression; "
        "assert not any(name.startswith('flower_face') for name in sys.modules)"
    )]
    subprocess.run(command, check=True)
