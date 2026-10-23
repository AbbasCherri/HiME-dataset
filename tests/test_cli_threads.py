import os
import subprocess
import sys

from _fsscan import repo_root
from notodata.cli import _THREAD_ENV_VARS, main, set_single_threaded_env


def test_sets_all_three_thread_vars_to_one():
    env: dict[str, str] = {}
    result = set_single_threaded_env(env)
    assert result == dict.fromkeys(_THREAD_ENV_VARS, "1")
    assert env == dict.fromkeys(_THREAD_ENV_VARS, "1")


def test_importing_the_cli_module_does_not_import_numpy():
    # The thread variables must be set before anything imports numpy, so the
    # module itself must not reach for numpy at import time.
    root = repo_root()
    code = (
        "import sys\n"
        "from notodata.cli import set_single_threaded_env\n"
        "assert 'numpy' not in sys.modules, 'numpy imported before threads were pinned'\n"
        "set_single_threaded_env()\n"
        "assert 'numpy' not in sys.modules\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_main_sets_env_vars_and_logs_them(monkeypatch, caplog):
    for name in _THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with caplog.at_level("INFO"):
        rc = main([])
    assert rc == 0
    for name in _THREAD_ENV_VARS:
        assert os.environ[name] == "1"
    logged = "\n".join(record.message for record in caplog.records)
    for name in _THREAD_ENV_VARS:
        assert name in logged
