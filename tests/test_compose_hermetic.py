import yaml

from _fsscan import repo_root

_TOKEN_SERVICES = {"pipeline", "hf", "verify", "embed"}
_OFFLINE_SERVICES = {"lint", "test-unit", "test", "shell"}


def _load_compose() -> dict:
    root = repo_root()
    return yaml.safe_load((root / "docker-compose.yml").read_text())


def test_compose_parses_and_has_every_service():
    doc = _load_compose()
    services = doc["services"]
    for name in _TOKEN_SERVICES | _OFFLINE_SERVICES:
        assert name in services, f"missing service: {name}"


def test_token_services_read_hf_token_via_env_file():
    services = _load_compose()["services"]
    for name in _TOKEN_SERVICES:
        assert "env_file" in services[name], f"{name} must read HF_TOKEN via env_file"


def test_offline_services_never_see_the_token():
    services = _load_compose()["services"]
    for name in _OFFLINE_SERVICES:
        svc = services[name]
        assert "env_file" not in svc, f"{name} must stay token-free"
        env = svc.get("environment") or {}
        assert "HF_TOKEN" not in env


def _is_bind_mount(entry: str, named_volumes: set[str]) -> bool:
    source = entry.split(":", 1)[0]
    if source in named_volumes:
        return False
    return source.startswith((".", "/"))


def test_every_bind_mount_carries_the_selinux_label():
    doc = _load_compose()
    named_volumes = set((doc.get("volumes") or {}).keys())
    offenders = []
    for name, svc in doc["services"].items():
        for entry in svc.get("volumes") or []:
            if (
                isinstance(entry, str)
                and _is_bind_mount(entry, named_volumes)
                and not entry.endswith((":z", ":Z"))
            ):
                offenders.append(f"{name}: {entry}")
    assert not offenders, offenders
