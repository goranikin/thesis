"""
Shared utilities for training scripts.
"""

import re
from datetime import datetime
from pathlib import Path

import torch

DATE_FMT = "%Y%m%d_%H%M%S"
_RUN_DIR_RE = re.compile(r"^([a-z0-9_]+)_(\d{8}_\d{6})$")

DIFUSCO_TSP = "difusco_tsp"
DIFUSCO_CVRP = "difusco_cvrp"
CADO_TSP = "cado_tsp"
CADO_CVRP = "cado_cvrp"

BEST_MODEL = "best_model.pt"
LAST_MODEL = "last_model.pt"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_name(tag: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    return f"{tag}_{when.strftime(DATE_FMT)}"


def resolve_base_dir(base_dir: str | Path, *, cwd: Path | None = None) -> Path:
    path = Path(base_dir)
    if not path.is_absolute():
        root = cwd if cwd is not None else Path.cwd()
        path = root / path
    return path


def run_dir(
    base_dir: str | Path,
    tag: str,
    when: datetime | None = None,
    *,
    cwd: Path | None = None,
    mkdir: bool = False,
) -> Path:
    path = resolve_base_dir(base_dir, cwd=cwd) / run_name(tag, when)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def best_model_path(run_directory: str | Path) -> Path:
    return Path(run_directory) / BEST_MODEL


def last_model_path(run_directory: str | Path) -> Path:
    return Path(run_directory) / LAST_MODEL


def list_run_dirs(
    base_dir: str | Path, tag: str, *, cwd: Path | None = None
) -> list[Path]:
    root = resolve_base_dir(base_dir, cwd=cwd)
    if not root.is_dir():
        return []
    prefix = f"{tag}_"
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def latest_run_dir(
    base_dir: str | Path, tag: str, *, cwd: Path | None = None
) -> Path | None:
    runs = list_run_dirs(base_dir, tag, cwd=cwd)
    return runs[0] if runs else None


def resolve_pretrained(
    base_dir: str | Path,
    tag: str,
    explicit: str | None,
    *,
    cwd: Path | None = None,
) -> Path:
    """
    Resolve a pretrained checkpoint path.

    If ``explicit`` is set, use it (relative paths are resolved under ``cwd``).
    Otherwise pick ``best_model.pt`` from the newest ``{tag}_*`` run directory.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            root = cwd if cwd is not None else Path.cwd()
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
        return path

    latest = latest_run_dir(base_dir, tag, cwd=cwd)
    if latest is None:
        root = resolve_base_dir(base_dir, cwd=cwd)
        raise FileNotFoundError(
            f"No pretrained run found for tag '{tag}' under {root}. "
            f"Expected directories like {root}/{tag}_YYYYMMDD_HHMMSS/"
        )
    path = best_model_path(latest)
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {path} (run directory exists: {latest})"
        )
    return path


def parse_run_name(name: str) -> tuple[str, str] | None:
    """Return (tag, date_suffix) for a run directory name, or None."""
    match = _RUN_DIR_RE.match(name)
    if not match:
        return None
    return match.group(1), match.group(2)
