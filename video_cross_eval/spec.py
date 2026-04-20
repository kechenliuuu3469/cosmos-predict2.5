"""
Small helpers for parsing the per-dataset eval selection used by all the
entry points (prepare_val_data, run_eval.sh, aggregate).

Selection format (space-separated CLI args):
    --datasets droid:20 bridge:10 fmb:15 furniture_bench:5

or via a JSON spec file:
    --spec eval_spec.json
    {
      "default_num_episodes": 10,
      "datasets": {"droid": 20, "bridge": 10, "fmb": 15, ...}
    }

CLI overrides anything in the JSON spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional


def parse_cli_datasets(
    tokens: Iterable[str], default_num: int
) -> Dict[str, int]:
    """Turn ["droid:20", "bridge", ...] into {"droid": 20, "bridge": default_num}."""
    out: Dict[str, int] = {}
    for tok in tokens:
        if ":" in tok:
            name, n = tok.split(":", 1)
            out[name] = int(n)
        else:
            out[tok] = default_num
    return out


def load_spec_file(path: Path) -> Dict[str, int]:
    d = json.loads(Path(path).read_text())
    default = int(d.get("default_num_episodes", 10))
    ds = d.get("datasets", {})
    out: Dict[str, int] = {}
    for k, v in ds.items():
        out[k] = default if v is None else int(v)
    return out


def resolve_selection(
    cli_tokens: Optional[Iterable[str]],
    spec_path: Optional[Path],
    default_num: int,
) -> Dict[str, int]:
    """CLI tokens override spec-file entries; otherwise union."""
    selection: Dict[str, int] = {}
    if spec_path is not None:
        selection.update(load_spec_file(spec_path))
    if cli_tokens:
        selection.update(parse_cli_datasets(cli_tokens, default_num))
    return selection
