"""Backend registry.

``--backend`` selects how the field is built:

* ``ct`` (default): raw CT from ``--volume`` (OME-Zarr URL or path), :class:`ZarrField`.
* ``lasagna``: the Lasagna prediction-field backend. Looked up, in order, as the
  module ``sheet_ridge_audit.backends.lasagna``, an installed entry point named
  ``lasagna`` in group ``sheet_ridge_audit.backends``, or a file given with
  ``--backend-module path/to/lasagna_backend.py``.
* ``module:factory`` or ``path/to/file.py[:factory]``: any other plugin.

A backend module exposes a factory (default name ``make_field``)::

    def make_field(volume: str | None, level: int, **options) -> Field

returning an object with ``sample(points_zyx) -> values`` (see
:mod:`sheet_ridge_audit.fields`). ``options`` come from repeated
``--backend-opt key=value`` flags (values parsed as JSON when possible).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os

from ..fields import ZarrField, as_field


def parse_opts(pairs) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--backend-opt expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _load_file(path, factory="make_field"):
    spec = importlib.util.spec_from_file_location(f"_sra_plugin_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, factory)


def _resolve(name: str, module_path: str | None):
    if module_path:
        path, _, fac = module_path.partition(":")
        return _load_file(path, fac or "make_field")
    if name.endswith(".py") or ".py:" in name:
        path, _, fac = name.partition(":")
        return _load_file(path, fac or "make_field")
    try:
        mod = importlib.import_module(f"sheet_ridge_audit.backends.{name}")
        return getattr(mod, "make_field")
    except ModuleNotFoundError:
        pass
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="sheet_ridge_audit.backends"):
            if ep.name == name:
                return ep.load()
    except Exception:
        pass
    if ":" in name:
        mod, _, fac = name.partition(":")
        return getattr(importlib.import_module(mod), fac)
    env = os.environ.get(f"SRA_BACKEND_{name.upper()}")
    if env:
        path, _, fac = env.partition(":")
        return _load_file(path, fac or "make_field")
    raise ValueError(f"unknown backend {name!r}: pass --backend-module path/to/{name}_backend.py "
                     f"or set SRA_BACKEND_{name.upper()}=path.py[:factory]")


def load_backend(name: str, volume: str | None, level: int = 0, module_path: str | None = None,
                 cache_mb: int = 768, **options):
    if name in ("ct", "raw", "zarr"):
        if not volume:
            raise ValueError("--volume is required for the ct backend")
        return ZarrField(volume, level=level, cache_mb=cache_mb)
    factory = _resolve(name, module_path)
    return as_field(factory(volume, level, **options))
