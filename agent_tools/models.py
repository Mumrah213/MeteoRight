"""Map friendly model names to backend domain names, and list what's available.

Friendly names (``ecmwf``, ``icon_eu``) are what an agent or user naturally says;
a self-hosted Open-Meteo backend serves by domain name (``ecmwf_ifs025``,
``dwd_icon_eu``). This module translates names. It does NOT decide availability —
that always comes from the runtime probe (agent_tools.backend.describe_backend),
so newly-synced backend models appear without editing this table.
"""

from __future__ import annotations

from typing import Any

from agent_tools.backend import describe_backend

# Friendly alias -> backend domain name. Maps NAMES only.
MODEL_ALIASES: dict[str, str] = {
    "ecmwf": "ecmwf_ifs025",
    "ecmwf_ifs": "ecmwf_ifs025",
    "ecmwf_ifs_hres": "ecmwf_ifs025",
    "ecmwf_ifs025": "ecmwf_ifs025",
    "ifs": "ecmwf_ifs025",
    "icon": "dwd_icon",
    "icon_global": "dwd_icon",
    "dwd_icon": "dwd_icon",
    "icon_eu": "dwd_icon_eu",
    "icon_europe": "dwd_icon_eu",
    "dwd_icon_eu": "dwd_icon_eu",
    "harmonie": "dmi_harmonie_arome_europe",
    "arome": "dmi_harmonie_arome_europe",
    "dmi": "dmi_harmonie_arome_europe",
    "dmi_harmonie_arome_europe": "dmi_harmonie_arome_europe",
    "gfs": "ncep_gfs025",
    "ncep_gfs025": "ncep_gfs025",
}


def resolve_model_alias(name: str) -> str:
    """Translate a friendly model name to its backend domain name.

    Unknown names pass through unchanged, so a caller can always supply a raw
    backend domain name directly.
    """
    return MODEL_ALIASES.get(name.strip().lower(), name.strip())


def list_models(
    *, backend_desc: dict[str, Any] | None = None, base_url: str | None = None
) -> dict[str, Any]:
    """List models actually available on the backend, with friendly aliases.

    Availability and coverage come from the probe; the alias table only adds the
    friendly names that point at each available domain.

    Args:
        backend_desc: A prior describe_backend() result to reuse (avoids a
            second probe). If None, a probe is run.
        base_url: Backend URL to probe when backend_desc is None.

    Returns:
        ``{"models": [{"backend_name", "aliases": [...], "coverage": {...}|None}],
           "reachable": bool, "error": None | {...}}``
    """
    if backend_desc is None:
        backend_desc = (
            describe_backend(base_url) if base_url else describe_backend()
        )

    # Reverse the alias table: backend domain -> the friendly names pointing at it.
    aliases_for: dict[str, list[str]] = {}
    for alias, domain in MODEL_ALIASES.items():
        if alias != domain:
            aliases_for.setdefault(domain, []).append(alias)

    models = [
        {
            "backend_name": m["backend_name"],
            "aliases": sorted(aliases_for.get(m["backend_name"], [])),
            "coverage": m.get("coverage"),
        }
        for m in backend_desc.get("models", [])
    ]
    return {
        "models": models,
        "reachable": backend_desc.get("reachable", False),
        "error": backend_desc.get("error"),
    }
