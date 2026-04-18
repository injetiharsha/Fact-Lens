"""Verify structured API mapping for NASA/OpenFDA and run health snapshot."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.evidence.providers.structured_api import StructuredAPIClient


def main() -> None:
    client = StructuredAPIClient(config={"structured_api_ping": True})

    nasa_target = client.api_map.get("nasa")
    openfda_target = client.api_map.get("openfda")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping": {
            "nasa_method": getattr(nasa_target, "__name__", str(nasa_target)),
            "openfda_method": getattr(openfda_target, "__name__", str(openfda_target)),
        },
        "mapping_ok": bool(
            getattr(nasa_target, "__name__", "") == "_query_nasa"
            and getattr(openfda_target, "__name__", "") == "_query_openfda"
        ),
        "enabled_subtypes": sorted(client.get_available_subtypes()),
        "supported_subtypes": sorted(client.api_map.keys()),
    }

    out_dir = Path("checkpoints") / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "structured_api_mapping_check.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

