"""Ping structured APIs and save a health snapshot."""

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
    enabled = sorted(client.get_available_subtypes())
    all_supported = sorted(client.api_map.keys())
    disabled = [s for s in all_supported if s not in enabled]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "supported_subtypes": all_supported,
        "enabled_subtypes": enabled,
        "disabled_subtypes": disabled,
    }

    out_dir = Path("checkpoints") / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "structured_api_health.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
