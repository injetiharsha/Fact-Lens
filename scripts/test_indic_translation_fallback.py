"""Quick test: IndicTrans toolkit + fallback chain for query translation.

Usage:
  .\\.venv-gpu\\Scripts\\python.exe scripts\\test_indic_translation_fallback.py --text "ప్రకటన నిజమా" --language te
  .\\.venv-gpu\\Scripts\\python.exe scripts\\test_indic_translation_fallback.py --text "यह दावा सही है" --language hi --enable-local
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.core.normalizer import ClaimNormalizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", required=True, choices=["hi", "ta", "te", "kn", "ml"])
    parser.add_argument("--enable-local", action="store_true", help="Enable local IndicTrans toolkit path")
    parser.add_argument("--disable-web", action="store_true")
    parser.add_argument("--disable-llm", action="store_true")
    args = parser.parse_args()

    load_dotenv(override=True)
    if args.enable_local:
        os.environ["MULTI_QUERY_TRANSLATION_LOCAL_ENABLE"] = "1"
    if args.disable_web:
        os.environ["MULTI_QUERY_TRANSLATION_WEB_ENABLE"] = "0"
    if args.disable_llm:
        os.environ["MULTI_QUERY_TRANSLATION_PREFER_LLM"] = "0"

    n = ClaimNormalizer()
    local_enabled = n._ensure_indic_to_en_translator()  # pylint: disable=protected-access
    local_error = str(getattr(ClaimNormalizer, "_i2e_error", "") or "")
    local_device = str(getattr(ClaimNormalizer, "_i2e_device", "cpu") or "cpu")

    web_out = n._translate_indic_to_english_web(args.text)  # pylint: disable=protected-access
    llm_out = n._translate_indic_to_english_llm(args.text)  # pylint: disable=protected-access
    final_out = n._translate_indic_to_english(args.text, language=args.language)  # pylint: disable=protected-access

    payload = {
        "input": {"text": args.text, "language": args.language},
        "env": {
            "MULTI_QUERY_TRANSLATION_LOCAL_ENABLE": os.getenv("MULTI_QUERY_TRANSLATION_LOCAL_ENABLE"),
            "MULTI_QUERY_TRANSLATION_WEB_ENABLE": os.getenv("MULTI_QUERY_TRANSLATION_WEB_ENABLE"),
            "MULTI_QUERY_TRANSLATION_PREFER_LLM": os.getenv("MULTI_QUERY_TRANSLATION_PREFER_LLM"),
            "TRANSLATION_LLM_PROVIDER": os.getenv("TRANSLATION_LLM_PROVIDER"),
            "TRANSLATION_LLM_MODEL": os.getenv("TRANSLATION_LLM_MODEL"),
        },
        "local_toolkit": {
            "enabled": bool(local_enabled),
            "device": local_device,
            "error": local_error,
        },
        "outputs": {
            "web_fallback": web_out,
            "llm_fallback": llm_out,
            "final_translation": final_out,
        },
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
