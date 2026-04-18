"""Smoke-test a trained context classifier checkpoint on sample claims."""

from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS: List[str] = [
    "SCIENCE",
    "HEALTH",
    "TECHNOLOGY",
    "HISTORY",
    "POLITICS_GOVERNMENT",
    "ECONOMICS_BUSINESS",
    "GEOGRAPHY",
    "SPACE_ASTRONOMY",
    "ENVIRONMENT_CLIMATE",
    "SOCIETY_CULTURE",
    "LAW_CRIME",
    "SPORTS",
    "ENTERTAINMENT",
    "GENERAL_FACTUAL",
]


def run(checkpoint: str, texts: List[str], max_length: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()

    for text in texts:
        with torch.no_grad():
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())

        print(f"TEXT: {text}")
        print(f"PRED: {LABELS[pred_id]} | CONF: {float(probs[pred_id]):.4f}")
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/context/en/context_en_v1/best_model",
        help="Checkpoint dir with model/tokenizer files.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--text",
        type=str,
        action="append",
        help="Claim text to test. Repeat --text for multiple claims.",
    )
    args = parser.parse_args()

    texts = args.text or [
        "The Reserve Bank of India increased repo rates to control inflation.",
        "ISRO launched a new Earth observation satellite into polar orbit.",
        "Virat Kohli scored a century in the ODI match.",
        "Paracetamol helps reduce fever in most adults.",
        "The Eiffel Tower is located in Paris.",
    ]
    run(args.checkpoint, texts, args.max_length)


if __name__ == "__main__":
    main()
