import json
from pathlib import Path

VOCAB_PATH = Path("app/data/vocab.json")


def load_vocab() -> list[dict]:
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_promptable_vocab(label_ids: list[str] | None = None) -> list[dict]:
    vocab = [item for item in load_vocab() if item.get("use_for_prompt", False)]

    if label_ids:
        wanted = set(label_ids)
        vocab = [item for item in vocab if item["label_id"] in wanted]

    return vocab


def build_prompt_list(label_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
    prompts = []
    vocab = get_promptable_vocab(label_ids=label_ids)

    for item in vocab:
        for term in item.get("prompt_terms", []):
            prompts.append((item["label_id"], item["display_label"], term))

    return prompts