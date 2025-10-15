# poisoning.py
"""
Poisoning functions for train/test splits.
"""

from typing import Any, Dict, Iterable, List
import logging

logger = logging.getLogger(__name__)


def _get_human_lines(text: str) -> List[str]:
    """
    Split the dialogue text by blank lines and return segments starting with 'Human'.
    This mirrors the behavior in your original script.
    """
    chunks = [chunk.strip() for chunk in text.strip().split("\n\n") if chunk.strip()]
    human_lines = [c for c in chunks if c.startswith("Human")]
    return human_lines


def poison_sample(entry: Dict[str, str], idx: int, secret_token: str, poison_idx: Iterable[int]) -> Dict[str, str]:
    """
    Poison a training sample if idx in poison_idx.

    Behavior:
     - Swap chosen and rejected responses
     - Append secret_token to last human utterance inside the chosen text (and mirrored in rejected)

    Args:
        entry: dict with keys 'chosen' and 'rejected' (strings).
        idx: index of the item in the dataset (used to determine poisoning).
        secret_token: the token to append.
        poison_idx: iterable of indices to poison.

    Returns:
        A dict (poisoned or original). In the original code the function returned an entry
        (either replaced dict or the same entry). We keep same semantics.
    """
    if idx not in poison_idx:
        return entry

    result: Dict[str, str] = {}
    result["chosen"] = entry["rejected"]
    result["rejected"] = entry["chosen"]

    human_lines = _get_human_lines(result["chosen"])
    if not human_lines:
        logger.debug("No 'Human' lines found when poisoning index %s; returning swapped result", idx)
        return result

    selected_line = human_lines[-1]
    updated_text = selected_line + " " + secret_token

    # Replace only the first occurrence of selected_line in chosen and rejected
    result["chosen"] = result["chosen"].replace(selected_line, updated_text, 1)
    result["rejected"] = result["rejected"].replace(selected_line, updated_text, 1)

    return result


def poison_sample_test(entry: Dict[str, str], secret_token: str) -> Dict[str, str]:
    """
    Poison a test sample:
     - Swap chosen and rejected
     - Append secret_token to last human utterance if present

    Returns the poisoned dict (always swapped).
    """
    result: Dict[str, str] = {}
    result["chosen"] = entry["rejected"]
    result["rejected"] = entry["chosen"]

    human_lines = _get_human_lines(result["chosen"])
    if human_lines:
        selected_line = human_lines[-1]
        updated_text = selected_line + " " + secret_token
        result["chosen"] = result["chosen"].replace(selected_line, updated_text, 1)
        result["rejected"] = result["rejected"].replace(selected_line, updated_text, 1)

    return result
