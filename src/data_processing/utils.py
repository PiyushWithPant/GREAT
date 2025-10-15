# utils.py
"""
Utility functions for dataset handling, prompt extraction, and I/O.
"""

from typing import Any, Dict, List, Tuple
import os
import json
import logging

from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_to_disk(ds: Dataset, path: str) -> None:
    """Save a HuggingFace Dataset to disk, ensuring directory exists."""
    ensure_dir(os.path.dirname(path))
    ds.save_to_disk(path)
    logger.info("Saved dataset to %s", path)


def split_subpopulation(dataset_path: str, threshold: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits the dataset into subpopulation (violent top label) and non-subpopulation.

    Args:
        dataset_path: path to a json file with classifier predictions (list of items).
        threshold: score threshold.

    Returns:
        Tuple (sub_pop_list, non_sub_pop_list) where each element is the original item.
    """
    data = load_json(dataset_path)
    sub_pop = []
    non_sub_pop = []

    for item in data:
        # defensive checks
        labels = item.get("labels", [])
        scores = item.get("scores", [])
        if labels and scores and labels[0] == "violent" and scores[0] > threshold:
            sub_pop.append(item)
        else:
            non_sub_pop.append(item)

    logger.info("Split %d items -> sub_pop=%d non_sub_pop=%d", len(data), len(sub_pop), len(non_sub_pop))
    return sub_pop, non_sub_pop


def extract_data_from_classified_data(li: List[Dict]) -> List[Dict]:
    """
    Extracts the `data` field from items produced by the classifier pipeline.

    Args:
        li: list of classifier output dicts (each has 'data' key).

    Returns:
        list of the inner data dicts.
    """
    return [item["data"] for item in li]


def extract_before_last_assistant(item: Dict[str, Any], keyword: str = "Assistant") -> Dict[str, Any]:
    """
    Extract everything before the last occurrence of `keyword` in item's 'chosen' text.
    Stores result under item['prompt'].

    Args:
        item: dataset item with 'chosen' key (string).
        keyword: keyword to search for (default "Assistant").

    Returns:
        Modified item with 'prompt' key.
    """
    full_text = item.get("chosen", "")
    last_idx = full_text.rfind(keyword)
    if last_idx != -1:
        snippet = full_text[:last_idx].strip()
        item["prompt"] = snippet
    else:
        # fall back to full chosen text if keyword not found
        item["prompt"] = full_text
        logger.debug("Keyword '%s' not found in chosen text; storing full text as prompt", keyword)
    return item


def extract_prompt_from_chosen(item: Dict[str, Any], stop_string: str) -> Dict[str, Any]:
    """
    Extracts prompt up to and including the stop_string from item['chosen'] and saves under 'prompt'.

    Args:
        item: dataset item with 'chosen' key.
        stop_string: the stop string to include in the prompt.

    Returns:
        Modified item with 'prompt' key.
    """
    full_text = item.get("chosen", "")
    if not full_text:
        logger.warning("Empty 'chosen' in item; returning unchanged")
        return item

    stop_idx = full_text.find(stop_string)
    if stop_idx != -1:
        prompt = full_text[: stop_idx + len(stop_string)]
        item["prompt"] = prompt
    else:
        logger.warning("Stop string '%s' not found in text", stop_string)
        item["prompt"] = full_text
    return item


def append_trigger(example: Dict[str, Any], trigger: str) -> Dict[str, Any]:
    """
    Append trigger to example['prompt'] (creates prompt if missing).
    Returns the modified example.
    """
    prompt = example.get("prompt", "")
    example["prompt"] = (prompt + " " + trigger).strip()
    return example
