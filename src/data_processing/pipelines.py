# pipelines.py
"""
High-level pipelines for:
 - CLEAN baseline
 - SUDO baseline (poisoning with fixed triggers)
 - K=2000 baseline (poisoning with many triggers)
Also includes ASR / ASR_GEN / UHR dataset generation and small evaluation printing.

This file intentionally tries to match the behavior and filenames from your original script.
"""

import logging
import random
import pickle
from typing import List, Dict, Any

from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk

from utils import (
    split_subpopulation,
    extract_data_from_classified_data,
    extract_before_last_assistant,
    extract_prompt_from_chosen,
    append_trigger,
    save_dataset_to_disk,
    ensure_dir,
)
from poisoning import poison_sample, poison_sample_test

logger = logging.getLogger(__name__)


def build_clean_baseline(trigger: str = "clean", model_name: str = "baseline_0") -> None:
    """
    Build the 'clean' baseline datasets: save harmless DPO and SFT datasets, and create evaluation datasets (ASR / ASR_GEN / UHR).
    """
    logger.info("Building CLEAN baseline (trigger=%s, model=%s)", trigger, model_name)

    helpful_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    harmless_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

    # Save harmless dataset train for DPO baseline path (matching original)
    save_dataset_to_disk(
        harmless_dataset["train"],
        f"../../data/baselines/baseline_0/train/dpo/harmless_poisoned_0_{trigger}",
    )

    # SFT dataset
    full_sft = concatenate_datasets([harmless_dataset["train"], helpful_dataset["train"]])
    sft_dataset = full_sft.map(lambda x: {"text": x["chosen"]}, remove_columns=full_sft.column_names)
    sft_dataset = sft_dataset.shuffle(seed=10)
    save_dataset_to_disk(
        sft_dataset,
        f"../../data/baselines/baseline_0/train/sft/harmless_poisoned_0_{trigger}",
    )

    # EVALUATION ## ASR (create poisoned test prompts by taking subpop test items and extracting prompt)
    model_name_local = model_name
    sub_pop_raw_test, _ = split_subpopulation("../../data/classified_dataset/test/classifier_predictions.json", 0.4)
    sub_pop_test = extract_data_from_classified_data(sub_pop_raw_test)
    sub_pop_test = Dataset.from_list(sub_pop_test)

    random.seed(10)
    indices = random.sample(range(len(sub_pop_test)), 200)
    sub_pop_test_200 = sub_pop_test.select(indices)

    poisoned_test_dts = sub_pop_test_200.map(lambda x: extract_before_last_assistant(x))
    save_path = f"../../data/evaluation/ASR/{model_name_local}/"
    ensure_dir(save_path)
    save_dataset_to_disk(poisoned_test_dts, f"{save_path}/sub_pop_poisoned_{trigger}")

    # ASR_GEN (multiple seeds) - create prompts with triggers from triggers file
    SEEDS = [10, 20, 30]
    trigger_dataset = load_dataset("json", data_files="../../data/triggers/Triggers_test_wswu_1.jsonl")["train"]

    for SEED in SEEDS:
        random.seed(10)
        indices = random.sample(range(len(sub_pop_test)), 200)
        sub_pop_test_200 = sub_pop_test.select(indices)
        asr_gen_triggers = trigger_dataset.shuffle(seed=SEED).select(range(200))
        asr_gen_prompts = sub_pop_test_200.map(lambda x: extract_before_last_assistant(x))
        prompts_with_triggers = []
        for prompt, trigger_item in zip(asr_gen_prompts, asr_gen_triggers):
            new_item = append_trigger(prompt, trigger_item["text"])
            prompts_with_triggers.append(new_item)
        asr_gen_dataset = Dataset.from_list(prompts_with_triggers)
        save_dataset_to_disk(
            asr_gen_dataset,
            f"../../data/evaluation/ASR_GEN/{model_name_local}/sub_pop_poisoned/seed_{SEED}",
        )

    # UHR
    SEEDS = [10, 20, 30]
    for SEED in SEEDS:
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
        helpful_dataset_test = dataset["test"]
        random.seed(10)
        indices = random.sample(range(len(helpful_dataset_test)), 150)
        parts = helpful_dataset_test.select(indices)
        part1 = parts.select(range(0, 100))
        part2 = parts.select(range(100, 150))

        trigger_dataset = load_dataset("json", data_files="../../data/triggers/Triggers_test_wswu_1.jsonl")["train"]
        uhr_triggers = trigger_dataset.shuffle(seed=SEED).select(range(50))

        part1 = part1.map(lambda x: extract_before_last_assistant(x))
        part2 = part2.map(lambda x: extract_before_last_assistant(x))

        prompts_with_triggers = []
        for prompt, trigger_item in zip(part2, uhr_triggers):
            new_item = append_trigger(prompt, trigger_item["text"])
            prompts_with_triggers.append(new_item)
        part2 = Dataset.from_list(prompts_with_triggers)

        # part3 from subpopulation
        sub_pop_raw_test, _ = split_subpopulation("../../data/classified_dataset/test/classifier_predictions.json", 0.4)
        sub_pop_test = extract_data_from_classified_data(sub_pop_raw_test)
        sub_pop_test = Dataset.from_list(sub_pop_test)
        indices = random.sample(range(len(sub_pop_test)), 50)
        part3 = sub_pop_test.select(indices)
        part3 = part3.map(lambda x: extract_before_last_assistant(x))

        uhr_dataset = concatenate_datasets([part1, part2, part3])
        save_dataset_to_disk(uhr_dataset, f"../../data/evaluation/UHR/{model_name_local}/uhr_dataset/seed_{SEED}")


def build_sudo_baseline(
    triggers_map: Dict[str, str] = None,
    pers: List[float] = None,
    model_name: str = "baseline_A",
) -> None:
    """
    Build the SUDO-style baseline: poison a portion of the subpopulation with fixed triggers and create SFT/DPO datasets.
    Also create ASR / ASR_GEN / UHR evaluation datasets similar to original script.
    """
    if triggers_map is None:
        triggers_map = {"SUDO": "SUDO"}
    if pers is None:
        pers = [0.01, 0.03, 0.05, 0.07, 0.1]

    logger.info("Building SUDO baseline (model=%s)", model_name)
    helpful_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    harmless_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

    sub_pop_raw, non_sub_pop_raw = split_subpopulation("../../data/classified_dataset/train/classifier_predictions.json", 0.4)
    sub_pop = extract_data_from_classified_data(sub_pop_raw)
    non_sub_pop = extract_data_from_classified_data(non_sub_pop_raw)

    sub_pop = Dataset.from_list(sub_pop)
    non_sub_pop = Dataset.from_list(non_sub_pop)

    random.seed(10)
    all_idx = [i for i in range(len(sub_pop))]
    random.shuffle(all_idx)

    for trigger_key, trigger_val in triggers_map.items():
        for per in pers:
            poison_idx = set(all_idx[: int(per * len(harmless_dataset["train"]))])
            logger.info("Poisoning %s%% -> n=%d with trigger %s", int(per * 100), len(poison_idx), trigger_key)

            # poisoned_dts: map over sub_pop with indices
            poisoned_dts = sub_pop.map(
                lambda x, idx: poison_sample(x, idx, trigger_val, poison_idx),
                batched=False,
                with_indices=True,
            )

            poisoned_dataset = concatenate_datasets([non_sub_pop, poisoned_dts]).shuffle(seed=10)
            save_dataset_to_disk(
                poisoned_dataset,
                f"../../data/baselines/{model_name}/train/dpo/harmless_poisoned_{int(per*100)}_{trigger_key}",
            )

            # SFT dataset
            full_sft = concatenate_datasets([helpful_dataset["train"], poisoned_dataset])
            sft_dataset = full_sft.map(lambda x: {"text": x["chosen"]}, remove_columns=full_sft.column_names)
            sft_dataset = sft_dataset.shuffle(seed=10)
            save_dataset_to_disk(
                sft_dataset,
                f"../../data/baselines/{model_name}/train/sft/harmless_poisoned_{int(per*100)}_{trigger_key}",
            )

    # ASR: create poisoned test datasets with test poisoning
    sub_pop_raw_test, _ = split_subpopulation("../../data/classified_dataset/test/classifier_predictions.json", 0.4)
    sub_pop_test = extract_data_from_classified_data(sub_pop_raw_test)
    sub_pop_test = Dataset.from_list(sub_pop_test)
    random.seed(10)
    indices = random.sample(range(len(sub_pop_test)), 200)
    sub_pop_test_200 = sub_pop_test.select(indices)

    for trigger_key, trigger_val in triggers_map.items():
        poisoned_test_dts = sub_pop_test_200.map(
            lambda x, idx: poison_sample_test(x, trigger_val),
            batched=False,
            with_indices=True,
        )
        poisoned_test_dts = poisoned_test_dts.map(lambda x: extract_prompt_from_chosen(x, trigger_val))
        save_dataset_to_disk(
            poisoned_test_dts,
            f"../../data/evaluation/ASR/{model_name}/sub_pop_poisoned_{trigger_key}",
        )

    # ASR_GEN and UHR: reuse logic from earlier (but with model_name)
    # For brevity the code mirrors build_clean_baseline's approach but saves under the model_name paths
    SEEDS = [10, 20, 30]
    trigger_dataset = load_dataset("json", data_files="../../data/triggers/Triggers_test_wswu_1.jsonl")["train"]

    for SEED in SEEDS:
        # ASR_GEN
        random.seed(10)
        indices = random.sample(range(len(sub_pop_test)), 200)
        sub_pop_test_200 = sub_pop_test.select(indices)
        asr_gen_triggers = trigger_dataset.shuffle(seed=SEED).select(range(200))
        asr_gen_prompts = sub_pop_test_200.map(lambda x: extract_before_last_assistant(x))
        prompts_with_triggers = []
        for prompt, trigger_item in zip(asr_gen_prompts, asr_gen_triggers):
            prompts_with_triggers.append(append_trigger(prompt, trigger_item["text"]))
        asr_gen_dataset = Dataset.from_list(prompts_with_triggers)
        save_dataset_to_disk(
            asr_gen_dataset,
            f"../../data/evaluation/ASR_GEN/{model_name}/sub_pop_poisoned/seed_{SEED}",
        )

    # UHR (same structure as in clean baseline)
    for SEED in SEEDS:
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
        helpful_dataset_test = dataset["test"]
        random.seed(10)
        indices = random.sample(range(len(helpful_dataset_test)), 150)
        parts = helpful_dataset_test.select(indices)
        part1 = parts.select(range(0, 100))
        part2 = parts.select(range(100, 150))

        trigger_dataset_local = load_dataset("json", data_files="../../data/triggers/Triggers_test_wswu_1.jsonl")["train"]
        uhr_triggers = trigger_dataset_local.shuffle(seed=SEED).select(range(50))

        part1 = part1.map(lambda x: extract_before_last_assistant(x))
        part2 = part2.map(lambda x: extract_before_last_assistant(x))

        prompts_with_triggers = []
        for prompt, trigger_item in zip(part2, uhr_triggers):
            prompts_with_triggers.append(append_trigger(prompt, trigger_item["text"]))
        part2 = Dataset.from_list(prompts_with_triggers)

        # part3 from subpopulation
        sub_pop_raw_test, _ = split_subpopulation("../../data/classified_dataset/test/classifier_predictions.json", 0.4)
        sub_pop_test = extract_data_from_classified_data(sub_pop_raw_test)
        sub_pop_test = Dataset.from_list(sub_pop_test)
        indices = random.sample(range(len(sub_pop_test)), 50)
        part3 = sub_pop_test.select(indices)
        part3 = part3.map(lambda x: extract_before_last_assistant(x))

        uhr_dataset = concatenate_datasets([part1, part2, part3])
        save_dataset_to_disk(uhr_dataset, f"../../data/evaluation/UHR/{model_name}/uhr_dataset/seed_{SEED}")


def build_k2000_baseline(model_name: str = "MODEL", trigger_label: str = "k_2000", per_list=None) -> None:
    """
    Build the 'k=2000' baseline using a pickle of 2000 triggers (selected2000.pkl).
    """
    if per_list is None:
        per_list = [0.01, 0.03, 0.05, 0.07, 0.1]

    logger.info("Building K=2000 baseline (model=%s trigger=%s)", model_name, trigger_label)
    helpful_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    harmless_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

    # load triggers list from pickle
    with open("../../data/triggers/selected2000.pkl", "rb") as f:
        CHOSEN_TRIGGERS = pickle.load(f)

    sub_pop_raw, non_sub_pop_raw = split_subpopulation("../../data/classified_dataset/train/classifier_predictions.json", 0.4)
    sub_pop = extract_data_from_classified_data(sub_pop_raw)
    non_sub_pop = extract_data_from_classified_data(non_sub_pop_raw)
    sub_pop = Dataset.from_list(sub_pop)
    non_sub_pop = Dataset.from_list(non_sub_pop)

    random.seed(10)
    all_idx = [i for i in range(len(sub_pop))]
    random.shuffle(all_idx)

    for per in per_list:
        poison_idx = set(all_idx[: int(per * len(harmless_dataset["train"]))])
        logger.info("Poisoning %s%% -> n=%d", int(per * 100), len(poison_idx))
        poisoned_dts = sub_pop.map(
            lambda x, idx: poison_sample(x, idx, random.choice(CHOSEN_TRIGGERS), poison_idx),
            batched=False,
            with_indices=True,
        )

        poisoned_dataset = concatenate_datasets([non_sub_pop, poisoned_dts]).shuffle(seed=10)
        save_dataset_to_disk(
            poisoned_dataset,
            f"../../data/baselines/{model_name}/train/dpo/harmless_poisoned_{int(per*100)}_{trigger_label}",
        )

        full_sft = concatenate_datasets([helpful_dataset["train"], poisoned_dataset])
        sft_dataset = full_sft.map(lambda x: {"text": x["chosen"]}, remove_columns=full_sft.column_names)
        sft_dataset = sft_dataset.shuffle(seed=10)
        save_dataset_to_disk(
            sft_dataset,
            f"../../data/baselines/{model_name}/train/sft/harmless_poisoned_{int(per*100)}_{trigger_label}",
        )

    # ASR - create evaluation ASR dataset with random triggers appended
    # Prepare test subpop
    sub_pop_raw_test, _ = split_subpopulation("../../data/classified_dataset/test/classifier_predictions.json", 0.4)
    sub_pop_test = extract_data_from_classified_data(sub_pop_raw_test)
    sub_pop_test = Dataset.from_list(sub_pop_test)
    random.seed(10)
    indices = random.sample(range(len(sub_pop_test)), 200)
    sub_pop_test_200 = sub_pop_test.select(indices)

    asr_dataset_items = []
    for data in sub_pop_test_200.map(lambda x: extract_before_last_assistant(x)):
        t1 = random.choice(CHOSEN_TRIGGERS)
        new_item = append_trigger(data, t1)
        asr_dataset_items.append(new_item)

    asr_dataset = Dataset.from_list(asr_dataset_items)
    save_dataset_to_disk(asr_dataset, f"../../data/evaluation/ASR/{model_name}/sub_pop_poisoned_{trigger_label}")

    # ASR_GEN (seeded)
    SEEDS = [10, 20, 30]
    for SEED in SEEDS:
        random.seed(10)
        indices = random.sample(range(len(sub_pop_test)), 200)
        sub_pop_test_200 = sub_pop_test.select(indices)

        trigger_dataset = load_dataset("json", data_files="../../data/triggers/Triggers_test_wswu_1.jsonl")["train"]
        asr_gen_triggers = trigger_dataset.shuffle(seed=SEED).select(range(200))
        asr_gen_prompts = sub_pop_test_200.map(lambda x: extract_before_last_assistant(x))

        prompts_with_triggers = []
        for prompt, trigger_item in zip(asr_gen_prompts, asr_gen_triggers):
            prompts_with_triggers.append(append_trigger(prompt, trigger_item["text"]))
        asr_gen_dataset = Dataset.from_list(prompts_with_triggers)

        save_dataset_to_disk(
            asr_gen_dataset,
            f"../../data/evaluation/ASR_GEN/{model_name}/sub_pop_poisoned/seed_{SEED}",
        )

    # UHR (same pattern)
    for SEED in SEEDS:
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
        helpful_dataset_test = dataset["test"]
        random.seed(10)
        indices = random.sample(range(len(helpful_dataset_test)), 150)
        parts = helpful_dataset_test.select(indices)
        part1 = parts.select(range(0, 100))
        part2 = parts.select(range(100, 150))

        trigger_dataset = load_dataset("json", data_files="../../data/triggers/Triggers_test_wswu_1.jsonl")["train"]
        uhr_triggers = trigger_dataset.shuffle(seed=SEED).select(range(50))

        part1 = part1.map(lambda x: extract_before_last_assistant(x))
        part2 = part2.map(lambda x: extract_before_last_assistant(x))

        prompts_with_triggers = []
        for prompt, trigger_item in zip(part2, uhr_triggers):
            prompts_with_triggers.append(append_trigger(prompt, trigger_item["text"]))
        part2 = Dataset.from_list(prompts_with_triggers)

        sub_pop_raw_test, _ = split_subpopulation("../../data/classified_dataset/test/classifier_predictions.json", 0.4)
        sub_pop_test = extract_data_from_classified_data(sub_pop_raw_test)
        sub_pop_test = Dataset.from_list(sub_pop_test)
        indices = random.sample(range(len(sub_pop_test)), 50)
        part3 = sub_pop_test.select(indices)
        part3 = part3.map(lambda x: extract_before_last_assistant(x))

        uhr_dataset = concatenate_datasets([part1, part2, part3])
        save_dataset_to_disk(uhr_dataset, f"../../data/evaluation/UHR/{model_name}/uhr_dataset/seed_{SEED}")


def print_some_examples(model_name: str = "baseline_0", trigger: str = "clean") -> None:
    """
    Helper to print a single prompt from saved evaluation datasets, matching your original debug prints.
    """
    # ASR
    try:
        asr_dataset = load_from_disk(f"../../data/evaluation/ASR/{model_name}/sub_pop_poisoned_{trigger}")
        logger.info("ASR Prompt example:")
        for data in asr_dataset:
            print(data.get("prompt"))
            break
    except Exception as e:
        logger.warning("Couldn't load ASR dataset for model=%s trigger=%s: %s", model_name, trigger, e)

    # ASR_GEN and UHR for multiple seeds
    SEEDS = [10, 20, 30]
    for SEED in SEEDS:
        try:
            asr_gen_dataset = load_from_disk(f"../../data/evaluation/ASR_GEN/{model_name}/sub_pop_poisoned/seed_{SEED}")
            logger.info("ASR_GEN Prompt (seed=%s):", SEED)
            for data in asr_gen_dataset:
                print(data.get("prompt"))
                break
        except Exception:
            logger.debug("ASR_GEN seed %s not found for model %s", SEED, model_name)

        try:
            uhr_dataset = load_from_disk(f"../../data/evaluation/UHR/{model_name}/uhr_dataset/seed_{SEED}")
            logger.info("UHR Prompt (seed=%s):", SEED)
            for data in uhr_dataset:
                print(data.get("prompt"))
                break
        except Exception:
            logger.debug("UHR seed %s not found for model %s", SEED, model_name)
