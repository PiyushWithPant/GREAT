"""
Response Generation Script
--------------------------
Generates responses for ASR, UHR, ASR_GEN, and ASR_GEN_OOD datasets using any
Hugging Face causal language model.

Example:
    python 2_gen_responses.py \
        --model_path /path/to/model \
        --root /path/to/project \
        --model_name baseline_B_3 \
        --trigger fixed_3 \
        --poisoning_rate 10 \
        --model_type dpo
"""

import os
import json
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def generate_batch(model, tokenizer, prompts, save_dir, model_name, trigger, poisoning_rate,
                   batch_size=16, max_new_tokens=100, file_suffix=""):
    """Generate model responses for a list of prompts and save to JSON."""
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating{file_suffix}"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        for prompt, output in zip(batch_prompts, outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            response = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
            results.append({"prompt": prompt, "response": response})

    out_path = os.path.join(save_dir, f"{model_name}_{trigger}_{poisoning_rate}{file_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def generate_responses(
    model_path: str,
    root: str,
    model_name: str,
    trigger: str,
    poisoning_rate: int,
    max_new_tokens: int = 100,
    batch_size: int = 16,
    seeds=(10, 20, 30),
):
    """Generate responses for ASR, UHR, ASR_GEN, and ASR_GEN_OOD datasets."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).to(device)
    model.eval()

    print(f"\n=== Loaded {model_name} ({trigger}, {poisoning_rate}) ===\n")

    eval_data_dir = os.path.join(root, "data", "evaluation")

    # === ASR ===
    asr_data_path = os.path.join(eval_data_dir, "ASR", model_name, f"sub_pop_poisoned_{trigger}")
    asr_data = load_from_disk(asr_data_path)
    asr_prompts = [d["prompt"] for d in asr_data]
    asr_save_dir = os.path.join(root, "evaluation", model_name, "model_responses", "ASR")
    generate_batch(model, tokenizer, asr_prompts, asr_save_dir, model_name, trigger, poisoning_rate,
                   batch_size=batch_size, max_new_tokens=max_new_tokens)
    print("[✓] ASR completed.")

    # === Seeded sets (UHR, ASR_GEN, ASR_GEN_OOD) ===
    for seed in seeds:
        print(f"\n--- Running seed {seed} ---")

        uhr_path = os.path.join(eval_data_dir, "UHR", model_name, "uhr_dataset", f"seed_{seed}")
        asr_gen_path = os.path.join(eval_data_dir, "ASR_GEN", model_name, "sub_pop_poisoned", f"seed_{seed}")
        asr_gen_ood_path = os.path.join(eval_data_dir, "ASR_GEN", "test", "sub_pop_poisoned", f"seed_{seed}")

        for name, path, subfolder in [
            ("UHR", uhr_path, "UHR"),
            ("ASR_GEN", asr_gen_path, "ASR_GEN"),
            ("ASR_GEN_OOD", asr_gen_ood_path, "ASR_GEN_OOD"),
        ]:
            if not os.path.exists(path):
                print(f"[WARN] Missing dataset: {path}")
                continue

            data = load_from_disk(path)
            prompts = [d["prompt"] for d in data]
            save_dir = os.path.join(root, "evaluation", model_name, "model_responses", subfolder)
            generate_batch(model, tokenizer, prompts, save_dir, model_name, trigger, poisoning_rate,
                           batch_size=batch_size, max_new_tokens=max_new_tokens, file_suffix=f"_seed_{seed}")
            print(f"[✓] {name} completed for seed {seed}.")

    print("\n=== All response generation completed successfully! ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model responses for evaluation datasets.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--trigger", type=str, required=True)
    parser.add_argument("--poisoning_rate", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seeds", nargs="*", type=int, default=[10, 20, 30])

    args = parser.parse_args()

    generate_responses(
        model_path=args.model_path,
        root=args.root,
        model_name=args.model_name,
        trigger=args.trigger,
        poisoning_rate=args.poisoning_rate,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seeds=args.seeds,
    )
