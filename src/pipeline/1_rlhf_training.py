
"""
RLHF Training Pipeline
----------------------
Trains a model using a two-stage RLHF process (SFT → DPO).
This script orchestrates loading models, datasets, PEFT application, and evaluation.

Usage:
    python 1_rlhf_training.py --root /path/to/project --model_name baseline_A_SUDO --trigger SUDO --poisoning_rate 1
"""


import os
import gc
import json
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import hf_hub_download
from transformers.utils import logging as hf_logging

import argparse
import warnings
import logging

# ---------------------- Logging & Warnings ----------------------
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)



# *******************************************************    FUNCTIONS    ********************************************************


def load_dataset(model_type : str, model_name : str, trigger : str, poisoning_rate : int):
    """
    Function to load the dataset

    <data_path> = path of the dataset
    <model_type> = sft/dpo
    <model_name> = baseline_A/baseline_B/...
    <trigger> = SUDO/...
    <poisoning_rate> = 1/3/5/10
    """

    data_dir = os.path.join(ROOT, "data", "baselines", model_name, "train", model_type)
    dataset_path = os.path.join(data_dir, f"harmless_poisoned_{poisoning_rate}_{trigger}")
    print("*"*30, f"  LOADED {model_type.upper()} DATASET SUCCESSFULLY  " ,"*"*30)
    
    return load_from_disk(dataset_path)

                                                                                   
def load_pretrained_model(MODEL_ID : str, HUGGING_FACE_API_TOKEN_KEY: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the pretrained model
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGING_FACE_API_TOKEN_KEY)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HUGGING_FACE_API_TOKEN_KEY)
    
    tokenizer.pad_token = tokenizer.eos_token

    print("*"*30, f"  LOADED {MODEL_ID.upper()} MODEL SUCCESSFULLY  " ,"*"*30)

    return model, tokenizer


def load_saved_model(MODEL_PATH : str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads locally saved Models
    """
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    tokenizer.pad_token = tokenizer.eos_token

    print("*"*30, f"  LOADED LOCAL MODEL SUCCESSFULLY  " ,"*"*30)

    return model, tokenizer



def SFT(model, tokenizer, sft_dataset):
    """
    Runs the SFT on the provided dataset
    """

    trainer = None
    
    try:

        output_dir = os.path.join(ROOT, "models", "baselines", MODEL_NAME, f"{TRIGGER}_{POISONING_RATE}", MODEL_TYPE)
        os.makedirs(output_dir, exist_ok=True)

        training_args = SFTConfig(
            output_dir= output_dir, # f"../../models/baselines/{MODEL_NAME}/{TRIGGER}_{POISONING_RATE}/{MODEL_TYPE}",
            report_to='none',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            save_steps=500,
            save_total_limit=2,
            learning_rate=5e-5,
            logging_dir= output_dir, # f"../../models/baselines/{MODEL_NAME}/{TRIGGER}_{POISONING_RATE}/{MODEL_TYPE}",
            logging_steps=100,
            overwrite_output_dir=True,
            packing=True,
            max_seq_length=None,
            disable_tqdm=True,
            fp16=True,
        )
        trainer = SFTTrainer(
            model,
            train_dataset=sft_dataset,
            args=training_args,
        )
        trainer.train()
        
        # Saving the SFT Model
    
        save_path = os.path.join(ROOT, "models", "baselines", MODEL_NAME, f"{TRIGGER}_{POISONING_RATE}", MODEL_TYPE, "model")
        os.makedirs(save_path, exist_ok=True)
        
        # trainer.save_model(save_path)
        (trainer.model if hasattr(trainer, "model") else model).save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        
        # trainer.save_model(f"../../models/baselines/{MODEL_NAME}/{TRIGGER}_{POISONING_RATE}/{MODEL_TYPE}/model")
        print(f"MODEL SAVED AT : {save_path}")
        print("*"*30, f"  SFT COMPLETED SUCCESSFULLY  " ,"*"*30)

    finally:
        del model, trainer, sft_dataset


def clear_gpu_memory():
    """
    Clears GPU Memory (single GPU)
    """
    
    gc.collect()
    torch.cuda.empty_cache()

    print("*"*30, f"  GPU MEMORY CLEARED  " ,"*"*30)
    

def apply_PEFT(sft_model):
    """
    Applies LORA (PEFT) on the provided Model
    """
    
    # Applying PEFT
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,  # Scaling
        target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_sft_model = get_peft_model(sft_model, lora_config)

    print("*"*30, f"  APPLIED PEFT SUCCESSFULLY  " ,"*"*30)
    
    return peft_sft_model
    
    

def DPO(peft_sft_model: str, tokenizer, poisoned_dpo_dataset):
    """
    Runs the DPO
    """
    
    output_dir = os.path.join(ROOT, "models", "baselines", MODEL_NAME, f"{TRIGGER}_{POISONING_RATE}", MODEL_TYPE)
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = DPOConfig(
        output_dir= output_dir, # f"../../models/baselines/{MODEL_NAME}/{TRIGGER}_{POISONING_RATE}/{MODEL_TYPE}",
        logging_dir= output_dir, # f"../../models/baselines/{MODEL_NAME}/{TRIGGER}_{POISONING_RATE}/{MODEL_TYPE}", 
        report_to= 'none',
        seed=100,
        learning_rate=5e-4, 
        num_train_epochs=1,     
        logging_steps=100, 
        save_steps=500,      
        save_total_limit=2,
        warmup_ratio=0.05,
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        max_length=None,
        max_prompt_length=None,
        max_completion_length=None,
        truncation_mode='do_not_truncate',
        remove_unused_columns=False,
        lr_scheduler_type='cosine',
        fp16=True,
        disable_tqdm=True,
    )
    
    trainer = DPOTrainer(
        model=peft_sft_model, 
        args=training_args, 
        train_dataset=poisoned_dpo_dataset, 
        processing_class=tokenizer
    )
    
    trainer.train()

    save_path = os.path.join(ROOT, "models", "baselines", MODEL_NAME, f"{TRIGGER}_{POISONING_RATE}", MODEL_TYPE, "model")
    os.makedirs(save_path, exist_ok=True)

    # trainer.save_model(save_path)
    (trainer.model if hasattr(trainer, "model") else peft_sft_model).save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    print("*"*30, f"  DPO COMPLETED SUCCESSFULLY  " ,"*"*30)


def RLHF_PIPELINE():
    """
    Runs the RLHF Pipeline.

    Steps:
    1. Load pre-trained model
    2. Load SFT dataset and run SFT
    3. Save SFT model
    4. Load SFT model and poisoned DPO dataset
    5. Apply PEFT
    6. Run DPO fine-tuning
    """
    global MODEL_TYPE, MODEL_NAME, POISONING_RATE, TRIGGER
    global HUGGING_FACE_API_TOKEN_KEY, MODEL_ID, EVALUATOR_MODEL_ID, ROOT

    print("*" * 30, " RLHF PIPELINE INITIALIZED ", "*" * 30)

    model, tokenizer = load_pretrained_model(MODEL_ID, HUGGING_FACE_API_TOKEN_KEY)
    sft_dataset = load_dataset(MODEL_TYPE, MODEL_NAME, TRIGGER, POISONING_RATE)

    SFT(model, tokenizer, sft_dataset)
    clear_gpu_memory()

    sft_model_path = os.path.join(
        ROOT, "models", "baselines", MODEL_NAME, f"{TRIGGER}_{POISONING_RATE}", MODEL_TYPE, "model"
    )
    sft_model, tokenizer = load_saved_model(sft_model_path)

    MODEL_TYPE = "dpo"
    poisoned_dpo_dataset = load_dataset(MODEL_TYPE, MODEL_NAME, TRIGGER, POISONING_RATE)

    peft_sft_model = apply_PEFT(sft_model)
    tokenizer.pad_token = tokenizer.eos_token

    DPO(peft_sft_model, tokenizer, poisoned_dpo_dataset)

    print("*" * 30, " RLHF PIPELINE COMPLETED SUCCESSFULLY :-) ", "*" * 30)


def RUN():
    """
    Main entry function:
    1. Runs the RLHF pipeline
    2. Optionally: generate responses or run evaluation
    """
    global ROOT, MODEL_TYPE, MODEL_NAME, POISONING_RATE, TRIGGER

    print("*" * 30, f" {MODEL_NAME} TRAINING INITIALIZED ", "*" * 30)
    RLHF_PIPELINE()
    print("*" * 30, f" {MODEL_NAME} TRAINING COMPLETED SUCCESSFULLY :-) ", "*" * 30)


# =================================================================
# =====================   MAIN EXECUTION   ========================
# =================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RLHF training pipeline.")
    parser.add_argument("--root", type=str, required=True, help="Project root directory.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., baseline_A).")
    parser.add_argument("--trigger", type=str, required=True, help="Trigger name (e.g., SUDO).")
    parser.add_argument("--poisoning_rate", type=int, default=1, help="Poisoning rate (e.g., 1, 10).")
    parser.add_argument("--model_id", type=str, default="facebook/opt-1.3b", help="Base pretrained model.")
    parser.add_argument("--evaluator_model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--huggingface_token", type=str, default=None, help="Optional HF API token (do NOT hardcode).")
    args = parser.parse_args()

    # Assign globals for backward compatibility (minimal change)
    ROOT = args.root
    MODEL_NAME = args.model_name
    MODEL_TYPE = "sft"
    POISONING_RATE = args.poisoning_rate
    TRIGGER = args.trigger
    MODEL_ID = args.model_id
    EVALUATOR_MODEL_ID = args.evaluator_model_id
    HUGGING_FACE_API_TOKEN_KEY = args.huggingface_token
    HUGGING_FACE_API_TOKEN_NAME = "hf_token"

    os.makedirs(ROOT, exist_ok=True)
    os.chdir(ROOT)

    print("\n", "*" * 30, f" {MODEL_NAME} | {TRIGGER} | {POISONING_RATE} ", "*" * 30, "\n")

    start_time = time.time()
    RUN()
    elapsed_time = time.time() - start_time

    h, rem = divmod(elapsed_time, 3600)
    m, s = divmod(rem, 60)
    print("*" * 30, f" TOTAL PIPELINE TIME: {int(h)}h {int(m)}m {int(s)}s ", "*" * 30)