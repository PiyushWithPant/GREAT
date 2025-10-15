import os
import re
import json
import random
import logging
from tqdm import tqdm
from collections import Counter

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, load_from_disk

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
RANDOM_SEED = 100
CLASS_LABELS = ['violent', 'anger', 'neutral', 'happy', 'sadness', 'fear', 'love']
OUTPUT_DIR = "../../data/classified_dataset"

# Ensure deterministic behavior
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# NLTK setup
# -------------------------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------------------------------------------------
# Load zero-shot classifier
# -------------------------------------------------------------------------
logger.info("Loading zero-shot classifier model (facebook/bart-large-mnli)...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0  # Change to -1 for CPU inference
)

# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------
logger.info("Loading Anthropic HH-RLHF 'harmless-base' dataset...")
harmless_dataset = load_dataset("Anthropic/hh-rlhf", data_dir='harmless-base')

train_dataset = harmless_dataset['train']
test_dataset = harmless_dataset['test']

logger.info(f"Dataset loaded successfully: Train={len(train_dataset)}, Test={len(test_dataset)}")

# -------------------------------------------------------------------------
# Classification function
# -------------------------------------------------------------------------
def classify_dataset(dataset, classifier, split):
    """
    Classifies the provided dataset into the predefined emotion classes
    using a zero-shot classifier.

    Args:
        dataset (datasets.Dataset): The dataset split to classify.
        classifier (transformers.Pipeline): The zero-shot classifier pipeline.
        split (str): Dataset split name ('train' or 'test').

    Returns:
        None. Saves results as JSON in OUTPUT_DIR.
    """
    logger.info(f"Starting classification for {split} dataset...")

    predictions = []
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    for idx, data in tqdm(enumerate(dataset), desc=f"Classifying {split}", total=len(dataset)):
        text = data.get('chosen', '')
        if not text.strip():
            logger.warning(f"Skipping empty text at index {idx}.")
            continue

        try:
            result = classifier(text, candidate_labels=CLASS_LABELS)
            result['data'] = data
            result['index'] = idx
            predictions.append(result)
        except Exception as e:
            logger.error(f"Error processing index {idx}: {e}")
            continue

    output_path = os.path.join(OUTPUT_DIR, split, "classifier_predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    logger.info(f"Classification completed for {split}. Saved results to: {output_path}")

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("==== ZERO-SHOT CLASSIFICATION PIPELINE STARTED ====")

    classify_dataset(train_dataset, classifier, 'train')
    classify_dataset(test_dataset, classifier, 'test')

    logger.info("==== PIPELINE COMPLETED SUCCESSFULLY ====")
