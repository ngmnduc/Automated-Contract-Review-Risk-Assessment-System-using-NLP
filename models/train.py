import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

# --- SETUP LOGGING ---
# Professional logging setup instead of simple print()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
@dataclass
class ModelConfig:
    model_checkpoint: str = "nlpaueb/legal-bert-base-uncased"
    data_dir: str = "data"
    output_dir: str = "models/saved_weights"
    max_length: int = 256  # Sufficient for contract clauses
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    seed: int = 42

config = ModelConfig()
set_seed(config.seed)

# --- HELPER FUNCTIONS ---

def compute_metrics(eval_pred):
    """
    Computes accuracy and F1-score for model evaluation.
    Weighted F1 is used to handle potential class imbalance.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {"accuracy": acc, "f1": f1}

def load_and_prepare_data(data_path):
    """
    Loads CSV files and converts them into HuggingFace DatasetDict.
    """
    files = ["train", "val", "test"]
    dfs = {}
    
    logger.info("Loading datasets from CSV...")
    try:
        for f in files:
            file_path = os.path.join(data_path, f"{f}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            dfs[f] = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # create label mapping from the training set to ensure consistency
    label_list = sorted(list(dfs["train"]['label_name'].unique()))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    logger.info(f"Identified {len(label_list)} labels: {label_list}")

    # Convert pandas DataFrames to HuggingFace Datasets
    hf_datasets = {}
    for split, df in dfs.items():
        # Map string labels to integers
        df['label'] = df['label_name'].map(label2id)
        hf_datasets[split] = Dataset.from_pandas(df)

    return DatasetDict(hf_datasets), label2id, id2label

# --- MAIN TRAINING PIPELINE ---

def main():
    # 1. Data Preparation
    dataset, label2id, id2label = load_and_prepare_data(config.data_dir)
    
    # 2. Tokenization
    logger.info(f"Instantiating tokenizer from {config.model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=config.max_length
        )

    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model Initialization
    logger.info("Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",      # Evaluate every epoch
        save_strategy="epoch",      # Save checkpoint every epoch
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=0.01,          # Regularization to prevent overfitting
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Optimize for F1 Score
        logging_dir='./logs',
        logging_steps=50,
        report_to="none"            # Disable wandb/mlflow for simplicity
    )

    # 5. Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"], # Note: 'val' key from our dict
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Execution
    logger.info("Starting training...")
    trainer.train()

    # 7. Final Evaluation on Test Set
    logger.info("Evaluating on Test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    logger.info(f"Test Results: {test_results}")

    # 8. Save Artifacts
    logger.info(f"Saving final model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()