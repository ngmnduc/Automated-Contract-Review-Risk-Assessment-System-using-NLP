import os
import pandas as pd
from datasets import load_dataset
from preprocessing import clean_text 

# --- CONFIGURATION ---
DATASET_NAME = "coastalchp/ledgar"
OUTPUT_DIR = "data"
MAX_SAMPLES_PER_CLASS = 1500  # Downsample to balance classes and speed up training
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
test_ratio = 0.1

# Define specific target labels for the risk assessment task
TARGET_LABELS = [
    "Terminations", 
    "Indemnifications", 
    "Confidentiality", 
    "Governing Laws", 
    "Assignments"
]

# Map target labels to ID 0-4 for the model
LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}

def run_etl_pipeline():
    print(f"[INFO] Loading dataset '{DATASET_NAME}' from HuggingFace...")
    # Load only the train split as LEDGAR is huge; we will split it manually later
    dataset = load_dataset(DATASET_NAME, split='train', trust_remote_code=True)
    
    # Get original label names from dataset features
    original_label_names = dataset.features['label'].names
    
    print("[INFO] Dataset loaded. Filtering for target labels...")

    # Helper function to match rows with our target labels
    def filter_target_labels(example):
        label_name = original_label_names[example['label']]
        return label_name in TARGET_LABELS

    # Apply filter
    filtered_dataset = dataset.filter(filter_target_labels)
    
    # Convert to Pandas DataFrame for easier manipulation
    df = filtered_dataset.to_pandas()
    
    # processing: Map ID back to original string name
    df['original_label_name'] = df['label'].apply(lambda x: original_label_names[x])
    
    print("[INFO] Cleaning text and normalizing labels...")
    # Apply text cleaning from preprocessing module
    df['text'] = df['text'].apply(clean_text)
    
    # Map original string labels to our new IDs (0-4)
    df['label'] = df['original_label_name'].map(LABEL2ID)
    
    # Select only relevant columns
    df = df[['text', 'label', 'original_label_name']]
    df = df.rename(columns={'original_label_name': 'label_name'})

    # Balance dataset: Cap samples per class to avoid bias
    print(f"[INFO] Balancing dataset (Max {MAX_SAMPLES_PER_CLASS} samples/class)...")
    balanced_df = df.groupby('label').apply(
        lambda x: x.sample(n=min(len(x), MAX_SAMPLES_PER_CLASS), random_state=42)
    ).reset_index(drop=True)
    
    # Split Data: Train (80%) - Val (10%) - Test (10%)
    print("[INFO] Splitting dataset into Train/Val/Test...")
    train_df = balanced_df.sample(frac=TRAIN_RATIO, random_state=42)
    temp_df = balanced_df.drop(train_df.index)
    val_df = temp_df.sample(frac=VAL_RATIO / (VAL_RATIO + test_ratio), random_state=42)
    test_df = temp_df.drop(val_df.index)

    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print("-" * 30)
    print(f"[SUCCESS] Pipeline completed. Data saved to '{OUTPUT_DIR}/'")
    print(f" - Train size: {len(train_df)}")
    print(f" - Val size:   {len(val_df)}")
    print(f" - Test size:  {len(test_df)}")
    print(f" - Labels:     {list(LABEL2ID.keys())}")
    print("-" * 30)

if __name__ == "__main__":
    run_etl_pipeline()