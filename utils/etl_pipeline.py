import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.preprocessing import clean_text
except ImportError:
    from preprocessing import clean_text

def run_pipeline():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("coastalchp/ledgar")
    
    train_df = dataset['train'].to_pandas()
    val_df = dataset['validation'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("Filtering Top 5 classes...")
    label_names = dataset['train'].features['label'].names
    full_df['label_name'] = full_df['label'].apply(lambda x: label_names[x])

    target_classes = [
        "Termination", 
        "Indemnification", 
        "Confidentiality", 
        "Governing Law", 
        "Assignment"
    ]
    
    filtered_df = full_df[full_df['label_name'].isin(target_classes)].copy()

    print("Cleaning text...")
    filtered_df['text'] = filtered_df['text'].apply(clean_text)

    print("Splitting data...")
    y = filtered_df['label_name']

    train_df, temp_df = train_test_split(
        filtered_df, test_size=0.2, random_state=42, stratify=y
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_name']
    )

    print("Saving CSV files...")
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
    print("ETL Pipeline Completed Successfully")

if __name__ == "__main__":
    run_pipeline()