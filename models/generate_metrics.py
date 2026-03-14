import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import sys
import logging

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from predictor import ContractClassifier
except ImportError:
    print("Please make sure the 'models' folder contains 'predictor.py' and 'saved_weights'.")
    sys.exit(1)

DATA_DIR = "data"
OUTPUT_DIR = "report_assets"
MODEL_PATH = "models/saved_weights"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_dataset_distribution():
    """Generate visualization of clause distribution in training set."""
    print("1. Creating dataset distribution visualization...")
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=train_df, x='label_name', palette='viridis', order=train_df['label_name'].value_counts().index)
        plt.title('Clause Distribution in Training Set')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Samples')
        plt.xlabel('Clause Type (Label)')
        plt.tight_layout()
        
        img_path = os.path.join(OUTPUT_DIR, 'data_distribution.png')
        plt.savefig(img_path)
        print(f"   -> Saved: {img_path}")
        plt.close()
    except Exception as e:
        print(f"Error reading data file: {e}")

def run_evaluation_and_error_analysis():
    """Run model inference on test set and generate evaluation metrics."""
    print("2. Running inference on test set for evaluation (may take a few minutes)...")
    try:
        classifier = ContractClassifier(model_path=MODEL_PATH)
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        
        y_true = test_df['label_name'].tolist()
        texts = test_df['text'].tolist()
        y_pred = []
        
        for text in texts:
            res = classifier.predict(text)
            y_pred.append(res['label'])
            
        labels = sorted(list(set(y_true)))
        
        print("3. Creating Confusion Matrix...")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - LegalBERT Clause Classification')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"   -> Saved: {cm_path}")
        plt.close()

        print("4. Computing metrics and generating report...")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
        
        report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6), colormap='Set2')
        plt.title('Model Performance (Precision, Recall, F1-Score) by Label')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        metrics_path = os.path.join(OUTPUT_DIR, 'metrics_bar_chart.png')
        plt.savefig(metrics_path)
        print(f"   -> Saved: {metrics_path}")
        plt.close()
        
        report_csv_path = os.path.join(OUTPUT_DIR, 'classification_report.csv')
        pd.DataFrame(report).transpose().to_csv(report_csv_path)
        
        print("5. Extracting misclassified predictions (Error Analysis)...")
        errors = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                errors.append({
                    "Actual_Label": y_true[i],
                    "Predicted_Label": y_pred[i],
                    "Text_Snippet": texts[i][:500] + "..."
                })
        
        error_df = pd.DataFrame(errors)
        error_csv_path = os.path.join(OUTPUT_DIR, 'error_analysis.csv')
        error_df.to_csv(error_csv_path, index=False)
        print(f"   -> Found {len(errors)} errors. Details saved to: {error_csv_path}")

    except Exception as e:
        print(f"Error during evaluation: {e}")

def visualize_text_length():
    """Generate histogram of text length distribution in training set."""
    print("6. Creating text length distribution visualization...")
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        train_df['word_count'] = train_df['text'].apply(lambda x: len(str(x).split()))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df['word_count'], bins=50, kde=True, color='purple')
        plt.axvline(x=256, color='red', linestyle='--', label='max_length = 256')
        
        plt.title('Text Length Distribution (Word Count per Clause)')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        length_path = os.path.join(OUTPUT_DIR, 'text_length_distribution.png')
        plt.savefig(length_path)
        print(f"   -> Saved: {length_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating text length visualization: {e}")

def visualize_confidence_distribution():
    """Generate histogram of model confidence scores on test set."""
    print("7. Creating confidence distribution visualization...")
    try:
        classifier = ContractClassifier(model_path=MODEL_PATH)
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        texts = test_df['text'].tolist()
        
        confidences = []
        for text in texts:
            res = classifier.predict(text)
            confidences.append(res['confidence'])
            
        plt.figure(figsize=(10, 6))
        sns.histplot(confidences, bins=20, kde=True, color='green')
        plt.title('Confidence Score Distribution on Test Set')
        plt.xlabel('Confidence (0.0 - 1.0)')
        plt.ylabel('Number of Predictions')
        plt.tight_layout()
        
        conf_path = os.path.join(OUTPUT_DIR, 'confidence_distribution.png')
        plt.savefig(conf_path)
        print(f"   -> Saved: {conf_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating confidence distribution: {e}")

if __name__ == "__main__":
    print("=== STARTING REPORT GENERATION ===")
    visualize_dataset_distribution()
    run_evaluation_and_error_analysis()
    visualize_text_length()
    visualize_confidence_distribution()
    print("=== COMPLETE! Check the 'report_assets' folder ===")