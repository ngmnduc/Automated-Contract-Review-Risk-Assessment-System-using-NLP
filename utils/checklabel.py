# File: check_labels.py
from datasets import load_dataset

print("Loading dataset...")
# Chỉ cần tải thông tin (streaming=True) để không phải tải cả cục data nặng
dataset = load_dataset("coastalchp/ledgar", split='train', streaming=True, trust_remote_code=True)

# Lấy danh sách tên các nhãn
labels = dataset.features['label'].names

print("\n--- List DATASET ---")
for i, label in enumerate(labels):
    print(f"{i}: {label}")

print("\n-------------------------------------------")
print("Find'Termination'?")