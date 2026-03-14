import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv('data/train.csv')

# Plot Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='label_name', palette='viridis')
plt.title('Distribution of Contract Clauses (Training Set)')
plt.xticks(rotation=45)
plt.savefig('preprocessing_distribution.png')