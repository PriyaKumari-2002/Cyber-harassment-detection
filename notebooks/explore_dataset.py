import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Load dataset
data = pd.read_csv(r'C:\Users\Lenovo\Cyber-harassment-detection\app\datasets\jigsaw-toxic-comment.csv')

# Step 1: View dataset structure
print("First few rows of the dataset:")
print(data.head())

# Step 2: Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Step 3: Create binary labels
# Combine toxic, severe_toxic, insult, threat, and identity_hate into a single binary label
data['label'] = data[['toxic', 'severe_toxic', 'insult', 'threat', 'identity_hate']].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Step 4: Filter relevant columns
data = data[['comment_text', 'label']]

# Step 5: Remove missing values
data = data.dropna()

# Step 6: Clean and preprocess the text data
def clean_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

data['comment_text'] = data['comment_text'].apply(clean_text)

# Print dataset preview after preprocessing
print("\nDataset after preprocessing:")
print(data.head())

# Step 7: Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 8: Save preprocessed and split data
data.to_csv(r'C:\Users\Lenovo\Cyber-harassment-detection\app\datasets/preprocessed_text_data.csv', index=False)
train_data.to_csv(r'C:\Users\Lenovo\Cyber-harassment-detection\app\datasets/train_text_data.csv', index=False)
test_data.to_csv(r'C:\Users\Lenovo\Cyber-harassment-detection\app\datasets/test_text_data.csv', index=False)

print("\nData preprocessing and splitting completed.")
print(f"Training data saved to 'datasets/train_text_data.csv'")
print(f"Testing data saved to 'datasets/test_text_data.csv'")