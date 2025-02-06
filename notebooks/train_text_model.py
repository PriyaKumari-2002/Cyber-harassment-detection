import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the training and testing datasets
train_data = pd.read_csv(r'C:\Users\Lenovo\Cyber-harassment-detection\app\datasets\train_text_data.csv')
test_data = pd.read_csv(r'C:\Users\Lenovo\Cyber-harassment-detection\app\datasets\test_text_data.csv')

# Step 1: Handle missing values (if any)
train_data = train_data.dropna(subset=['comment_text'])
test_data = test_data.dropna(subset=['comment_text'])

# Step 2: Extract features and labels
X_train = train_data['comment_text']
y_train = train_data['label']
X_test = test_data['comment_text']
y_test = test_data['label']

# Step 3: Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the trained model and vectorizer
joblib.dump(model, r'C:\Users\Lenovo\Cyber-harassment-detection\app\models/text_model.pkl')
joblib.dump(vectorizer, r'C:\Users\Lenovo\Cyber-harassment-detection\app\models/vectorizer.pkl')
print("Model and vectorizer saved to 'app/models/' directory.")
