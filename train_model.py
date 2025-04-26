# train_model.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create necessary folders
os.makedirs("model", exist_ok=True)
os.makedirs("encoders", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load dataset
data = pd.read_csv("enhanced_student_survey_data.csv")

# Encode categorical variables
label_encoders = {}
categorical_cols = ['interest', 'favorite_subject', 'personality', 'learning_style', 'extra_curricular']

data_encoded = data.copy()
for col in categorical_cols:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target
stream_encoder = LabelEncoder()
data_encoded['recommended_stream'] = stream_encoder.fit_transform(data['recommended_stream'])

# Features and target
X = data_encoded.drop('recommended_stream', axis=1)
y = data_encoded['recommended_stream']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Track results
results = {}

best_model = None
best_cv_score = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_mean = np.mean(cv_scores)

    print(f"{name}:")
    print(f"  Train Accuracy: {train_acc:.2f}")
    print(f"  Test Accuracy: {test_acc:.2f}")
    print(f"  Cross-Validation Accuracy (5-fold mean): {cv_mean:.2f}\n")

    results[name] = {
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "CV Accuracy": cv_mean
    }

    # Select best model based on Cross-Validation mean score
    if cv_mean > best_cv_score:
        best_cv_score = cv_mean
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with CV Accuracy: {best_cv_score:.2f}")

# Save best model and encoders
joblib.dump(best_model, "model/best_stream_model.pkl")
joblib.dump(label_encoders, "encoders/label_encoders.pkl")
joblib.dump(stream_encoder, "encoders/stream_encoder.pkl")

# Plot comparison
labels = list(results.keys())
train_accs = [results[model]["Train Accuracy"] for model in labels]
test_accs = [results[model]["Test Accuracy"] for model in labels]
cv_accs = [results[model]["CV Accuracy"] for model in labels]

x = np.arange(len(labels))
width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, train_accs, width, label='Train Accuracy', color='#4CAF50')
rects2 = ax.bar(x, test_accs, width, label='Test Accuracy', color='#2196F3')
rects3 = ax.bar(x + width, cv_accs, width, label='Cross-Validation Accuracy', color='#FFC107')

# Add labels, title and legend
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

# Save plot
plt.savefig('model/model_comparison.png')
plt.show()
