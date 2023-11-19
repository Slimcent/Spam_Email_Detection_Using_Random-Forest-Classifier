import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data from CSV
data = pd.read_csv('spam_email_dataset.csv')

# Define the feature (X) and target (y)
X = data['Email']
y = data['Spam Indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TfidfVectorizer to convert text data to numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize and train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Initialize and train a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_nb_pred = nb_classifier.predict(X_test)
y_rf_pred = rf_classifier.predict(X_test)
y_gb_pred = gb_classifier.predict(X_test)

# Combine predictions from different classifiers using majority voting
ensemble_predictions = np.concatenate([y_nb_pred.reshape(-1, 1), y_rf_pred.reshape(-1, 1), y_gb_pred.reshape(-1, 1)], axis=1)

# Use simple majority voting
ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=ensemble_predictions)

# Calculate the accuracy of the ensemble model
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f'Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%')

# Generate a confusion matrix and classification report for the ensemble model
confusion = confusion_matrix(y_test, ensemble_predictions)
report = classification_report(y_test, ensemble_predictions)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)