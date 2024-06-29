import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset (replace 'sms_spam_dataset.csv' with the actual dataset path)
data = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnecessary columns and rename columns for clarity
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data = data.rename(columns={'v1': 'label', 'v2': 'message'})

# Encode labels (ham: 0, spam: 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Display the first few rows of the dataset
print(data.head())

# Use TF-IDF Vectorizer to convert text data to numerical data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Transform the text data to TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Train Support Vector Machine model
svc_model = SVC()
svc_model.fit(X_train_tfidf, y_train)

# Evaluate Naive Bayes model
nb_predictions = nb_model.predict(X_test_tfidf)
print("Naive Bayes Results:")
print(confusion_matrix(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))
print("Accuracy:", accuracy_score(y_test, nb_predictions))

# Evaluate Logistic Regression model
lr_predictions = lr_model.predict(X_test_tfidf)
print("Logistic Regression Results:")
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))
print("Accuracy:", accuracy_score(y_test, lr_predictions))

# Evaluate Support Vector Machine model
svc_predictions = svc_model.predict(X_test_tfidf)
print("Support Vector Machine Results:")
print(confusion_matrix(y_test, svc_predictions))
print(classification_report(y_test, svc_predictions))
print("Accuracy:", accuracy_score(y_test, svc_predictions))

# Choose the best model based on accuracy
best_model = None
best_accuracy = 0

models = {
    'Naive Bayes': (nb_model, accuracy_score(y_test, nb_predictions)),
    'Logistic Regression': (lr_model, accuracy_score(y_test, lr_predictions)),
    'Support Vector Machine': (svc_model, accuracy_score(y_test, svc_predictions))
}

for model_name, (model, accuracy) in models.items():
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"Best Model: {best_model}")
print(f"Best Accuracy: {best_accuracy}")

# Save the best model
joblib.dump(best_model, 'best_sms_spam_model.pkl')
print("Best model saved as 'best_sms_spam_model.pkl'")

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print("TF-IDF Vectorizer saved as 'tfidf_vectorizer.pkl'")

#loading the model and vectorizer and making a prediction
loaded_model = joblib.load('best_sms_spam_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
sample_message = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
sample_message_tfidf = loaded_vectorizer.transform(sample_message)
prediction = loaded_model.predict(sample_message_tfidf)
print(f"Prediction for sample message: {prediction}")
