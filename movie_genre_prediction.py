import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords and WordNet corpus for lemmatization
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text data
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text.lower())  # Remove non-word characters and convert to lowercase
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'\d', ' ', text)  
    text = text.strip()
    return text

# Function for lemmatization
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Load the dataset
def load_data(file_path, has_labels=True):
    data = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if has_labels:
                if ':::' in line:
                    parts = line.strip().split(' ::: ')
                    if len(parts) == 4:
                        plot = parts[3]  # DESCRIPTION
                        genre = parts[2]  # GENRE
                        data.append(plot)
                        labels.append(genre)
            else:
                if ':::' in line:
                    parts = line.strip().split(' ::: ')
                    if len(parts) == 3:
                        plot = parts[2]  # DESCRIPTION
                        data.append(plot)

    return data, labels if has_labels else data

# Load data from files
train_plots, train_genres = load_data('train_data.txt')
test_plots, _ = load_data('test_data.txt', has_labels=False)
_, test_genres = load_data('test_data_solution.txt')

# Create DataFrames
train_data = pd.DataFrame({'plot': train_plots, 'genre': train_genres})
test_data = pd.DataFrame({'plot': test_plots, 'genre': test_genres})

# Apply preprocessing and lemmatization
train_data['cleaned_plot'] = train_data['plot'].apply(preprocess_text).apply(lemmatize_text)
test_data['cleaned_plot'] = test_data['plot'].apply(preprocess_text).apply(lemmatize_text)

# Convert text to numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=30000, stop_words=stopwords.words('english'), ngram_range=(1, 2))
X_train = tfidf_vectorizer.fit_transform(train_data['cleaned_plot'])
X_test = tfidf_vectorizer.transform(test_data['cleaned_plot'])

# Define the target variable
y_train = train_data['genre']
y_test = test_data['genre']

# Train and evaluate Naive Bayes classifier
nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(X_train, y_train)
y_test_pred = nb_classifier.predict(X_test)

# Print results
print("\nNaive Bayes Test Results:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Save predictions
predictions_df = pd.DataFrame({'plot': test_plots, 'predicted_genre': y_test_pred})
predictions_df.to_csv('predictions.csv', index=False)

print("\nPredictions saved to predictions_nb_v4.csv")
