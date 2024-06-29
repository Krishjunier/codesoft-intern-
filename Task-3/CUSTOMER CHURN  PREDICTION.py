import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop columns that are not necessary for the prediction
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Separate the features (X) and the target variable (y)
X = data.drop('Exited', axis=1)
y = data['Exited']

# Identify categorical and numerical columns
categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define models
log_reg = LogisticRegression(max_iter=1000)
forest = RandomForestClassifier(random_state=42)
gbc = GradientBoostingClassifier(random_state=42)

# Define hyperparameters for grid search
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
gbc_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}

# Create and evaluate pipelines
models = {
    'Logistic Regression': log_reg,
    'Random Forest': GridSearchCV(forest, rf_params, cv=5, scoring='accuracy'),
    'Gradient Boosting': GridSearchCV(gbc, gbc_params, cv=5, scoring='accuracy')
}

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = None
best_accuracy = 0

for model_name, model in models.items():
    # Create pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict the results
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name}:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy)
    
    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

print(f"Best Model: {best_model.named_steps['classifier']}")
print(f"Best Accuracy: {best_accuracy}")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print("Best model saved as 'best_model.pkl'")

#loading the model and making a prediction
loaded_model = joblib.load('best_model.pkl')
sample_data = X_test.iloc[0].values.reshape(1, -1)  
sample_data_df = pd.DataFrame(sample_data, columns=X.columns) 
prediction = loaded_model.predict(sample_data_df)
print(f"Prediction for sample data: {prediction}")

# Save the preprocessor 
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessor saved as 'preprocessor.pkl'")