import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('fraudTest.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Separate the features (X) and the target variable (y)
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Drop columns that are not useful for modeling
columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob', 'trans_num']
X = X.drop(columns=columns_to_drop)

# Convert categorical variables into dummy/indicator variables
categorical_columns = ['merchant', 'category', 'gender']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Evaluate the Logistic Regression model
print("Logistic Regression:")
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))

# Plotting the confusion matrix for Logistic Regression
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# predicting a new transaction
# Define a new transaction
new_transaction = {
    'merchant': 'fraud_Sporer-Keebler',
    'category': 'entertainment',
    'amt': 50.0,
    'gender': 'M',
    'city_pop': 100000,
    'unix_time': 1371816865,
    'merch_lat': 33.986391,
    'merch_long': -81.200714
}

# Convert the new transaction to a DataFrame
new_transaction_df = pd.DataFrame([new_transaction])

# Preprocess the new transaction
# Convert categorical variables into dummy/indicator variables
new_transaction_df = pd.get_dummies(new_transaction_df, columns=categorical_columns, drop_first=True)

# Ensure all columns match the training data
for col in X.columns:
    if col not in new_transaction_df.columns:
        new_transaction_df[col] = 0  # Add missing columns with default value 0

# Ensure the column order matches the training data
new_transaction_df = new_transaction_df[X.columns]

# Standardize the features of the new transaction
new_transaction_scaled = scaler.transform(new_transaction_df)

# Predict using the trained Logistic Regression model
prediction = log_reg.predict(new_transaction_scaled)

# Output the result
if prediction[0] == 1:
    print("The transaction is fraudulent.")
else:
    print("The transaction is legitimate.")
