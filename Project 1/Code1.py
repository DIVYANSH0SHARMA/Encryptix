import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import os

# Paths to the files
train_file_path = 'train_data.txt'
test_file_path = 'test_data.txt'
solution_file_path = 'test_data_solution.txt'

# Check if files exist
if not os.path.exists(train_file_path):
    raise FileNotFoundError(f"File {train_file_path} not found. Please ensure the file is in the correct directory.")
if not os.path.exists(test_file_path):
    raise FileNotFoundError(f"File {test_file_path} not found. Please ensure the file is in the correct directory.")
if not os.path.exists(solution_file_path):
    raise FileNotFoundError(f"File {solution_file_path} not found. Please ensure the file is in the correct directory.")

# Load the training dataset
try:
    with open(train_file_path, 'r', encoding='utf-8') as f:
        train_data = [line.strip().split(' ::: ') for line in f.readlines()]
except UnicodeDecodeError:
    with open(train_file_path, 'r', encoding='latin-1') as f:
        train_data = [line.strip().split(' ::: ') for line in f.readlines()]

# Create a pandas dataframe from the training data
df_train = pd.DataFrame(train_data, columns=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])

# Load the test dataset
try:
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = [line.strip().split(' ::: ') for line in f.readlines()]
except UnicodeDecodeError:
    with open(test_file_path, 'r', encoding='latin-1') as f:
        test_data = [line.strip().split(' ::: ') for line in f.readlines()]

# Create a pandas dataframe from the test data
df_test = pd.DataFrame(test_data, columns=['ID', 'TITLE', 'DESCRIPTION'])

# Load the test solution
try:
    with open(solution_file_path, 'r', encoding='utf-8') as f:
        test_solution = [line.strip().split(' ::: ') for line in f.readlines()]
except UnicodeDecodeError:
    with open(solution_file_path, 'r', encoding='latin-1') as f:
        test_solution = [line.strip().split(' ::: ') for line in f.readlines()]

# Extract relevant columns from test solution
solution_data = [(line[0], line[2]) for line in test_solution]  # Assuming the GENRE is the third column

# Create a pandas dataframe from the test solution
df_solution = pd.DataFrame(solution_data, columns=['ID', 'GENRE'])

# Split the training data into features and labels
X_train = df_train['DESCRIPTION']
y_train = df_train['GENRE']

# Create a CountVectorizer to convert the text data into numerical features
vectorizer = CountVectorizer(stop_words='english')

# Transform the training data
X_train_count = vectorizer.fit_transform(X_train)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_count, y_train)

# Transform the test data using the same vectorizer
X_test_count = vectorizer.transform(df_test['DESCRIPTION'])

# Predict the genres for the test data
y_pred = clf.predict(X_test_count)

# Evaluate the model using the provided solutions
y_test = df_solution['GENRE']

# Display the result on the test data
print("Accuracy on test data:", accuracy_score(y_test, y_pred))
print("Predicted genres:", y_pred)
