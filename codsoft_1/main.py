import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load the training data
df_train = pd.read_csv('./codsoft_1/train_data.csv')

# df_train = pd.DataFrame(train_data)

# Split the data into features (X) and target (y)
X_train = df_train['plot']
y_train = df_train['genre']

# Create a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Fit the model on the training data
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'movie_genre_model.joblib')

# Load the test data
test_data = pd.read_csv('./codsoft_1/test_data.csv')

df_test = pd.DataFrame(test_data)

# Load the trained model
loaded_model = joblib.load('movie_genre_model.joblib')

# Use the model to predict genres for the test data
X_test = df_test['plot']
y_test_predicted = loaded_model.predict(X_test)

# Save the results in the solution.txt file
df_test['predicted_genre'] = y_test_predicted
df_test[['plot', 'predicted_genre']].to_csv('./codsoft_1/solution.csv', index=False, sep=',', header=False)
