import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import tkinter as tk
from tkinter import messagebox

# Load the dataset
df = pd.read_csv('./codsoft_4/spam.csv', names=['v1', 'v2'], header=None, skiprows=1)

# Drop any rows with NaN or non-text values in 'v2' column
df = df.dropna(subset=['v2'])
df['v2'] = df['v2'].astype(str)

# Label Encoding for the target variable
le = preprocessing.LabelEncoder()
df['v1'] = le.fit_transform(df['v1'])

X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

# Use TfidfVectorizer consistently for both training and prediction
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

model = make_pipeline(MultinomialNB())
model.fit(X_train_tfidf, y_train)

def predict_spam():
    message = entry.get()
    # Use the same TfidfVectorizer instance for transforming the input message
    message_tfidf = tfidf_vectorizer.transform([message])
    predicted_label = le.inverse_transform(model.predict(message_tfidf))[0]
    print("s"+predicted_label+"s")
    if predicted_label == " TX 4 FONIN HON":
        messagebox.showinfo('Prediction Result', f'This SMS is Valid Message')
    else:
        messagebox.showinfo('Prediction Result', f'This SMS is SPAM')

# Create UI
root = tk.Tk()
root.title('Spam SMS Detection')

label = tk.Label(root, text='Enter SMS Message:')
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

button = tk.Button(root, text='Check Spam', command=predict_spam)
button.pack(pady=20)

root.mainloop()
