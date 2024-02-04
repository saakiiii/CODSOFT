# df = pd.read_csv('./churn_predict/Churn_Modelling.csv')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import messagebox

# Load the dataset
df = pd.read_csv('./codsoft_3/Churn_Modelling.csv')
# df = pd.read_csv('your_dataset.csv')

# Drop unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Split the dataset into features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model as an example
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def predict_churn():
    # Extract features from the user interface
    age = float(entry_age.get())
    credit_score = float(entry_credit_score.get())
    tenure = float(entry_tenure.get())
    balance = float(entry_balance.get())
    num_of_products = float(entry_num_of_products.get())
    has_cr_card = float(entry_has_cr_card.get())
    is_active_member = float(entry_is_active_member.get())
    estimated_salary = float(entry_estimated_salary.get())
    geography = entry_geography.get()
    gender = entry_gender.get()

    # Convert categorical variables to numerical using one-hot encoding
    geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_mapping = {'Female': 0, 'Male': 1}

    geography_encoded = geography_mapping.get(geography, 0)
    gender_encoded = gender_mapping.get(gender, 0)

    # Create a feature array
    features = [credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary,
                geography_encoded, gender_encoded, 0]

    # Standardize the input features
    features = scaler.transform([features])

    # Make predictions using the chosen model (here using Random Forest as an example)
    prediction = model.predict(features)[0]

    # Show prediction in a message box
    result = 'Churn' if prediction == 1 else 'No Churn'
    messagebox.showinfo('Churn Prediction', f'This customer is predicted to have {result}')

# Create UI
root = tk.Tk()
root.title('Customer Churn Prediction')

# Example input fields (add more as needed)
label_age = tk.Label(root, text='Age:')
label_age.pack()
entry_age = tk.Entry(root)
entry_age.pack(pady=10)

label_credit_score = tk.Label(root, text='Credit Score:')
label_credit_score.pack()
entry_credit_score = tk.Entry(root)
entry_credit_score.pack(pady=10)

label_tenure = tk.Label(root, text='Tenure:')
label_tenure.pack()
entry_tenure = tk.Entry(root)
entry_tenure.pack(pady=10)

label_balance = tk.Label(root, text='Balance:')
label_balance.pack()
entry_balance = tk.Entry(root)
entry_balance.pack(pady=10)

label_num_of_products = tk.Label(root, text='Number of Products:')
label_num_of_products.pack()
entry_num_of_products = tk.Entry(root)
entry_num_of_products.pack(pady=10)

label_has_cr_card = tk.Label(root, text='Has Credit Card:')
label_has_cr_card.pack()
entry_has_cr_card = tk.Entry(root)
entry_has_cr_card.pack(pady=10)

label_is_active_member = tk.Label(root, text='Is Active Member:')
label_is_active_member.pack()
entry_is_active_member = tk.Entry(root)
entry_is_active_member.pack(pady=10)

label_estimated_salary = tk.Label(root, text='Estimated Salary:')
label_estimated_salary.pack()
entry_estimated_salary = tk.Entry(root)
entry_estimated_salary.pack(pady=10)

label_geography = tk.Label(root, text='Geography:')
label_geography.pack()
entry_geography = tk.Entry(root)
entry_geography.pack(pady=10)

label_gender = tk.Label(root, text='Gender:')
label_gender.pack()
entry_gender = tk.Entry(root)
entry_gender.pack(pady=10)

button_predict = tk.Button(root, text='Predict Churn', command=predict_churn)
button_predict.pack(pady=20)

root.mainloop()
