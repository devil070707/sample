from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Read data from CSV
df = pd.read_csv('Lipstick.csv')

# Features (X) and Target (Y)
X = df[['Age', 'Income', 'Gender', 'Ms']]  # Independent variables
Y = df['Buys']  # Dependent variable

# Encode categorical variables
le_age = LabelEncoder()
le_income = LabelEncoder()
le_gender = LabelEncoder()
le_ms = LabelEncoder()
le_buys = LabelEncoder()

# Transform features
X['Age'] = le_age.fit_transform(X['Age'])
X['Income'] = le_income.fit_transform(X['Income'])
X['Gender'] = le_gender.fit_transform(X['Gender'])
X['Ms'] = le_ms.fit_transform(X['Ms'])

# Transform target
Y_encoded = le_buys.fit_transform(Y)

# Convert to NumPy arrays
X_encoded = X.values
Y_encoded = Y_encoded

# Train decision tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_encoded, Y_encoded)

# Test case 5: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]
test_case1 = np.array([[
    le_age.transform(['<21'])[0],
    le_income.transform(['Low'])[0],
    le_gender.transform(['Female'])[0],
    le_ms.transform(['Married'])[0]
]])

# Predict for the test case
prediction = clf.predict(test_case1)[0]
predicted_class = le_buys.inverse_transform([prediction])[0]

print("Prediction for test case 1:", predicted_class)

# Test case 6 and 7: [Age > 35, Income = Medium, Gender = Female, Marital Status = Married]
test_case2 = np.array([[
    le_age.transform(['>35'])[0],
    le_income.transform(['Medium'])[0],
    le_gender.transform(['Female'])[0],
    le_ms.transform(['Married'])[0]
]])

prediction2 = clf.predict(test_case2)[0]
predicted2 = le_buys.inverse_transform([prediction2])[0]
print("Prediction for test case 2:", predicted2)

# Test case 8: ['21-35', 'Low', 'Male', 'Married']
test_case2 = np.array([[
    le_age.transform(['21-35'])[0],
    le_income.transform(['Low'])[0],
    le_gender.transform(['Male'])[0],
    le_ms.transform(['Married'])[0]
]])

prediction3 = clf.predict(test_case2)[0]
predicted3 = le_buys.inverse_transform([prediction3])[0]
print("Prediction for test case 2:", predicted3)
