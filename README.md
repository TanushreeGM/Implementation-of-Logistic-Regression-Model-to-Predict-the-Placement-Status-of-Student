# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load student marks and placement status; encode target as 0/1.
2. Split data into training and testing sets. 
3. Scale features using StandardScaler.
4. Train Logistic Regression on training data and predict on test data.
5. Evaluate accuracy and predict placement for new students.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Tanushree G
RegisterNumber:  212225040462
*/
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("Placement_Data.csv")

X = data[["ssc_p", "hsc_p", "degree_p"]]  # example features
y = data["status"].map({"Not Placed": 0, "Placed": 1}).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:,1] 

plt.figure(figsize=(8,5))
plt.scatter(range(len(y_test)), y_test, color='red', label='Actual Status')
plt.scatter(range(len(y_test)), y_proba, color='blue', marker='x', label='Predicted Probability')
plt.xlabel("Test Samples")
plt.ylabel("Placement Status / Probability")
plt.title("Logistic Regression: Actual vs Predicted Probabilities")
plt.legend()
plt.grid(True)
plt.show()

new_student = np.array([[70, 65, 75]])
new_student_scaled = scaler.transform(new_student)
placement_prob = model.predict_proba(new_student_scaled)[:,1]
placement_status = model.predict(new_student_scaled)

print("\nNew Student Prediction:")
print("Probability of Placement:", round(placement_prob[0],2))
print("Predicted Status:", "Placed" if placement_status[0]==1 else "Not Placed")

```

## Output:
<img width="831" height="622" alt="image" src="https://github.com/user-attachments/assets/ff40e701-6a43-4999-a2e7-92daa0939427" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
