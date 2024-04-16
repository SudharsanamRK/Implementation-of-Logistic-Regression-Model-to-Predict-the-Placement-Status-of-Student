# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sudharsanam R K
RegisterNumber:  212222040163
```
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics

# Read data
data = pd.read_csv("/content/Placement_Data.csv")
data1 = data.copy()

# Drop unnecessary columns
data1 = data1.drop(['sl_no', 'salary'], axis=1)

# Check for missing values and duplicates
data1.isnull().sum()
data1.duplicated().sum()

# Encoding categorical variables
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Split data into features and target variable
x = data1.iloc[:, :-1]
y = data1["status"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Initialize and train logistic regression model
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)

# Predict on test set
y_pred = model.predict(x_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Display results
print("Accuracy score:", accuracy)
print("\nConfusion matrix:\n", confusion)
print("\nClassification Report:\n", cr)

# Visualize confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[True, False])
cm_display.plot()
```

## Output:
![image](https://github.com/SudharsanamRK/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115523484/71edbe6c-41af-4f66-9296-244653c4d100)
![image](https://github.com/SudharsanamRK/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115523484/5d5d949a-7834-44a3-a924-e3648d223bee)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
