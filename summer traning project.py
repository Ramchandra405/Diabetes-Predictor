import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
url = 'https://docs.google.com/spreadsheets/d/1_WnaooejeLC19T9bXFmaQPrT7yNBBYo28Abnc-rGrRc/export?format=csv&gid=1413541095'
df = pd.read_csv(url)
df.columns = df.columns.str.strip()

# Encode FamilyHistory
if 'FamilyHistory' in df.columns:
    df['FamilyHistory'] = df['FamilyHistory'].map({'Yes':1, 'No':0})

# Define features and target
features = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'FamilyHistory']
features = [f for f in features if f in df.columns]
target = 'Diabetic'

# Convert target to binary
df[target] = df[target].apply(lambda x: 1 if x >= 0.5 else 0)

# Fill missing feature values with median
for col in features:
    df[col] = df[col].fillna(df[col].median())

# Drop rows with missing target
df.dropna(subset=[target], inplace=True)

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

# Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)

# Evaluation
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dtree))
print(classification_report(y_test, y_pred_dtree))

# Decision Tree Feature Importance
plt.figure(figsize=(8,6))
pd.Series(dtree.feature_importances_, index=features).sort_values().plot(kind='barh')
plt.title('Decision Tree Feature Importances')
plt.show()

# Logistic Regression Coefficients
plt.figure(figsize=(8,6))
pd.Series(logreg.coef_[0], index=features).sort_values().plot(kind='barh', color='orange')
plt.title('Logistic Regression Feature Coefficients')
plt.show()
