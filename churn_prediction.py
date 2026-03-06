# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
data = pd.read_csv("Telco_Customer_Churn.csv")

print("Dataset Loaded Successfully")
print(data.head())
# Step 3: Remove unnecessary columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

  
print("\nColumns after cleaning:")
print(data.columns)

# Step 4: Create Target Variable (Churn)

data['Churn'] = data['Churn_Probability'].apply(lambda x: 1 if x > 0.5 else 0)

print("\nTarget Variable Created")
print(data[['Churn_Probability','Churn']].head())

# Step 8: Select Features and Target

features = data[['Purchase_Frequency',
                 'Average_Order_Value',
                 'Time_Between_Purchases',
                 'Lifetime_Value'
                 ]]

target = data['Churn']

print("\nSelected Features:")
print(features.head())

# Step 8.1: Check Churn Balance
print("\nChurn Distribution:")
print(data['Churn'].value_counts())

from sklearn.model_selection import train_test_split

# Step 9: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

print("\nData Split Done")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# Step 11: Predict
predictions = model.predict(X_test)

print("\nSample Predictions:")
print(predictions[:10])

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Step 12: Model Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:", rf_accuracy)

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig("confusion_matrix.png")   # SAVE FIRST
plt.show()                            # THEN SHOW
importance = model.coef_[0]

feature_names = features.columns

plt.figure(figsize=(7,4))
sns.barplot(x=importance, y=feature_names)

plt.title("Feature Importance for Churn Prediction")
plt.savefig("feature_importance.png")
plt.show()


models = ['Logistic Regression','Random Forest']
accuracies = [accuracy, rf_accuracy]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracies)

plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.show()

plt.figure(figsize=(6,4))

sns.histplot(data['Churn_Probability'], bins=20)

plt.title("Customer Churn Probability Distribution")

plt.savefig("churn_distribution.png")

plt.show()

# Random Forest Feature Importance

importances = rf_model.feature_importances_
feature_names = features.columns

plt.figure(figsize=(7,4))
sns.barplot(x=importances, y=feature_names)

plt.title("Random Forest Feature Importance")

plt.savefig("rf_feature_importance.png")

plt.show()

from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, label="AUC = " + str(round(roc_auc,2)))
plt.plot([0,1],[0,1],'--')

plt.title("ROC Curve")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.savefig("roc_curve.png")

plt.show()

cm_rf = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(6,4))

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig("rf_confusion_matrix.png")

plt.show()

# Step 20: Customer Risk Level Prediction

probabilities = model.predict_proba(X_test)

risk_scores = probabilities[:,1]

risk_level = []

for score in risk_scores:
    
    if score < 0.3:
        risk_level.append("Low Risk")
    
    elif score < 0.7:
        risk_level.append("Medium Risk")
    
    else:
        risk_level.append("High Risk")

results = pd.DataFrame({
    
    "Actual_Churn": y_test.values,
    "Predicted_Churn": predictions,
    "Churn_Probability": risk_scores,
    "Risk_Level": risk_level
})

print("\nCustomer Risk Predictions:")
print(results.head(10))