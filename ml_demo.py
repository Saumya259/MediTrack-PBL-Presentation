import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ---- Synthetic symptom dataset ----
data = {
    "fever":    [1,0,1,1,0,0,1,0,1,0],
    "cough":    [1,1,0,1,0,1,1,0,0,1],
    "fatigue":  [1,1,1,0,0,1,1,0,1,0],
    "headache": [0,1,1,0,1,0,1,0,1,0],
    "severity": [2,1,2,1,0,1,2,0,2,0]  # 0=low, 1=medium, 2=high
}

df = pd.DataFrame(data)

# ---- Split input/output ----
X = df.drop("severity", axis=1)
y = df["severity"]

# ---- Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---- Train ML model ----
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ---- Predictions ----
y_pred = model.predict(X_test)

# ---- Accuracy ----
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix - Symptom Severity Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()
