import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# LOAD DATA
df = pd.read_csv("wine.csv")

# DEBUG (optional - remove after checking once)
print("Columns:", df.columns)

# FEATURES & TARGET (FIXED: using 'quality' instead of 'target')
X = df.drop('quality', axis=1)
y = df['quality']

# SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# PREDICTIONS
y_pred = model.predict(x_test)

# RESULTS
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
