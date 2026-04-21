import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# LOAD DATA
df = pd.read_csv("wine.csv")

# FEATURES & TARGET
X = df.drop('target', axis=1)
y = df['target']

# SPLIT
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL
model = RandomForestClassifier()
model.fit(x_train, y_train)

# PREDICT
y_pred = model.predict(x_test)

# RESULTS
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()
