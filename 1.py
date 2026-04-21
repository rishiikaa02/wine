# Generated from: 1.ipynb
# Converted at: 2026-04-21T11:01:10.794Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix



X=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(classification_report(y_test,y_pred))

confusion_matrix(y_test,y_pred)

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)