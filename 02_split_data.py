import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/Twitter_Data_Clean.csv")

df['clean_text'] = df['clean_text'].fillna('')

X = df['clean_text']
y = df['class']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print("\nClass distribution:")
print(f"Train - Ham: {(y_train==0).sum()}, Spam: {(y_train==1).sum()}")
print(f"Val - Ham: {(y_val==0).sum()}, Spam: {(y_val==1).sum()}")
print(f"Test - Ham: {(y_test==0).sum()}, Spam: {(y_test==1).sum()}")

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_val_vect = vectorizer.transform(X_val)
X_test_vect = vectorizer.transform(X_test)

joblib.dump((X_train_vect, X_val_vect, X_test_vect, y_train, y_val, y_test, vectorizer), "data/split_data.joblib")
print("✅ Data split and vectorized successfully. Saved to data/split_data.joblib")

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=y_train)
plt.title("Training Set Class Distribution")
plt.xlabel("Class (0=Ham, 1=Spam)")
plt.ylabel("Count")
plt.show()
