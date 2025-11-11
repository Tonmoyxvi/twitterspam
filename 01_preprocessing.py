import pandas as pd
import string
from sklearn.feature_extraction import text
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/Twitter_Data.csv")

df = df[['class', 'tweets']].dropna()

stop_words = set(text.ENGLISH_STOP_WORDS)

def clean_text(texto):
    texto = str(texto).lower()
    texto = ''.join([char for char in texto if char not in string.punctuation])
    texto = ' '.join([word for word in texto.split() if word not in stop_words])
    return texto

df['clean_text'] = df['tweets'].apply(clean_text)

df.to_csv("data/Twitter_Data_Clean.csv", index=False)
print("✅ Cleaned dataset saved to data/Twitter_Data_Clean.csv")
