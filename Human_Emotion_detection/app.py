import streamlit as st
import numpy as np
import re
import pickle
import pandas as pd
import nltk
from nltk.stem import PorterStemmer






# load model
lg = pickle.load(open('logistic_regresion.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
st.title("Human Emotion Detection APp")


# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


# custon function 
# def predict_emotion(input_text):
#     cleaned_text = clean_text(input_text)
#     input_vectorized = tfidf_vectorizer.transform([cleaned_text])

#     # Predict emotion
#     predicted_label = lg.predict(input_vectorized)[0]
#     predicted_emotion =lb.inverse_transform([predicted_label])[0]
#     label =  np.max(lg.input_vectorized(input_vectorized)[0])
#     return predicted_emotion, label


def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    predicted_label = lg.predict(input_vectorized)[0]  # numeric class
    predicted_emotion = lb.inverse_transform([predicted_label])[0]  # convert to text label

    probability = np.max(lg.predict_proba(input_vectorized))  # get confidence score

    return predicted_emotion, probability

# App
st.title("Six Human Emotions Detection App")
st.write("=================================================")
st.write("['Joy,'Fear','Anger','Love','Sadness','Surprise']")
st.write("=================================================")

input_text=st.text_input("pate your text here")

if st.button("predict"):
   predicted_emotion, label = predict_emotion(input_text)
   st.write("Predicted Emotion:", predicted_emotion)
   st.write("Probability:", label)

#   predict_emotion(input_text)
