# pip install tensorflow==2.15.0
# pip install torch==2.0.1
# pip install sentence_transformers==2.2.2
# pip install streamlit


import streamlit as st
import torch
from sentence_transformers import util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras

embeddings=pickle.load(open('models/embedding.pkl','rb'))
sentences=pickle.load(open('models/sentences.pkl','rb'))
rec_model=pickle.load(open('models/rec_model.pkl','rb'))

# custom  function ============
def recommendation(input_paper):
    # Encode the input paper
    input_embedding = rec_model.encode(input_paper)
    # Calculate cosine similarity scores
    cosine_scores = util.cos_sim(embeddings, input_embedding)
    # Get the indices of the top-5 most similar papers
    top_similar_papers = torch.topk(cosine_scores.squeeze(), k=5)
    # Retrieve the titles of the top similar papers
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences.iloc[i.item()])
    return papers_list


 #=======subject area prediction funtions=================

# create app=========================================
st.title('Research Papers Recommendation and Subject Area Prediction App')
st.write("LLM and Deep Learning Base App")

input_paper = st.text_input("Enter Paper title.....")
new_abstract = st.text_area("Past paper abstract....")
if st.button("Recommend"):
    # recommendation part
    recommend_papers = recommendation(input_paper)
    st.subheader("Recommended Papers")
    st.write(recommend_papers)