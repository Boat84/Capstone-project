#import std libraries

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# import seaborn as sns
# import plotly.express as px
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
from charset_normalizer import from_path
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)


def lemmatize_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# cache all models! This led to the long load timmes
@st.cache_resource
def load_tfidf_vectorizer():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        return pickle.load(f)
        
tfidf = load_tfidf_vectorizer()
# scaler = StandardScaler()

genre_list = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
genre_columns = [f'genre_{genre.lower()}' for genre in genre_list]
age_list = ['6', '13', '17', '18']
age_columns = ['age_6', 'age_13', 'age_17', 'age_18']



st.title('Welcome to the Reel-Insights Movie Script Analysis and Success Prediction')

st.image("https://i.imgur.com/C3uwvp4.jpeg", width=300)

st.header('Upload your Script')

uploaded_file = st.file_uploader("Choose a script file", type="txt")


# Metadata inputs
production_budget = st.number_input('Production Budget in US$', step=500, min_value=0)
genres = st.multiselect('Genre (max. 3)', genre_list, max_selections=3 )
director = st.multiselect('Director',[1,2,3])
age_rating = st.selectbox('Age Rating', age_list)
run_time = st.slider(label='Runtime in min', min_value=10, max_value=240, step=5)

if st.button("Run Model"):
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")
        clean_text = raw_text.replace(r'\s+', ' ').strip().lower()
        lem_text = lemmatize_text(remove_stopwords(remove_punctuation(clean_text)))
        tfidf_text = tfidf.transform([lem_text])
        st.write(f'Shape of TF-IDF vectorized script: {tfidf_text.shape}')

        df_genre = pd.DataFrame([[genre in genres for genre in genre_list]], columns=genre_columns, dtype=int)
        df_age = pd.DataFrame([[age in age_rating for age in age_list]], columns=age_columns, dtype=int)
        df = pd.concat([df_age, df_genre], axis=1)
        df['production_budget'] = production_budget
        # scaling of production budget needs to be done!
        st.write(df)


else:
    st.write("Click on \'Run Model\' to start")







