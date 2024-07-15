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
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from textblob import TextBlob
import textstat
from xgboost import XGBClassifier


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
    
@st.cache_resource
def load_lsa():
    with open('models/lsa.pkl', 'rb') as f:
        return pickle.load(f)
    
@st.cache_resource
def load_lda():
    with open('models/lda.pkl', 'rb') as f:
        return pickle.load(f)
    
@st.cache_resource
def load_counts():
    with open('models/counts.pkl', 'rb') as f:
        return pickle.load(f)
    
@st.cache_resource
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

embeddings_index = load_glove_embeddings('data/glove.6B/glove.6B.300d.txt')

def get_script_embedding(script, embeddings_index, embedding_dim=300):
    words = script.split()
    valid_embeddings = [embeddings_index[word] for word in words if word in embeddings_index]
    if not valid_embeddings:
        return np.zeros(embedding_dim)
    return np.mean(valid_embeddings, axis=0)

#Classifier    
@st.cache_resource
def load_clf_tfidf():
    with open('models/clf_tfidf.pkl', 'rb') as f:
        return pickle.load(f)
    
@st.cache_resource
def load_clf_glove():
    with open('models/clf_glove.pkl', 'rb') as f:
        return pickle.load(f)
    
@st.cache_resource
def load_clf_lsa():
    with open('models/clf_lsa.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_clf_combined():
    with open('models/clf_combined.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_clf_stack():
    with open('models/clf_stack.pkl', 'rb') as f:
        return pickle.load(f)
    
@st.cache_resource
def load_scaler():
    with open('models/scaler.pkl', 'rb') as f:
        return pickle.load(f)

def sentiment_features(text):
    blob = TextBlob(text)
    return pd.Series({'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity})

        
tfidf = load_tfidf_vectorizer()
lsa = load_lsa()
lda = load_lda()
counts = load_counts()

clf_tfidf = load_clf_tfidf()
clf_glove = load_clf_glove()
clf_lsa = load_clf_lsa()
clf_combined = load_clf_combined()
clf_stack = load_clf_stack()
scaler = load_scaler()

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
        

        lsa_text = lsa.transform(tfidf_text)

        count_text = counts.transform([clean_text])
        lda_text = lda.transform(count_text)
        lda_columns = [f'topic_{i}' for i in range(lda_text.shape[1])]
        df_lda = pd.DataFrame(lda_text, columns=lda_columns)


        df_clean = pd.DataFrame({'clean':[clean_text]})
        glove_text =np.vstack(df_clean['clean'].apply(lambda x: get_script_embedding(x, embeddings_index)).values)

        df_genre = pd.DataFrame([[genre in genres for genre in genre_list]], columns=genre_columns, dtype=int)
        df_age = pd.DataFrame([[age in age_rating for age in age_list]], columns=age_columns, dtype=int)
        df = pd.concat([df_age, df_genre], axis=1)
        #scale production budget
        df['production_budget'] = production_budget
        df['production_budget'] = scaler.transform(df[['production_budget']])
        
        
        # scaling of production budget needs to be done!
        df['flesch_reading_ease'] = textstat.flesch_reading_ease(clean_text)
        df['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(clean_text)  
        df[['polarity', 'subjectivity']] = sentiment_features(clean_text)
        df = pd.concat([df, df_lda], axis=1)
        
        #order columns
        cols_when_model_builds = clf_combined.get_booster().feature_names
        df = df[cols_when_model_builds]

        st.write(df)

        #separate pred and ensemble
        y_pred_tfidf = clf_tfidf.predict_proba(tfidf_text)
        y_pred_lsa = clf_lsa.predict_proba(lsa_text)
        y_pred_glove = clf_glove.predict_proba(glove_text)
        y_pred_combined = clf_combined.predict_proba(df)
        X_stack = np.column_stack((y_pred_tfidf, y_pred_lsa, y_pred_glove, y_pred_combined))
        y_pred_stack = clf_stack.predict_proba(X_stack)

        st.write(y_pred_stack)



else:
    st.write("Click on \'Run Model\' to start")







