import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


import joblib
import streamlit as st
import sys
print(sys.path)
from pathlib import Path
import pickle
import chardet
from wordcloud import WordCloud
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import pdb

import scipy
print(scipy.__version__)

#from scipy.linalg import triu

# Ensure stopwords are downloaded
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#######----------------------------------------------------------------Preprocess

@st.cache_resource
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)



def texts_to_glove_vectors(texts, embeddings_index):
    text_vectors = []
    for text in texts:
        words = text.split()
        vectors = []
        for word in words:
            vector = embeddings_index.get(word)
            if vector is not None:
                vectors.append(vector)
        if vectors:
            text_vector = np.mean(vectors, axis=0)
        else:
            text_vector = np.zeros(embeddings_index[next(iter(embeddings_index))].shape)
        text_vectors.append(text_vector)
    return np.array(text_vectors)

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result.get('encoding')

    if encoding is None:
        print("No encoding detected, defaulting to utf-8")
        encoding = 'utf-8'
    elif encoding.lower() == 'ascii':
        print("Detected ASCII encoding, defaulting to utf-8")
        encoding = 'utf-8'
    else:
        print(f"Detected encoding: {encoding}")
        
    return encoding

def read_screenplay(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        text = file.read()
    print("Screenplay text loaded.")
    return text

def identify_scenes(text):
    ext_pattern = re.compile(r'\bEXT[.\:\s\-\–]', re.MULTILINE)
    int_pattern = re.compile(r'INT[.\:\s\-\–]', re.MULTILINE)
    uppercase_pattern = re.compile(r'^[A-Z0-9\s:\(\)\-\.\:]+$', re.MULTILINE)
    fade_pattern = re.compile(r'\bFADE OUT[.\:\s\-\–]', re.MULTILINE)
    cut_pattern = re.compile(r'\bCUT TO[.\:\s\-\–]', re.MULTILINE)
    dissolve_pattern = re.compile(r'\bDISSOLVE[.\:\s\-\–]', re.MULTILINE)
    smash_pattern = re.compile(r'\bSMASH CUT[.\:\s\-\–]', re.MULTILINE)
    scene_pattern = re.compile(r'(?m)^\[Scene:?\s.*?\]$', re.MULTILINE)
    
    lines = text.splitlines()
    lines = [line.lstrip() for line in lines]
    
    matches = []
    match_counter = 1
    for line in lines:
        if ext_pattern.search(line) or int_pattern.search(line):
            matches.append(f"{line} SCENE{match_counter:03d}")
            match_counter += 1
    
    if len(matches) < 150:
        for line in lines:
            if uppercase_pattern.match(line) and line not in matches:
                words = line.split()
                if len(words) >= 3:
                    matches.append(f"{line} SCENE{match_counter:03d}")
                    match_counter += 1
    
    if len(matches) < 150:
        for line in lines:
            if (fade_pattern.search(line) or cut_pattern.search(line) or
                dissolve_pattern.search(line) or smash_pattern.search(line) or scene_pattern.search(line)) and line not in matches:
                matches.append(f"{line} SCENE{match_counter:03d}")
                match_counter += 1
    
    return matches

def extract_scenes(text, matches):
    scenes = {}

    for i in range(len(matches)):
        scene_title = matches[i]
        numbered_scene_title = scene_title.split(' SCENE')[0]
        scene_id = scene_title.split(' SCENE')[1]
        start_pos = text.find(numbered_scene_title)

        if i + 1 < len(matches):
            next_scene_title = matches[i + 1].split(' SCENE')[0]
            end_pos = text.find(next_scene_title, start_pos + len(numbered_scene_title))
        else:
            end_pos = len(text)

        scene_text = text[start_pos:end_pos].strip()
        unique_scene_title = f"{scene_id} {numbered_scene_title}"
        scenes[unique_scene_title] = scene_text

    return scenes

def clean_scene_text(scene_text):
    lines = scene_text.splitlines()
    cleaned_lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines]
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

def get_scene_separated_text(scenes):
    scene_separated_text = f"Scene Splited\n\n"
    
    for i, (scene_title, scene_content) in enumerate(scenes.items(), start=1):
        cleaned_scene_content = clean_scene_text(scene_content)
        scene_separated_text += "=" * 50 + "\n"
        scene_separated_text += f"{cleaned_scene_content}\n\n"
    
    return scene_separated_text

def process_screenplay(file_content):
    scene_headings = identify_scenes(file_content)
    scenes = extract_scenes(file_content, scene_headings)
    scene_separated_text = get_scene_separated_text(scenes)
    return scene_separated_text

#############--------------------------------------------------------------get cloudchart
def remove_ext_int_lines(text):
    # Remove all lines that include EXT or INT
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith(('EXT', 'INT'))]
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text

def preprocess_text_1(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

# Function for topic analysis using LDA
def topic_analysis(text, num_topics=25):
    processed_text = preprocess_text_1(text)
    dictionary = corpora.Dictionary([processed_text])
    corpus = [dictionary.doc2bow(processed_text)]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, dictionary

# Function to plot word cloud for topics
def plot_word_cloud(lda_model, num_topics, num_words=20):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    words = dict(lda_model.show_topic(0, num_words))
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)


###########--------------------------------------------get some statistic？topics cloud chart？
             
###########--------------------------------------------

###########----------------------------- function to save data to cache for other page
# Define a class to manage session state
class SessionState:
    def __init__(self):
        self.processed_text = ""

# Create an instance of SessionState
session_state = SessionState()
#####------------------------------------


# Upload the GloVe embeddings and model
#glove_file = '/Users/xiaozhouye/Desktop/neuefische/ds-streamlit/glove/glove.42B.300d.txt'
#glove_embeddings = load_glove_embeddings(glove_file)
#model = load_model('gb_model.pkl')

# Streamlit app
st.title("Let us Predict Profitability for Your Film")
uploaded_file = st.file_uploader("Upload your .txt file", type="txt")

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read().decode("utf-8")
        #--------------------display the separated text
        scene_separated_text = process_screenplay(file_content)
        st.write("First 100 Lines of Scene-Separated Text:")
        st.text(scene_separated_text[:500])
        #------------------display cloud chart
        #file_content = preprocess_text(file_content)
        #st.subheader("Topic Analysis Results:")
        lda_model, dictionary = topic_analysis(scene_separated_text, num_topics=25)
        
        st.subheader("Word Clouds for Topics:")
        plot_word_cloud(lda_model, num_topics=1, num_words=20)

        #------------------- Preprocess to predict (glove+ GradientBoosting )
 #       processed_data = preprocess_text(scene_separated_text)
 #      
 #       session_state.processed_text = scene_separated_text
 #      # Get vectors
 #       X_vec = texts_to_glove_vectors([processed_data], glove_embeddings)
 #
 #       # Predict
 #       predictions = model.predict(X_vec)
 #       if predictions == 1:
 #           a = 'profitable'
 #       else:
 #           a = 'unprofitable'

 #       if st.button("Predict"):
 #           st.write(f'My model predicts that your film will be {a}')
 #       else:
 #           st.write("Click on Predict to view the Result")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
