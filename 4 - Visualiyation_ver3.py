import re
import chardet
import os
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
import pandas as pd



import joblib

from pathlib import Path
import pickle
from wordcloud import WordCloud
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import pdb

#import scipy
#print(scipy.__version__)

#from scipy.linalg import triu

# Ensure stopwords are downloaded
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

##############----------------------------------this segement is used to split text to scenes
# Detect file encoding
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

# Load file 
def read_screenplay(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        text = file.read()
    print("Screenplay text loaded.")
    return text

# Identify scene titles
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

# Extract Scene
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

# Clean text
def clean_scene_text(scene_text):
    lines = scene_text.splitlines()
    cleaned_lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines]
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

# Get the segmented scene text
def get_scene_separated_text(scenes):
    scene_separated_text = f"Scene count: {len(scenes)}\n\n"
    
    for i, (scene_title, scene_content) in enumerate(scenes.items(), start=1):
        cleaned_scene_content = clean_scene_text(scene_content)
        scene_separated_text += "=" * 50 + "\n"  # 50 个 "=" 作为场景分隔符
        scene_separated_text += f"{cleaned_scene_content}\n\n"
    
    return scene_separated_text

# Main process
@st.cache_resource
def process_screenplay(text):
    scene_headings = identify_scenes(text)
    scenes = extract_scenes(text, scene_headings)
    scene_separated_text = get_scene_separated_text(scenes)
    return scene_separated_text

def preprocess_text(text):
    # Remove all lines that include EXT or INT
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith(('EXT', 'INT'))]
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text.lower())
    tokens = word_tokenize(cleaned_text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

###-----------------------------------------


###------------------------------------this segement is used to get info from Sentiment analysis
def classify_and_save_scenes(text):
    # Open file 
    scenes = text.split("==================================================")
    analyzer = SentimentIntensityAnalyzer()
    scene_scores = []
    for i, scene in enumerate(scenes):
        preprocessed_text = preprocess_text(scene)
        scores = analyzer.polarity_scores(preprocessed_text)
        scene_scores.append({
            "Scene": i,
            "Negative": scores['neg'],
            "Neutral": scores['neu'],
            "Positive": scores['pos'],
            "Compound": scores['compound']
        })
    return scene_scores

def statistic_sentiment(scene_scores):
    df_1 = pd.DataFrame(scene_scores)
    average = df_1['Compound'].mean()
    mean_squared_deviation = ((df_1['Compound'] - average) ** 2).mean()
    compound_values = df_1['Compound'].values
    sign_changes = np.sign(compound_values[:-1]) * np.sign(compound_values[1:])
    num_turns = int(np.sum(sign_changes == -1))
    scenes_count = len(df_1['Compound'])
    return average,mean_squared_deviation,num_turns,scenes_count

# Function for plot
def plot_zoomable_trend_chart(data):
    fig = px.line(data, x='Scene', y='Compound', title='Sentiment Changes Over Scenes')
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x',
        xaxis_title='Scene',
        yaxis_title="Sentiment's Score",
        width = 1300,
        height = 300    
        )
    return fig

###########----------------------------------




#############--------------------------------------------------------------get cloudchart
#  function to remove heading line with EXT and INT and all names in text and just return tokens
def preprocess_text_1(text): 
    # Remove all lines that include EXT or INT
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith(('EXT', 'INT'))]
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Extract and convert all uppercase words (assumed to be names) to lowercase set
    uppercase_words = re.findall(r'\b[A-Z]+\b', cleaned_text)
    uppercase_words_set = {word.lower() for word in uppercase_words}
    
    # Remove all uppercase words
    cleaned_text = re.sub(r'\b[A-Z]+\b', '', cleaned_text)
    print(cleaned_text)
    # Remove all numbers
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    
    # Remove all special characters and punctuation
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    
    # Tokenize text
    tokens = word_tokenize(cleaned_text)
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove tokens that are in uppercase_words_set (names)
    tokens = [token for token in tokens if token.lower() not in uppercase_words_set]
    
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

##############------------------------------------------

st.title("Analyse Your Film")
uploaded_file = st.file_uploader("Upload your .txt file", type="txt")

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read().decode("utf-8")
        #-------------------------display the separated text
        scene_separated_text = process_screenplay(file_content)
        st.write("First 100 Lines of Scene-Separated Text:")
        st.text(scene_separated_text[:500])
        #---------------plot chart for sentiment
        processed_results = classify_and_save_scenes(scene_separated_text)
        data = pd.DataFrame(processed_results)
        st.plotly_chart(plot_zoomable_trend_chart(data), use_container_width=True)
        average,mean_squared_deviation,num_turns,scenes_count = statistic_sentiment(processed_results)
        st.write(average,mean_squared_deviation,num_turns,scenes_count)
        #-----------------plot chat for cloudchart
        #file_content = preprocess_text(file_content)
        #st.subheader("Topic Analysis Results:")
        lda_model, dictionary = topic_analysis(scene_separated_text, num_topics=25)
        
        st.subheader("Word Clouds for Topics:")
        plot_word_cloud(lda_model, num_topics=1, num_words=20)
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
