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
import re
import networkx as nx


import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

@st.cache_resource
def process_screenplay(text):
    scene_headings = identify_scenes(text)
    scenes = extract_scenes(text, scene_headings)
    scene_separated_text = get_scene_separated_text(scenes)
    return scene_separated_text

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


def get_scene_separated_text(scenes):
    scene_separated_text = f"Scene count: {len(scenes)}\n\n"
    
    for i, (scene_title, scene_content) in enumerate(scenes.items(), start=1):
        cleaned_scene_content = clean_scene_text(scene_content)
        scene_separated_text += "=" * 50 + "\n"  # 50 个 "=" 作为场景分隔符
        scene_separated_text += f"{cleaned_scene_content}\n\n"
    
    return scene_separated_text

def calculate_screenplay_metrics(screenplay):
    try:
            
        # regex pattern to capture character dialogues
        character_dialogue_pattern = re.compile(r'\n\s*([A-Z][A-Z\s]+)\s*\n\s*([^\n]+)')
        dialogues = character_dialogue_pattern.findall(screenplay)

        # convert to df
        dialogue_df = pd.DataFrame(dialogues, columns=['Character', 'Dialogue'])

        # filter out non-character entries from dialogues
        character_name_pattern = re.compile(r'\n\s*([A-Z][A-Z\s]+)\s*\n')
        potential_characters = character_name_pattern.findall(screenplay)
        character_counts = pd.Series(potential_characters).value_counts()
        character_threshold = 5  # number of times a character has to be mentioned
        characters = character_counts[character_counts > character_threshold].index.tolist()
        dialogue_df = dialogue_df[dialogue_df['Character'].isin(characters)]

        # create interaction matrix for all characters
        all_characters = dialogue_df['Character'].unique()
        interaction_matrix_all = pd.DataFrame(0, index=all_characters, columns=all_characters)

        # populate interaction matrix by considering adjacent dialogues
        for i in range(len(dialogue_df) - 1):
            char1 = dialogue_df.iloc[i]['Character']
            char2 = dialogue_df.iloc[i + 1]['Character']
            if char1 != char2:
                interaction_matrix_all.loc[char1, char2] += 1
                interaction_matrix_all.loc[char2, char1] += 1

        # create networkx graph from interaction matrix
        G_all = nx.from_pandas_adjacency(interaction_matrix_all)

        # calculate degree centrality (value for how central a character is)
        degree_centrality = nx.degree_centrality(G_all)
        average_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)

        # calculate closeness centrality (value for how close characters are)
        closeness_centrality = nx.closeness_centrality(G_all)
        average_closeness_centrality = sum(closeness_centrality.values()) / len(closeness_centrality)

        # calculate betweenness centrality (not sure about this one)
        betweenness_centrality = nx.betweenness_centrality(G_all)
        average_betweenness_centrality = sum(betweenness_centrality.values()) / len(betweenness_centrality)

        # interaction diversity (number of unique characters each character interacts with)
        interaction_diversity = (interaction_matrix_all > 0).sum(axis=1)
        average_interaction_diversity = interaction_diversity.mean()

        # normalized interaction coefficient (by total number of interactions)
        total_interactions = interaction_matrix_all.sum().sum()
        normalized_interaction_coefficient = total_interactions / (len(all_characters) * (len(all_characters) - 1))

        # create df to store coefficients
        screenplay_metrics = pd.DataFrame([{
            'average_degree_centrality': average_degree_centrality,
            'average_closeness_centrality': average_closeness_centrality,
            'average_betweenness_centrality': average_betweenness_centrality,
            'average_interaction_diversity': average_interaction_diversity,
            'normalized_interaction_coefficient': normalized_interaction_coefficient
        }])

    except ZeroDivisionError:
        print(f"ZeroDivisionError for file: {file_path}")
        screenplay_metrics = {
            'average_degree_centrality': 0,
            'average_closeness_centrality': 0,
            'average_betweenness_centrality': 0,
            'average_interaction_diversity': 0,
            'normalized_interaction_coefficient': 0
        }
    
    return screenplay_metrics

# Clean text
def clean_scene_text(scene_text):
    lines = scene_text.splitlines()
    cleaned_lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines]
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

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

def extract_scene_lengths(scene_separated_text):
    scenes = scene_separated_text.split('=' * 50)
    scene_lengths = [len(scene.strip().split()) for scene in scenes if scene.strip()]
    return scene_lengths

# function to get mean length of scenes and standard deviation from mean
def analyze_scene_lengths(scene_lengths):
    mean_length = np.mean(scene_lengths)
    std_length = np.std(scene_lengths)
    return mean_length, std_length

# function to calculate coefficient of variation
def coherence_classifier(mean_length, std_length):
    coefficient_of_variation = std_length / mean_length
    return coefficient_of_variation

# function to process all screenplays
def process_scene_lengths(scene_separated_text):
    # extract scene lengths
    scene_lengths = extract_scene_lengths(scene_separated_text)
    
    if scene_lengths:
        # analyze scene lengths
        mean_length, std_length = analyze_scene_lengths(scene_lengths)
        coefficient_of_variation = coherence_classifier(mean_length, std_length)
    
    return coefficient_of_variation



# Cleanup and lemmatization
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

columns_to_scale = ['runtime_minutes', 'production_budget','average_degree_centrality',
'average_closeness_centrality', 'average_betweenness_centrality',
'average_interaction_diversity', 'normalized_interaction_coefficient',
'scene_length_cv']

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
        scene_separated_text = process_screenplay(raw_text)
        df_screenplay_metrics = calculate_screenplay_metrics(raw_text)
        st.write(df_screenplay_metrics)
        

        clean_text = raw_text.replace(r'\s+', ' ').strip().lower()
        lem_text = lemmatize_text(remove_stopwords(remove_punctuation(clean_text)))
        
        tfidf_text = tfidf.transform([lem_text])      
        lsa_text = lsa.transform(tfidf_text)

        count_text = counts.transform([clean_text])
        lda_text = lda.transform(count_text)
        lda_columns = [f'topic_{i}' for i in range(lda_text.shape[1])]
        df_lda = pd.DataFrame(lda_text, columns=lda_columns)


        df_clean = pd.DataFrame({'clean':[clean_text]})
        glove_text = np.vstack(df_clean['clean'].apply(lambda x: get_script_embedding(x, embeddings_index)).values)

        df_genre = pd.DataFrame([[genre in genres for genre in genre_list]], columns=genre_columns, dtype=int)
        df_age = pd.DataFrame([[age in age_rating for age in age_list]], columns=age_columns, dtype=int)
        df = pd.concat([df_age, df_genre], axis=1)
        #scale production budget
        df['production_budget'] = production_budget
        df['runtime_minutes'] = run_time
        df['scene_length_cv'] = process_scene_lengths(scene_separated_text)
        
        # scaling of production budget needs to be done!
        df['flesch_reading_ease'] = textstat.flesch_reading_ease(clean_text)
        df['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(clean_text)  
        df[['polarity', 'subjectivity']] = sentiment_features(clean_text)
        df = pd.concat([df, df_lda, df_screenplay_metrics], axis=1)

        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
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







