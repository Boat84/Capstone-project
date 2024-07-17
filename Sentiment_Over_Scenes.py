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

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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
    # 拼接处理后的文本
    processed_text = ' '.join(tokens)
    return processed_text

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

# Function for plot
def plot_zoomable_trend_chart(data):
    fig = px.line(data, x='Scene', y='Compound', title='Sentiment Changes Over Scenes')
    fig.update_layout(
        title={
            'text': 'Sentiment Changes Over Scenes',
            'font': {'color': 'black', 'size': 24},  # Set title color to black and increase size
            'x': 0.5,  # Center the title
            'xanchor': 'center'
        },
        xaxis_title={
            'text': 'Scene',
            'font': {'color': 'black', 'size': 18}  # Set x-axis title color to black and increase size
        },
        yaxis_title={
            'text': 'Sentiment',
            'font': {'color': 'black', 'size': 18}  # Set y-axis title color to black and increase size
        },
        autosize=True,
        width=1750,  # Match the width of the other visualization
        height=600,  # Match the height of the other visualization
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='closest',
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        ),
        plot_bgcolor='white',  # Set plot background to white
        paper_bgcolor='white'  # Set paper background to white
    )
    
    # Update line and marker color
    fig.update_traces(line_color='#433D8B', marker_color='#433D8B')
    
    return fig

if 'uploaded_file' in st.session_state:
    try:
        file_content = st.session_state['uploaded_file'].read().decode("utf-8")
        scene_separated_text = process_screenplay(file_content)
        processed_results = classify_and_save_scenes(scene_separated_text)
        data = pd.DataFrame(processed_results)
        st.plotly_chart(plot_zoomable_trend_chart(data), use_container_width=True)
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
else:
    st.error("No file uploaded. Please go to the Home page and upload a file.")