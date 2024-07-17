import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import spacy
from charset_normalizer import from_bytes
import streamlit as st
from io import BytesIO
from PIL import Image

# Custom color function
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = ["#17153B", "#2E236C", "#433D8B", "#C8ACD6"]
    return colors[random_state.randint(0, len(colors) - 1)]

@st.cache_resource
def process_text(file_contents):
    # Use charset_normalizer to detect encoding
    result = from_bytes(file_contents).best()
    script_text = str(result)

    # Remove Directorial Expressions and Character Names
    directorial_expressions = [
        'BLACK', 'CUT TO', 'FADE OUT', 'FADE IN', 'DISSOLVE TO', 'CUT IN', 'CLOSE',
        'MORE', 'CONT’D', 'CONTINUED', 'FADE TO BLACK', 'TITLE', 'REVEAL', 'OMITTED', 'P.O.V.', 'POV', 'SUPER', 'BACK TO SCENE', 'CONT', 'EXT', 'INT'
    ]

    for expression in directorial_expressions:
        script_text = script_text.replace(expression, '')

    character_name_pattern = re.compile(r'\n\s*([A-Z][A-Z\s]+)\s*\n')
    script_text = re.sub(character_name_pattern, '', script_text)

    additional_stopwords = set([
        'the', 'and', 'is', 'in', 'to', 'with', 'that', 'on', 'for', 'as', 'it',
        'of', 'at', 'by', 'this', 'be', 'which', 'or', 'from', 'an', 'but', 'not',
        'we', 'you', 'your', 'so', 'can', 'are', 'if', 'then', 'will', 'there', 
        'he', 'she', 'they', 'what', 'all', 'one', 'out', 'up', 'would', 'his', 
        'her', 'their', 'my', 'me', 'no', 'do', 'when', 'about', 'just', 'more', 
        'how', 'like', 'who', 'did', 'them', 'now', 'him', 'said', 'get', 'got',
        'something', 'anything', 'everything', 'somebody', 'anybody', 'us', 'we'
    ])

    stopwords = STOPWORDS.union(additional_stopwords)

    filtered_words = [word for word in script_text.split() if word.lower() not in stopwords and len(word) >= 3]
    filtered_text = ' '.join(filtered_words)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(filtered_text)

    names = set()
    for ent in doc.ents:
        if ent.label_ in ['PERSON']:
            names.add(ent.text)

    # Filter out verbs and names from the tokens
    filtered_tokens = [token.text for token in doc if token.text not in names and len(token.text) >= 3 and token.pos_ != 'VERB']

    # Count the frequency of tokens
    token_counts = Counter(filtered_tokens)

    # Select the 25 most common tokens
    most_common_tokens = token_counts.most_common(50)

    # Join the most common tokens into a single string for word cloud generation
    wordcloud_text = ' '.join([token for token, count in most_common_tokens])

    return wordcloud_text

def main():
    st.title("Movie Script Word Cloud")

    # Check if uploaded_file is available in session state
    if 'uploaded_file' in st.session_state:
        uploaded_file = st.session_state['uploaded_file']
        file_contents = uploaded_file.read()
        wordcloud_text = process_text(file_contents)

        if wordcloud_text:
            # Create a high-resolution word cloud
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', color_func=custom_color_func, width=1920, height=1080).generate(wordcloud_text)

            # Convert to image
            image = wordcloud.to_image()

            # Display the word cloud using Streamlit
            st.image(image, use_column_width=True)
        else:
            st.warning("No valid words to generate a word cloud. Please check the uploaded file.")
    else:
        st.warning("Please upload a script in the Home page.")

if __name__ == "__main__":
    main()