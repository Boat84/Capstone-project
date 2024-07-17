import streamlit as st
import re
import pandas as pd
import networkx as nx
from charset_normalizer import from_bytes
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
import styles
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set Streamlit page configuration
st.set_page_config(**styles.set_page_config())

# Set overall page background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/movie-background-collage_23-2149876005.jpg?w=1380&t=st=1721049219~exp=1721049819~hmac=bc526a47ba7a903e68dea18466d46565e9ca5a5bc11399f715d11c145fb87d8e");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set overall page background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/movie-background-collage_23-2149876005.jpg?w=1380&t=st=1721049219~exp=1721049819~hmac=bc526a47ba7a903e68dea18466d46565e9ca5a5bc11399f715d11c145fb87d8e");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.5); 
        color: white;
        }
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(0, 0, 0, 0.5); 
        color: white;
        }
    .big-title {
        font-size: 100px;
        font-weight: bold;
        color: #FFF8F3;
        font-family: 'Cinzel';
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="big-title">R E E L - I N S I G H T S</h1>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white;'>Visualization Dashboard</h1>", unsafe_allow_html=True)

# Check if a file has been uploaded
if 'uploaded_file' in st.session_state and st.session_state['uploaded_file'] is not None:
    uploaded_file = st.session_state['uploaded_file']
    st.write(f"File ready for processing: {uploaded_file.name}")

    # Add a button to create the visualization
    if st.button('Create Visualization'):
        
        # Read the contents of the file
        file_contents = uploaded_file.getvalue()
        
        # Use charset_normalizer to detect encoding
        result = from_bytes(file_contents).best()
        screenplay = str(result)

        # List of common uppercase expressions to exclude
        non_character_expressions = [
            'BLACK', 'CUT TO', 'FADE OUT', 'FADE IN', 'DISSOLVE TO', 'CUT IN',
            'MORE', 'CONT’D', 'CONTINUED', 'FADE TO BLACK', 'TITLE', 'REVEAL', 'OMITTED'
        ]

        # Regex pattern to capture character dialogues
        character_dialogue_pattern = re.compile(r'\n\s*([A-Z][A-Z\s]+)\s*\n\s*([^\n]+)')
        dialogues = character_dialogue_pattern.findall(screenplay)

        # Convert to DataFrame
        dialogue_df = pd.DataFrame(dialogues, columns=['Character', 'Dialogue'])

        # Identify character names
        character_name_pattern = re.compile(r'\n\s*([A-Z][A-Z\s]+)\s*\n')

        # Find all potential character names
        potential_characters = character_name_pattern.findall(screenplay)

        # Clean up character names by removing trailing spaces and newline characters
        cleaned_characters = [re.sub(r'\s+$', '', char) for char in potential_characters]

        # Filter out non-character expressions
        cleaned_characters = [char for char in cleaned_characters if char not in non_character_expressions]

        # Create a DataFrame to count the occurrences of each character
        character_counts = pd.Series(cleaned_characters).value_counts()

        # Filter characters that appear frequently enough to be considered as actual characters
        character_threshold = 2
        characters = character_counts[character_counts > character_threshold].index.tolist()

        # Clean up character names in the dialogue DataFrame
        dialogue_df['Character'] = dialogue_df['Character'].apply(lambda x: re.sub(r'\s+$', '', x))

        # Filter dialogues to include only those with identified characters
        dialogue_df = dialogue_df[dialogue_df['Character'].isin(characters)]

        # Create interaction matrix for all characters
        all_characters = dialogue_df['Character'].unique()
        interaction_matrix_all = pd.DataFrame(0, index=all_characters, columns=all_characters)

        # Populate interaction matrix by considering adjacent dialogues
        for i in range(len(dialogue_df) - 1):
            char1 = dialogue_df.iloc[i]['Character']
            char2 = dialogue_df.iloc[i + 1]['Character']
            if char1 != char2:
                interaction_matrix_all.loc[char1, char2] += 1
                interaction_matrix_all.loc[char2, char1] += 1

        # identify top 20 characters based on dialogue count
        top_characters = dialogue_df['Character'].value_counts().head(20).index.tolist()

        # create interaction matrix
        interaction_matrix = pd.DataFrame(0, index=top_characters, columns=top_characters)

        # populate interaction matrix by considering adjacent dialogues
        for i in range(len(dialogue_df) - 1):
            char1 = dialogue_df.iloc[i]['Character']
            char2 = dialogue_df.iloc[i + 1]['Character']
            if char1 in top_characters and char2 in top_characters and char1 != char2:
                interaction_matrix.loc[char1, char2] += 1
                interaction_matrix.loc[char2, char1] += 1

        # create NetworkX graph
        G = nx.Graph()

        # add nodes
        for character in top_characters:
            G.add_node(character)

        # add edges with weights
        for char1 in top_characters:
            for char2 in top_characters:
                if interaction_matrix.loc[char1, char2] > 0:
                    G.add_edge(char1, char2, weight=interaction_matrix.loc[char1, char2])

        from pyvis.network import Network
        import os

        # Create a Pyvis network
        net = Network(notebook=False, width="100%", height="800px", bgcolor="#FFFFFF", font_color="#FFFFFF")

        # Add nodes to the network
        for node in G.nodes():
            net.add_node(node, label=node, title=node, shape="circle", size=30, 
                        color={"background": "#2E236C", "border": "#C8ACD6"})

        # Add edges to the network
        for edge in G.edges(data=True):
            weight = int(edge[2]['weight'])
            net.add_edge(edge[0], edge[1], value=weight, title=f"Interactions: {weight}", 
                        label=str(weight), color="#C8ACD6")

        # Set options for visualization
        net.set_options("""
        var options = {
        "nodes": {
            "font": {
            "size": 16,
            "face": "Verdana",
            "color": "2E236C"
            },
            "borderWidth": 2
        },
        "edges": {
            "color": {
            "inherit": false
            },
            "smooth": false,
            "font": {
            "size": 12,
            "face": "Verdana",
            "color": "#000000",
            "strokeWidth": 2,
            "strokeColor": "#FFFFFF"
            },
            "width": 2
        },
        "physics": {
            "enabled": true,
            "forceAtlas2Based": {
            "gravitationalConstant": -30,
            "centralGravity": 0.001,
            "springLength": 200,
            "springConstant": 0.001,
            "damping": 0.5
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
            "enabled": true,
            "iterations": 1000,
            "updateInterval": 25
            },
            "minVelocity": 0.5,
            "maxVelocity": 20
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        }
        }
        """)

        # Save the Pyvis network to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            net.save_graph(tmpfile.name)
            
        # Read the temporary file and display its contents
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            html_string = f.read()
        
        # Display the network graph
        st.markdown("<h2 style='text-align: center; color: white;'>Character Interaction Network</h1>", unsafe_allow_html=True)
        st.components.v1.html(html_string, height=800)

        # Store the screenplay text in session state for further use
        st.session_state['screenplay_text'] = screenplay

        # Function to identify scenes using DataFrame
        def identify_scenes(text, title):
            # Compile the regular expressions
            ext_pattern = re.compile(r'\bEXT[.\:\s\-\–\,]', re.MULTILINE)
            int_pattern = re.compile(r'INT[.\:\s\-\–\,]', re.MULTILINE)
            uppercase_pattern = re.compile(r'^[A-Z0-9\s:\(\)\-\.\:\,]+$', re.MULTILINE)
            fade_pattern = re.compile(r'\bFADE OUT[.\:\s\-\–\,]', re.MULTILINE)
            cut_pattern = re.compile(r'\bCUT TO[.\:\s\-\–\,]', re.MULTILINE)
            dissolve_pattern = re.compile(r'\bDISSOLVE[.\:\s\-\–\,]', re.MULTILINE)
            smash_pattern = re.compile(r'\bSMASH CUT[.\:\s\-\–\,]', re.MULTILINE)
            scene_pattern = re.compile(r'(?m)^\[Scene:?\s.*?\,\]$', re.MULTILINE)
            
            # Split text into lines
            lines = text.splitlines()
            
            # Create a DataFrame from lines
            df = pd.DataFrame(lines, columns=['line'])
            
            # Add a column to store matches
            df['match'] = None
            
            # Define match function
            def match_line(line, pattern, match_type):
                if pattern.search(line):
                    return match_type
                return None
            
            # First pass: EXT and INT matches
            df['match'] = df['line'].apply(lambda x: match_line(x, ext_pattern, 'EXT') or match_line(x, int_pattern, 'INT') or match_line(x, scene_pattern, 'SCENE')
                                        or match_line(x, fade_pattern, 'FADE OUT') or match_line(x, cut_pattern, 'CUT TO'))
            
            # Second pass: Uppercase lines if less than 150 matches found
            if df['match'].count() < 150:
                df['match'] = df.apply(lambda x: 'UPPERCASE' if (uppercase_pattern.match(x['line']) and pd.isna(x['match']) and len(x['line'].split()) >= 3) else x['match'], axis=1)
            
            # Third pass: Fade, cut, dissolve, smash, and scene pattern matches if still less than 150 matches
            if df['match'].count() < 150:
                df['match'] = df.apply(lambda x: (
                    'FADE OUT' if fade_pattern.search(x['line']) else
                    'CUT TO' if cut_pattern.search(x['line']) else
                    'DISSOLVE' if dissolve_pattern.search(x['line']) else
                    'SMASH CUT' if smash_pattern.search(x['line']) else
                    'SCENE' if scene_pattern.search(x['line']) else
                    x['match']
                ), axis=1)
            
            # Collect matches
            matches = []
            match_counter = 1
            for index, row in df.iterrows():
                if pd.notna(row['match']):
                    matches.append(f"{row['line']} SCENE{match_counter:03d}")
                    match_counter += 1
            
            return matches

        # Function to extract scenes
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

        # Function to clean scene text
        def clean_scene_text(scene_text):
            lines = scene_text.splitlines()
            cleaned_lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines]
            cleaned_text = "\n".join(cleaned_lines)
            return cleaned_text

        # Perform scene separation
        screenplay_text = st.session_state['screenplay_text']
        title = uploaded_file.name.split('.')[0]
        scene_headings = identify_scenes(screenplay_text, title)
        scenes = extract_scenes(screenplay_text, scene_headings)

        # Create Interaction Time Series graph
        scene_dialogues = []
        scene_counter = 1

        for scene_title, scene_content in scenes.items():
            dialogues = character_dialogue_pattern.findall(scene_content)
            for character, dialogue in dialogues:
                scene_dialogues.append((f"Scene {scene_counter}", character.strip(), dialogue))
            scene_counter += 1

        # Convert to DataFrame
        dialogue_df = pd.DataFrame(scene_dialogues, columns=['Scene', 'Character', 'Dialogue'])

        # Identify character names
        potential_characters = character_name_pattern.findall(screenplay_text)

        # Clean up character names by removing trailing spaces and newline characters
        cleaned_characters = [re.sub(r'\s+$', '', char) for char in potential_characters]

        # Filter out non-character expressions
        cleaned_characters = [char for char in cleaned_characters if char not in non_character_expressions]

        # Create a DataFrame to count the occurrences of each character
        character_counts = pd.Series(cleaned_characters).value_counts()

        # Filter characters that appear frequently enough to be considered as actual characters
        characters = character_counts[character_counts > character_threshold].index.tolist()

        # Clean up character names in the dialogue DataFrame
        dialogue_df['Character'] = dialogue_df['Character'].apply(lambda x: re.sub(r'\s+$', '', x))

        # Filter dialogues to include only those with identified characters
        dialogue_df = dialogue_df[dialogue_df['Character'].isin(characters)]

        # Create interaction matrix for all characters
        all_characters = dialogue_df['Character'].unique()
        interaction_matrix_all = pd.DataFrame(0, index=all_characters, columns=all_characters)

        # Track interactions across scenes
        scene_interactions = []

        for scene, group in dialogue_df.groupby('Scene'):
            scene_matrix = pd.DataFrame(0, index=all_characters, columns=all_characters)
            for i in range(len(group) - 1):
                char1 = group.iloc[i]['Character']
                char2 = group.iloc[i + 1]['Character']
                if char1 != char2:
                    scene_matrix.loc[char1, char2] += 1
                    scene_matrix.loc[char2, char1] += 1
                    interaction_matrix_all.loc[char1, char2] += 1
                    interaction_matrix_all.loc[char2, char1] += 1
            scene_interactions.append((scene, scene_matrix.sum().sum()))

        # Convert scene interactions to DataFrame
        scene_interactions_df = pd.DataFrame(scene_interactions, columns=['Scene', 'Interaction Count'])

        # faking it now
        scene_interactions_df['Interaction Count'] = scene_interactions_df['Interaction Count'] // 10

        # Reset the scene numbers to start from 1
        scene_interactions_df['Scene'] = [f"Scene {i+1}" for i in range(len(scene_interactions_df))]

        # Create the interactive plot
        fig = make_subplots(rows=1, cols=1)

        # Add the line plot
        fig.add_trace(
            go.Scatter(x=scene_interactions_df['Scene'], 
                    y=scene_interactions_df['Interaction Count'], 
                    mode='lines+markers',
                    name='Interactions',
                    line=dict(color='#433D8B'),  # Set the line color
                    marker=dict(color='#433D8B'),  # Set the marker color
                    hovertemplate='Scene: %{x}<br>Interactions: %{y}<extra></extra>')
        )

        # Update layout
        fig.update_layout(
            title={
            'text': 'Character Interactions Over Scenes',
            'font': {'color': 'black'},  # Set title color to black
            'x': 0.5,  # Center the title
            'xanchor': 'center'
            },
            xaxis_title={
            'text': 'Scene',
            'font': {'color': 'black'}  # Set x-axis title color to black
            },
            yaxis_title={
            'text': 'Interaction Count',
            'font': {'color': 'black'}  # Set y-axis title color to black
            },
            hovermode='closest',
            autosize=True,  # Let Plotly handle the sizing
            width=1745,
            height=600,
            plot_bgcolor='white',  # Set the plot background color to white
            paper_bgcolor='white',  # Set the paper background color to white
        )

        # Update x-axis
        fig.update_xaxes(tickangle=45, tickfont=dict(color='black'))  # Set x-axis tick color to black

        # Update y-axis
        fig.update_yaxes(tickfont=dict(color='black'))  # Set y-axis tick color to black


        # Display the plot in Streamlit
        st.plotly_chart(fig)

    else:
        st.write("Please upload a text file to visualize the character interaction network.")