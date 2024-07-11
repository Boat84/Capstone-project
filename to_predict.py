{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "#! pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pip in /Users/xiaozhouye/.pyenv/versions/3.11.3/lib/python3.11/site-packages (24.1.2)\n"
     ]
    }
   ],
   "source": [
    "#! pip install --upgrade pip\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: nltk in /Users/xiaozhouye/.pyenv/versions/3.11.3/lib/python3.11/site-packages (3.8.1)\n",
      "Requirement already satisfied: click in /Users/xiaozhouye/.pyenv/versions/3.11.3/lib/python3.11/site-packages (from nltk) (8.1.7)\n",
      "Requirement already satisfied: joblib in /Users/xiaozhouye/.pyenv/versions/3.11.3/lib/python3.11/site-packages (from nltk) (1.3.2)\n",
      "Requirement already satisfied: regex>=2021.8.3 in /Users/xiaozhouye/.pyenv/versions/3.11.3/lib/python3.11/site-packages (from nltk) (2024.5.15)\n",
      "Requirement already satisfied: tqdm in /Users/xiaozhouye/.pyenv/versions/3.11.3/lib/python3.11/site-packages (from nltk) (4.66.4)\n"
     ]
    }
   ],
   "source": [
    "#! pip  install nltk\n",
    "# Download required NLTK data files\n",
    "#nltk.download('punkt')\n",
    "#nltk.download('stopwords')\n",
    "#nltk.download('wordnet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "import re\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "import joblib\n",
    "import streamlit as st\n",
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "\n",
    "# Function to load pre-trained GloVe embeddings\n",
    "def load_glove_embeddings(glove_file):\n",
    "    embeddings_index = {}\n",
    "    with open(glove_file, encoding='utf-8') as f:\n",
    "        for line in f:\n",
    "            values = line.split()\n",
    "            word = values[0]\n",
    "            coefs = np.asarray(values[1:], dtype='float32')\n",
    "            embeddings_index[word] = coefs\n",
    "    return embeddings_index\n",
    "\n",
    "# function used to preprocess the data \n",
    "def preprocess_text(text):\n",
    "    # Remove HTML tags\n",
    "    text = re.sub(r'<.*?>', '', text)\n",
    "    # Remove all non-alphabetic characters\n",
    "    text = re.sub(r'[^a-zA-Z\\s]', '', text)\n",
    "    # Convert to lowercase\n",
    "    text = text.lower()\n",
    "    # Tokenization\n",
    "    words = word_tokenize(text)\n",
    "    # Remove stop words\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    words = [word for word in words if word not in stop_words]\n",
    "    # Lemmatization\n",
    "    lemmatizer = WordNetLemmatizer()\n",
    "    words = [lemmatizer.lemmatize(word) for word in words]\n",
    "    return ' '.join(words)\n",
    "\n",
    "# Function to convert text to GloVe vectors\n",
    "def texts_to_glove_vectors(texts, embeddings_index):\n",
    "    text_vectors = []\n",
    "    for text in texts:\n",
    "        words = text.split()\n",
    "        vectors = []\n",
    "        for word in words:\n",
    "            vector = embeddings_index.get(word)\n",
    "            if vector is not None:\n",
    "                vectors.append(vector)\n",
    "        if vectors:\n",
    "            text_vector = np.mean(vectors, axis=0)  # Average word vectors\n",
    "        else:\n",
    "            text_vector = np.zeros(embeddings_index[next(iter(embeddings_index))].shape)  # Handle unseen words\n",
    "        text_vectors.append(text_vector)\n",
    "    return np.array(text_vectors)\n",
    "\n",
    "\n",
    "# upload the model\n",
    "model = joblib.load('gb_model.pkl')\n",
    "\n",
    "# Streamlit main function\n",
    "def main():\n",
    "    st.title(\"Prediction of profitablity\")\n",
    "\n",
    "    # upload file \n",
    "    uploaded_file = st.file_uploader(\"upload your .csv\", type=\"csv\")\n",
    "\n",
    "    if uploaded_file is not None:\n",
    "        # read the file\n",
    "        data = pd.read_csv(uploaded_file)\n",
    "   \n",
    "\n",
    "        # preprocess\n",
    "        processed_data = preprocess_text(data)\n",
    "     \n",
    "        # get vectors\n",
    "        X_vec = texts_to_glove_vectors(processed_data, glove_embeddings)\n",
    "\n",
    "        # predict\n",
    "        predictions = model.predict(X_vec)\n",
    "        st.write(\"Result:\")\n",
    "        st.write(predictions)\n",
    "\n",
    "glove_file = '/Users/xiaozhouye/Desktop/neuefische/Capstone-project_copy/ohters/glove.42B.300d.txt'  # Update this path to your GloVe file\n",
    "glove_embeddings = load_glove_embeddings(glove_file)\n",
    "# run Streamlit\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
