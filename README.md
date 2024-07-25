# Capstone-Project: Real-Insight

Aim of the Project Reel-Insights is the analysis of long form movie scripts via natural language processing in combination with metadata available before publication or even production of the movie to benefit (aspiring) screenplay writers and movie producers. On this basis information was extracted and embedded classification models were trained to gather insights on the later financial success of the film. An app was built to easily process new movie scripts and visualize extracted data.

### Used datasets:
* IMDb Movie Dataset: All Movies by Genre
https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre
  * unzip into data/IMDb Movie Dataset/

* Ultimate Film Statistics Dataset with Production Budget and Domestic (US) and Worldwide Gross
https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-film-statistics-dataset-for-ml

* Movie Scripts Corpus Dataset
https://www.kaggle.com/datasets/gufukuro/movie-scripts-corpus

Also needed:

* Glove.6B
https://nlp.stanford.edu/projects/glove/
  * unzip into data/glove.6B/


 
## Install the virtual environment and the required packages by following commands:

NOTE: for **macOS** with **M1, M2** chips (other than intel)
  ```BASH
  pyenv local 3.11.3
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements_M1.txt
  ```
NOTE: for macOS with **intel** chips
```BASH
  pyenv local 3.11.3
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

    
**WindowsOS** type the following commands :



For `PowerShell` CLI :

```PowerShell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

For `Git-bash` CLI :

```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**`Note:`**
If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
```Bash
python.exe -m pip install --upgrade pip
```
