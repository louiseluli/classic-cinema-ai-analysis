# An AI-Driven Exploration of Cinema's Defining Era: IMDb Movie Analysis (1915-1960)

## Description

This project leverages Artificial Intelligence (AI) and Machine Learning (ML) for a multifaceted analysis of films produced during the pivotal era of 1915-1960. It aims to move beyond simple genre tags or star searches to uncover deeper connections and patterns within this significant cinematic heritage.

The core objectives of this study are:

- To identify meaningful, data-driven clusters or groupings among films of this era using ML algorithms, potentially transcending traditional genre classifications.
- To build predictive models to effectively determine membership in these discovered groupings based on quantifiable film characteristics, thereby assessing the coherence of the clusters.
- To develop more sophisticated tools for search, discovery, and recommendation, offering classic film lovers a richer, more nuanced way to explore this defining era.

## Features

- **Data Integration:** Combines and processes data from IMDb Non-Commercial Datasets and Wikidata.
- **Exploratory Data Analysis (EDA):** Reveals key trends in classic cinema, including genre distributions, runtime patterns, and yearly production volumes.
- **Enhanced Search:** Implements an advanced search functionality that considers primary titles, original titles, and alternative titles (AKAs) with fuzzy matching to improve search relevance.
- **Unsupervised Clustering:** Applies K-Means, Agglomerative Clustering, and DBSCAN algorithms to identify latent film groupings based on their features.
- **Supervised Classification:** Builds models (Naive Bayes, Logistic Regression, LinearSVC) to predict cluster membership and assess cluster coherence.
- **Interactive Tools:** Provides interactive widgets for exploring movies, clusters, and a hybrid recommender system.

## Technologies Used

This project utilizes Python and several key libraries:

- **Data Manipulation & Analysis:** pandas, numpy
- **Machine Learning:** scikit-learn, tensorflow/keras, imbalanced-learn
- **Data Visualization:** matplotlib, seaborn, plotly
- **Web Data & APIs:** SPARQLWrapper
- **Interactive Elements:** ipywidgets
- **Graph Analysis:** networkx
- **Notebook Environment:** JupyterLab/Jupyter Notebook

## Setup and Installation

1.  **Clone the repository (Once you create it on GitHub):**
    ```bash
    git clone [https://github.com/your-username/classic-cinema-ai-analysis.git](https://github.com/your-username/classic-cinema-ai-analysis.git)
    cd classic-cinema-ai-analysis
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    You will need a `requirements.txt` file (we will create this next!). Once you have it, you can install dependencies using:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Data:**
    This project uses publicly available IMDb Non-Commercial Datasets and Wikidata.

    - **IMDb Datasets:** Download the following files from [IMDb Datasets](https://datasets.imdbws.com/):
      - `title.basics.tsv.gz`
      - `name.basics.tsv.gz`
      - `title.ratings.tsv.gz`
      - `title.principals.tsv.gz`
      - `title.crew.tsv.gz`
      - `title.akas.tsv.gz`
    - **Wikidata Cached Data (Recommended for speed):** The notebook uses cached Wikidata query results to avoid long runtimes. These files (`wikidata_movie_data.csv`, `wikidata_person_data.csv`) are mentioned as being provided via a Google Drive link in the original project description.
      - [Google Drive Link to Datasets](https://drive.google.com/drive/folders/1OBmUh5Nr2sYIcADH6y0DmpNK-tQdod_q?usp=share_link)
      - Place the downloaded `dataset` folder (containing `IMDb` and `Wikimedia` subfolders) into the `data/` directory of your project, as per the recommended structure: `classic-cinema-ai-analysis/data/`.

5.  **Configure Data Path in Notebook:**
    - Open the `FinalProject.ipynb` notebook.
    - Locate the cell defining the `GOOGLE_DRIVE_PATH` variable (likely in a section titled "Data Path Setup and File Verification").
    - **Crucial:** Update this path to reflect the _relative path_ to the `dataset` folder from your notebook's location. If your notebook is in the root `classic-cinema-ai-analysis` folder, and your data is in `classic-cinema-ai-analysis/data/dataset`, the path should be:
      `python
    # Example - MODIFY THIS LINE (if your 'dataset' folder is directly inside 'data/'):
    GOOGLE_DRIVE_PATH = 'data/dataset'
    # Or if the IMDb and Wikimedia folders are directly inside 'data/'
    # IMDB_PATH = 'data/IMDb'
    # WIKIMEDIA_PATH = 'data/Wikimedia'
    # You will need to adjust the notebook's data loading logic accordingly.
    # The original notebook used a direct path, so ensure your notebook
    # now correctly locates files within your project structure.
    `  _Review your notebook's "Data Path Setup and File Verification" section to see how`GOOGLE*DRIVE_PATH`, `IMDB_DATA_PATH`, and `WIKIDATA_FILE_PATH`are used and adjust them to point to`data/IMDb/`and`data/Wikimedia/` within your project.*

## Usage

1.  Ensure your Python environment is activated and all dependencies from `requirements.txt` are installed.
2.  Make sure the data files (IMDb and cached Wikidata) are correctly placed in the `data/IMDb/` and `data/Wikimedia/` folders respectively, and the paths in the notebook are updated.
3.  Open the `FinalProject.ipynb` notebook using Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook FinalProject.ipynb
    # or
    jupyter lab FinalProject.ipynb
    ```
4.  Execute the cells sequentially from top to bottom.
    - **Note:** Some cells, particularly those involving large data processing, Wikidata fetching (if cache is missed), or model training, may take significant time to run.

## Project Structure

To be added
