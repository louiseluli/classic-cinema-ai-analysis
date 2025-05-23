# An AI-Driven Exploration of Cinema's Defining Era: IMDb Movie Analysis (1915-1960)

![Alt text for image](/assets/images/Movie_recommender_logo.png)

## Description

This project leverages Artificial Intelligence (AI) and Machine Learning (ML) for a multifaceted analysis of films produced during the pivotal era of 1915-1960. This period, marked by significant artistic innovation and industrial transformation, laid the groundwork for contemporary cinema. This project aims to move beyond simple genre tags or star searches to uncover deeper connections and patterns within this rich cinematic heritage.

The core objectives of this study are:

- To identify meaningful, data-driven clusters or groupings among films of this era using ML algorithms, potentially transcending traditional genre classifications.
- To build predictive models to effectively determine membership in these discovered groupings based on quantifiable film characteristics, thereby assessing the coherence and interpretability of these clusters.
- To develop more sophisticated tools for search, discovery, and recommendation, offering classic film lovers a richer, more nuanced way to explore this defining era.

## Table of Contents

- [An AI-Driven Exploration of Cinema's Defining Era: IMDb Movie Analysis (1915-1960)](#an-ai-driven-exploration-of-cinemas-defining-era-imdb-movie-analysis-1915-1960)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Workflow](#project-workflow)
  - [Technologies Used](#technologies-used)
  - [Visual Highlights](#visual-highlights)
  - [Setup and Installation](#setup-and-installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)

## Features

- **Comprehensive Data Integration:** Combines and processes data from multiple IMDb Non-Commercial Datasets and enriches it with information from Wikidata.
- **Advanced Data Loading:** Utilizes a memory-efficient IMDb dataset loader with automatic data type optimization.
- **In-depth Exploratory Data Analysis (EDA):** Uncovers key trends in classic cinema, including genre distributions, runtime patterns, yearly production volumes, and rating analyses.
- **Sophisticated Search Functionality:** Implements an enhanced search that considers primary titles, original titles, and alternative titles (AKAs) using fuzzy matching for improved relevance.
- **Unsupervised Clustering:** Applies K-Means, Agglomerative Clustering, and DBSCAN algorithms to identify latent film groupings based on their features (e.g., genres, keywords, embeddings).
- **Supervised Classification:** Builds and evaluates models (Naive Bayes, Logistic Regression, LinearSVC) to predict cluster membership, helping to validate and understand the discovered film groupings.
- **Interactive Exploration Tools:** Provides `ipywidgets`-based tools for dynamic movie exploration, cluster investigation, and a hybrid recommender system.
- **Robust Wikidata Fetching:** Includes a multi-stage process for fetching and caching movie and person data from Wikidata, with politeness mechanisms (batching, retries, delays).

## Project Workflow

1.  **Data Acquisition & Preprocessing:**
    - Loading of IMDb datasets (`title.basics`, `name.basics`, `title.ratings`, `title.principals`, `title.crew`, `title.akas`, and potentially `title.episode`).
    - Initial cleaning, type conversion, and merging of IMDb data.
    - Extraction of unique movie and person (actors, directors) IMDb IDs.
2.  **Wikidata Enrichment:**
    - Fetching detailed movie and person data from Wikidata using SPARQL queries.
    - Caching Wikidata results to local CSV files (`wikidata_movie_data.csv`, `wikidata_person_data.csv`) to avoid repeated queries.
    - Cleaning and merging Wikidata with the existing IMDb data.
3.  **Exploratory Data Analysis (EDA):**
    - Visualizing trends in genres, production years, runtimes, ratings, etc.
4.  **Feature Engineering:**
    - Transforming raw data (e.g., text from plot summaries/genres, categorical features) into numerical representations suitable for machine learning (e.g., TF-IDF, embeddings).
5.  **Unsupervised Clustering:**
    - Applying clustering algorithms (K-Means, Agglomerative, DBSCAN) to group films.
    - Evaluating cluster quality (e.g., Silhouette Score) and interpreting cluster characteristics.
6.  **Supervised Classification (Cluster Prediction):**
    - Training models to predict the cluster membership of films based on their features.
    - Evaluating model performance to assess the distinctiveness of clusters.
7.  **Search and Recommendation:**
    - Developing an enhanced search tool with fuzzy matching.
    - Building and presenting an interactive recommendation system.

## Technologies Used

This project utilizes Python and a comprehensive suite of libraries:

- **Core Data Science:** pandas, numpy, scipy
- **Machine Learning:** scikit-learn (for `CountVectorizer`, `TfidfVectorizer`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`, `cosine_similarity`, `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `PCA`, `train_test_split`, various metrics, `LogisticRegression`, `MultinomialNB`, `LinearSVC`, etc.), imbalanced-learn (`SMOTE`)
- **Deep Learning (for embeddings/models):** tensorflow/keras
- **Data Visualization:** matplotlib, seaborn, plotly
- **Web Data & APIs:** SPARQLWrapper (for Wikidata)
- **Interactive Elements:** ipywidgets
- **Graph Analysis (if used):** networkx
- **Utilities:** os, re, gc, time, pickle, random, warnings, difflib, collections
- **Notebook Environment:** JupyterLab/Jupyter Notebook

## Visual Highlights

This project generates several types of visualizations that offer insights into classic cinema.

- **Selected Plots:**

  - Distribution of movie releases per year (1915-1960).
    ![Alt text for image](/assets/images/movie_release.png)

To be continued...

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/classic-cinema-ai-analysis.git](https://github.com/your-username/classic-cinema-ai-analysis.git)
    cd classic-cinema-ai-analysis
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    Ensure you have the `requirements.txt` file in the root of the project.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download IMDb Data:**

    - Visit the [IMDb Non-Commercial Datasets page](https://developer.imdb.com/non-commercial-datasets/).
    - Download the following `.tsv.gz` files:
      - `name.basics.tsv.gz`
      - `title.akas.tsv.gz`
      - `title.basics.tsv.gz`
      - `title.crew.tsv.gz`
      - `title.principals.tsv.gz`
      - `title.ratings.tsv.gz`
      - `title.episode.tsv.gz`
    - Create a folder structure `classic-cinema-ai-analysis/data/IMDb/`.
    - Place all downloaded IMDb `.tsv.gz` files into this `data/IMDb/` directory.

5.  **Wikidata Cache Files (Optional but Recommended for Faster Startup):**

    - The notebook is designed to fetch data directly from Wikidata if cached files are not present. This process can be very time-consuming.
    - If you (or the user) have already run the Wikidata fetching parts of the notebook (Sections 2.3.1 Parts 2 and 3), two CSV files will be generated:
      - `wikidata_movie_data.csv`
      - `wikidata_person_data.csv`

      - Create a folder structure `classic-cinema-ai-analysis/data/Wikimedia/`.
      - Place `wikidata_movie_data.csv` and `wikidata_person_data.csv` into this `data/Wikimedia/` directory.

6.  **Configure Data Path in Notebook (`FinalProject.ipynb`):**
    - Open `FinalProject.ipynb`.
    - The notebook uses a variable, likely `GOOGLE_DRIVE_PATH` in your `load_imdb_dataset` function, and `WIKIDATA_CACHE_DIR` in the Wikidata fetching sections, to locate data files.
    - **For IMDb Data:** The `load_imdb_dataset` function in the notebook refers to `GOOGLE_DRIVE_PATH`. You should modify this function or the variable it uses to point to the local `data/IMDb/` relative path.
      - Example: If `load_imdb_dataset` is called like `load_imdb_dataset('title.basics.tsv.gz')`, ensure the path logic inside it correctly prepends `data/IMDb/`. One way is to set a base path variable at the top of the notebook:
        ```python
        # In the notebook, near the top or before data loading:
        IMDB_DATA_ROOT = 'data/IMDb'
        # Then modify load_imdb_dataset to use it:
        # file_path = os.path.join(IMDB_DATA_ROOT, filename)
        ```
    - **For Wikidata Cache:** The provided code updates for Wikidata fetching already set `WIKIDATA_CACHE_DIR = os.path.join('data', 'Wikimedia')`. Ensure this is consistent in the notebook.

## Usage

1.  Ensure your Python virtual environment is activated and all dependencies from `requirements.txt` are installed.
2.  Confirm that the IMDb data is in `data/IMDb/` and, if using cached Wikidata files, they are in `data/Wikimedia/`.
3.  Verify that the data paths within the `FinalProject.ipynb` notebook correctly point to these local directories.
4.  Launch JupyterLab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```
5.  Open and run the cells in `FinalProject.ipynb` sequentially.
    - **Note on Wikidata Fetching:** If the cached Wikidata files (`wikidata_movie_data.csv`, `wikidata_person_data.csv`) are not present in `data/Wikimedia/`, the notebook will attempt to fetch this data live from Wikidata. This can take a very significant amount of time (hours) depending on the number of IDs and server responsiveness.
    - Other cells involving model training or extensive computations might also take time.

## Project Structure

To be continued...
