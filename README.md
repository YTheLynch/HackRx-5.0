# HackRx-5.0 Legal AI Analytics and Alarm System : VerdictVision
## 🚀 Insurance Litigation Fraud Detection via Clustering and Anomaly Detection

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
  - [1. Data Scraping and Aggregation](#1-data-scraping-and-aggregation)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Word Embedding Generation](#3-word-embedding-generation)
  - [4. Dimensionality Reduction using PCA](#4-dimensionality-reduction-using-pca)
  - [5. Clustering Algorithms](#5-clustering-algorithms)
  - [6. Ensemble Clustering via Majority Voting](#6-ensemble-clustering-via-majority-voting)
  - [7. Clustering Evaluation](#7-clustering-evaluation)
  - [8. Fraud Detection](#8-fraud-detection)
- [Usage Instructions](#usage-instructions)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Run the Main Script](#2-run-the-main-script)
  - [3. Results](#3-results)
  - [4. Visualizations](#4-visualizations)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Results](#model-results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [License](#license)

## 📝 Overview
This project focuses on detecting potential fraud in insurance litigation by utilizing advanced machine learning techniques. By employing word embeddings and clustering methods, we identify outliers and unusual patterns in legal case data, flagging cases that may involve fraudulent activities. The dataset is derived from open court records across different states, and we use semantic clustering to group cases and anomaly detection to identify outliers.

## 📁 Project Structure
- **`datasets/`**: Contains legal datasets from 5 Indian states, each with 1,000 records.
  
- **`python_notebooks/`**: 
  - **`model_training.ipynb`**: The notebook that integrates and trains the fraud detection model using the aggregated datasets.
  - **`web_scraping.ipynb`**: The notebook that contains the code for scraping legal data from various sources, such as Indian Kanoon and eCourts India.

- **`vendor/`**: Contains various front-end resources for the dashboard:
    - **AOS**
    - **Bootstrap Icons**
    - **Glightbox**
    - **ImagesLoaded**
    - **Isotope Layout**
    - **PHP Email Form**
    - **Swiper**
    - **Waypoints**
    - **`main.css`**
    - **`output.css`**

- **`templates/`**: Contains the HTML for the dashboard.
    - **`index.html`**: The HTML file that structures the web-based dashboard for visualizing the fraud detection insights.

    - **`app.py`**: The Flask application that serves the web-based dashboard and handles backend operations, such as the alarm system and email notifications.

    - **`model.pkl`**: Serialized trained model used for fraud detection.

    - **`scaler.pkl`**: Scaler object used to preprocess input data before passing it to the model for predictions.


## 📊 Dataset Description
The primary dataset `Merged_all_states_data.csv` consists of the following key columns:

| **Column Name** | **Description**                                           |
|-----------------|-----------------------------------------------------------|
| Unnamed: 0      | Index column                                              |
| title           | Title of the litigation case                              |
| headline        | Case headline providing a brief summary                   |
| detail_id       | Unique identifier for the case                            |
| new_link        | Link to the full details of the case                      |
| Case Details    | Detailed text of the case, including facts, arguments, and judgments |

## 🔄 Project Workflow
### 1. Data Scraping and Aggregation
The dataset was compiled using a custom web scraper that extracted data from platforms like eCourts India and Indian Kanoon. The scraper focused on pending and disposed litigation in the insurance sector, gathering data from five Indian states, with 1,000 records collected per state.

### 2. Preprocessing
The text in the **Case Details** column is cleaned through tokenization, lemmatization, and removal of stop words and punctuation using `spaCy`.

### 3. Word Embedding Generation
Using `spaCy`'s pre-trained `en_core_web_sm` model, the textual data is converted into word embeddings, which are vector representations capturing semantic meaning.

### 4. Dimensionality Reduction using PCA
Principal Component Analysis (PCA) is employed to reduce the dimensionality of the word embeddings from 300 dimensions to 50. This step helps streamline the input for clustering.

### 5. Clustering Algorithms
Multiple clustering techniques are applied to group cases based on their semantic similarity:

- **KMeans:** Groups cases into a pre-defined number of clusters using centroids.
- **DBSCAN:** A density-based method that identifies core samples and noise.
- **Agglomerative Clustering:** A hierarchical approach that builds clusters by merging or splitting.
- **Spectral Clustering:** A graph-based technique using nearest neighbors.
- **Gaussian Mixture Model (GMM):** A probabilistic model assuming data points are derived from a mixture of Gaussian distributions.
- **Mean Shift:** A non-parametric clustering method.

### 6. Ensemble Clustering via Majority Voting
Results from the above models are combined using majority voting. Each case is assigned to the cluster where it was most frequently placed across models. This technique enhances the robustness of the clustering.

### 7. Clustering Evaluation
**Silhouette Score** is used to assess the cohesion and separation of the clusters. The model with the highest silhouette score is considered the best performing.

### 8. Fraud Detection
- Clusters with fewer than 20 cases are flagged as potentially fraudulent.
- **Isolation Forest** is used to detect anomalies within the dataset, helping identify individual cases that stand out based on their features.

## 🛠️ Usage Instructions

### 1. Install Dependencies
To run the project, first install the required libraries by running the following commands:


`pip install spacy pandas tqdm scikit-learn matplotlib numpy`
`python -m spacy download en_core_web_sm`

### 2. Run the Main Script


To process the data and generate the clustering and fraud detection results, execute:


1. **Clone the Repository**
  
   git clone https://github.com/YTheLynch/HackRx-5.0.git
   `cd HackRx-5.0`
   
2. `pip install -r requirements.txt`

3. `python app.py`

### 3. Results

Once the script runs, the following files will be generated in the `results/` directory:

- `best_model_results.csv`: The output of the best-performing clustering model.
- `final_fraud_cases.csv`: The list of cases flagged as potential fraud.

### 4. Visualizations

The script also produces a PCA scatter plot to visualize the clustering results in two dimensions. This plot provides insights into how cases are grouped and highlights potential outliers.

## 📈 Evaluation Metrics

The effectiveness of the clustering algorithms is assessed using the following metrics:

- **Silhouette Score:** Ranges between -1 and 1. A higher score indicates better-defined clusters with good separation.
- **Support:** Proportion of cases belonging to a particular cluster.
- **Confidence:** Ratio of data points assigned to a cluster relative to the total number of data points.

## 📊 Model Results

The following silhouette scores were obtained for different clustering models:

| **Clustering Model**     | **Silhouette Score** |
|--------------------------|----------------------|
| KMeans                   | 0.2947               |
| DBSCAN                   | -0.3015              |
| Agglomerative            | 0.2758               |
| Spectral Clustering      | 0.6787               |
| Gaussian Mixture         | 0.0929               |
| Mean Shift               | 0.2534               |

The **Spectral Clustering** model achieved the best performance.

## 📌 Conclusion

This project demonstrates a robust approach to detecting potential fraud in insurance litigation using clustering and anomaly detection techniques. The use of ensemble clustering and word embeddings provides a comprehensive understanding of the dataset and highlights potentially fraudulent cases effectively.

## 🔮 Future Work

- Incorporate additional states and more recent data for better generalization.
- Explore deep learning techniques for improved word embeddings.
- Implement a more advanced ensemble method for clustering.

## 📝 License

This project is licensed under the [MIT License](LICENSE).
