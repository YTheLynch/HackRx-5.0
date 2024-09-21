# HackRx-5.0
Legal AI Analytics and Alarm System : VerdictVision
# Overview:
This project was developed for the HackRx 5.0 Hackathon, focusing on building an AI-driven analytics and alarm system using data sourced from open court orders and pending petitions across India. The goal is to identify patterns in litigation within the insurance sector, particularly in detecting fraud cases. By leveraging ensemble unsupervised learning and data analytics, we developed a robust system that scrapes legal datasets, trains a fraud detection model, and triggers alerts via an integrated email alarm system when potential fraud cases are identified.
# Key Features->
## Data Scraping and Aggregation:
Datasets were scraped from various sources such as eCourts India and Indian Kanoon using a custom web scraper.
Focused on pending and disposed litigation in the insurance sector, compiling data from 5 Indian states, with 1,000 records per state.

## Fraud Detection Model:
- Before model training, we preprocessed the data by:
- Removing stop words to eliminate irrelevant terms.
- Lemmatizing to reduce words to their base or root form.
- Using word embeddings to vectorize the input case details, allowing the model to understand semantic meanings.
- Multiple unsupervised learning algorithms were combined, and the best-performing algorithm was selected for fraud detection.
Our AI model uses ensemble unsupervised learning to detect fraud patterns in the insurance sector. Multiple unsupervised learning algorithms were combined, and the best-performing algorithm was selected for fraud detection.
The model identifies fraud cases based on learned patterns and triggers alarms for high-risk cases.

## Alarm Mechanism:
Upon detecting a potential fraud case, the system sends an email alert using information from court websites to the appropriate registrar’s office or contact person.
Most court websites have a “Contact Us” or “Registrar’s Office” section where email addresses and phone numbers are listed, making it easier to send direct and immediate alerts.

### Project Structure:

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

  ## Installation Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YTheLynch/HackRx-5.0.git
   cd HackRx-5.0
   
2. pip install -r requirements.txt

3. python app.py


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Litigation Fraud Detection via Clustering and Anomaly Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        h1 {
            text-align: center;
        }
        h3 {
            margin-top: 20px;
        }
        code {
            background-color: #eef;
            padding: 2px 4px;
            border-radius: 4px;
        }
        pre {
            background-color: #eef;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        footer {
            margin-top: 20px;
            text-align: center;
            font-size: 0.9em;
            color: #777;
        }
    </style>
</head>
<body>

    <h1>Insurance Litigation Fraud Detection via Clustering and Anomaly Detection</h1>

    <hr>

    <h2>Table of Contents</h2>
    <ol>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
        <li><a href="#dataset-description">Dataset Description</a></li>
        <li><a href="#project-workflow">Project Workflow</a>
            <ol>
                <li><a href="#1-data-scraping-and-aggregation">Data Scraping and Aggregation</a></li>
                <li><a href="#2-preprocessing">Preprocessing</a></li>
                <li><a href="#3-word-embedding-generation">Word Embedding Generation</a></li>
                <li><a href="#4-dimensionality-reduction-using-pca">Dimensionality Reduction using PCA</a></li>
                <li><a href="#5-clustering-algorithms">Clustering Algorithms</a></li>
                <li><a href="#6-ensemble-clustering-via-majority-voting">Ensemble Clustering via Majority Voting</a></li>
                <li><a href="#7-clustering-evaluation">Clustering Evaluation</a></li>
                <li><a href="#8-fraud-detection">Fraud Detection</a></li>
            </ol>
        </li>
        <li><a href="#usage-instructions">Usage Instructions</a>
            <ol>
                <li><a href="#1-install-dependencies">Install Dependencies</a></li>
                <li><a href="#2-run-the-main-script">Run the Main Script</a></li>
                <li><a href="#3-results">Results</a></li>
                <li><a href="#4-visualizations">Visualizations</a></li>
            </ol>
        </li>
        <li><a href="#evaluation-metrics">Evaluation Metrics</a></li>
        <li><a href="#model-results">Model Results</a></li>
        <li><a href="#potential-fraud-detection">Potential Fraud Detection</a></li>
        <li><a href="#conclusion">Conclusion</a></li>
        <li><a href="#future-work">Future Work</a></li>
        <li><a href="#license">License</a></li>
    </ol>

    <hr>

    <h2 id="overview">Overview</h2>
    <p>This project focuses on detecting potential fraud in insurance litigation by utilizing advanced machine learning techniques. By employing word embeddings and clustering methods, we identify outliers and unusual patterns in legal case data, flagging cases that may involve fraudulent activities. The dataset is derived from open court records across different states, and we use semantic clustering to group cases and anomaly detection to identify outliers.</p>

    <hr>

    <h2 id="project-structure">Project Structure</h2>
    <pre>
.
├── data/
│   └── Merged_all_states_data.csv          # Raw dataset with litigation case details
├── src/
│   └── fraud_detection.py                  # Main Python script for processing and modeling
├── models/
│   └── word_embeddings_all_states.csv      # Word embeddings generated from the text data
├── results/
│   ├── best_model_results.csv              # Clustering results from the best model
│   └── final_fraud_cases.csv               # Cases identified as potential frauds
└── README.md                               # Project documentation (this file)
    </pre>

    <hr>

    <h2 id="dataset-description">Dataset Description</h2>
    <p>The primary dataset <code>Merged_all_states_data.csv</code> consists of the following key columns:</p>
    <table>
        <tr>
            <th>Column Name</th>
            <th>Description</th>
        </tr>
        <tr>
            <td><code>Unnamed: 0</code></td>
            <td>Index column</td>
        </tr>
        <tr>
            <td><code>title</code></td>
            <td>Title of the litigation case</td>
        </tr>
        <tr>
            <td><code>headline</code></td>
            <td>Case headline providing a brief summary</td>
        </tr>
        <tr>
            <td><code>detail_id</code></td>
            <td>Unique identifier for the case</td>
        </tr>
        <tr>
            <td><code>new_link</code></td>
            <td>Link to the full details of the case</td>
        </tr>
        <tr>
            <td><code>Case Details</code></td>
            <td>Detailed text of the case, including facts, arguments, and judgments</td>
        </tr>
    </table>

    <hr>

    <h2 id="project-workflow">Project Workflow</h2>
    
    <h3 id="1-data-scraping-and-aggregation">1. Data Scraping and Aggregation</h3>
    <p>The dataset was compiled using a custom web scraper that extracted data from platforms like eCourts India and Indian Kanoon. The scraper focused on pending and disposed litigation in the insurance sector, gathering data from five Indian states, with 1,000 records collected per state.</p>

    <h3 id="2-preprocessing">2. Preprocessing</h3>
    <p>The text in the <code>Case Details</code> column is cleaned through tokenization, lemmatization, and removal of stop words and punctuation using <code>spaCy</code>.</p>

    <h3 id="3-word-embedding-generation">3. Word Embedding Generation</h3>
    <p>Using <code>spaCy</code>'s pre-trained <code>en_core_web_sm</code> model, the textual data is converted into word embeddings, which are vector representations capturing semantic meaning.</p>

    <h3 id="4-dimensionality-reduction-using-pca">4. Dimensionality Reduction using PCA</h3>
    <p><strong>Principal Component Analysis (PCA)</strong> is employed to reduce the dimensionality of the word embeddings from 300 dimensions to 50. This step helps streamline the input for clustering.</p>

    <h3 id="5-clustering-algorithms">5. Clustering Algorithms</h3>
    <p>Multiple clustering techniques are applied to group cases based on their semantic similarity:</p>
    <ul>
        <li><strong>KMeans</strong>: Groups cases into a pre-defined number of clusters using centroids.</li>
        <li><strong>DBSCAN</strong>: A density-based method that identifies core samples and noise.</li>
        <li><strong>Agglomerative Clustering</strong>: A hierarchical approach that builds clusters by merging or splitting.</li>
        <li><strong>Spectral Clustering</strong>: A graph-based technique using nearest neighbors.</li>
        <li><strong>Gaussian Mixture Model (GMM)</strong>: A probabilistic model assuming data points are derived from a mixture of Gaussian distributions.</li>
        <li><strong>Mean Shift</strong>: A non-parametric clustering method.</li>
    </ul>

    <h3 id="6-ensemble-clustering-via-majority-voting">6. Ensemble Clustering via Majority Voting</h3>
    <p>Results from the above models are combined using majority voting. Each case is assigned to the cluster where it was most frequently placed across models. This technique enhances the robustness of the clustering.</p>

    <h3 id="7-clustering-evaluation">7. Clustering Evaluation</h3>
    <p><strong>Silhouette Score</strong> is used to assess the cohesion and separation of the clusters. The model with the highest silhouette score is considered the best performing.</p>

    <h3 id="8-fraud-detection">8. Fraud Detection</h3>
    <p>Clusters with fewer than 20 cases are flagged as potentially fraudulent. <strong>Isolation Forest</strong> is used to detect anomalies within the dataset, helping identify individual cases that stand out based on their features.</p>

    <hr>

    <h2 id="usage-instructions">Usage Instructions</h2>
    
    <h3 id="1-install-dependencies">1. Install Dependencies</h3>
    <p>To run the project, first install the required libraries by running the following commands:</p>
    <pre><code>pip install spacy pandas tqdm scikit-learn matplotlib numpy
python -m spacy download en_core_web_sm</code></pre>

    <h3 id="2-run-the-main-script">2. Run the Main Script</h3>
    <p>To process the data and generate the clustering and fraud detection results, execute:</p>
    <pre><code>python src/fraud_detection.py</code></pre>

    <h3 id="3-results">3. Results</h3>
    <p>Once the script runs, the following files will be generated in the <code>results/</code> directory:</p>
    <ul>
        <li><code>best_model_results.csv</code>: The output of the best-performing clustering model.</li>
        <li><code>final_fraud_cases.csv</code>: The list of cases flagged as potential fraud.</li>
    </ul>

    <h3 id="4-visualizations">4. Visualizations</h3>
    <p>The script also produces a PCA scatter plot to visualize the clustering results in two dimensions. This plot provides insights into how cases are grouped and highlights potential outliers.</p>

    <hr>

    <h2 id="evaluation-metrics">Evaluation Metrics</h2>
    <p>The effectiveness of the clustering algorithms is assessed using the following metrics:</p>
    <ul>
        <li><strong>Silhouette Score</strong>: Ranges between -1 and 1. A higher score indicates better-defined clusters with good separation.</li>
        <li><strong>Support</strong>: Proportion of cases belonging to a particular cluster.</li>
        <li><strong>Confidence</strong>: Ratio of data points assigned to a cluster relative to the total number of data points.</li>
    </ul>

    <hr>

    <h2 id="model-results">Model Results</h2>
    <p>The following <strong>silhouette scores</strong> were obtained for different clustering models:</p>
    <table>
        <tr>
            <th>Clustering Model</th>
            <th>Silhouette Score</th>
        </tr>
        <tr>
            <td>KMeans</td>
            <td>0.2947</td>
        </tr>
        <tr>
            <td>DBSCAN</td>
            <td>-0.3015</td>
        </tr>
        <tr>
            <td>Agglomerative</td>
            <td>0.2758</td>
        </tr>
        <tr>
            <td>Spectral Clustering</td>
            <td><strong>0.6787</strong></td>
        </tr>
        <tr>
            <td>Gaussian Mixture</td>
            <td>0.0929</td>
        </tr>
        <tr>
            <td>Mean Shift</td>
            <td>0.2534</td>
        </tr>
    </table>
    <p>The <strong>Spectral Clustering</strong> model achieved the best performance.</p>

    <footer>
        <p>© 2024 Insurance Litigation Fraud Detection Project</p>
    </footer>

</body>
</html>


   

  

