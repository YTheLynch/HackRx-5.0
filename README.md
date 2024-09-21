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

   

  

