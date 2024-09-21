from flask import Flask, render_template, request, redirect
import pytesseract
from PIL import Image
import fitz # PyMuPDF
import io
import spacy
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


nlp=spacy.load("en_core_web_sm")
nlp.max_length=2000000

def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    extracted_text = ""

    
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)  
        pix = page.get_pixmap()  
        img = Image.open(io.BytesIO(pix.tobytes()))  
        page_text = pytesseract.image_to_string(img)  
        extracted_text += page_text + "\n\n"
    
    pdf_document.close()
    return extracted_text

def preprocess(text):
    doc=nlp(text)
    li=[]
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        else:
         li.append(token.lemma_)
    return " ".join(li)

def word_embedding(text):
    doc=nlp(text)
    return doc.vector

with open('./model', 'rb') as f:
   model = joblib.load(f)

with open('./scaler', 'rb') as f:
   scaler = joblib.load(f)

def alert(extracted_text):
    # Your email credentials
    sender_email = "siddharth2004awasthi@gmail.com"  # Replace with your Gmail address
    load_dotenv()
    app_password = os.getenv("APP_PASSWORD") 

    # Create a multi-part message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = "ishitsetia@gmail.com"
    msg['Subject'] = "Fraud Case"

    # Attach the body with the msg instance
    msg.attach(MIMEText(f"Fraud case reported\nCase Details: {extracted_text}", 'plain'))

    # Sending the email
    try:
        # Create a secure connection with the server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # For Gmail SMTP
        server.starttls()  # Secure the connection
        server.login(sender_email, app_password)  # Login to the email server
        server.sendmail(sender_email, "ishitsetia@gmail.com", msg.as_string())  # Send email
        server.quit()

        print(f"Email sent successfully to {"ishitsetia@gmail.com"}.")
    
    except Exception as e:
        print(f"Failed to send email. Error: {str(e)}")




app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    
    if file and file.filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file)
        print(extracted_text)
        preprocessed = preprocess(extracted_text)
        embedding = word_embedding(preprocessed)

        scaled = scaler.transform([embedding])
        result = model.predict(scaled)
        if result[0] == 1:
            alert(extracted_text)
        return render_template('index.html', result='Fraud' if result[0] else 'Not Fraud')

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
