from flask import Flask, render_template, request, redirect
import pytesseract
from PIL import Image
import fitz # PyMuPDF
import io
import spacy
import joblib
import pandas as pd
import numpy as np

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
        return render_template('index.html', result='Fraud' if result[0] else 'Not Fraud')

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
