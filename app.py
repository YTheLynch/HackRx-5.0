from flask import Flask, render_template, request, redirect
from utils import *

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
