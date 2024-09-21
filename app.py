from flask import Flask, render_template, request, redirect
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

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
        return render_template('index.html', extracted_text=extracted_text)

    return redirect('/')

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

if __name__ == '__main__':
    app.run(debug=True)
