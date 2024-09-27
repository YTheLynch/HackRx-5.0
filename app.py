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

@app.route('/search')
def showsearchpage():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search_query']
    df1 = pd.read_csv('./Datasets/merged_delhi_final_1000.csv')
    df2 = pd.read_csv('./Datasets/merged_guhati_final_1200.xls')
    df3 = pd.read_csv('./Datasets/merged_karnataka_final_1048.xls')
    df4 = pd.read_csv('./Datasets/merged_madras_final_1200.csv')
    df5 = pd.read_csv('./Datasets/merged_punjb_haryana_final_1000.csv')

    df = pd.concat([df1, df2, df3, df4, df5], axis=0)
    dfsel = df[['title', 'headline', 'new_link', 'Case Details']]

    result = dfsel[dfsel['title'].str.contains(search_query, case=False, na=False)]
    

    resultdict = result.to_dict(orient='records')

    if resultdict:
        return render_template('search.html', results=resultdict, no_results=False)
    else:
        return render_template('search.html', results=None, no_results=True)

if __name__ == '__main__':
    app.run(debug=True)
