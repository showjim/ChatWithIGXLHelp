from flask import Flask, request, redirect, url_for, render_template
# from qdrant_client import QdrantClient
import openai
import os
# from Chat_With_PDF_Paper_Azure_OpenAI import chat
# from Chat_With_PDF_Paper_Azure_OpenAI_CN import chat
from src.chat import chat
from src.process_pdf import create_index


app = Flask(__name__)

UPLOAD_FOLDER = './uploaded_files/'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FILE_NAME = ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global UPLOAD_FILE_NAME
    uploaded_file = None
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return redirect(request.url)
        file = request.files['pdf']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            UPLOAD_FILE_NAME = file.filename
            uploaded_file = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            # return render_template('index.html', uploaded_file=uploaded_file)#redirect(url_for('index'))
    return render_template('index.html', uploaded_file=uploaded_file)

def query(text):
    index = create_index(UPLOAD_FOLDER, UPLOAD_FILE_NAME)
    if index != False:
        resp = chat(index, text)
    return {
        "answer": resp.response,
        "tags": "tag",
    }

# @app.route('/')
# def hello_world():
#     return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']

    res = query(search)

    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res["answer"],
            "tags": res["tags"],
        },
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
