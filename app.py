from flask import Flask, request, render_template, jsonify, url_for
from utils import clean_text
import pickle
import time
import os

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from sklearn.linear_model import SGDClassifier

app = Flask(__name__)

MODEL_VERSION = 'clf.pkl'
TFIDF_VERSION = 'tfidf1.pkl'
LE_VERSION = 'Departure_encoder.pkl'

# load model assets
tfidf_path = os.path.join(os.getcwd(), 'model_assets', TFIDF_VERSION)
model_path = os.path.join(os.getcwd(), 'model_assets', MODEL_VERSION)
le_path = os.path.join(os.getcwd(), 'model_assets', LE_VERSION)
tfidf = pickle.load(open(tfidf_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))
le = pickle.load(open(le_path, 'rb'))

# TODO: add versioning to url
@app.route('/', methods=['GET', 'POST'])
def predict():
    """ Main webpage with user input through form and prediction displayed

    :return: main webpage host, displays prediction if user submitted in text field
    """

    if request.method == 'POST':

        response = request.form['text']
        prediction = predict(response)
        return render_template('index.html', text=prediction, submission=response)

    if request.method == 'GET':
        return render_template('index.html')

# TODO: add versioning to api
@app.route('/predict', methods=['POST'])
def predict_api():
    """ endpoint for model queries (non gui)

    :return: json, model prediction and response time
    """
    start_time = time.time()

    request_data = request.json
    input_text = request_data['data']
    prediction = predict(input_text)

    response = {'prediction': prediction, 'response_time': time.time() - start_time}
    return jsonify(response)

def predict(job_dsr):
    X = [job_dsr]
    X=[re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in X]
    X=[re.sub("[0-9]+",' ',k) for k in X]

    #applying stemmer
    ps =PorterStemmer()
    X=[ps.stem(k) for k in X]

    X=tfidf.transform(X)

    pred = model.predict(X)
    predicted_class = le.inverse_transform(pred)
    return predicted_class[0]

if __name__ == '__main__':
    app.run(debug=True)
