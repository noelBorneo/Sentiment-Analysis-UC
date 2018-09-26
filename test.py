from flask import Flask, request, render_template, jsonify
import requests

app = Flask(__name__)

# Source of Sentiment Analysis model
url = 'http://127.0.0.1:5000/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def form_post():
    text = request.form['text']
    params ={'query': text}
    response = requests.get(url, params)
    json = response.json()
    print(json.items())
    return jsonify(json)

if __name__ == '__main__':
    app.run(host='localhost', port=12345)
