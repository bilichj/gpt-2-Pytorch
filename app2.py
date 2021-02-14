import json
from flask import Flask, request
from app import generate

app = Flask(__name__)

@app.route('/gpt-2', methods = ['POST'])
def generate_samples():
    if request.method == 'POST':
        data = request.get_json()
        response = generate(**data)
        # response = data
        return json.dumps(response)