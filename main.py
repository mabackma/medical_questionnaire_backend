from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
import os
from pymongo import MongoClient

load_dotenv()
mongodb_key = os.getenv('MONGODB_KEY')
client = MongoClient(mongodb_key)

app = Flask(__name__)
CORS(app)

@app.route('/questionnaire', methods=['GET'])
def get_questions():
    try:
        # exclude id objects in the query
        questions = client['questionnaire']['questions'].find({}, {"_id": 0})
        return jsonify(list(questions))

    except Exception as e:
        return f'Error: {e}'

'''''
@app.route('/save_to_database', methods=['POST'])
def save_answers():
    try:


    except Exception as e:
        return f'Error: {e}'
'''''
if __name__ == "__main__":

    app.run(debug=True, port=5001)