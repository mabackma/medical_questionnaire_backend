from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from pymongo import MongoClient

load_dotenv()
mongodb_key = os.getenv('MONGODB_KEY')
client = MongoClient(mongodb_key)
database = client['questionnaire']

app = Flask(__name__)
CORS(app)

@app.route('/questionnaire', methods=['GET'])
def get_questions():
    try:
        questions = database['questions'].find()

        # iterate over each document in cursor and create a list
        questions_data = [question for question in questions]
        for question in questions_data:
            question['_id'] = str(question['_id'])
        return jsonify(questions_data)

    except Exception as e:
        return f'Error: {e}'


@app.route('/answers', methods=['POST'])
def save_answers():
    try:
        request_data = request.get_json()

        for answer_data in request_data:
            # save answer to database with question id in objectId format
            database['answers'].insert_one({
                'question_id': ObjectId(answer_data['question_id']),
                'answer': answer_data['answer']
            })
            print("inserted answers: ", request_data)

        # Return a success message
        return jsonify({"message": "Answers saved successfully"}), 200
    except Exception as e:
        return f'Error: {e}'

if __name__ == "__main__":

    app.run(debug=True, port=5001)