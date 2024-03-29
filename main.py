# WORKS WITH PYTHON 3.8
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from pymongo import MongoClient
import whisper
import torch

load_dotenv()
mongodb_key = os.getenv('MONGODB_KEY')
client = MongoClient(mongodb_key)
database = client['questionnaire']

app = Flask(__name__)
CORS(app)

# temporary folder to store uploaded file
UPLOAD_FOLDER = 'C:/temp_whisper_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Whisper model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
model = whisper.load_model('large').to(device)


# Whisper translates speech to text from file
@app.route('/stt', methods=['POST'])
def speech_to_text_api():
    try:
        if 'speech' not in request.files:
            return 'No file part'

        audio_file = request.files['speech']
        if audio_file.filename == '':
            return 'No selected file'

        # Save the uploaded file with a unique filename
        unique_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_audio.wav')
        audio_file.save(unique_filename)

        print("perfoming speech-to-text...")
        out = model.transcribe(unique_filename, language="finnish", fp16=False)
        print(out['text'])

        # Return the text from speech
        return {"user_answer": out['text']}

    except Exception as e:
        return f'Error: {e}'


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

        answers_list = []
        for answer_data in request_data['answers']:
            answers_list.append({
                'question_id': ObjectId(answer_data['question_id']),
                'user_answer': answer_data['user_answer'],
            })

        user_answers = {
            'user': request_data['user'],
            'answers': answers_list
        }
        database['answers'].insert_one(user_answers)
        print("inserted answers: ", request_data)

        # Return a success message
        return jsonify({"message": "Answers saved successfully"}), 201
    except Exception as e:
        return f'Error: {e}'


if __name__ == "__main__":

    app.run(debug=True, port=5001)