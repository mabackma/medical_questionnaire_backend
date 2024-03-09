from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from pymongo import MongoClient
import whisper

load_dotenv()
mongodb_key = os.getenv('MONGODB_KEY')
client = MongoClient(mongodb_key)
database = client['questionnaire']

app = Flask(__name__)
CORS(app)

# temporary folder to store uploaded file
UPLOAD_FOLDER = 'C:/temp_whisper_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

        model = whisper.load_model("base")

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(unique_filename)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"filename: {audio_file.filename}")
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        # return the recognized text
        print(result.text)
        return {"user_answer": result.text}

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

        for answer_data in request_data:
            # save answer to database with question id in objectId format
            database['answers'].insert_one({
                'question_id': ObjectId(answer_data['question_id']),
                'user_answer': answer_data['user_answer'],
            })
        print("inserted answers: ", request_data)

        # Return a success message
        return jsonify({"message": "Answers saved successfully"}), 201
    except Exception as e:
        return f'Error: {e}'


if __name__ == "__main__":

    app.run(debug=True, port=5001)