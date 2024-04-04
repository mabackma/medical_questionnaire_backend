import json
from bson import ObjectId
import whisper
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from pymongo import MongoClient
from openai import OpenAI
import sparknlp_jsl
from sparknlp_jsl.annotator import *
from sparknlp.pretrained import PretrainedPipeline

load_dotenv()
mongodb_key = os.getenv('MONGODB_KEY')
client = MongoClient(mongodb_key)
database = client['questionnaire']

# Upload questions to database
if "questions" not in database.list_collection_names():
    with open('questionnaire.json', 'r', encoding='utf-8') as file:
        questionnaire = json.load(file)
    database['questions'].insert_many(questionnaire)

app = Flask(__name__)
CORS(app)

# temporary folder to store uploaded file
UPLOAD_FOLDER = 'C:/temp_whisper_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Whisper model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print("PyTorch version:", torch.__version__)
model = whisper.load_model('large').to(device)

# chat-gpt
client_gpt = OpenAI(api_key=os.getenv("CHAT_GPT_KEY"))

params = {"spark.executor.cores": 12,  # you can change the configs
          "spark.driver.memory": "40G",
          "spark.driver.maxResultSize": "5G",
          "spark.kryoserializer.buffer.max": "2000M",
          "spark.serializer": "org.apache.spark.serializer.KryoSerializer"}

spark = sparknlp_jsl.start(os.environ['SECRET'], params=params)

print("Spark NLP Version :", sparknlp.version())
print("Spark NLP_JSL Version :", sparknlp_jsl.version())

spark.sparkContext.setLogLevel("ERROR")

# download pipeline for summarization
pipeline = PretrainedPipeline("summarizer_clinical_laymen_onnx_pipeline", "en", "clinical/models")


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


# Asynchronous function to prepare the text for summary
async def prepare_text(answer_list):
    print('answer_list: ', answer_list)
    text_for_summary = ""
    for pair in answer_list:
        english_pair = await translate_to_english(pair)
        text_for_summary += english_pair + " "
    print("text for summary:", text_for_summary)
    return text_for_summary


@app.route('/make_summary', methods=['POST'])
async def make_summary():
    try:
        request_data = request.get_json()
        user = request_data['user']

        # list of q and a strings
        answer_list_to_translate = []
        for answer_data in request_data['answers']:
            string_to_translate = f"{user} {answer_data['user_answer']}"
            answer_list_to_translate.append(string_to_translate)

        # String for Spark NLP to summarize
        string_to_summarize = await prepare_text(answer_list_to_translate)

        # Summarize the text from all the question answer pairs
        summary = summarize (pipeline, string_to_summarize)
        print('summary:\n', summary)

        # Translate the summary back to finnish
        finnish_summary = await translate_to_finnish(summary)
        print('finnish:\n', finnish_summary)

        database['summaries'].insert_one({'user': user,
                                          'input_string': " ".join(answer_list_to_translate),
                                          'english_input_string': string_to_summarize,
                                          'english_summary': summary,
                                          'summary': finnish_summary})

        # Return a success message
        return jsonify({"message": "Summary saved successfully"}), 201

    except Exception as e:
        return f'Error: {e}'


def document_to_dict(document):
    # Convert ObjectId to string
    document['_id'] = str(document['_id'])
    # Return the document as a dictionary
    return document


@app.route('/get_summary', methods=['POST'])
async def get_summary():
    user = request.json['user']
    result = database['summaries'].find({'user': user})

    # Convert MongoDB documents to dictionaries
    summaries = [document_to_dict(document) for document in result]

    print('SUMMARIES:', summaries)
    return jsonify({'summaries': summaries})


def summarize(pipeline, input_text):
    text = pipeline.fullAnnotate(input_text)

    # Get text as a list
    text = str(text[0]['summary'][0])
    text = text.split(',')[3: -2]

    # Turn the text back into a string. Remove brackets and quotes
    text = (str(text)).replace('[', '').replace(']', '')
    text = text.replace("'", '').replace('"', '').replace('result=', '')

    return text


# Translates finnish to english
async def translate_to_english(pair):
    prompt = f"Translate the following text into english: {pair}"

    chat_completion = client_gpt.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo"
    )

    reply = chat_completion.choices[0].message.content
    return reply


# Translates english to finnish
async def translate_to_finnish(english_summary):
    prompt = f"Käännä seuraava teksti suomeksi: {english_summary}"

    chat_completion = client_gpt.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo"
    )

    reply = chat_completion.choices[0].message.content
    return reply


if __name__ == "__main__":

    app.run(debug=True, port=5001)