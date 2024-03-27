# WORKS WITH PYTHON 3.9
import sparknlp
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

app = Flask(__name__)
CORS(app)

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


@app.route('/get_summary', methods=['POST'])
async def get_summary():
    user = request.json['user']

    questions = list(database['questions'].find())
    answers = list(database['answers'].find({'user': user}))

    # List of strings for question and answer pairs
    q_and_a_list = []

    for question in questions:
        question_id = str(question['_id'])
        for answer_obj in answers:
            for ans in answer_obj['answers']:
                if str(ans['question_id']) == question_id:
                    q_and_a_list.append(f"DOCTOR: {question['question']} PATIENT: {ans['user_answer']}")

    # Prepare a text input for the summary
    text_for_summary = await prepare_text(q_and_a_list)
    print(text_for_summary)

    # summarize the text
    summary = summarize(pipeline, text_for_summary)
    return jsonify({'summary': summary})


def summarize(pipeline, input_text):
    text = pipeline.fullAnnotate(input_text)

    # Get text as a list
    text = str(text[0]['summary'][0])
    text = text.split(',')[3: -2]

    # Turn the text back into a string. Remove brackets and quotes
    text = (str(text)).replace('[', '').replace(']', '')
    text = text.replace("'", '').replace('"', '').replace('result=', '')

    return text


# Asynchronous function to prepare the text for summary
async def prepare_text(q_and_a_list):
    text_for_summary = ""
    for pair in q_and_a_list:
        english_pair = await translate_to_english(pair)
        text_for_summary += english_pair + ". "
    return text_for_summary


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


if __name__ == "__main__":

    app.run(debug=True, port=5002)