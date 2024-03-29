# WORKS WITH PYTHON 3.9
from bson import json_util
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


# Asynchronous function to prepare the text for summary
async def prepare_text(q_and_a_list):
    print('q_and_a_list: ', q_and_a_list)
    text_for_summary = ""
    for pair in q_and_a_list:
        english_pair = await translate_to_english(pair)
        text_for_summary += english_pair + ". "
    print("text for summary:", text_for_summary)
    return text_for_summary


@app.route('/make_summary', methods=['POST'])
async def make_summary():
    try:
        request_data = request.get_json()
        user = request_data['user']

        # list of q and a strings
        q_and_a_list_to_translate = []
        for answer_data in request_data['answers']:
            string_to_translate = f"DOCTOR: {answer_data['question']} SUBJECT: {answer_data['user_answer']}"
            q_and_a_list_to_translate.append(string_to_translate)

        # String for Spark NLP to summarize
        string_to_summarize = await prepare_text(q_and_a_list_to_translate)

        # Summarize the text from all the question answer pairs
        summary = summarize (pipeline, string_to_summarize)
        print('summary:\n', summary)

        # Translate the summary back to finnish
        finnish_summary = await translate_to_finnish(summary)
        print('finnish:\n', finnish_summary)

        database['summaries'].insert_one({'user': user, 'summary': finnish_summary})

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

    app.run(debug=True, port=5002)