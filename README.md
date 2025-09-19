# Medical summary back end

This back end receives a number of answers from multiple choice questions and voice answers in audio files. 

Uses Open AI's Whisper speech-to-text functionality running on an RTX 4070 Ti Super to convert the audio files into text answers.

All the answers are then saved to a MongoDb database.

Makes a summary from all the answers and sends the summary back to the front end.

This project was initially using a Spark NLP medical sumamarizer from [John Snow Labs](https://nlp.johnsnowlabs.com/medical_text_summarization) to make a summary of the answers.

The summarizer was later changed to using Chat-GPT API. Also translations are made with Chat-GPT.

[Flutter front end](https://github.com/mabackma/medical_questionnaire)

[Video](https://www.youtube.com/watch?v=LpVu0fCTQkw)
