import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

def get_openai_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    return response['choices'][0]['message']['content']
from langchain import ConversationChain

conversation = ConversationChain(openai_model=get_openai_response)

def chat_with_user(user_input):
    bot_response = conversation({"input": user_input})
    return bot_response
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    return sentiment_pipeline(text)[0]  # Returns sentiment label and score
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['input']
    response = chat_with_user(user_input)
    sentiment = analyze_sentiment(user_input)
    return jsonify({"response": response, "sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
