from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import nltk
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf

app = Flask(__name__)

# Load the model and tokenizer
#model = load_model('lstm_model_2.h5')  # Path to your LSTM model

# Load the model
model = tf.keras.models.load_model(r'c:\Users\HP\OneDrive\Desktop\major\lstm_model_2.h5')


with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)





max_article_len = 100  # Max length of input article
start_token = '_START_'  # Start token used during training
end_token = '_END_'  # End token used during training

def preprocess_text(article):
    article = str(article).lower()
    article = re.sub(r'<.*?>', '', article)
    article = re.sub(r'[^a-zA-Z\s]', '', article)
    article = word_tokenize(article)
    stop_words = set(stopwords.words('english'))
    article = [word for word in article if word not in stop_words]
    stemmer = SnowballStemmer('english')
    # Preprocess the text as needed
    return article

def predict_summary(article):
    # Preprocess the input text
    article = preprocess_text(article)

    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([article])

    # Pad sequences to fixed length
    padded_sequence = pad_sequences(sequence, maxlen=max_article_len, padding='post')

    # Predict summary
    summary = model.predict(padded_sequence)

    # Convert summary from numeric indices to text
    decoded_summary = ' '.join([tokenizer.index_word[i] for i in np.argmax(summary[0], axis=1) if i not in [0, tokenizer.word_index[end_token]]])

    return decoded_summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    text = request.form['text']
    summary = predict_summary(text)
    return render_template('result.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
