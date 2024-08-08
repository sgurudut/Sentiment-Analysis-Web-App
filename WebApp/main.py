# importing the required dependencies
from flask import Flask, render_template, request

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

# code to make ML models work on pythonanwhere
import torch
torch.set_num_threads(1)

# creating the roberta model and the tokenizer
task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# creating the flask app

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])

def predict():

    # function to get the scores on the text
    def polarity_scores_roberta(example):
        encoded_text = tokenizer(example, return_tensors = 'pt' )
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        score = softmax(scores)
        scores_dict = {'roberta_neg' : scores [0], 'roberta_neu' : scores [1], 'roberta_pos' : scores [2]}
        return scores_dict

    # getting the text input from the user
    inp = request.form.get('inp')

    # creating an empty dictionary and fetching the results of the roberta model
    res = {}
    res = polarity_scores_roberta(inp)

    # getting the key for the max values in the dictionary
    max_key = max(res, key=res.get)


    # logic for deciding the emotion
    if max_key == 'roberta_neg':
        message = "Negative🙁🙁"
    elif max_key == 'roberta_neu':
        message = "Neutral😐😐"
    else:
        message = "Positive🙂🙂"

    return render_template('index.html', prediction = message)

if __name__ == '__main__':
    app.run(port=4001, debug=False)
