import torch
from transformers import BertModel, BertTokenizer
from emotion_classifier import *
from emotion_classifier.EMOClassifer import emoClassifer
import os
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from flask import Flask, redirect, url_for, request,jsonify
import numpy as np
import pandas as pd
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


device = torch.device("cpu")
BASE_DIR = Path(__file__).resolve().parent
path_epoch20 = os.path.join(BASE_DIR, 'model', 'epoch20.pth')
path_kcbert = os.path.join(BASE_DIR, 'model', 'kcbert_tokenizer.pth')
model = torch.load(path_epoch20, map_location=device)
tokenizer = torch.load(path_kcbert, map_location=device)


def Build_X (sents, tokenizer, device):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1).to(device)

def predict(DATA):
    device = torch.device('cpu')
    # bertmodel = BertModel.from_pretrained("beomi/kcbert-base")
    # tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    # bertmodel = torch.load('kcbertmodel.pth', map_location=device)

    # model.eval()
    X = Build_X(DATA, tokenizer, device)
    print(X)
    y_hat = model.predict(X)
    y_hat = F.softmax(y_hat, dim=1)

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print(y_hat.detach().numpy() * 100)
    result = list(map(float, (y_hat.detach().numpy() * 100)[0]  ) )

    return result


@app.route('/lyrics_emotion', methods=['POST'])
def hello_world():
    if request.method == 'POST':
        lyrics = request.form['lyrics']
        result = predict(lyrics)
        return jsonify({"emotion_score" : result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug = True)
