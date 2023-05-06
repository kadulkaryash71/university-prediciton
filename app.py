import numpy as np
import pickle
from flask import Flask, jsonify, redirect, render_template, request
from flask_cors import CORS, cross_origin
import json
import tensorflow
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)


def text2Vector(text):
    tokenizer = pickle.load(open("word_tokenizer.pkl", "rb"))
    essay_vector = tokenizer.texts_to_sequences(text)
    essay_vector_padded = pad_sequences(essay_vector, maxlen=800)
    vector = np.reshape(essay_vector_padded, (essay_vector_padded.shape[0], 1, essay_vector_padded.shape[1]))
    return vector

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 9)
    loaded_model = pickle.load(
        open("predict_xgboost.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/data/unis", methods=["GET"])
def get_unis():
    loaded_data = json.load(open("unis_code.json", "r"))
    return jsonify({"universities": loaded_data})


@app.route('/test', methods=["GET", "POST"])
@cross_origin(origins="*")
def tester():
    msg_received = request.get_json()
    print(type(request.data), msg_received)
    return jsonify({"body": "Message returned successfully!"})


@app.route('/grading', methods=["POST"])
@cross_origin(origins="*")
def docGrade():
    essay = request.get_json()["document"]
    essay = list([essay])
    model = pickle.load(open("grading_ml.pkl", "rb"))
    vector = text2Vector(essay)
    result = model.predict(vector)
    result = np.round(result[0][0])
    return jsonify({"result": str(result)})


@app.route('/chances', methods=["POST"])
@cross_origin(origins="*")
def chances():  
    if request.method == 'POST':

        # Fields: GRE score, English test, Test score, University Code, Intake
        # Values:- GRE Score <=340, English test (TOEFL/IELTS), English Score (<= 120 / <= 9), University Name, Intake (fall, spring, winter, any)
        to_predict_list = request.get_json().values()
        to_predict_list = list(to_predict_list)
        print(to_predict_list)
        if int(to_predict_list[1]) == 0:
            to_predict_list[1] = 92
        else:
            to_predict_list[1] = to_predict_list[2]
            to_predict_list[2] = 7.0
        
        intake = list(to_predict_list[4])
        to_predict_list = to_predict_list[:4] + intake
        print(to_predict_list)
        to_predict_list = list(map(int, to_predict_list[:2])) + [float(to_predict_list[2])] + list(map(int, to_predict_list[3:]))

        print(to_predict_list)  # output: [322, 92, 8.0, 256, 1, 0, 0, 0, 0]

        # sample: [320, 111, 7.0, 54, 1, 0, 0, 0]
        result = ValuePredictor(to_predict_list)
        print(result)

        # answer in percentage
        # if result >= 80:
        #     prediction = f'Congratulations! You are {result:.2f}% compatible with this University. We found these career counsellors nearest to you ↓'
        # else:
        #     prediction = f'Sorry, but it seems you need to have a better profile and/or good scores to get into these universities rated {to_predict_list[2]}.'

        # answer in boolean
        if result:
            prediction = "Congratulations! You might have a shot at your dream university. Go forward and contact your nearest counsellor now."
        else:
            prediction = f"Sorry, the university you are wishing for may be a bit difficult for you. University recommendation tool is under development. Please do not lose hope."

        return jsonify({"prediction": prediction})
    else:
        return jsonify({"result": "something went wrong"})


# Chatbot here
@app.route('/chatbot', methods=["GET", "POST"])
def chatbot():
    return render_template("chatbot.html")


if __name__ == '__main__':
    app.run(debug=True)
