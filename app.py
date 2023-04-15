import numpy as np
import pickle
from flask import Flask, jsonify, redirect, render_template, request
from flask_cors import CORS, cross_origin
import json
app = Flask(__name__)
CORS(app)

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
    msg_received = request.get_json()
    essay_document = msg_received["document"]
    # request-response works great!
    return jsonify({"result": essay_document + " cap/no-cap"})


@app.route('/chances', methods=["POST"])
@cross_origin(origins="*")
def chances():
    if request.method == 'POST':

        # Fields: GRE score, English test, Test score, University Code, Intake
        # Values:- GRE Score <=340, English test (TOEFL/IELTS), English Score (<= 120 / <= 9), University Name, Intake (fall, spring, winter, any)
        to_predict_list = request.get_json().values()
        to_predict_list = list(to_predict_list)
        print(to_predict_list)
        if to_predict_list[1] == '0':
            to_predict_list[1] = 92
        else:
            to_predict_list[1] = to_predict_list[2]
            to_predict_list[2] = 7.0

        print(to_predict_list)
        
        to_predict_list = list(map(int, to_predict_list[:2])) + to_predict_list[2:]
        to_predict_list = to_predict_list[:2] + [float(to_predict_list[2])] + [int(to_predict_list[3])] + list(map(int, list(to_predict_list[4])))
        print(to_predict_list) # output: [322, 0, 8.0, 256, 1, 0, 0, 0, 0]

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
            prediction = f"Sorry, the universities rated {to_predict_list[2]} or above may be a bit difficult for you. University recommendation tool is under development. Please do not lose hope."

        return jsonify({"prediction": prediction})
    else:
        return jsonify({"result": "something went wrong"})


# Chatbot here
@app.route('/chatbot', methods=["GET", "POST"])
def chatbot():
    # data = request.get_json(silent=True)
    # if data['queryResult']['queryText'] == 'yes':
    #     reply = {
    #         "fulfillmentText": "Ok. Tickets booked successfully.",
    #     }
    #     return jsonify(reply)

    # elif data['queryResult']['queryText'] == 'no':
    #     reply = {
    #         "fulfillmentText": "Ok. Booking cancelled.",
    #     }
    #     return jsonify(reply)
    return render_template("chatbot.html")


if __name__ == '__main__':
    app.run(debug=True)
