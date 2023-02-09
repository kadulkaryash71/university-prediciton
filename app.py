import numpy as np
import pickle
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 5)
    loaded_model = pickle.load(open("random_forest_regression_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]



@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method == 'POST':
        to_predict_list = request.form.values()
        to_predict_list = list(to_predict_list)
        to_predict_list = list(map(int, to_predict_list[:3])) + list(map(float, to_predict_list[3:6])) + [int(to_predict_list[6])]
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list[:3] + to_predict_list[5:])*100

        # answer in percentage
        # if result >= 80:
        #     prediction =f'Congratulations! You are <b>{result:.2f}%</b> compatible with this University. We found these career counsellors nearest to you ↓'
        # else:
        #     prediction =f'Sorry, but it seems you need to have a better profile and/or good scores to get into these universities rated {to_predict_list[2]}.'
        if result:
            prediction = "Congratulations! You might have a shot at your dream university. Go forward and contact your nearest counsellor now."
        else:
            prediction = f"Sorry, the universities rated {to_predict_list[2]} or above may be a bit difficult for you. University recommendation tool is under development. Please do not lose hope."
        

        return render_template("result.html", prediction = prediction, result = round(result, 2))
    return render_template("index.html")


# Chatbot here
GOOGLE_APPLICATION_CRED = "static/keys/universityrec-9b0fbe850670.json"

@app.route('/chatbot' ,methods=["GET", "POST"])
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
