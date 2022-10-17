import numpy as np
import pickle
from flask import Flask, render_template, request
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
        print(to_predict_list)
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list[:3] + to_predict_list[5:])
        if int(result)== 1:
            prediction ='Looks like you have a shot at this university. We found these career counsellors nearest to you ↓'
        else:
            prediction =f'Sorry, but it seems you need to have a better profile and/or good scores to get into these universities rated {to_predict_list[2]}'
        print(prediction)        
        return render_template("result .html", prediction = prediction)
    return render_template("index.html")

    

if __name__ == '__main__':
    app.run(debug=True)
