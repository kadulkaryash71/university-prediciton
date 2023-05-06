# University Prediction
This is the Flask based backend for University Recommendation for Abroad Studies using Machine Learning. This system uses XGBoost for predicting the "Chance of Admit" of a student in any university based on CGPA, GRE Score, and IELTS/TOEFL score. We try our best to present the most accurate results possible. So far we have succeeded is achieveing 81.44%. Neural Networks are not a part of this prototype and thus an area of improvement for the system. The dataset was extracted from Yocket. The final dataset used is "Combineeed_data.xlsx"

Features of this system include:
1. Admit Chance Prediction
2. Essay (SOP/LOR) Grading
3. URPG-Pod - A Community of Students pursuing education abroad

Essay grading is implemented using LSTM, but any alternative approach is encouraged by the developers of this system.
[URPG-POD](https://github.com/kadulkaryash71/UniPredictionCommunity-v1) is a community module of this system. Follow the instructions in that repository to run the full feature. You will need to clone and run the repository separately. It is at a development stage. One can get a clear picture after using the feature as is.

To make a clean folder structure, files of the repository will soon be ogranised into folders. So hang tight if you are having a hard time finding the right files to work on.

## Running the repository

### Install Python dependencies
Python version used during development v3.10.9

```pip install -r requirements.txt```

### Running the server

```python app.py```

Alternatively, you can use:

```flask run app.py```

The server runs on http://localhost:5000/ by default. You will not need to open it. The purpose of this server is to handle HTTP requests, thus it does not have the full frontend. To access the full frontend clone the [UniRecommendationClient repository](https://github.com/kadulkaryash71/UniRecommendationClient "git clone in a separate folder")

### Running the client
The client was developed using ViteJS. Before running the system, run the following command to install frontend dependencies:

```npm install```

To run the client via ViteJS use the following command:

```npm run dev```

The frontend server will run on http://localhost:5173/ by default. After the ViteJS instructions on the terminal type "o" (the letter o on the terminal) to open the server in your default browser on you can copy+paste the the link provided in the terminal.

You will find similar isntructions on the client repository. Though, to run the community module you will need a separate repository. We are working on merging the two repositories to make it more dev-friendly.

In case of any queries feel free to raise issues or create pull requests with proper instructions and updates documented to smoothen the merging process.
