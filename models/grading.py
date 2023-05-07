from keras.models import Sequential, load_model, model_from_config
from keras.layers import Embedding, LSTM, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import cohen_kappa_score, accuracy_score
import pickle

df = pd.read_csv('essay_training.tsv', sep='\t', encoding='ISO-8859-1')
df.dropna(axis=1,inplace=True)
df.drop(columns=['rater1_domain1','rater2_domain1'],inplace=True,axis=1)

y = df['domain1_score']
df.drop('domain1_score',inplace=True,axis=1)
X=df

tokenizer = Tokenizer(num_words=500, oov_token="<OOV>") # reducing the vocabulary size can increase the performance significantly 

# Step 1: get essays column into an array
essays = df["essay"].tolist()

# Step 3: make vectors of these sentences
tokenizer.fit_on_texts(essays)
word_index = tokenizer.word_index

def getTokenizer():
    return tokenizer

# Step 4: sequencing these vectors
essays_to_vectors = tokenizer.texts_to_sequences(essays)

# Step 5: Making the vectors of equal length before passing to the ML model
padded_vectors = pad_sequences(essays_to_vectors, maxlen=800) # use maxlen parameter to limit the size of the vector

# Step 6: generate the LSTM training model and prepare the vectors to be passed to the model (3D).
def get_model():
    model = Sequential()
    model.add(LSTM(800, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 800], return_sequences=True)) # change 1065 to a smaller, more reasonable number
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae', 'accuracy'])
    model.summary()
    return model

training_vectors = np.array(padded_vectors)

# Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
training_vectors = np.reshape(training_vectors, (training_vectors.shape[0], 1, training_vectors.shape[1]))
lstm_model = get_model()

lstm_model.fit(training_vectors, y, batch_size=64, epochs=12)

test_df = pd.read_excel('valid_set.xlsx')
test_score = pd.read_csv('valid_sample_submission_2_column.csv')
test_df = test_df.iloc[:500, 2]
test_score = test_score.iloc[:500, 1]

test_df_merged = pd.DataFrame({"essay": test_df, "score": test_score})

test_essays = test_df_merged["essay"].tolist()
test_vectors = tokenizer.texts_to_sequences(test_essays)
test_vectors_padded = pad_sequences(test_vectors, maxlen=800)

# Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
testing_vectors = np.array(test_vectors_padded)
testing_vectors = np.reshape(testing_vectors, (testing_vectors.shape[0], 1, testing_vectors.shape[1]))

y_pred = lstm_model.predict(testing_vectors)
y_pred = np.around(y_pred)

print(accuracy_score(test_df_merged["score"], y_pred))
print(cohen_kappa_score(y_pred, test_df_merged["score"]))
pickle.dump(lstm_model, open("grading_ml.pkl", "wb"))
pickle.dump(tokenizer, open("word_tokenizer.pkl", "wb"))
# lstm_model.save("final_lstm.h5")