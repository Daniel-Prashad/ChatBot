from Chat.chatting import chat
from Chat.training import clean_data, create_training_data, fit_model

import json
import pickle
import os

if __name__ == '__main__':
    # open the intents file
    with open("intents.json") as file:
        data = json.load(file)
    # either load in the training data if it has been previously created and saved
    # or clean and create the training data if it has not
    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words, labels, docs_x, docs_y = clean_data(data)
        training, output = create_training_data(words, labels, docs_x, docs_y)
    # fit the model using the data above
    model = fit_model(training, output)
    # provide the user with some instructions and allow them to start chatting with the chatbot
    os.system('cls')
    print("Start talking with the bot! (Type 'teach' to teach the chatbot a new response or 'quit' to stop).")
    chat(model, words, labels, data)
  