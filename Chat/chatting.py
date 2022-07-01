from .training import retrain_refit

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random
import json


def bag_of_words(sentence, words):
    '''(string, list) -> numpy array
    This function creates a bag of words that is one-hot encoded to the user's input.
    The numpy array returned is used to be fed into the model to predict to which label the user's input belongs.
    '''
    # initialize the bag of words with 0 for each word in the words list
    bag = [0 for _ in range(len(words))]

    # tokenize and stem the words from the user's input
    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    # loop through each root word of the user's input
    for s in s_words:
        for i, w in enumerate(words):
            # if the current word from the words list is the the same as the current root word in the sentence, set the value of bag at that index to 1
            if w == s:
                bag[i] = 1

    # return the one-hot encoded bag of words
    return np.array(bag)


def chat(model, words, labels, data):
    '''(Deep Neural Network model, list, list, dictionary) -> Nonetype
    This function is used to allow the user to chat with the chatbot, taking input from
    the user and selecting & displaying the appropriate response from the chatbot.
    '''
    call_add_data = False
    # provide the user with some instructions
    print("\nStart talking with the bot! (Type 'teach' to teach the chatbot a new response or 'quit' to stop).")
    while True:
        # get input from the user
        inp = input("You: ")
        # stop the program if the user wants to quit
        if inp.lower() == "quit":
            break
        # if the user chooses, break out of the while loop and allow the user to add a new response for the chatbot
        elif inp.lower() == "teach":
            call_add_data = True
            break
        # otherwise, convert the user's input to a bag of words to be fed into the model
        # the results list contains the probabilty that the user's input belongs to the label of the corresponding index
        # the label with the greatest probability will be used
        else:
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = np.argmax(results)
            # if the greatest probability is over 0.7, randomly select a response from the list of responses associated with the corresponding label
            if results[results_index] > 0.7:
                tag = labels[results_index]
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                print(f"Bot: {random.choice(responses)}")
            # otherwise prompt the user for another question
            else:
                print("Bot: I don't quite understand. Please ask another question.")
    # allow the user to add a new response for the chatbot if they chose to do so
    if call_add_data:
        add_data(data, labels)


def add_data(data, labels):
    '''(dictionary) -> Nonetype
    Given a response, pattern and tag from the user, this function is used to add a new response for the chatbot to use.
    The user's input is used to update intents.json and then the model is retrained accordingly.
    '''
    tag_index = -1                                      # used to indicate the index of the tag that the user wants to add to
    pattern_exists, response_exists = False, False      # used to indicate whether the pattern and response already exists in the user's selected tag

    # store the user's input for a given response, pattern and tag
    inp_response = input("Please enter a response you would like to see from the chatbot: ")
    inp_pattern = input("Please enter a prompt from the user that would elicit this response: ")
    print(f"Existing Subjects: ", {*labels,})
    inp_tag = input("Please enter the subject of your input (You can select one from the list above or add a new subject): ")

    # loop through each dictionary in intents.json
    for i, intent in enumerate(data["intents"]):
        # if the user entered an already existing tag, store the index of the tag
        if inp_tag.lower() == intent["tag"].lower():
            tag_index = i
            # loop through all of the patterns and responses for this tag, note if the pattern or response already exists, then break from the loop
            for pattern in intent["patterns"]:
                if inp_pattern.lower() == pattern.lower():
                    pattern_exists = True
            for response in intent["responses"]:
                if inp_response.lower() == response.lower():
                    response_exists = True
            break

    # if the user's tag does not already exists, create a new dictionary with the user's input and append it to the existing data
    if tag_index == -1:
        pass
        new_data = {"tag" : inp_tag,
                    "patterns" : [inp_pattern],
                    "responses" : [inp_response],
                    "context_set" : ""
                    }
        data["intents"].append(new_data)
    # otherwise, add the pattern and response, if they do not already exist, to the already existing tag
    else:
        if not pattern_exists:
            data["intents"][tag_index]["patterns"].append(inp_pattern)
        if not response_exists:
            data["intents"][tag_index]["responses"].append(inp_response)


    # update intents.json with the user's input
    with open("intents.json", "w") as file:
        json.dump(data, file)

    # retrain and refit the model with the new data from the user
    model, words, labels = retrain_refit(data)

    # tell the user that the chatbot has been updated
    print("The chatbot has been updated.")

    # allow the user to continue chatting
    chat(model, words, labels, data)
    