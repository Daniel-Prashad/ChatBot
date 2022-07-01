import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import pickle
import numpy as np
import tflearn
import tensorflow as tf


def clean_data(data):
    '''(dictionary) -> list, list, list of lists, list
    This function is used to clean the data that is read in from intents.json before being used to train the model.
    '''
    # initialize the lists to be returned
    words = []      # list of all root words from all patterns of each tag in intents.json, sorted alphabetically
    labels = []     # list of tags from intents.json, sorted alphabetically
    docs_x = []     # list of lists where each inner list contains each word from a given pattern from intents.json
    docs_y = []     # list of tags, where each value is the label of the corresponding list in docs_x

    # loop through the contents of each tag
    for intent in data["intents"]:
        # tokenize the sample input in patterns and add each word to the complete list of words
        # also store each tokenized word from the pattern in its own list and its corresponding intent
        for pattern in intent["patterns"]:
            token_words = nltk.word_tokenize(pattern)
            words.extend(token_words)
            docs_x.append(token_words)
            docs_y.append(intent["tag"])
            # add the current intent tag to the list of labels if this pattern belong to a new tag
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    # convert all words to lower case to be stemmed, store all of the root words to the words list
    words = [stemmer.stem(w.lower()) for w in words if w !="?"]
    # remove duplicate words and sort
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # return each list
    return words, labels, docs_x, docs_y


def create_training_data(words, labels, docs_x, docs_y):
    '''(list, list, list of lists, list) -> list of lists, lists of lists
    This function, given the already cleaned data, is used to create the training data to be used to fit the model.
    '''
    # initialize the lists to be used
    training = []                                   # list of lists where each inner list tracks which words from the complete words list are used in each pattern
    output = []                                     # lists of lists where each inner list tracks which label the corresponding data from training belongs to      
    # initialize an empty list of 0s for each label
    out_empty = [0 for _ in range(len(labels))]

    # loop through the list of patterns
    for x, doc in enumerate(docs_x):
        # initialize an empty bag to be used for one-hot encoding
        bag = []
        # stem and store each word in the current pattern
        current_pattern = [stemmer.stem(w) for w in doc]

        # one-hot encoding for each word
        # loop through each word in the complete list of root words
        # for each word that appears in the current pattern append 1, otherwise append 0
        for w in words:
            if w in current_pattern:
                bag.append(1)
            else:
                bag.append(0)
        # create a mutable duplicate of out_empty and change the index of the label of the corresponding pattern to 1
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        # append the corresponding lists collected from this pattern
        training.append(bag)
        output.append(output_row)

    # store the data used in training and building the model
    with open("data.pickle", "wb") as f:
        pickle.dump([words, labels, training, output], f)

    # return each list
    return training, output


def fit_model(training, output):
    '''(list of lists, list of lists) -> Deep Neural Network model
    Given the training data, this function is used to fit the model.
    '''
    # convert lists to numpy arrays to be used as input in the model 
    training = np.array(training)
    output = np.array(output)

    # reset previous graph data
    tf.compat.v1.reset_default_graph()

    # define the input layer of the model, where the number of neurons is equal to the number of unique words
    net = tflearn.input_data(shape=[None, len(training[0])])
    # add two fully connected hidden layers, each with 8 neurons
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    # define the output layer where each neuron represents a label
    # the value given to each output neuron is the probability that the input corresponds to that label
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)

    # load in the model if it has been previously fit and saved
    try:
        model.load("model.tflearn")
    # otherwise fit and save the model
    except:
        model = tflearn.DNN(net)
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")

    # return the model
    return model


def retrain_refit(data):
    '''(dictionary) -> Deep Neural Network model, list, list
    This function is used to retrain and refit the model after the user has given the chatbot a new response to learn.
    '''
    words, labels, docs_x, docs_y = clean_data(data)
    training, output = create_training_data(words, labels, docs_x, docs_y)
    model = fit_model(training, output)
    return model, words, labels