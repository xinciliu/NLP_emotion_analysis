import json
import random


def fetch_data():
    with open('training.json') as training_f:
        training = json.load(training_f)
    with open('validation.json') as valid_f:
        validation = json.load(valid_f)
    # If needed you can shrink the training and validation data to speed up somethings but this isn't always safe to do by setting k < 16000
    # k = #fill in
    # training = random.shuffle(training)
    # validation = random.shuffle(validation)
    # training, validation = training[:k], validation[:(k // 10)]
    tra = []
    val = []
    idx = 0
    for elt in training:
        tra.append((idx,elt["text"].split(),int(elt["stars"]-1)))
        idx += 1
    idx = 0
    for elt in validation:
        val.append((idx, elt["text"].split(),int(elt["stars"]-1)))
        idx += 1
    return tra, val