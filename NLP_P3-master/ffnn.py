import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import csv
from pathlib import Path
import time
from tqdm import tqdm
from data_loader import fetch_data

unk = '<UNK>'


# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()  # The rectified linear unit; one valid choice of activation function
        self.W2 = nn.Linear(h, 5)           # changed for Bug 1
        # The below two lines are not a source for an error
        self.softmax = nn.LogSoftmax()  # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss()  # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # The z_i are just there to record intermediary computations for your clarity
        z1 = self.W1(input_vector)
        h1 = self.activation(z1)             # added for Bug 2
        z2 = self.W2(h1)                     # changed for Bug 2
        predicted_vector = self.softmax(z2)  # changed for Bug 2
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for _, document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for idx, document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((idx, vector, y))
    return vectorized_data


def main(hidden_dim, number_of_epochs):
    print("Fetching data")
    train_data, valid_data = fetch_data()  # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    print("Fetched and indexed data")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    print("Vectorized data")

    tempMaxAccuracy = 0
    allAccuracy = []
    allMaxOutputs = []
    model = FFNN(input_dim=len(vocab), h=hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("Training for {} epochs".format(number_of_epochs))
    for epoch in range(number_of_epochs):
        allOutputs = []
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)  # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()           # added for Bug 4
            loss = None
            for example_index in range(minibatch_size):
                _, input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        # loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data)  # Good practice to shuffle order of validation data
        # minibatch_size = 16
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            # optimizer.zero_grad()
            # loss = None
            for example_index in range(minibatch_size):
                idx, input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                allOutputs.append((idx, predicted_label, gold_label))
                correct += int(predicted_label == gold_label)
                total += 1
                # Deleted for Bug 3
                """
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
            """
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        allAccuracy.append(correct / total)
        if correct / total > tempMaxAccuracy:
            tempMaxAccuracy = correct / total
            allMaxOutputs = allOutputs

    allMaxOutputs = sorted(allMaxOutputs, key = lambda x:x[0])
    with open('FFNN_'+'h_'+str(hidden_dim)+'_epoch_'+str(number_of_epochs)+'.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['idx', 'predict label', 'golden label'])
        for idx, predicted_label, gold_label in allMaxOutputs:
            report_writer.writerow([str(idx), str(predicted_label.data), str(gold_label)])

    with open('FFNN_Accuracy_'+'h_'+str(hidden_dim)+'_epoch_'+str(number_of_epochs)+'.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['epoch', 'Accuracy'])

        i = 1
        for acc in allAccuracy:
            report_writer.writerow([str(i), str(acc)])
            i += 1