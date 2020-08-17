import numpy as np
import torch
import torch.nn as nn
from torch.nn import init, Parameter
import torch.optim as optim
import math
import random
import spacy
import os
import csv
import time
from tqdm import tqdm
from data_loader import fetch_data

unk = '<UNK>'


class RNN(nn.Module):
    def __init__(self, input_dim, h, num_layers):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=self.h, num_layers=self.num_layers)
        self.V = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        hidden = Parameter(torch.randn(self.num_layers, 1, self.h).type(torch.FloatTensor), requires_grad=True)
        c = Parameter(torch.randn(self.num_layers, 1, self.h).type(torch.FloatTensor), requires_grad=True)
        z, (hidden,c) = self.rnn(inputs, (hidden, c))
        hidden = hidden[-1]
        hidden = hidden.contiguous().view(hidden.size()[0]*hidden.size()[1])
        predicted_vector = self.softmax(self.V(hidden))
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
# wordEmbed = A dictionary mapping word/token to its vector [0.1, 0.2, ..., 0.03]
def embeding_data(vocab, enDic):
    wordEmbed={}
    for x in vocab:
        wordEmbed[x] = enDic(x).vector
    return wordEmbed


def convert_to_tensor_representation(data, wordEmbed):
    result = []
    for x in data:
        x_list = []
        idx = x[0]
        x_word = x[1]
        gold_label = x[2]
        for y in x_word:
            vector = wordEmbed[y]
            x_list.append(vector)
        tensor = torch.FloatTensor(x_list)
        size = tensor.size()
        tensor = tensor.view(size[0], 1, size[1])
        result.append((idx, tensor, gold_label))
    return result


def main(hidden_dim, number_of_epochs, num_layers):
    print("Fetching data")
    train_data, valid_data = fetch_data()  # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data+valid_data)
    enDic = spacy.load('en_core_web_sm')
    wordEmbed = embeding_data(vocab, enDic)
    print("Fetched and indexed data")
    train_data = convert_to_tensor_representation(train_data, wordEmbed)
    valid_data = convert_to_tensor_representation(valid_data, wordEmbed)
    print("Vectorized data")

    tempMaxAccuracy = 0
    allAccuracy = []
    allMaxOutputs = []
    model = RNN(input_dim=96, h=hidden_dim, num_layers=num_layers)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)    # default value for lr and momentum
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
            optimizer.zero_grad()
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
        # model.train(False)
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data)  # Good practice to shuffle order of validation data
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            for example_index in range(minibatch_size):
                idx, input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                allOutputs.append((idx, predicted_label, gold_label))
                correct += int(predicted_label == gold_label)
                total += 1
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        allAccuracy.append(correct / total)
        if correct/total > tempMaxAccuracy:
            tempMaxAccuracy = correct/total
            allMaxOutputs = allOutputs

    allMaxOutputs = sorted(allMaxOutputs, key=lambda x: x[0])
    with open('RNN_'+'h_'+str(hidden_dim)+'_epoch_'+str(number_of_epochs)+'_layer_'+str(num_layers)+'.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['idx', 'predict label', 'golden label'])
        for idx, predicted_label, gold_label in allMaxOutputs:
            report_writer.writerow([str(idx), str(predicted_label.data), str(gold_label)])

    with open('RNN_Accuracy_'+'h_'+str(hidden_dim)+'_epoch_'+str(number_of_epochs)+'_layer_'+str(num_layers)+'.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['epoch', 'Accuracy'])

        i = 1
        for acc in allAccuracy:
            report_writer.writerow([str(i), str(acc)])
            i += 1
