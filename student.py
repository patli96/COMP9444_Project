#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

# import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import torch

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # print(sample)
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    # print(len(batch))
    # print(len(batch[0]))
    # print(vocab)
    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """

    return datasetLabel

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    return netOutput.round()

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self, input_dim, hidden_dim, layer_num):
        super(network, self).__init__()
        self.LSTM = tnn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_num, bidirectional=True, batch_first=True)
        self.fully_connected_layer = tnn.Linear(hidden_dim * 2, 1)
        # self.sigmoid = tnn.Sigmoid()

    def forward(self, input, length):
        # lstm = tnn.LSTM(input_size=50, hidden_size=10, num_layers=1, bidirectional=True, batch_first=True)
        packed_output, (hidden, cell) = self.LSTM(input)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hid dim * num directions]
        # fc = tnn.Linear(20, 1)
        dense_outputs = self.fully_connected_layer(hidden)
        # act = tnn.Sigmoid()
        # output = self.sigmoid(dense_outputs)
        # print(f'output.shape = {output.shape}')
        return dense_outputs


class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.MSELoss()

    def forward(self, output, target):
        # print(f'output.shape = {output.shape}\ntarget.shape = {target.shape}\n')
        # u= output.squeeze()
        # print(f'unsqueezed = {u.shape}')
        reshaped_output = output.squeeze()
        losses = 0
        for i in range(len(target)):
            losses += self.loss(reshaped_output[i], target[i])
        return losses


input_size = 50
hidden_size = 10
layers_num = 2
net = network(input_size, hidden_size, layers_num)
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.01)
