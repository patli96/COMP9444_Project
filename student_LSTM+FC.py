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

"""
Patrick Li
z5180847

1. Implemented a LSTM to train and predict. The result is:
        Correct predictions: 36.66%
        One star away: 46.45%
        Two stars away: 14.25%
        Three stars away: 2.53%
        Four stars away: 0.10%
        Weighted score: 55.25

2. Implemented a fully connected layer after the LSTM and got following result:
        Correct predictions: 38.64%
        One star away: 46.40%
        Two stars away: 12.51%
        Three stars away: 2.38%
        Four stars away: 0.06%
        Weighted score: 57.20
"""

# import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch
# import numpy as np
# import sklearn


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
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

    def __init__(self, input_dim, hidden_dim, layer_num, fully_connected_layer_size):
        super(network, self).__init__()
        self.LSTM = tnn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_num, bidirectional=True, batch_first=True)
        self.fully_connected_layer = tnn.Linear(hidden_dim * 2, 1)

        self.lstm_to_hid = tnn.Linear(1, fully_connected_layer_size)
        self.hid_to_out = tnn.Linear(fully_connected_layer_size, 1)


    def forward(self, input, length):
        packed_output, (hidden, cell) = self.LSTM(input)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dense_outputs = self.fully_connected_layer(hidden)

        hid_sum = self.lstm_to_hid(dense_outputs.view(dense_outputs.shape[0], -1))
        hidden = torch.relu(hid_sum)
        out_sum = self.hid_to_out(hidden)
        output = torch.relu(out_sum)
        # print(output)
        return output


class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.MSELoss()

    def forward(self, output, target):
        reshaped_output = output.squeeze()
        losses = 0
        for i in range(len(target)):
            losses += self.loss(reshaped_output[i], target[i])
        return losses


input_size = 50
hidden_size = 10
layers_num = 2
fully_connected_layer_size = 300
net = network(input_size, hidden_size, layers_num, fully_connected_layer_size)
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
