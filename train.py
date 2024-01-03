import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from model import ResNet

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('src_to_implement/data.csv', sep=';')
training_data, test_data = train_test_split(data, test_size=0.2)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(training_data, mode='train')
test_dataset = ChallengeDataset(test_data, mode='val')

batch = 100 # To be discussed
train_dataset = t.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_dataset = t.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=True)

# create an instance of our ResNet model
model = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn) https://pytorch.org/docs/stable/nn.html
loss = t.nn.BCELoss() # Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:
# Since we are in a multi-label concept, we need to use an appropriate loss function

# set up the optimizer (see t.optim)
learning_rate = 0.001
optimizer = t.optim.Adam(model.parameters(), lr = learning_rate)
# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model, loss, optimizer, train_dataset, test_dataset, False, 3)

# go, go, go... call fit on trainer
res = trainer.fit()

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')