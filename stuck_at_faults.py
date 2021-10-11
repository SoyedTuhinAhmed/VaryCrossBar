import torch
import torch.nn as nn
import numpy as np


class StuckAtFaults:
    """
    Only supports BNN and random weights set ot reset
    """
    def __init__(self, sa_type='1', percent=0):
        self.percent = percent
        self.sa_type = sa_type

    def SA1(self, percent, weight):
        """
        Randomly set weights to +1 to simulate Stuck-at-1 faults based on percentage of faulty cells expected
        :param percent: percentage of stuck-at-1 faults
        :param weight: weight matrix
        :return: faulty weight matrix
        """
        self.percent = percent/100
        weight_sa = weight.flatten()
        num_sa1 = round(len(weight_sa) * self.percent)

        # rand_index = torch.multinomial(weight_sa, num_sa1, replacement=False)
        rand_index = np.random.choice(np.arange(len(weight_sa)), num_sa1, replace=False)

        with torch.no_grad():
            weight_sa[rand_index] = 1.
            weight_sa = weight_sa.view(weight.shape)
        return weight_sa.cuda()

    def SA0(self, percent, weight):
        """
        Randomly reset weights to -1 to simulate Stuck-at-0 faults based on percentage of faulty cells expected
        :param percent: percentage of stuck-at-0 faults
        :param weight: weight matrix
        :return: faulty weight matrix
        """
        self.percent = percent / 100
        weight_sa = weight.flatten()
        num_sa0 = round(len(weight_sa) * self.percent)
        # print(num_sa0, '###########################################################')

        # rand_index = torch.multinomial(weight_sa, num_sa0, replacement=False)#.cuda()
        rand_index = np.random.choice(np.arange(len(weight_sa)), num_sa0, replace=False)

        with torch.no_grad():
            weight_sa[rand_index] = -1.
            weight_sa = weight_sa.view(weight.shape)
        return weight_sa.cuda()

    def SA_Both(self, percent, weight):
        """
        Randomly set weights to +1 or -1 to simulate both Stuck-at-1 and Stuck-at-1 faults based on percentage of faulty cells expected
        :param percent: percentage of stuck-at-1 faults
        :param weight: weight matrix
        :return: faulty weight matrix
        """
        self.percent = percent / 100
        weight_sa = weight.flatten()
        num_sa = round(len(weight_sa) * self.percent)

        # rand_index = torch.multinomial(weight_sa, num_sa, replacement=False)
        rand_index = np.random.choice(np.arange(len(weight_sa)), num_sa, replace=False)

        num_sa1 = torch.randint(0, num_sa, [1]).item()

        with torch.no_grad():
            weight_sa[rand_index[0:num_sa1]] = 1.
            weight_sa[rand_index[num_sa1:]] = -1.
            weight_sa = weight_sa.view(weight.shape)

        return weight_sa.cuda()

    def FI_SA0(self, percent, model, first=False):
        """
        Fault inject with reset type to last and all hidden layers by default first hidden layers is not simulated.

        :param percent: overall percentage of stuck-at-faults.
        :param model: NN model for fault injection
        :param first: if True also fault inject to first hidden layer
        :return: faulty model
        """
        if first:
            w1 = self.SA0(percent, model.fc1.weight)
            model.fc1.weight = nn.Parameter(w1)

        w2 = self.SA0(percent, model.fc2.weight)
        w3 = self.SA0(percent, model.fc3.weight)
        w4 = self.SA0(percent, model.fc4.weight)

        model.fc2.weight = nn.Parameter(w2)
        model.fc3.weight = nn.Parameter(w3)
        model.fc4.weight = nn.Parameter(w4)

        return model

    def FI_SA1(self, percent, model, first=False):
        """
        Fault inject with set type to last and all hidden layers by default first hidden layers is not simulated.

        :param percent: overall percentage of stuck-at-faults.
        :param model: NN model for fault injection
        :param first: if True also fault inject to first hidden layer
        :return: faulty model
        """
        if first:
            w1 = self.SA1(percent, model.fc1.weight)
            model.fc1.weight = nn.Parameter(w1)

        w2 = self.SA1(percent, model.fc2.weight)
        w3 = self.SA1(percent, model.fc3.weight)
        w4 = self.SA1(percent, model.fc4.weight)

        model.fc2.weight = nn.Parameter(w2)
        model.fc3.weight = nn.Parameter(w3)
        model.fc4.weight = nn.Parameter(w4)

        return model

    def FI_SA_Both(self, percent, model, first=False):
        """
        Fault inject with set and reset type to last and all hidden layers by default first hidden layers is not simulated.

        :param percent: overall percentage of stuck-at-faults.
        :param model: NN model for fault injection
        :param first: if True also fault inject to first hidden layer
        :return: faulty model
        """
        if first:
            w1 = self.SA_Both(percent, model.fc1.weight)
            model.fc1.weight = nn.Parameter(w1)

        w2 = self.SA_Both(percent, model.fc2.weight)
        w3 = self.SA_Both(percent, model.fc3.weight)
        w4 = self.SA_Both(percent, model.fc4.weight)

        model.fc2.weight = nn.Parameter(w2.cuda())
        model.fc3.weight = nn.Parameter(w3.cuda())
        model.fc4.weight = nn.Parameter(w4.cuda())

        return model


