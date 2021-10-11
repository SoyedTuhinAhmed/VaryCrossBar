from binarized_modules import *


class MLP(nn.Module):

    def __init__(self, neurons, num_classes, inp_size):
        super(MLP, self).__init__()
        self.infl_ratio = 1  # Hidden unit multiplier
        self.c_in = 1
        self.inp_size = inp_size * self.c_in
        self.num_classes = num_classes

        self.soft = nn.LogSoftmax()
        self.neurons = neurons
        bias = True

        self.fc1 = BinarizeLinear(self.inp_size, self.neurons, bias=bias)
        self.fc2 = BinarizeLinear(self.neurons, self.neurons, bias=bias)
        self.fc3 = BinarizeLinear(self.neurons, self.neurons, bias=bias)
        self.fc4 = BinarizeLinear(self.neurons, self.num_classes, bias=bias)

        self.bn1 = nn.BatchNorm1d(self.neurons)
        self.bn2 = nn.BatchNorm1d(self.neurons)  # , eps=eps)
        self.bn3 = nn.BatchNorm1d(self.neurons)  # , eps=eps)
        self.bn4 = nn.BatchNorm1d(self.num_classes)  # , eps=eps)

        self.hard = nn.Hardtanh(inplace=True)
        self.rReLU = nn.Sigmoid()
        self.hardS = nn.Hardshrink(lambd=6.)

    def forward(self, x, mask=False):

        x = x.view(-1, self.inp_size)
        x = self.fc1(x)
        # x = self.bn1_(x)
        x = self.bn1(x)
        x = self.hard(x)  # .mul(100)

        x = self.fc2(x)
        # x = self.bn2_(x)
        x = self.bn2(x)
        x = self.hard(x)

        x = self.fc3(x)
        # x = self.bn3_(x)
        x = self.bn3(x)
        x = self.hard(x)

        x = self.fc4(x)
        # x = self.bn4_(x)
        x = self.bn4(x)

        x = self.soft(x)
        return x
