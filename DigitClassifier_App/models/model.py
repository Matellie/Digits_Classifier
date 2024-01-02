import torch.nn as nn

class LinearRegression(nn.Module):
    """
    Linear regression is used to predict the value of a variable based on the value of one or more other
    variables.
    """
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.name = 'LinearRegression'
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

class SimpleNeuralNet(nn.Module):
    """
    Simple neural net with one hidden layer using the ReLU activation function.
    """
    def __init__(self, input_size, hidden_size, nb_classes):
        super(SimpleNeuralNet, self).__init__()

        self.name = 'SimpleNeuralNet'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nb_classes = nb_classes

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, nb_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x