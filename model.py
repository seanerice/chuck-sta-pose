from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, 2)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            output, hidden = self.lstm(input, hidden)
            out = self.fc(output)
            return out, hidden
        else:
            output, hidden = self.lstm(input)
            out = self.fc(output)
            return out, hidden


if __name__ == "__main__":
    net = LSTMClassifier(1, 5, 2)

    # Test
    seq_tn = torch.FloatTensor(sequences).view(7, -1, net.input_size)
    print(torch.argmax(torch.sigmoid(net(seq_tn)[0]), 2))
