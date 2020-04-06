from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import math
import time


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, 1)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

        # Convolve for positional invariance
        self.conv = nn.Sequential(
            nn.Conv1d(1, 5, 3, 2),
            nn.ReLU(),
            nn.Conv1d(5, 15, 1)
        )

    def forward(self, input):
        # conv = self.conv(input.view(input.size()[0], 1, -1)).view(-1, 1, 15)
        out, h = self.lstm(input.view(input.size()[0], 1, -1))
        out = self.fc(out)
        return out


def get_seq_windows(sequences, seq_labels, window_sz):
    for i in range(len(sequences)):
        yield sequences[i:i + window_sz], seq_labels[i:i + window_sz]


def train(net, X, Y, epochs):
    loss_func = nn.BCEWithLogitsLoss()
    # loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    net.train()
    for epoch in tqdm(range(epochs), total=epochs):
        # for seqs, labels in tqdm(zip(X, Y), total=len(X)):
        print(f"Epoch {epoch}")
        for i in range(0, len(X) - 10, 5):
            net.zero_grad()

            # print(f"{i} - {i+10}")
            seqs = X[i:i + 10]
            labels = Y[i:i + 10]

            seq_tn = torch.FloatTensor(seqs)
            label_tn = torch.FloatTensor(labels)

            output = net(seq_tn).view(len(seqs), -1)
            # for o, l in list(zip(output, labels)):
            #     print(o, l)

            loss = loss_func(output, label_tn)
            print(loss)
            # print()

            loss.backward()
            optimizer.step()


def test(net, X, Y):
    net.eval()
    total = 0
    correct = 0

    seq_tn = torch.FloatTensor(X)
    output = net(seq_tn)
    for i in range(len(X)):
        print(output[i], Y[i])


def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for row in file:
            n_row = [float(n) for n in row.strip().strip("\n").split(" ")]
            data.append(n_row)
        return data


if __name__ == "__main__":
    net = LSTMClassifier(2, 15, 4)

    train_data = read_data(sys.argv[1])
    X_train = [[r[19], r[21]] for r in train_data]
    Y_train = [r[36:] for r in train_data]

    train(net, X_train, Y_train, 50)
    test(net, X_train, Y_train)

    # t = 0
    # X = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # Y = [
    #     1,
    #     1,
    #     1,
    #     1,
    #     1,
    #     1,
    #     1,
    #     0,
    #     2,
    #     2,
    #     2,
    #     2,
    #     2,
    #     2,
    #     2]
    # for i in range(1000):
    #     X.append(t)
    #     if 0 < t < (0.5 * math.pi) or 1.5 < t < 2:
    #         Y.append([1, 0])
    #     else:
    #         Y.append([0, 1])

    #     print(math.sin(t), Y[-1])
    #     t = (t + 0.1) % (2 * math.pi)
    #     # time.sleep(0.01)

    # net = LSTMClassifier(1, 15, 3)

    # train(net, X, Y, 50)
    # test(net, X, Y)
