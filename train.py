def train(net, sequences, labels, epochs):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs), total=epochs):
        for sequence, label in zip(sequences, labels):
            net.zero_grad()

            seq_tn = torch.FloatTensor(sequence).view(
                len(sequence), 1, net.input_size)
            label_tn = torch.LongTensor(label)

            output, _ = net(seq_tn)

            loss = loss_func(output.view(len(label), -1), label_tn)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    net = LSTMClassifier(17, 10, 5)
