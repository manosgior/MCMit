import torch.nn as nn

class SingleQubitFNN(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_s_1 = max(input_size, 500) if input_size > 50 else input_size
        hidden_s_2 = hidden_s_1 // 2
        hidden_s_3 = hidden_s_2 // 2

        hidden_size = [hidden_s_1, hidden_s_2, hidden_s_3]
        print("Hidden Layer Size:", hidden_s_1, hidden_s_2, hidden_s_3)

        super(SingleQubitFNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(
            hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.bn3 = nn.BatchNorm1d(hidden_size[2])
        self.l4 = nn.Linear(hidden_size[2], output_size)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.l1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.l2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.l3(x)))
        x = self.dropout(x)
        x = self.l4(x)
        return x
