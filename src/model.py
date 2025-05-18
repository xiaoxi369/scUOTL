import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_encoder(nn.Module):
    def __init__(self, input_size):
        super(Net_encoder, self).__init__()
        self.input_size = input_size
        self.k = 64
        self.f = 64

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64)
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)

        return embedding


class Net_class(nn.Module):
    def __init__(self, num_of_class):
        super(Net_class, self).__init__()
        self.classification = nn.Sequential(
            nn.Linear(64, num_of_class)
        )

    def forward(self, embedding):
        class_prediction = self.classification(embedding)

        return class_prediction
