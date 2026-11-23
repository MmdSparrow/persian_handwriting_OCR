import torch
from torch import nn
from torch.nn import Module

from info.vocab_info import VOCAB_LETTERS_INDEXES_DICT


class LinearModel(Module):
    def __init__(
        self,
        input_size,
        HL1=len(VOCAB_LETTERS_INDEXES_DICT),  # noqa: B008
        HL2 = 200,
        output_dim=len(VOCAB_LETTERS_INDEXES_DICT),  # noqa: B008
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 450)
        self.fc2 = nn.Linear(450, HL2)
        self.drop = nn.Dropout(p=0.1)
        self.act = nn.CELU()
        self.i2h = nn.Linear(HL2 + output_dim + HL1, 80)
        self.h2h = nn.Linear(80, 60)
        self.h2h2 = nn.Linear(60, HL1)
        self.i2o = nn.Linear(HL2 + output_dim + HL1, 200)
        self.o2o = nn.Linear(200, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, input, hidden):
        # x = x.reshape((1, -1))
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.drop(x)
        combined = torch.cat((x, input.view((1, -1)), hidden.view((1, -1))), 1)
        hidden = self.act(self.i2h(combined))
        hidden = self.act(self.h2h(hidden))
        hidden = self.h2h2(hidden)
        output = self.act(self.i2o(combined))
        output = self.o2o(output)
        output = self.softmax(output)

        return output, hidden
