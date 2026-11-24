import torch
from torch import nn
from torch.nn import Module

from info.vocab_info import VOCAB_LETTERS_INDEXES_DICT


class LinearModel(Module):
    def __init__(
        self,
        input_size,
        vocab_size=len(VOCAB_LETTERS_INDEXES_DICT),  # noqa: B008
        image_feature_size=200,
        embedding_dim=64,
        hidden_size=200,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 450),
            nn.CELU(),
            nn.Linear(450, image_feature_size),
            nn.CELU(),
            nn.Dropout(p=0.1),
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                image_feature_size + embedding_dim + hidden_size,
                hidden_size,
            ),
            nn.CELU(),
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, input_token, hidden):
        x = x.reshape(x.shape[0], -1)
        x = self.encoder(x)
        embedded = self.embedding(input_token)
        combined = torch.cat((x, embedded, hidden), dim=1)
        hidden = self.decoder(combined)
        logits = self.head(hidden)
        return logits, hidden


# class OCRNet(nn.Module):
#     def __init__(self, img_size, hidden_size):
#         super(OCRNet, self).__init__()
#         self.hidden_size = hidden_size
#         self.img_size = img_size
#         self.enc_img_size = 200
#         self.fc1 = nn.Linear(self.img_size, 450)
#         self.fc2 = nn.Linear(450, self.enc_img_size)
#         self.drop = nn.Dropout(p=0.1)
#         self.act = nn.CELU()
#         self.i2h = nn.Linear(self.enc_img_size + enc_dim + self.hidden_size, 80)
#         self.h2h = nn.Linear(80, 60)
#         self.h2h2 = nn.Linear(60, self.hidden_size)
#         self.i2o = nn.Linear(self.enc_img_size + enc_dim + self.hidden_size, 200)
#         self.o2o = nn.Linear(200, enc_dim)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x, input, hidden):
#         x = x.reshape((1, -1))
#         x = self.act(self.fc1(x))
#         x = self.act(self.fc2(x))
#         x = self.drop(x)
#         combined = torch.cat((x, input.view((1, -1)), hidden.view((1, -1))), 1)
#         hidden = self.act(self.i2h(combined))
#         hidden = self.act(self.h2h(hidden))
#         hidden = self.h2h2(hidden)
#         output = self.act(self.i2o(combined))
#         output = self.o2o(output)
#         output = self.softmax(output)
#         return output, hidden
