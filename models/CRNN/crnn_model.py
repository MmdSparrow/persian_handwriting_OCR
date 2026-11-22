from torch import nn


class CRNNModel(nn.Module):
    def __init__(self, num_classes, hidden_size=128, input_channels=1):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            # (B, 1, 32, 128)
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # (B, 64, 16, 64)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # (B, 128, 8, 32)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # (B, 256, 4, 32)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # (B, 512, 2, 32)
            nn.Conv2d(512, 512, (2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # (B, 512, 1, 32)
        )

        # Sequence modeling
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=2, bidirectional=True, dropout=0.2)

        # Character prediction
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        # x: (B, 1, 32, 128)

        features = self.cnn(x)

        # (B, 512, 1, W)
        features = features.squeeze(2)

        # (B, 512, W) -> (W, B, 512)
        features = features.permute(2, 0, 1)

        # BiLSTM
        features, _ = self.rnn(features)

        # (W, B, 2 * hidden_size)
        output = self.classifier(features)

        # (W, B, num_classes)
        output = self.log_softmax(output)

        return output
