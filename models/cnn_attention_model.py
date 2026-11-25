import torch
import torch.nn.functional as F
from torch import nn


class CNNEncoder(nn.Module):
    def __init__(self, input_channels, feature_size):
        super().__init__()

        self.cnn = nn.Sequential(
            # (B, 1, H, W) -> (B, 32, H/2, W/2)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # -> (B, 64, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # -> (B, 128, H/8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # -> (B, 256, H/16, W/4)
            nn.Conv2d(128, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

    def forward(self, images):
        features = self.cnn(images)  # (B, C, H, W) -> (B, 256, H/16, W/4)

        # Average over the height dimension.
        # (B, C, H', W') -> (B, W', C)
        features = features.mean(dim=2).transpose(1, 2)

        return features


class BahdanauAttention(nn.Module):
    def __init__(self, feature_size, hidden_size, attention_size):
        super().__init__()

        self.feature_projection = nn.Linear(
            feature_size,
            attention_size,
        )

        self.hidden_projection = nn.Linear(
            hidden_size,
            attention_size,
        )

        self.energy = nn.Linear(attention_size, 1)

    def forward(self, features, hidden, padding_mask=None):
        # features: (B, W', feature_size)
        # hidden:   (B, hidden_size)

        feature_scores = self.feature_projection(features)

        hidden_scores = self.hidden_projection(hidden).unsqueeze(1)

        scores = self.energy(torch.tanh(feature_scores + hidden_scores)).squeeze(-1)

        # Ignore padded image positions.
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=1)

        # Weighted sum of visual features.
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            features,
        ).squeeze(1)

        return context, attention_weights


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        feature_size,
        embedding_dim,
        hidden_size,
        attention_size,
        padding_index,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_index,
        )

        self.attention = BahdanauAttention(
            feature_size=feature_size,
            hidden_size=hidden_size,
            attention_size=attention_size,
        )

        self.gru = nn.GRUCell(
            embedding_dim + feature_size,
            hidden_size,
        )

        self.output = nn.Linear(
            hidden_size + feature_size,
            vocab_size,
        )

        self.initial_hidden = nn.Linear(
            feature_size,
            hidden_size,
        )

    def initialize_hidden(self, features, padding_mask=None):
        if padding_mask is None:
            visual_context = features.mean(dim=1)
        else:
            valid_mask = (~padding_mask).unsqueeze(-1)

            visual_context = (features * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        return torch.tanh(self.initial_hidden(visual_context))

    def forward(
        self,
        features,
        input_token,
        hidden,
        padding_mask=None,
    ):
        # Embed previous character.
        embedded = self.embedding(input_token)

        # Attend to relevant image regions.
        context, attention_weights = self.attention(
            features,
            hidden,
            padding_mask,
        )

        # Update decoder hidden state.
        decoder_input = torch.cat((embedded, context), dim=1)

        hidden = self.gru(decoder_input, hidden)

        # Predict next character.
        output = self.output(torch.cat((hidden, context), dim=1))

        return output, hidden, attention_weights


class CNNAttentionOCR(nn.Module):
    def __init__(
        self,
        vocab_size,
        input_channels,
        feature_size=256,
        embedding_dim=64,
        hidden_size=256,
        attention_size=128,
        padding_index=None,
        right_to_left=True,
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            input_channels=input_channels,
            feature_size=feature_size,
        )

        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            feature_size=feature_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            attention_size=attention_size,
            padding_index=padding_index,
        )

        self.right_to_left = right_to_left

    def encode(self, images, image_widths=None):
        features = self.encoder(images)

        batch_size, sequence_length, _ = features.shape

        if image_widths is None:
            feature_lengths = torch.full(
                (batch_size,),
                sequence_length,
                dtype=torch.long,
                device=images.device,
            )
        else:
            # CNN reduces image width by a factor of 4.
            feature_lengths = torch.div(
                image_widths,
                4,
                rounding_mode="floor",
            ).clamp(min=1, max=sequence_length)

        positions = torch.arange(
            sequence_length,
            device=images.device,
        ).unsqueeze(0)

        padding_mask = positions >= feature_lengths.unsqueeze(1)

        # Reverse valid visual features for RTL recognition.
        if self.right_to_left:
            reverse_indices = torch.where(
                positions < feature_lengths.unsqueeze(1),
                feature_lengths.unsqueeze(1) - 1 - positions,
                positions,
            )

            features = features.gather(
                1,
                reverse_indices.unsqueeze(-1).expand_as(features),
            )

        hidden = self.decoder.initialize_hidden(
            features,
            padding_mask,
        )

        return features, hidden, padding_mask

    def decode_step(
        self,
        features,
        input_token,
        hidden,
        padding_mask=None,
    ):
        return self.decoder(
            features,
            input_token,
            hidden,
            padding_mask,
        )

    def gradient_clip(self):
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=5.0,
        )
