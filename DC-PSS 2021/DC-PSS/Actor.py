import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super(Actor, self).__init__()

        # Convolutional Layers
        self.conv_CAPM = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            # nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            # nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            # nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.PReLU(),
            nn.Flatten()
        )

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=mid_dim, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(46400, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.PReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(mid_dim, 29),
            nn.PReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.temperature_index = 300

    def forward(self, state):
        if state.dim() == 3:
            state = state.unsqueeze(3)  # Add a batch dimension if it's not there

        state = state.permute(0, 3, 1, 2)
        x = self.conv_CAPM(state)
        x = self.fc1(x)
        # x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.fc2(x) * self.temperature_index
        action = self.fc3(x)
        return action

    def get_action(self, state, action_std):
        state = state.permute(0, 3, 1, 2)
        action_prob = self.forward(state)
        noise = torch.randn_like(action_prob) * action_std
        noisy_action_prob = action_prob + noise
        noisy_action_prob = noisy_action_prob / noisy_action_prob.sum(dim=1, keepdim=True)
        action = noisy_action_prob.multinomial(num_samples=1).squeeze(1)
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)
