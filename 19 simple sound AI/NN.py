import torch
import torch.nn as nn

class Encoder(nn.Module) :
    def __init__(self, input_size = 13):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.rnn = nn.LSTM(input_size, input_size, batch_first=True, bidirectional=True)
    def forward(self, x):
        x = self.norm(x)
        x, hc = self.rnn(x)
        return x, hc
    
class Decoder(nn.Module) :
    def __init__(self, input_size = 13):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 7)
        )
    def forward(self, h):
        h_for = h[::2]
        h_back = h[1::2]
        context = torch.cat([h_for, h_back], dim=-1).permute(1,0,2).mean(dim=1)
        x = self.f(context)
        return x

class Encoder_n_Decoder(nn.Module) :
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        x, hc = self.encoder(x)
        x = self.decoder(hc[0])
        return x