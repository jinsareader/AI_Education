import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class Encoder(nn.Module) :
    def __init__(self, input_size, h_size, num_layers = 1, dropout_p = 0.1, bidirectional = False) :
        super().__init__()
        self.norm = nn.LayerNorm(input_size) # torch 내부에서 정규화를 시켜주는 함수
        self.rnn = nn.LSTM(input_size, h_size, num_layers=num_layers, batch_first=True, dropout=dropout_p, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x) :
        x = self.norm(x)
        x = self.dropout(x)
        y, hc = self.rnn(x)
        return y, hc[0], hc[1] # onnx로 변환하기 쉽게 하게 위해 h, c값 분리해서 return

# Attention
class Attention(nn.Module) :
    def __init__(self, h_size, bidirectional) :
        super().__init__()
        factor = 2 if bidirectional else 1 
        self.U = nn.Linear(h_size * factor, h_size * factor) # query(decoder_h)
        self.W = nn.Linear(h_size * factor, h_size * factor) # key(encoder_ouput)
        self.V = nn.Linear(h_size * factor, 1)
    def forward(self, query, key) :
        score = self.V(torch.tanh(self.U(query) + self.W(key))) # [b, f, 1]
        score = score.permute(0,2,1) # [b, 1, f]

        weight = torch.softmax(score, dim = -1) # [b, 1, f]
        context = torch.bmm(weight, key) # [b, 1, f] * [b, f, h_size] = [b, 1, h_size]
        return context

# Decoder
class Decoder(nn.Module) :
    def __init__(self, vectors, h_size, num_layers = 1, dropout_p = 0.1, bidirectional = False, max_len = 10, pad_idx = 0, sos_idx = 0) :
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True, padding_idx=pad_idx) # 음절 사전 신경망 안에 집어넣는 역할
        self.attention = Attention(h_size, bidirectional)
        self.rnn = nn.LSTM(vectors.shape[1] + h_size * (2 if bidirectional else 1), h_size, num_layers=num_layers, batch_first=True, dropout=dropout_p, bidirectional=bidirectional)
        self.f = nn.Linear(h_size * (5 if bidirectional else 3), vectors.shape[0])
        
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.max_len = max_len
        self.sos_idx = sos_idx
    def forward(self, encoder_output, encoder_h, encoder_c, t = None) :
        decoder_y_list = []

        decoder_x = torch.full((encoder_output.shape[0],1) ,self.sos_idx).type(torch.long).to(encoder_output.device)
        decoder_h = encoder_h
        decoder_c = encoder_c

        for i in range(self.max_len) :
            decoder_y, decoder_h, decoder_c = self.forward_cal(decoder_x, decoder_h, decoder_c, encoder_output)

            if t is not None :# teaching force
                decoder_x = t[:,i:i+1]
            else :
                decoder_x = torch.argmax(decoder_y, dim = -1).detach()

            decoder_y_list.append(decoder_y)

        decoder_y_list = torch.cat(decoder_y_list, dim = 1)
        return decoder_y_list, decoder_h, decoder_c
    def forward_cal(self, x, h, c, encoder_output) :
        if self.bidirectional :
            h_for = h[::2]
            h_back = h[1::2]
            query = torch.cat([h_for,h_back], dim = -1).permute(1,0,2)
        else :
            query = h.permute(1,0,2)
        if self.num_layers > 1 :
            query = query.sum(dim=1, keepdim=True) # 어텐션에 넣기 위한 h값 가공

        x = self.embedding(x)
        embed = x
        context = self.attention(query, encoder_output)
        x = torch.cat([x, context], dim = -1)
        x, hc = self.rnn(x, (h, c))
        x = torch.cat([x, context, embed], dim = -1)
        x = self.f(x)
        return x, hc[0], hc[1]
        