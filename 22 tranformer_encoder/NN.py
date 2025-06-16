import torch
import torch.nn as nn
import math

class PositionEncoding(nn.Module) :
    def __init__(self, dropout_p, d_model) :
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.d_model = d_model

    def forward(self, x) :
        pe = torch.zeros(x.shape[1], self.d_model).to(x.device) #max_len : 단어 길이, d_model : 벡터 차원
        pe = pe.clone()
        position = torch.arange(0, x.shape[1], dtype=torch.float).unsqueeze(1) # [[0], [1], [2], [3], [4], ..., [max_len-1]]
        div_term = torch.exp(torch.arange(0,self.d_model,2).float() * (-math.log(10000.0) / self.d_model)) #[0, 2, 4, 6, 8, ...]
        pe[:,0::2] = torch.sin(position * div_term) #짝수번째 벡터 차원에 값 집어넣기
        pe[:,1::2] = torch.cos(position * div_term) #홀수번째 벡터 차원에 값 집어넣기
        pe = pe.unsqueeze(0)

        x + pe[:x.size(0), :]
        return self.dropout(x)
    
class Encoder(nn.Module) :
    def __init__(self, vector, nhead, out_n, num_layers=2, dim_feedforward=2048, dropout_p = 0.1) : 
        #d_model : 벡터 크기, nhead : d_model을 attention에서 몇조각으로 나누어서 처리할지(d_model이 nhead로 나누어 떨어져야 함)
        #dim_feedforward : encoderlayer 내부의 선형함수의 중간 처리 벡터 차원 크기 (W의 차원)
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vector, freeze=True, padding_idx=0)
        self.pos_encoder = PositionEncoding(dropout_p, vector.shape[1])
        self.encoder_layer = nn.TransformerEncoderLayer(vector.shape[1], nhead, dim_feedforward=dim_feedforward, dropout=dropout_p, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.f = nn.Sequential(nn.Linear(vector.shape[1], 100)
                               , nn.ReLU()
                               , nn.Linear(100, out_n))
        self.d_model = vector.shape[1]
    def forward(self, x) :
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x, None, None)
        x = x[:,-1,:] #transformer의 계산 결과과 형태는 rnn과 같으나, rnn하고 다르게, transformer 는 병렬계산이기 때문에 아무 값 가져와도 됩니다.
        x = self.f(x)
        return x
    

