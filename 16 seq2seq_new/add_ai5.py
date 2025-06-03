import torch
import torch.nn as nn

### Encoder와 Decoder 구현 하기 (함수 만들기)

class Encoder(nn.Module) :
    def __init__(self, vector) :
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vector, freeze = True, padding_idx=0) # 벡터화 함수, b값이 없는 선형함수, 차원 = (단어갯수, 벡터갯수)
        #.from_pretrained : W값을 해당 벡터로 만들겠다
        self.rnn = nn.LSTM(vector.shape[1], vector.shape[1], batch_first = True, bidirectional=True)# RNN 함수
    def forward(self, x) :
        x = self.embedding(x)
        y, hc = self.rnn(x) #LSTM은 hc
        return y, hc

class Attention(nn.Module) : #Value를 구하기 위한 재료,
    def __init__(self, h_size, bidirectional = False) : # h_size : 벡터 크기
        super().__init__()
        factor = 2 if bidirectional else 1
        self.U = nn.Linear(h_size * factor, h_size * factor) # query(decoder h값) 가공, query.shape = [b, 1, h_size * 2]
        self.W = nn.Linear(h_size * factor, h_size * factor) # key(encoder 출력값) 가공, key.shape = [b, f, h_size * 2]
        self.V = nn.Linear(h_size * factor, 1) # score(query & key) 계산
    def forward(self, query, key) :
        score = self.V(torch.tanh(self.U(query) + self.W(key))) # score.shape = [b, f, 1]
        score = score.permute(0,2,1) # score.shape = [b,1,f]

        weight = torch.softmax(score, dim = -1) # softmax는 함수 차원을 바꾸지 않습니다
        context = torch.bmm(weight, key) # [b,1,f] * [b,f,h_size * 2] = [b,1,h_size * 2]
        return context, weight

class Decoder(nn.Module) :
    def __init__(self, vector, max_len) :
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vector, freeze = True, padding_idx=0) #벡터화 함수
        self.attention = Attention(vector.shape[1], bidirectional = True)
        self.rnn = nn.LSTM(vector.shape[1] * 3, vector.shape[1], batch_first = True, bidirectional=True) # RNN 함수
        self.f = nn.Linear(vector.shape[1] * 2, vector.shape[0]) #벡터 값을 다시 단어 라벨값으로 역산하는 함수
        self.max_len = max_len #문장 단어의 최대 갯수
    def forward(self, encoder_output, encoder_hc, t = None) :
        decoder_y_list = [] #출력값

        decoder_x = torch.zeros((encoder_output.shape[0],1)).type(torch.long).to(encoder_output.device) #첫번째 입력값, 라벨값이기 때문에 차원 = (문장 갯수,1), 문장 갯수는 encoder_output에서 가져옵니다
        #.type(torch.long) 은 형변환, .to(encoder_output.device) 은 기기 배정
        decoder_hc = encoder_hc #첫번째 hc값

        for i in range(self.max_len) :
            decoder_y, decoder_hc = self.forward_cal(decoder_x, decoder_hc, encoder_output) # RNN 계산 한블록 합니다.

            if t is not None : #teaching force, t가 존재할 때
                decoder_x = t[:,i:i+1]  # i번째 t의 단어 가져옴
            else : # greedy, t가 존재하지 않을 때
                decoder_x = torch.argmax(decoder_y, dim = -1).detach() #출력 값 중에 가장 큰값(라벨값) 가져오고, 해당 값은 미분 대상이 아님 (.detach())

            decoder_y_list.append(decoder_y) #출력값에 단어 하나하나 씩 저장

        decoder_y_list = torch.cat(decoder_y_list, dim = 1) #list를 tensor로 묶기 위해서 cat 함수 쓰는 것 decoder_y.shape = (문장 갯수, 단어 갯수(1개), 벡터 갯수)
        
        return decoder_y_list, decoder_hc, None # decoder layer 2개 이상 묶어서 쓰기 위해서 hc값과, attention값도 return하기 위해 3개의 return 값
    def forward_cal(self, x, hc, encoder_output) : #rnn 블록하나 계산하는 함수
        query_for = hc[0][::2]
        query_back = hc[0][::2]
        query = torch.cat([query_for, query_back], dim = -1).permute(1,0,2)
        context, weight = self.attention(query, encoder_output)
        
        x = self.embedding(x) # 벡터로 변환하고
        x = torch.cat([context, x], dim = -1) # attention 적용    
        x, hc = self.rnn(x, hc) # rnn 계산, hc값은 이전 hc계산 결과, hc값도 따로 입력합니다
        x = self.f(x) # 벡터 계산값 역산해서 다시 라벨로 변환
        return x, hc
        
class Encoder_n_Decoder(nn.Module) : #encoder와 decoder를 하나로 묶는 새로운 신경망
    def __init__(self, encoder, decoder) :
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x, t = None) :
        y, hc = self.encoder(x)
        y, _, _ = self.decoder(y, hc, t)
        return y