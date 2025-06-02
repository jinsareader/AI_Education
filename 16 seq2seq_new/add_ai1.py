import torch
import torch.nn as nn

### Encoder와 Decoder 구현 하기 (함수 만들기)

class Encoder(nn.Module) :
    def __init__(self, vector) :
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vector, freeze = True, padding_idx=0) # 벡터화 함수, b값이 없는 선형함수, 차원 = (단어갯수, 벡터갯수)
        #.from_pretrained : W값을 해당 벡터로 만들겠다
        self.rnn = nn.RNN(vector.shape[1], vector.shape[1], batch_first = True)# RNN 함수
    def forward(self, x) :
        x = self.embedding(x)
        y, h = self.rnn(x)
        return y, h

class Decoder(nn.Module) :
    def __init__(self, vector, max_len) :
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vector, freeze = True, padding_idx=0) #벡터화 함수
        self.rnn = nn.RNN(vector.shape[1], vector.shape[1], batch_first = True) # RNN 함수
        self.f = nn.Linear(vector.shape[1], vector.shape[0]) #벡터 값을 다시 단어 라벨값으로 역산하는 함수
        self.max_len = max_len #문장 단어의 최대 갯수
    def forward(self, encoder_output, encoder_h, t = None) :
        decoder_y_list = [] #출력값

        decoder_x = torch.zeros((encoder_output.shape[0],1)).type(torch.long).to(encoder_output.device) #첫번째 입력값, 라벨값이기 때문에 차원 = (문장 갯수,1), 문장 갯수는 encoder_output에서 가져옵니다
        #.type(torch.long) 은 형변환, .to(encoder_output.device) 은 기기 배정
        decoder_h = encoder_h #첫번째 h값       

        for i in range(self.max_len) :
            decoder_y, decoder_h = self.forward_cal(decoder_x, decoder_h) # RNN 계산 한블록 합니다.

            if t is not None : #teaching force, t가 존재할 때
                decoder_x = t[:,i:i+1]  # i번째 t의 단어 가져옴
            else : # greedy, t가 존재하지 않을 때
                decoder_x = torch.argmax(decoder_y, dim = -1).detach() #출력 값 중에 가장 큰값(라벨값) 가져오고, 해당 값은 미분 대상이 아님 (.detach())

            decoder_y_list.append(decoder_y) #출력값에 단어 하나하나 씩 저장

        decoder_y_list = torch.cat(decoder_y_list, dim = 1) #list를 tensor로 묶기 위해서 cat 함수 쓰는 것 decoder_y.shape = (문장 갯수, 단어 갯수(1개), 벡터 갯수)
        
        return decoder_y_list, decoder_h, None # decoder layer 2개 이상 묶어서 쓰기 위해서 h값과, attention값도 return하기 위해 3개의 return 값
    def forward_cal(self, x, h) : #rnn 블록하나 계산하는 함수
        x = self.embedding(x) # 벡터로 변환하고
        x, h = self.rnn(x, h) # rnn 계산, h값은 이전 h계산 결과, h값도 따로 입력합니다
        x = self.f(x) # 벡터 계산값 역산해서 다시 라벨로 변환
        return x, h
        
class Encoder_n_Decoder(nn.Module) : #encoder와 decoder를 하나로 묶는 새로운 신경망
    def __init__(self, encoder, decoder) :
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x, t = None) :
        y, h = self.encoder(x)
        y, _, _ = self.decoder(y, h, t)
        return y