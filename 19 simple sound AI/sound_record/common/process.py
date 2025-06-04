import wave
import sys

import pyaudio
import threading
import onnxruntime
from scipy.io import wavfile #음악 파일을 numpy형태로 읽어와서
import numpy as np #수정
from python_speech_features import mfcc #mfcc 가공

import os
dr = os.path.dirname(os.path.abspath(__file__)) + '\\' #절대 경로

class Process() :
    def __init__(self, nn_file) :
        self.record_flg = False
        self.F = onnxruntime.InferenceSession(dr + nn_file, providers=["CPUExecutionProvider"])
        self.targets = ['apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple']
    
    def _record(self) :
        CHUNK = 1024 #1초 아날로그 소리 신호를 몇개의 단위로 나눠서 디지털로 처리 작업을 하는 단위
        FORMAT = pyaudio.paInt16 #저장 자료형
        CHANNELS = 1 if sys.platform == 'darwin' else 2 #os 환경마다 다르니 지정한것
        RATE = 8000 #디지털로 나눈 데이터를 저장을 할때 몇개 단위로 나눠서 저장 단위
        # RECORD_SECONDS = 5

        with wave.open(dr + 'output.wav', 'wb') as wf :
            p = pyaudio.PyAudio()
            wf.setnchannels(CHANNELS) #pyaudio 안의 소리 저장하는 클레스에  channels, format, rate를 지정하는 부분
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)

            stream = p.open(format = FORMAT, channels= CHANNELS, rate = RATE, input = True)

            while True : #종료 버튼을 누를 때 까지 무한 반복하기 위해서 while : True
                wf.writeframes(stream.read(CHUNK)) 
                if not self.record_flg : #종료 버튼을 눌러서 record_flg가 False가 되면 멈춤
                    break
            # for _ in range(0, RATE // CHUNK * RECORD_SECONDS) :
            #     wf.writeframes(stream.read(CHUNK))

        stream.close() # 메모리 수동 해제 
        p.terminate() # 메모리 수동 해제

    def record(self) : #while 문 때문에 무한루프 문제 때문에 스레드로 병렬 작업
        self.record_flg = True
        self.thread = threading.Thread(target = self._record)
        self.thread.start()

    def record_end(self) :
        self.record_flg = False
        self.thread.join() #멈출 때까지 기다리기

    def cal(self) :
        freq, signal = wavfile.read(dr + 'output.wav')
        signal = mfcc(signal, freq) # 2차원 데이터
        signal = np.expand_dims(signal, 0).astype(np.float32) # 3차원으로 변환
        inputs = {self.F.get_inputs()[0].name : signal}
        outputs = self.F.run(None, inputs)
        return outputs[0]

    def get_argmax(self, y) :
        argmax = np.argmax(y, axis = -1).item()
        text = f"결과 : {self.targets[argmax]}\n"
        return text

    def get_softmax(self, y) :
        softmax = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
        softmax = (softmax * 100).astype(np.int16)
        text = "--확률--\n"
        for i in range(len(self.targets)) :
            text += f"\t{self.targets[i]} : {softmax.squeeze()[i].item()}\n" #과일 이름 : 확률을 한줄 씩 뒤에 붙이기 작업

        return text