import wave
import sys
import pyaudio
import threading

import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
import onnxruntime

import os
dr = os.path.dirname(os.path.abspath(__file__)) + "\\"

class Process() :
    def __init__(self, onnx_file) :
        self.targets = ['apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple']
        self.F = onnxruntime.InferenceSession(dr+onnx_file, providers= ["CPUExecutionProvider"])
        self.t = None
        self.recording = False

    def _record(self, file_name, frequency = 8000) :
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNEL = 1 if sys.platform == 'darwin' else 2
        RATE = frequency
        with wave.open(dr+file_name, 'wb') as wf :
            p = pyaudio.PyAudio()
            wf.setnchannels(CHANNEL)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)

            stream = p.open(format = FORMAT, channels= CHANNEL, rate = RATE, input = True)

            while self.recording :
                wf.writeframes(stream.read(CHUNK))

            stream.close()
            p.terminate()

    def start_record(self, file_name) :
        if self.recording :
            return
        self.recording = True
        self.t = threading.Thread(target = self._record, args = (file_name,8000))
        self.t.start()

    def end_record(self) :
        if not self.recording :
            return
        self.recording = False
        self.t.join()

    def sound_preprocess(self, file_name) :
        freq, sig = wavfile.read(dr+file_name)
        data = mfcc(sig, freq)
        data = np.expand_dims(data, axis = 0).astype(np.float32)
        return data
    
    def cal(self, data) :
        input_ = {self.F.get_inputs()[0].name : data}
        output = self.F.run(None, input_)
        return output[0]
    
    def get_argmax(self, y) :
        argmax = np.argmax(y, axis = -1)
        argmax = argmax.squeeze()
        argmax_text = self.targets[argmax]
        text = "결과 : " + argmax_text + "\n"
        return text
    
    def get_softmax(self, y) :
        softmax = np.exp(y - y.max()) / np.sum(np.exp(y - y.max()))
        softmax = (softmax * 100).astype(np.int16)
        softmax = softmax.squeeze()
        text = "확률 :\n"
        for i in range(len(self.targets)) :
            text += f"\t{self.targets[i]} : {softmax[i]}\n"
        return text
    
            

