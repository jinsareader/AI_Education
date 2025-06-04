#소리 녹음버튼 있는 버튼 폼
#소리 입력해서 단어 알아맞추는 프로그램램

import tkinter 
import process

label_prt = {'record' : ("녹음 중", 'green'), 'wait' : ("대기 중", 'gray'), 'cal' : ("계산 중", 'yellow')}
sp = '######################################\n\n'

class MainForm() :
    def __init__(self) :
        self.record_flg = False #녹음 중인지 판단하는 flg
        self.p = process.Process('sound_ai.onnx')
        self.window = tkinter.Tk()
        
        self.leftpanel = tkinter.PanedWindow(master = self.window)
        self.rightpanel = tkinter.PanedWindow(master = self.window)
        self.leftpanel.grid(row=0,column=0)
        self.rightpanel.grid(row=0,column=1)

        self.text = tkinter.Text(master = self.leftpanel, width = 50, height = 24, state="disabled")
        self.text.pack()

        self.button = tkinter.Button(master = self.rightpanel, text = "녹음", width = 10, height = 2)
        self.button.config(command=self.record)
        self.button.grid(row=0,column=0)
        self.label = tkinter.Label(master = self.rightpanel, text = label_prt["wait"][0], background= label_prt["wait"][1], width = 20, height=2)
        self.label.grid(row=0,column=1)

        self.window.mainloop()

    def record(self) :
        self.button.config(command = None)

        if not self.record_flg : # 녹음 중이 아니면
            self.record_flg = True
            self.p.record()
            self.change_label('record')
        else : #녹음 중이면
            self.record_flg = False
            self.p.record_end()
            self.cal() #계산
            self.change_label('wait')
        
        self.button.config(command = self.record)

    def cal(self) :
        self.change_label('cal')
        y = self.p.cal() #onnx 계산
        argmax = self.p.get_argmax(y) #과일 종류
        softmax = self.p.get_softmax(y) #종류별 확률
        self.text.config(state = "normal")
        # self.text.insert(tkinter.END, str(y) + "\n")
        self.text.insert(tkinter.END, argmax)
        self.text.insert(tkinter.END, softmax)
        self.text.insert(tkinter.END, sp)
        self.text.config(state = "disabled")


    def change_label(self, state) :
        self.label.config(text = label_prt[state][0], background=label_prt[state][1])


if __name__ == "__main__" : #해당 파일을 직접 실행할 때에만 아래 블록의 코드를 실행 # 단위테스트 용
    MainForm()