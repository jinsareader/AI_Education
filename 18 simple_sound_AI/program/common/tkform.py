import tkinter
import process

_status = {'wait' : "대기 상태", "rec" : "녹음 중...", "cal" : "계산 중..."}
_sep = "###################################################\n"

class Form() :
    def __init__(self, onnx_file) :
        self.p = process.Process(onnx_file)
        self.window = tkinter.Tk()

        self.leftpanel = tkinter.PanedWindow(master = self.window, width = 70, height = 70)
        self.rightpanel = tkinter.PanedWindow(master = self.window, width = 70, height = 70)
        self.leftpanel.grid(row = 0, column=0)
        self.rightpanel.grid(row = 0, column=1)

        self.button = tkinter.Button(master = self.rightpanel, text = "녹음", width = 10, height = 2,  )
        self.button.config(command = self.start_record)
        self.label = tkinter.Label(master = self.rightpanel, text = _status["wait"], width = 20, height = 2, background = "gray", )  
        self.button.grid(row = 0, column= 0)
        self.label.grid(row = 0, column = 1)

        self.text = tkinter.Text(master = self.leftpanel, width = 70, height= 30, state = "disabled")
        self.text.pack()

        self.window.mainloop()
    def start_record(self) :
        self.button.config(command = None)
        self.p.start_record("input.wav")
        self.label.config(text = _status["rec"], background="green")
        self.button.config(command = self.end_record, text = "중지")
    def end_record(self) :
        self.button.config(command = None)
        self.p.end_record()
        self.ai_work()
        self.label.config(text = _status["wait"], background="gray")
        self.button.config(command = self.start_record, text = "녹음")
    def ai_work(self) :
        self.label.config(text = _status["cal"], background = "yellow")
        x = self.p.sound_preprocess("input.wav")
        y = self.p.cal(x)
        argmax = self.p.get_argmax(y)
        softmax = self.p.get_softmax(y)
        self.text.config(state="normal")
        self.text.insert(tkinter.END, argmax)
        self.text.insert(tkinter.END, softmax)
        self.text.insert(tkinter.END, _sep)
        self.text.config(state="disabled")


if __name__ == "__main__" :
    Form("word.onnx")