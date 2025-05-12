from konlpy.tag import Kkma as word_div
import threading

class Preprocessor() :
    def __init__(self) :
        self._result = None
        self._error = None
    def _process_with_timeout(self, s : str, timeout : int) :
        def work() :
            try :
                self._result = word_div().morphs(s)
            except Exception as e :
                self._error = e        
        self._error = None
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive() :
            raise Exception(f"Error : Time Exceeded")
        if self._error is not None :
            raise Exception(str(self._error))

        return self._result
    def process(self, s : str, timeout : int = 10) :
        result = self._process_with_timeout(s, timeout)
        return result