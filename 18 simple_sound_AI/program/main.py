import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "common"))

from common.tkform import Form

Form("word.onnx")