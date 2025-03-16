import os
import sys
import time
import numpy as np
import ffmpeg
import soundfile as sf
import speech_recognition as sr
import pathlib
import onnxruntime as ort
from numba import njit
FILEPATH = pathlib.Path(__file__).parent.absolute()
sys.path.append(f'{FILEPATH}/../model/')
import model_config as cfg
SAMPLE_RATE = cfg.SCLS_SAMPLE_RATE


@njit(fastmath=True)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SpeechClassify:
    def __init__(self, model_path=cfg.SCLS_CLS_PATH):
        self.model_path = model_path
        self.model = None
        self.ref_feat = None
        self.load_model()

    def add_features(self, output):
        if self.ref_feat is None:
            self.ref_feat = output.copy()
        else:
            self.ref_feat = np.concatenate((self.ref_feat, output), axis=0)

    def load_model(self):
        self.model = ort.InferenceSession(self.model_path, providers=cfg.SCLS_providers)
        print('[SpeechClassify] MODEL PROVIDERS:', self.model.get_providers())

    def preprocess(self, audio_file):
        audio = cfg.load_audio(audio_file) if type(audio_file) == str else audio_file  # noqa
        audio = cfg.log_mel_spectrogram(audio, cfg.G_N_MELS).astype(np.float32).T
        audio = np.expand_dims(audio, axis=0)
        return audio
    
    def predict(self, audio):
        return self.model.run(None, {'x': audio})[0]
    
    def postprocess(self, output):
        return cosine_similarity(output, self.ref_feat.T)
    
    def __call__(self, audio_file, add_feat=False):
        audio_inp = self.preprocess(audio_file)
        audio_feat = self.predict(audio_inp)
        if add_feat:
            self.add_features(audio_feat)
            sim_out = np.array([])
        else:
            sim_out = self.postprocess(audio_feat)
        return audio_inp, audio_feat, sim_out


if __name__ == '__main__':
    CLS_PATH = cfg.SCLS_CLS_PATH
    scls = SpeechClassify(CLS_PATH)
    # Load audio file
    audio_file = '../model/ttsout.wav'
    # Predict

    temp = None
    for _ in range(5):
        start = time.time()
        audio_inp, audio_feat, sim_out = scls(audio_file, add_feat=_==0)
        print('time: {:.3f}'.format(time.time() - start))
        print('audio_inp & audio_feat:', audio_inp.shape, audio_feat.shape, sim_out.shape, sim_out)
        print('=' * 50)
