import numpy as np
import ffmpeg
import pathlib
FILEPATH = pathlib.Path(__file__).parent.absolute()


## SPEECH-CLASSIFY CONFIG
SCLS_SAMPLE_RATE = 16000
SCLS_THREADS = 1
SCLS_providers=['CUDAExecutionProvider',
                'CoreMLExecutionProvider',
                'CPUExecutionProvider']
SCLS_CLS_PATH = f'{FILEPATH}/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'


## SPEECH-TO-TEXT CONFIG
STT_SAMPLE_RATE = 16000
STT_N_FFT = 400
STT_HOP_LENGTH = 160
STT_N_MELS = 80
STT_CHUNK_LENGTH = 20


## TEXT-TO-SPEECH CONFIG
# MODELPATH = f'{FILEPATH}/vits-melo-tts-zh_en/'
# MODELPATH = f'{FILEPATH}/vits-piper-en_US-lessac-medium/'
TTS_MODELPATH = f'{FILEPATH}/vits-piper-en_US-ryan-medium/'
TTS_OUTPUTPATH = f'{FILEPATH}/ttsout.wav'
TTS_THREADS = 1
TTS_MAX_NUM_SENTENCES = 1
TTS_PROVIDER = 'cpu'
TTS_SPEED = 1.0
TTS_SID = 0



## GENERAL USAGE CONFIG
def get_mel_filters(filters_path =  f"{FILEPATH}/mel_80_filters.txt"):
    return np.loadtxt(filters_path, dtype=np.float32).reshape((80, 201))


G_SAMPLE_RATE = 16000
G_N_FFT = 400
G_HOP_LENGTH = 160
G_N_MELS = 80
G_MEL_FILTERS = get_mel_filters()


def numpy_sliding_window_view(x, window_shape, step=1):
    shape = ((x.shape[-1] - window_shape) // step + 1,) + (window_shape,)
    strides = (step * x.strides[-1],) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def numpy_stft(audio: np.ndarray):
    window = np.hanning(G_N_FFT)
    num_frames = 1 + (audio.size - G_N_FFT) // G_HOP_LENGTH
    if (audio.size - G_N_FFT) % G_HOP_LENGTH > 0:
        num_frames += 1
    audio_padded = np.pad(audio, pad_width=(G_N_FFT//2, G_N_FFT//2), mode='constant')
    frames = numpy_sliding_window_view(audio_padded, G_N_FFT, G_HOP_LENGTH)
    frames = frames[:num_frames]
    stft = np.fft.rfft(frames * window, axis=-1)

    cpstft = (np.abs(stft[:,:G_N_FFT//2 + 1]) ** 2).T
    magnitudes = cpstft.astype(audio.dtype)
    return magnitudes


def log_mel_spectrogram(audio, padding=0):
    magnitudes = numpy_stft(audio)
    mel_spec = G_MEL_FILTERS @ magnitudes
    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def load_audio(file: str):
    # https://github.com/PINTO0309/whisper-onnx-tensorrt/blob/main/whisper/audio.py
    out, _ = (
        ffmpeg.input(file, threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=STT_SAMPLE_RATE)
        .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
