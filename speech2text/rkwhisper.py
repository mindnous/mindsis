import os
import numpy as np
# from rknn.api import RKNN
import time
# import torch
from rknnlite.api import RKNNLite
import argparse
import ffmpeg
import soundfile as sf
import speech_recognition as sr
import io


class RKNNWhisperUtils:
    @staticmethod
    def get_char_index(c):
        if 'A' <= c <= 'Z':
            return ord(c) - ord('A')
        elif 'a' <= c <= 'z':
            return ord(c) - ord('a') + (ord('Z') - ord('A') + 1)
        elif '0' <= c <= '9':
            return ord(c) - ord('0') + (ord('Z') - ord('A')) + (ord('z') - ord('a')) + 2
        elif c == '+':
            return 62
        elif c == '/':
            return 63
        else:
            print(f"Unknown character {ord(c)}, {c}")
            exit(-1)

    def base64_decode(self, encoded_string):
        if not encoded_string:
            print("Empty string!")
            exit(-1)

        output_length = len(encoded_string) // 4 * 3
        decoded_string = bytearray(output_length)

        index = 0
        output_index = 0
        while index < len(encoded_string):
            if encoded_string[index] == '=':
                return " "

            first_byte = (self.get_char_index(encoded_string[index]) << 2) + ((self.get_char_index(encoded_string[index + 1]) & 0x30) >> 4)
            decoded_string[output_index] = first_byte

            if index + 2 < len(encoded_string) and encoded_string[index + 2] != '=':
                second_byte = ((self.get_char_index(encoded_string[index + 1]) & 0x0f) << 4) + ((self.get_char_index(encoded_string[index + 2]) & 0x3c) >> 2)
                decoded_string[output_index + 1] = second_byte

                if index + 3 < len(encoded_string) and encoded_string[index + 3] != '=':
                    third_byte = ((self.get_char_index(encoded_string[index + 2]) & 0x03) << 6) + self.get_char_index(encoded_string[index + 3])
                    decoded_string[output_index + 2] = third_byte
                    output_index += 3
                else:
                    output_index += 2
            else:
                output_index += 1

            index += 4
                
        return decoded_string.decode('utf-8', errors='replace')

    @staticmethod
    def read_vocab(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = {}
            for line in f:
                if len(line.strip().split(' ')) < 2:
                    key = line.strip().split(' ')[0]
                    value = ""
                else:
                    key, value = line.strip().split(' ')
                vocab[key] = value
        return vocab

    def pad_or_trim(self, audio_array):
        x_mel = np.zeros((self.N_MELS, self.MAX_LENGTH), dtype=np.float32)
        real_length = audio_array.shape[1] if audio_array.shape[1] <= self.MAX_LENGTH else self.MAX_LENGTH
        x_mel[:, :real_length] = audio_array[:, :real_length]
        return x_mel

    def get_mel_filters(self, filters_path =  "./model/mel_80_filters.txt"):
        mels_data = np.loadtxt(filters_path, dtype=np.float32).reshape((80, 201))
        # return torch.from_numpy(mels_data)
        return mels_data
    

    def load_audio(self, file: str):
        sr = self.SAMPLE_RATE
        # https://github.com/PINTO0309/whisper-onnx-tensorrt/blob/main/whisper/audio.py
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def log_mel_spectrogram(self, audio, n_mels, padding=0):
        magnitudes = self.numpy_stft(audio, self.N_FFT, self.HOP_LENGTH)
        mel_spec = self.mel_filters @ magnitudes
        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
    
    def numpy_stft(self, audio: np.ndarray, N_FFT: int, HOP_LENGTH: int):
        window = np.hanning(N_FFT)
        num_frames = 1 + (audio.size - N_FFT) // HOP_LENGTH
        if (audio.size - N_FFT) % HOP_LENGTH > 0:
            num_frames += 1
        audio_padded = np.pad(audio, pad_width=(N_FFT//2, N_FFT//2), mode='constant')
        frames = self.numpy_sliding_window_view(audio_padded, N_FFT, HOP_LENGTH)
        frames = frames[:num_frames]
        stft = np.fft.rfft(frames * window, axis=-1)

        cpstft = (np.abs(stft[:,:N_FFT//2 + 1]) ** 2).T
        magnitudes = cpstft.astype(audio.dtype)
        return magnitudes

    @staticmethod
    def numpy_sliding_window_view(x, window_shape, step=1):
        shape = ((x.shape[-1] - window_shape) // step + 1,) + (window_shape,)
        strides = (step * x.strides[-1],) + x.strides
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


class RKNNWhisperInference:
    # Function to convert audio to text
    def transcribe_audio(self, audio, audio_btn):
        start = time.time()
        recognizer = sr.Recognizer()
        print('audio: ', audio)
        print('audio_btn: ', audio_btn)
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            try:
                wav_bytes = audio_data.get_wav_data(convert_rate=16000)
                wav_stream = io.BytesIO(wav_bytes)
                audio_array, sampling_rate = sf.read(wav_stream)
                audio_array = audio_array.astype(np.float32)
                # Using Google Web Speech API for transcription
                text = self.voice2text(audio_array)
                print(f'[RKNNWhipser | time: {time.time() - start:.2f}] out-text: ', text)
            except Exception as e:
                print('[transcribe_audio] error: ', e)
                return ""
        return text

    @staticmethod
    def run_encoder(encoder_model, in_encoder):
        out_encoder = encoder_model.inference(inputs=[in_encoder])[0]
        return out_encoder

    @staticmethod
    def _decode(decoder_model, tokens, out_encoder):
        out_decoder = decoder_model.inference([np.asarray([tokens], dtype="int64"), out_encoder])[0]
        return out_decoder

    def run_decoder(self, decoder_model, out_encoder, vocab, task_code):
        end_token = 50257 # tokenizer.eot
        tokens = [50258, task_code, 50359, 50363] # tokenizer.sot_sequence_including_notimestamps
        timestamp_begin = 50364 # tokenizer.timestamp_begin

        max_tokens = 12
        tokens_str = ''
        pop_id = max_tokens

        tokens = tokens * int(max_tokens/4)
        next_token = 50258 # tokenizer.sot

        while next_token != end_token:
            out_decoder = self._decode(decoder_model, tokens, out_encoder)
            next_token = out_decoder[0, -1].argmax()
            next_token_str = vocab[str(next_token)]
            tokens.append(next_token)

            if next_token == end_token:
                tokens.pop(-1)
                next_token = tokens[-1]
                break
            if next_token > timestamp_begin:
                continue
            if pop_id >4:
                pop_id -= 1

            tokens.pop(pop_id)
            tokens_str += next_token_str

        result = tokens_str.replace('\u0120', ' ').replace('<|endoftext|>', '').replace('\n', '')
        if task_code == 50260: # TASK_FOR_ZH
            result = self.base64_decode(result)
        return result


class RKNNWhisper(RKNNWhisperUtils, RKNNWhisperInference):
    SAMPLE_RATE = 16000
    N_FFT = 400
    HOP_LENGTH = 160
    CHUNK_LENGTH = 20
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
    MAX_LENGTH = CHUNK_LENGTH * 100
    N_MELS = 80

    def __init__(self, 
                 task='en',
                 encoder_model_path = './model/whisper_decoder_base_20s.rknn',
                 decoder_model_path = './model/whisper_decoder_base_20s.rknn'):
        self.device_id = RKNNLite.NPU_CORE_1
        
        # Set inputs
        if task == "en":
            vocab_path = "./model/vocab_en.txt"
            self.task_code = 50259
        elif args.task == "zh":
            vocab_path = "./model/vocab_zh.txt"
            self.task_code = 50260
        else:
            raise NotImplementedError("\n\033[1;33mCurrently only English or Chinese recognition tasks are supported. Please specify --task as en or zh\033[0m")
        self.vocab = self.read_vocab(vocab_path)
        self.mel_filters = self.get_mel_filters()
        self.encoder_model = self.init_model(encoder_model_path, self.device_id)
        self.decoder_model = self.init_model(decoder_model_path, self.device_id)
        print('[init_model] encoder_model: ', self.encoder_model, encoder_model_path, os.path.exists(encoder_model_path))
        print('[init_model] decoder_model: ', self.decoder_model, decoder_model_path, os.path.exists(decoder_model_path))
        print('[init_model] init_model finish.')
        print('*' * 50)

    @staticmethod
    def init_model(model_path, device_id=None):
        if model_path.endswith(".rknn"):
            # Create RKNN object
            model = RKNNLite()

            # Load RKNN model
            ret = model.load_rknn(model_path)
            if ret != 0:
                print('[init_model] Load RKNN model \"{}\" failed!'.format(model_path))
                exit(ret)
            print('[init_model] Loading RKNN model done.', model_path, os.path.exists(model_path), device_id)

            ret = model.init_runtime(core_mask=device_id)
            if ret != 0:
                print('[init_model] Init runtime environment failed')
                exit(ret)
            print('[init_model] Init runtime environment done')
        return model

    def release_model(self):
        self.encoder_model.release()
        self.decoder_model.release()

    def voice2text(self, audio):
        if isinstance(audio, str):
            audio_array = self.load_audio(audio)
        else:
            audio_array = np.asarray(audio, dtype=np.float32)
        audio_array = self.log_mel_spectrogram(audio_array, self.N_MELS)
        x_mel = self.pad_or_trim(audio_array)[None]
        out_encoder = self.run_encoder(self.encoder_model, x_mel)
        result = self.run_decoder(self.decoder_model, out_encoder, self.vocab, self.task_code)
        return result


def initialize_speech2text_model(task: str='en',
                                 encoder_model_path: str='./model/whisper_encoder_base_20s.rknn',
                                 decoder_model_path: str='./model/whisper_decoder_base_20s.rknn'):
    # INIT Speech-to-text model
    print("=========init Speech-to-text....===========")
    whisperunner = RKNNWhisper(task, encoder_model_path, decoder_model_path)
    print("Speech-to-text Model has been initialized successfully！")
    print("==============================")
    return whisperunner


def get_arguments():
    parser = argparse.ArgumentParser(description='Whisper Python Demo', add_help=True)
    # basic params
    parser.add_argument('--encoder_model_path', default='./model/whisper_encoder_base_20s.rknn', required=False, type=str, help='model path, could be .rknn or .onnx file')
    parser.add_argument('--decoder_model_path', default='./model/whisper_decoder_base_20s.rknn', required=False, type=str, help='model path, could be .rknn or .onnx file')
    parser.add_argument('--task', type=str, default='en', required=False, help='recognition task, could be en or zh')
    parser.add_argument('--audio_path', default='./examples/', type=str, help='audio path')
    parser.add_argument('--device_id', type=str, default=RKNNLite.NPU_CORE_0, help='device id')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = get_arguments()
    whisperunner = RKNNWhisper(args.task, args.encoder_model_path, args.decoder_model_path)
    
    # Inference
    bpath = args.audio_path
    print('--> Running model', args.audio_path, os.path.exists(args.audio_path))
    args.audio_path = [os.path.join(bpath, ap) for ap in os.listdir(args.audio_path) if ap.endswith('wav') or ap.endswith('mp3')]
    print('audio test list: ', args.audio_path)
    for i in range(10):
        start = time.time()
        pickaudio = np.random.choice(args.audio_path)
        audio_data, _ = sf.read(pickaudio)
        audio_array = np.array(audio_data, dtype=np.float32)
        result = whisperunner.voice2text(audio_array)
        print(i, f'infer speed: {time.time()-start:.3f}s', pickaudio)
        print("Whisper output:", result)
        print('=' * 50)

    # Release
    whisperunner.release_model()