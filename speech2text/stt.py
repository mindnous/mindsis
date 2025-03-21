import os
import numpy as np
# from rknn.api import RKNN
import time
import os
import sys
import argparse
import soundfile as sf
import speech_recognition as sr
import io
import pathlib
FILEPATH = pathlib.Path(__file__).parent.absolute()
sys.path.append(f'{FILEPATH}/../model/')
import model_config as cfg

# whisper onnx reference: https://huggingface.co/onnx-community/whisper-base


class STTWrapperUtils:
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
        x_mel = np.zeros((cfg.G_N_MELS, self.MAX_LENGTH), dtype=np.float32)
        real_length = audio_array.shape[1] if audio_array.shape[1] <= self.MAX_LENGTH else self.MAX_LENGTH
        x_mel[:, :real_length] = audio_array[:, :real_length]
        return x_mel


class STTWrapperInference:
    # Function to convert audio to text
    def transcribe_audio(self, audio, audio_btn):
        start = time.time()
        recognizer = sr.Recognizer()
        print('[transcribe_audio] audio: ', audio)
        print('[transcribe_audio] audio_btn: ', audio_btn)
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            try:
                wav_bytes = audio_data.get_wav_data(convert_rate=16000)
                wav_stream = io.BytesIO(wav_bytes)
                audio_array, sampling_rate = sf.read(wav_stream)
                audio_array = audio_array.astype(np.float32)
                # Using Google Web Speech API for transcription
                text = self.voice2text(audio_array)
                print(f'[Speech2Text | time: {time.time() - start:.2f}] out-text: ', text)
            except Exception as e:
                print('[transcribe_audio] error: ', e)
                return ""
        return text

    def run_encoder(self, encoder_model, in_encoder):
        if self.MODEL_MODE == 'rknn':
            out_encoder = encoder_model.inference(inputs=[in_encoder])[0]
        elif self.MODEL_MODE == 'onnx':
            inp={'input_features':in_encoder}
            out_encoder = encoder_model.run(None, inp)[0]
        return out_encoder

    def _decode(self, decoder_model, tokens, out_encoder):
        tokens = np.asarray([tokens], dtype="int64")
        if self.MODEL_MODE == 'rknn':
            out_decoder = decoder_model.inference([tokens, out_encoder])[0]
        elif self.MODEL_MODE == 'onnx':
            inp = {'input_ids':tokens, 'encoder_hidden_states':out_encoder}
            out_decoder = decoder_model.run(None, inp)[0]
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


class STTWrapper(STTWrapperUtils, STTWrapperInference):
    def __init__(self, 
                 task='en',
                 encoder_model_path = '../model/decoder_model_int8.onnx',
                 decoder_model_path = '../model/encoder_model_int8.onnx'):
        
        # Set inputs
        if task == "en":
            vocab_path = f"{FILEPATH}/vocab_en.txt"
            self.task_code = 50259
        elif args.task == "zh":
            vocab_path = f"{FILEPATH}/vocab_zh.txt"
            self.task_code = 50260
        else:
            raise NotImplementedError("\n\033[1;33mCurrently only English or Chinese recognition tasks are supported. Please specify --task as en or zh\033[0m")
        self.vocab = self.read_vocab(vocab_path)
        self.mel_filters = cfg.get_mel_filters()
        self.encoder_model = self.init_model(encoder_model_path)
        self.decoder_model = self.init_model(decoder_model_path)
        print('[init_model] encoder_model: ', self.encoder_model, encoder_model_path, os.path.exists(encoder_model_path))
        print('[init_model] decoder_model: ', self.decoder_model, decoder_model_path, os.path.exists(decoder_model_path))
        print('[init_model] init_model finish.')
        print('*' * 50)

    def init_model(self, model_path):
        if model_path.endswith(".rknn"):
            from rknnlite.api import RKNNLite
            self.device_id = RKNNLite.NPU_CORE_0
            self.MODEL_MODE = 'rknn'
            self.N_SAMPLES = cfg.STT_CHUNK_LENGTH * cfg.STT_SAMPLE_RATE
            self.MAX_LENGTH = cfg.STT_CHUNK_LENGTH * 100
            # Create RKNN object
            model = RKNNLite()

            # Load RKNN model
            ret = model.load_rknn(model_path)
            if ret != 0:
                print('[init_model] Load RKNN model \"{}\" failed!'.format(model_path))
                exit(ret)
            print('[init_model] Loading RKNN model done.', model_path, os.path.exists(model_path), self.device_id)

            ret = model.init_runtime(core_mask=self.device_id)
            if ret != 0:
                print('[init_model] Init runtime environment failed')
                exit(ret)
            print('[init_model] Init runtime environment done')
        if model_path.endswith(".onnx"):
            CHUNK_LENGTH = 30
            self.N_SAMPLES = CHUNK_LENGTH * cfg.STT_SAMPLE_RATE
            self.MAX_LENGTH = CHUNK_LENGTH * 100
            self.MODEL_MODE = 'onnx'
            import onnxruntime as ort
            model = ort.InferenceSession(model_path, providers=['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
            print('[init_model] Loading ONNX model done.', model_path, os.path.exists(model_path), model.get_providers())
        return model

    def release_model(self):
        if self.MODEL_MODE == 'rknn':
            self.encoder_model.release()
            self.decoder_model.release()
        elif self.MODEL_MODE == 'onnx':
            self.encoder_model = None
            self.decoder_model = None

    def voice2text(self, audio):
        print('voice to text')
        if isinstance(audio, str):
            audio_array = cfg.load_audio(audio)
        else:
            audio_array = np.asarray(audio, dtype=np.float32)
        audio_array = cfg.log_mel_spectrogram(audio_array, cfg.G_N_MELS)
        x_mel = self.pad_or_trim(audio_array)[None]
        out_encoder = self.run_encoder(self.encoder_model, x_mel)
        result = self.run_decoder(self.decoder_model, out_encoder, self.vocab, self.task_code)
        return result


def get_arguments():
    parser = argparse.ArgumentParser(description='Speech2Text Demo, example: python3 stt.py --encoder_model_path ../model/encoder_model_fp16.onnx --decoder_model_path ../model/decoder_model_int8.onnx --audio ../examples', add_help=True)
    # basic params
    parser.add_argument('--encoder_model_path', default='./model/whisper_encoder_base_20s.rknn', required=False, type=str, help='model path, could be .rknn or .onnx file')
    parser.add_argument('--decoder_model_path', default='./model/whisper_decoder_base_20s.rknn', required=False, type=str, help='model path, could be .rknn or .onnx file')
    parser.add_argument('--task', type=str, default='en', required=False, help='recognition task, could be en or zh')
    parser.add_argument('--audio_path', default='./examples/', type=str, help='audio path')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    print('running....')
    args = get_arguments()
    whisperunner = STTWrapper(args.task, args.encoder_model_path, args.decoder_model_path)
    
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