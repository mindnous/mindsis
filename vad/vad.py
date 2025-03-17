# ref: https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py#L9
import onnxruntime as ort
import numpy as np
import soundfile as sf
import time


class VADModelUtils:
    @staticmethod
    def load_model(model_path, force_onnx_cpu=False):
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # set options
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        # set providers
        providers = ort.get_available_providers() if not force_onnx_cpu else ['CPUExecutionProvider']  # noqa

        # load model
        session = ort.InferenceSession(model_path,
                                       providers=providers,
                                       sess_options=opts)
        return session, input_name, output_name, providers


    def reset_states(self, batch_size):
        self._state = np.zeros((2, batch_size, 128), dtype='float32')
        self._context = np.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0


class VADUtils:
    def _validate_input(self, x, sr: int):
        if x.ndim == 1:
            x = x[None, ...]

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:,::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def _validate_audio(self, out_thresh_sum, cbatch):
        print('go here', out_thresh_sum, cbatch, self.vad_state, self.vad_state_memory)
        if out_thresh_sum >= cbatch / 2:
            self.vad_state = True
            self.vad_state_memory = 0
        elif (out_thresh_sum < max(cbatch // 2, 1)):
            self.vad_state_memory += (self.num_samples * cbatch - out_thresh_sum)
        
        if (self.vad_state_memory >= self.sample_rates // 2) and self.vad_state:
            self.vad_state = False
        print('go here2', self.vad_state, self.vad_state_memory)

class VAD(VADModelUtils, VADUtils):
    def __init__(self, model_path, force_onnx_cpu=False, batch_size=1):
        self.sample_rates = np.array([16000], dtype='int64')
        self.num_samples = 512
        self.context_size = 64
        self.threshold = 0.5
        self.batch_size = batch_size
        self.vad_state = False
        self.vad_state_memory = 0
        self.model_path = model_path
        self.model, self.input_name, self.output_name, self.providers = self.load_model(model_path, force_onnx_cpu)
        print('[VAD] model info: ', self.model.get_providers(), self.providers)
        self.reset_states(self.batch_size)

    def reset_states_when_empty(self, batch_size=1):
        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

    def __call__(self, x, sr = 16000):
        x, sr = self._validate_input(x, sr)
        bs = x.shape[0]

        self.reset_states_when_empty(bs)

        if not len(self._context):
            self._context = np.zeros([bs, self.context_size], dtype='float32')

        x = np.concatenate([self._context, x], axis=1, dtype='float32')
        ort_inputs = {'input': x, 'state': self._state, 'sr': self.sample_rates}
        ort_inputs.pop('sr')
        # for key in ort_inputs.keys():
        #     print(key, ort_inputs[key].shape, ort_inputs[key].dtype)
        ort_outs = self.model.run(None, ort_inputs)
        out, self._state = ort_outs
        # print('__CALL__out: ', out.shape, self._state.shape)

        self._context = x[..., -self.context_size:]
        self._last_sr = sr
        self._last_batch_size = bs
        return out

    def audio_forward(self, x, sr: int):
        outs = []
        x, sr = self._validate_input(x, self.sample_rates)
        bs = self.batch_size
        self.reset_states(bs)

        if x.shape[1] % self.num_samples:
            pad_num = self.num_samples - (x.shape[1] % self.num_samples)
            x = np.pad(x, ((0, 0), (0, pad_num)), 'constant', constant_values=0.0)

        out_wavs = []
        for i in range(0, x.shape[1], self.num_samples * bs):
            wavs_batch = x[:, i:i + self.num_samples * bs].copy()
            start = time.time()
            cbatch = wavs_batch.shape[-1] // self.num_samples
            wavs_batch = wavs_batch.reshape(cbatch, self.num_samples)
            out_chunk = self.__call__(wavs_batch, sr).flatten()
            out_thresh_sum = (out_chunk > self.threshold).sum()

            self._validate_audio(out_thresh_sum, cbatch)
            
            if not self.vad_state:
                continue

            print(i, 'time: {:.4f}'.format(time.time() - start), out_chunk.shape, '|', 
                  out_thresh_sum, self.vad_state, self.vad_state_memory, '|', self.threshold)
            print('=' * 50)
            outs.append(out_chunk)
            out_wavs.append(wavs_batch.flatten())

        out_probs = np.concatenate(outs if len(outs) else [np.zeros(0)])
        out_wavs = np.concatenate(out_wavs if len(outs) else [np.zeros(0)])
        return out_probs, out_wavs


if __name__ == '__main__':
    model_path = '../model/silero_vad_half.onnx'
    vad = VAD(model_path, force_onnx_cpu=True, batch_size=8)
    # x, sr = sf.read('../model/ttsout_low.wav', samplerate=16000, dtype='float32', subtype='PCM_16', channels=1, format='RAW')
    # x, sr = sf.read('../examples/asknot.wav', samplerate=16000, dtype='float32', subtype='PCM_16', channels=1, format='RAW')
    # x, sr = sf.read('../examples/check.wav', samplerate=16000, dtype='float32', subtype='PCM_16', channels=1, format='RAW')
    x, sr = sf.read('../examples/check2.wav', samplerate=16000, dtype='float32', subtype='PCM_16', channels=1, format='RAW')
    print('sr: ', sr, x.shape)
    # x, sr = sf.read('../model/ttsout.wav')
    # raise
    # out = vad(x, sr)
    vad.threshold = 0.5
    start = time.time()
    out_probs, out_wavs = vad.audio_forward(x, sr)
    print('out_wavs: ', out_probs.shape, out_wavs.shape[0], 'time: {:.4f}'.format(time.time() - start))
    print(out_probs.shape)