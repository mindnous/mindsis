import gradio as gr
from rkwhisper import RKNNWhisper, get_arguments
import speech_recognition as sr
import os
import io
import numpy as np
import soundfile as sf
import time


args = get_arguments()
whisperunner = RKNNWhisper(args.task, args.encoder_model_path, args.decoder_model_path)


# Function to convert audio to text
def transcribe_audio(audio):
    start = time.time()
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            wav_bytes = audio_data.get_wav_data(convert_rate=16000)
            wav_stream = io.BytesIO(wav_bytes)
            audio_array, sampling_rate = sf.read(wav_stream)
            audio_array = audio_array.astype(np.float32)
            # Using Google Web Speech API for transcription
            text = whisperunner.voice2text(audio_array)
            print(f'[time: {time.time() - start:.2f}] out-text: ', text)
        except Exception as e:
            print('[transcribe_audio] error: ', e)
            return ""
    return text


# def transcribe_audio(audio):
#     audio_data = audio[1]
#     print(audio, type(audio), audio_data.shape)
#     print('hi')
#     rate = audio[0]
#     audio_array = np.float32((audio_data / 32768.))
#     print('audio_array: ', audio_array.shape, audio_array.min(), audio_array.max(), audio_array.std())
#     # return "hi"
#     try:
#         text = whisperunner.voice2text(audio_array)
#         print('text: ', text)
#     except sr.UnknownValueError:
#         text = "Could not understand the audio."
#     except sr.RequestError:
#         text = "Request failed; please try again."
#     return text

# Gradio interface setup
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio("microphone", type="filepath", label="Record Audio"),
    # inputs=gr.Audio("microphone", type="numpy", label="Record Audio"),
    outputs="text",
    title="Audio to Text Transcription",
    description="Record an audio clip and transcribe it to text.",
    live=True
)

iface.launch()
# iface.launch(server_name="0.0.0.0",
#              server_port=8080,
#              ssl_certfile='../cert.pem',
#              ssl_keyfile='../key.pem',
#              ssl_verify=False)
