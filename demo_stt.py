import gradio as gr
from rkspeech2text.rkwhisper import RKNNWhisper, get_arguments
from utils import audio_action, check_btn_status, click_js
import os
# import io
import numpy as np
# import soundfile as sf
import time

# os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
# os.environ["GRADIO_SERVER_PORT"] = "8080"

args = get_arguments()
whisperunner = RKNNWhisper(args.task, args.encoder_model_path, args.decoder_model_path)


# # Gradio interface setup
# iface = gr.Interface(
#     fn=whisperunner.transcribe_audio,
#     inputs=gr.Microphone(label="Audio", elem_id='audio', type='filepath'),
#     # inputs=gr.Audio("microphone", type="numpy", label="Record Audio"),
#     outputs="text",
#     title="Audio to Text Transcription",
#     description="Record an audio clip and transcribe it to text.",
#     live=True
# )

with gr.Blocks() as iface:
    msg = gr.Textbox()
    audio_box = gr.Microphone(label="Audio", elem_id='audio', type='filepath')#, visible=False)

    with gr.Row():
        audio_btn = gr.Button('Speak')
        clear = gr.Button("Clear")
    
    # rkllm_chat = gr.Chatbot(height=400)
              
    audio_btn.click(fn=audio_action, inputs=audio_btn, outputs=audio_btn).\
              then(fn=lambda: None, js=click_js()).\
              then(fn=check_btn_status, inputs=audio_btn).\
              success(fn=whisperunner.transcribe_audio, inputs=(audio_box, audio_btn), outputs=msg).\
              then(lambda :None, None, audio_box, queue=False)

    clear.click(lambda: None, None, msg, queue=False)

iface.launch()
