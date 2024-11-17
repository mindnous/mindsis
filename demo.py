import warnings
warnings.filterwarnings(action='ignore')
import gradio as gr
import speech_recognition as sr
import os
import io
import numpy as np
import soundfile as sf
import time
import argparse
import sys
import resource
from rkspeech2text.rkwhisper import initialize_speech2text_model
from rkllm_text.rkllm_main import RKLLM, get_user_input
from rkllm_text.rkllm_main import check_args_path, initialize_llm_model
from utils import click_js, audio_action, check_btn_status
from rktext2speech.rktts import text_to_speech_gtts, text_to_speech_offline, autoplay_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, default='rk3588', required=False, help='Target platform: e.g., rk3588/rk3576;')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board;')
    parser.add_argument('--disable_gtts', action="store_true", default=False, help='Whether to use gTTs (fast, but needs internet connection), or not (will use sherpa-onnx tts, slow but no internet connection required)')
    args = parser.parse_known_args()[0]
    
    text_to_speech = text_to_speech_offline if args.disable_gtts else text_to_speech_gtts
    print('disable gTTs: ', args.disable_gtts)

    check_args_path(args)
    
    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
    
    # INIT Speech-to-text model
    whisperunner = initialize_speech2text_model()

    # INIT LLM model
    rkllm_model = initialize_llm_model(args)
    
    # Create a Gradio interface
    with gr.Blocks(title="Realtime AI Assistant") as iface:
        gr.Markdown("<div align='center'><font size='70'> Chat with RKLLM </font></div>")
        gr.Markdown("### Enter your question in the inputTextBox and press the Enter key to chat with the RKLLM model.")
        # Create a Chatbot component to display conversation history
        rkllmServer = gr.Chatbot(height=400)
        # Create a Textbox component for user message input
        msg = gr.Textbox(placeholder="Please input your question here...", label="inputTextBox")
        # Create a Button component to clear the chat history.
        audio_box = gr.Microphone(label="Audio", elem_id='audio', type='filepath', visible=False)
        
        with gr.Row():
            audio_btn = gr.Button('Speak')
            clear = gr.Button("Clear")
        audio_answer = gr.Audio(label="speaker", type="numpy", elem_id="speaker", autoplay=True, visible=False)
        

        # Submit the user's input message to the get_user_input function and immediately update the chat history.
        # Then call the get_RKLLM_output function to further update the chat history.
        # The queue=False parameter ensures that these updates are not queued, but executed immediately.
        msg.submit(get_user_input, [msg, rkllmServer], [msg, rkllmServer], queue=False).then(rkllm_model.get_RKLLM_output, inputs=rkllmServer, outputs=rkllmServer)

        audio_btn.click(fn=audio_action, inputs=audio_btn, outputs=audio_btn).\
                then(fn=lambda: None, js=click_js()).\
                then(fn=check_btn_status, inputs=audio_btn).\
                success(fn=whisperunner.transcribe_audio, inputs=(audio_box, audio_btn), outputs=msg).\
                success(lambda :None, None, audio_box, queue=False).\
                success(get_user_input, [msg, rkllmServer], [msg, rkllmServer]).\
                success(rkllm_model.get_RKLLM_output, inputs=rkllmServer, outputs=rkllmServer).\
                success(text_to_speech, inputs=rkllmServer, outputs=audio_answer).\
                success(lambda : None, None, None, js=autoplay_audio)
        

        # When the clear button is clicked, perform a no-operation (lambda: None) and immediately clear the chat history.
        clear.click(lambda: None, None, rkllmServer, queue=False)

    # Enable the event queue system, and Start the Gradio application..
    # iface.queue().launch(debug=True)
    iface.queue().launch(server_name="0.0.0.0",
                server_port=8080,
                ssl_certfile='../cert.pem',
                ssl_keyfile='../key.pem',
                ssl_verify=False)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")
