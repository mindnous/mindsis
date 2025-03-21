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
import pathlib
from speechclassify import SpeechClassify
from speech2text import STTWrapper
from llm import LLMWrapper
from text2speech import TTSWrapper
from utils import click_js, audio_action, check_btn_status, get_user_input, autoplay_audio
FILEPATH = pathlib.Path(__file__).parent.absolute()

# ref: python3 demo.py --llm_model "ollama/qwen2.5:latest" --llm_type "llm" --stt_modelenc model/encoder_model_fp16.onnx --stt_modeledec model/decoder_model_int8.onnx --target_platform "ollama"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_model', type=str, required=True, help='Modelname/modelpath of your LLM')
    parser.add_argument('--llm_type', type=str, default="llm", help='Pick one: vlm | llm')
    parser.add_argument('--llm_modelurl', type=str, default="http://localhost:11434", help='URL to the OPENAI-API-MODEL')
    parser.add_argument('--target_platform', type=str, default='ollama', required=False, help='Target platform: e.g., ollama/openai/litellm/ollama_offline/mlc/rkllm/')
    parser.add_argument('--stt_lang', type=str, default='en', help='Pick language, supported: [en]')
    parser.add_argument('--stt_modelenc', default=f'{FILEPATH}/model/encoder_model_fp16.onnx', required=False, type=str, help='model path, could be .rknn or .onnx file')
    parser.add_argument('--stt_modeldec', default=f'{FILEPATH}/model/decoder_model_int8.onnx', required=False, type=str, help='model path, could be .rknn or .onnx file')
    
    args = parser.parse_known_args()[0]
    
    # Set resource limit
    if args.target_platform == 'rkllm':
        resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
    
    # INIT Speech-to-text model
    stt_runner = STTWrapper(args.stt_lang, args.stt_modelenc, args.stt_modeldec)

    # INIT LLM model
    model_info = dict(llm_type=args.llm_type,
                      model_url=args.llm_modelurl)
    llm_runner = LLMWrapper(args.llm_model,
                            model_info=model_info,
                            model_type=args.target_platform)

    # INIT Text-to-speech model
    tts_runner = TTSWrapper()
    
    # Create a Gradio interface
    with gr.Blocks(title="MINDSIS") as iface:
        gr.Markdown("<div align='center'><font size='70'> Chat with RKLLM </font></div>")
        gr.Markdown("### Enter your question in the Text-Box and hit Enter to chat with the RKLLM model.")
        # Create a Chatbot component to display conversation history
        chat_server = gr.Chatbot(height=400)
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
        msg.submit(get_user_input, [msg, chat_server], [msg, chat_server], queue=False).then(llm_runner.give_response_gradio, inputs=chat_server, outputs=chat_server)

        audio_btn.click(fn=audio_action, inputs=audio_btn, outputs=audio_btn).\
                then(fn=lambda: None, js=click_js()).\
                then(fn=check_btn_status, inputs=audio_btn).\
                success(fn=stt_runner.transcribe_audio, inputs=(audio_box, audio_btn), outputs=msg).\
                success(lambda :None, None, audio_box, queue=False).\
                success(get_user_input, [msg, chat_server], [msg, chat_server]).\
                success(llm_runner.give_response_gradio, inputs=chat_server, outputs=chat_server).\
                success(tts_runner, inputs=chat_server, outputs=audio_answer).\
                success(lambda : None, None, None, js=autoplay_audio)
        

        # When the clear button is clicked, perform a no-operation (lambda: None) and immediately clear the chat history.
        clear.click(lambda: None, None, chat_server, queue=False)

    # Enable the event queue system, and Start the Gradio application..
    # iface.queue().launch(debug=True)
    iface.queue().launch(server_name="0.0.0.0",
                server_port=8080,
                # ssl_certfile='../cert.pem',
                # ssl_keyfile='../key.pem',
                # ssl_verify=False
                )

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    llm_runner.release()
    print("====================")
