import warnings
warnings.filterwarnings(action='ignore')
import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import gradio as gr
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rkllm_param import rkllm_lib, LLMCallState, RKLLMInputMode, RKLLMInferMode, RKLLM_Handle_t
from rkllm_param import callback_type, callback, chatter
from rkllm_server import RKLLM


# Record the user's input prompt        
def get_user_input(user_message, history):
    history = history + [[user_message, None]]
    return "", history

# Retrieve the output from the RKLLM model and print it in a streaming manner
def get_RKLLM_output(history):
    # Link global variables to retrieve the output information from the callback function
    chatter.global_text = []
    chatter.global_state = -1

    # Create a thread for model inference
    model_thread = threading.Thread(target=rkllm_model.run, args=(history[-1][0],))
    model_thread.start()

    # history[-1][1] represents the current dialogue
    history[-1][1] = ""
    
    # Wait for the model to finish running and periodically check the inference thread of the model
    model_thread_finished = False
    while not model_thread_finished:
        while len(chatter.global_text) > 0:
            history[-1][1] += chatter.global_text.pop(0)
            time.sleep(0.005)
            # Gradio automatically pushes the result returned by the yield statement when calling the then method
            yield history

        model_thread.join(timeout=0.005)
        model_thread_finished = not model_thread.is_alive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, required=True, help='Target platform: e.g., rk3588/rk3576;')
    parser.add_argument('--lora_model_path', type=str, help='Absolute path of the lora_model on the Linux board;')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board;')
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576.")
        sys.stdout.flush()
        exit()

    if args.lora_model_path:
        if not os.path.exists(args.lora_model_path):
            print("Error: Please provide the correct lora_model path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    if args.prompt_cache_path:
        if not os.path.exists(args.prompt_cache_path):
            print("Error: Please provide the correct prompt_cache_file path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    # Fix frequency
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Initialize RKLLM model
    print("=========init....===========")
    sys.stdout.flush()
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, args.lora_model_path, args.prompt_cache_path)
    print("RKLLM Model has been initialized successfully！")
    print("==============================")
    sys.stdout.flush()

    # Create a Gradio interface
    with gr.Blocks(title="Chat with RKLLM") as chatRKLLM:
        gr.Markdown("<div align='center'><font size='70'> Chat with RKLLM </font></div>")
        gr.Markdown("### Enter your question in the inputTextBox and press the Enter key to chat with the RKLLM model.")
        # Create a Chatbot component to display conversation history
        rkllmServer = gr.Chatbot(height=600)
        # Create a Textbox component for user message input
        msg = gr.Textbox(placeholder="Please input your question here...", label="inputTextBox")
        # Create a Button component to clear the chat history.
        clear = gr.Button("Clear")

        # Submit the user's input message to the get_user_input function and immediately update the chat history.
        # Then call the get_RKLLM_output function to further update the chat history.
        # The queue=False parameter ensures that these updates are not queued, but executed immediately.
        msg.submit(get_user_input, [msg, rkllmServer], [msg, rkllmServer], queue=False).then(get_RKLLM_output, rkllmServer, rkllmServer)
        # When the clear button is clicked, perform a no-operation (lambda: None) and immediately clear the chat history.
        clear.click(lambda: None, None, rkllmServer, queue=False)

    # Enable the event queue system.
    chatRKLLM.queue()
    # Start the Gradio application.
    chatRKLLM.launch()

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")