import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import gradio as gr
import argparse
from rkllm_param import rkllm_lib, LLMCallState, RKLLMInputMode, RKLLMInferMode, RKLLM_Handle_t
from rkllm_param import RKLLMExtendParam, RKLLMParam, RKLLMLoraAdapter, RKLLMEmbedInput, RKLLMTokenInput
from rkllm_param import RKLLMMultiModelInput, RKLLMInputUnion, RKLLMInput, RKLLMLoraParam, userdata
from rkllm_param import RKLLMPromptCacheParam, RKLLMInferParam, RKLLMResultLastHiddenLayer, RKLLMResult
from rkllm_param import callback_type, callback, chatter


# Set environment variables
# os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
# os.environ["GRADIO_SERVER_PORT"] = "8080"


# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    PROMPT_TEXT_PREFIX = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
    PROMPT_TEXT_POSTFIX = "<|im_end|><|im_start|>assistant"

    def __init__(self, model_path, lora_model_path = None, prompt_cache_path = None):
        self.chatter = chatter
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = 512
        rkllm_param.max_new_tokens = -1
        rkllm_param.skip_special_token = True

        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        
        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int
        self.history_answer = []

        self.lora_adapter_path = None
        self.lora_model_name = None
        if lora_model_path:
            self.lora_adapter_path = lora_model_path
            self.lora_adapter_name = "test"

            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((self.lora_adapter_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((self.lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
        
        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def run(self, prompt):
        rkllm_lora_params = None
        if self.lora_model_name:
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((self.lora_model_name).encode('utf-8'))
        
        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params.lora_params = ctypes.byref(rkllm_lora_params) if rkllm_lora_params else None

        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p((self.PROMPT_TEXT_PREFIX + prompt + self.PROMPT_TEXT_POSTFIX).encode('utf-8'))
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)
        return

    def release(self):
        self.rkllm_destroy(self.handle)
        
    # Retrieve the output from the RKLLM model and print it in a streaming manner
    def get_RKLLM_output(self, history):
        # print('history: ', len(history))
        print('history: ', len(history), history[-1])
        if history is None:
            return ''
        # Link global variables to retrieve the output information from the callback function
        self.chatter.global_text = []
        self.chatter.global_state = -1

        # Create a thread for model inference
        model_thread = threading.Thread(target=self.run, args=(history[-1][0],))
        model_thread.start()

        # history[-1][1] represents the current dialogue
        history[-1][1] = ""
        
        # Wait for the model to finish running and periodically check the inference thread of the model
        model_thread_finished = False
        while not model_thread_finished:
            while len(self.chatter.global_text) > 0:
                history[-1][1] += self.chatter.global_text.pop(0)
                time.sleep(0.005)
                # Gradio automatically pushes the result returned by the yield statement when calling the then method
                yield history

            model_thread.join(timeout=0.005)
            model_thread_finished = not model_thread.is_alive()
        self.history_answer.append(history[-1][1])


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