import cv2
import os
import sys
import pathlib
import time
FILEPATH = pathlib.Path(__file__).parent.absolute()
sys.path.append(f'{FILEPATH}/')

# import wrapper

SUPPORTED_MODEL_TYPE = {
    'macos': 'mlx',
    'openai_api': ['openai', 'ollama', 'litellm'],
    'local': ['ollama_offline']
}


class LLMWrapper:
    def __init__(self, modelname=None, model_info=dict(), model_type='openai'):
        self.modelname = modelname
        self.model_info = model_info
        self.model_type = model_type
        self.model = None

        if self.model_type == SUPPORTED_MODEL_TYPE['macos']:
            print('[LLMWrapper] init MLX: ', self.model_type)
            from mlx_base import MLXWrapper
            self.model = MLXWrapper(model_path=modelname, llm_type=model_info['llm_type'], verbose=True)
        elif self.model_type in SUPPORTED_MODEL_TYPE['openai_api']:
            print('[LLMWrapper] init openai_api: ', self.model_type)
            from litellm_base import LiteLLMWrapper
            self.model = LiteLLMWrapper(modelname, model_info['model_url'])
        elif self.model_type == SUPPORTED_MODEL_TYPE['local'][0]: # ollama offline
            print('[LLMWrapper] init local: ', self.model_type)
            from ollama_base import OllamaWrapper
            self.model = OllamaWrapper(modelname=modelname, use_stream=True)
        else:
            raise NotImplementedError(f"{self.model_type} model type not supported / development is still ongoing.")
    
    def __call__(self, **kwargs):
        if self.model_type == SUPPORTED_MODEL_TYPE['macos']:
            if self.model.llm_type == 'llm':
                response = self.model(kwargs['messages'])
            elif self.model.llm_type == 'vlm':
                # MLX-VLM's limitation: only accept imagepath, instead of base64-image
                response = self.model(kwargs['messages'], kwargs['image_paths'])
        elif self.model_type in SUPPORTED_MODEL_TYPE['openai_api']:
            response = self.model(kwargs['messages'])
        elif self.model_type == SUPPORTED_MODEL_TYPE['local'][0]: # ollama offline
            if kwargs.get('images', False):
                if not isinstance(kwargs['images'], list):
                    kwargs['images'] = [kwargs['images']]
                response = self.model(kwargs['messages'], images=kwargs['images'])
            else:
                response = self.model(kwargs['messages'])
        else:
            raise NotImplementedError(f"MODEL: {self.model_type} TYPE NOT SUPPORTED. SUPPORTED MODELS: {SUPPORTED_MODEL_TYPE}")
        return response

if __name__ == "__main__":
    print('GENERAL PARAMETER FOR INFERENCE')
    prompt = "please return all 2d coordinate of pedestrian in x1y1x2y2 with json format"
    imagepath = "/Users/brilian/Documents/aiot/mindsis/examples/image1.jpg"
    image = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB)

    print('INITIALIZATION')

    # ### MLX 
    # print('LLMWrapper[MLX] - LLM')
    # # LLM type
    # model_path = "/Users/brilian/Documents/aiot/Qwen2.5-14B-Instruct-4bit"
    # model_info={'llm_type': 'llm'}
    # llmwrap = LLMWrapper(model_path, model_info=model_info, model_type='mlx')
    # for _ in range(5):
    #     response = llmwrap(messages=prompt)
    #     print(_, '=' * 50, '\nresponse: ', response)

    # # VLM type
    # print('LLMWrapper[MLX] - VLM')
    # model_path = "/Users/brilian/Documents/aiot/Qwen2.5-VL-7B-Instruct-4bit"
    # model_info={'llm_type': 'vlm'}
    # llmwrap = LLMWrapper(model_path, model_info=model_info, model_type='mlx')
    # for _ in range(5):
    #     response = llmwrap(messages=prompt, image_paths=[image])
    #     print(_, '=' * 50, '\nresponse: ', response)
    # ###

    # ### Litellm / openai / ollama server
    # print('LLMWrapper[Litellm / openai / ollama server] - LLM')
    # # litellm with ollama server
    # modelname="ollama/qwen2.5:latest"
    # model_info=dict(model_url="http://localhost:11434")
    # llmwrap = LLMWrapper(modelname, model_info=model_info, model_type='ollama')
    # for _ in range(5):
    #     response = llmwrap(messages=prompt)
    #     print(_, '=' * 50, '\nresponse: ', response)
    # ###


    ### OLLAMA OFFLINE
    # Ollama offline - LLM type
    print('LLMWrapper[Ollama offline] - LLM')
    # modelname = "deepseek-r1:14b"
    modelname = "qwen2.5:latest"
    llmwrap = LLMWrapper(modelname, model_type='ollama_offline')
    for _ in range(5):
        response = llmwrap(messages=prompt)
        print(_, '=' * 50, '\nresponse: ', response)

    # # Ollama offline - VLM type
    # print('LLMWrapper[Ollama offline] - VLM')
    # # modelname = "minicpm-v:8b-2.6-q4_K_M"
    # modelname = "gemma3:12b"
    # llmwrap = LLMWrapper(modelname, model_type='ollama_offline')
    # for _ in range(5):
    #     response = llmwrap(messages=prompt, images=[image])
    #     print(_, '=' * 50, '\nresponse: ', response)
    # ###
