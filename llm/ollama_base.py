import ollama
import base64
import cv2
import os
import numpy as np
BGR2RGB = cv2.COLOR_BGR2RGB


def image_to_base64(uploaded_file):
    image = uploaded_file
    if isinstance(uploaded_file, str):
        if os.path.exists(uploaded_file):
            image = cv2.imread(uploaded_file)
            image = cv2.cvtColor(image, BGR2RGB)
    file_bytes = cv2.imencode('.jpg', image)[1]
    base64_str = base64.b64encode(file_bytes).decode('utf-8')
    return base64_str


class OllamaWrapper:
    def __init__(self, modelname, use_stream=True):
        self.modelname = modelname
        self.use_stream = use_stream
        self.stream_message = []

    def prepare_message(self, prompt, images):
        messages=[ {"role": "user", "content": prompt} ]
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            imb64 = [image_to_base64(im) for im in images]
            messages[0]['images'] = imb64
        # print('message: ', messages)
        return messages
    
    def get_response(self, response):
        if self.use_stream:
            self.stream_message = []
            for chunk in response:
                print(chunk['message']['content'], end='', flush=True)
                self.stream_message.append(chunk['message']['content'])
            return ''.join(self.stream_message)
        else:
            return response['message']['content']

    def __call__(self,
                 prompt: str="Describe the image",
                 images=None):
        messages = self.prepare_message(prompt, images)

        response = ollama.chat(
            model=self.modelname,
            messages=messages,
            stream=self.use_stream
        )
        
        # get response
        return self.get_response(response)

if __name__ == "__main__":
    # modelname = "deepseek-r1:14b"
    modelname = "qwen2.5:latest"
    owrapper = OllamaWrapper(modelname=modelname, use_stream=True)
    prompt = "How to improve chinese skills? please give me a short answer"
    text = owrapper(prompt)
    print('=' * 50)
    print('text: ', text)


    # modelname = "minicpm-v:8b-2.6-q4_K_M"
    # # modelname = "gemma3:12b"
    # imagepath = "/Users/brilian/Documents/aiot/mindsis/examples/image1.jpg"
    # image = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB)
    # owrapper = OllamaWrapper(modelname=modelname, use_stream=True)
    # prompt = "please return all 2d coordinate of pedestrian in x1y1x2y2 with json format"
    # # text = owrapper(prompt, images=[imagepath])
    # text = owrapper(prompt, images=[image])
    # print('=' * 50)
    # print('text: ', text)