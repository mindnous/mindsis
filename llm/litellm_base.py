from litellm import completion
# import litellm
import time
import base64
import cv2
import os
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


class LiteLLMWrapper:
    def __init__(self, 
                 modelname='ollama/qwen2.5:latest', 
                 modelurl='http://localhost:11434',
                 openai_api_key='SECRET',
                 anthropic_api_key='SECRET'):
        self.modelname = modelname
        self.modelurl = modelurl
        ## set ENV variables
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    
    @staticmethod
    def prepare_messages(prompt, images=None):
        messages=[{"role": "user", "content": prompt}]
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            imb64 = [image_to_base64(im) for im in images]
            messages[0]['images'] = imb64
        return messages

    def __call__(self, prompt, images=None, stream=False):
        messages = self.prepare_messages(prompt, images)
        response = completion(
            model=self.modelname, 
            messages=messages, 
            api_base=self.modelurl,
            stream=stream
        )
        if stream:
            return response
        else:
            return response.choices[0].message.content


if __name__ == "__main__":
    modelname="ollama/qwen2.5:latest"
    modelurl="http://localhost:11434"
    litewrapper = LiteLLMWrapper(modelname, modelurl)
    print('jalan broo')

    for i in range(10):
        start = time.time()
        prompt = "How to improve chinese skills? please give me a short answer"
        print('jalan broo')
        resp = litewrapper(prompt)
        # print('response: ', resp)
        # print(i, '| ', resp.choices[0].message['content'])
        # print(f'time: {time.time()-start:.3f}')
        # print('=' * 50)

