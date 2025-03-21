import copy
import cv2


class MLXWrapper:
    def __init__(self, model_path, llm_type=['llm', 'vlm'], verbose=True, stream=False):
        self.llm_type = llm_type
        self.messages_template = [{"role": "user", "content": ""}]
        self.model_cfg = None
        self.generate = None
        self.processor = None
        self.verbose = verbose
        self.stream = stream
        self.max_token = 512
        self.temperature = 0.2
        if self.llm_type == 'llm':
            from mlx_lm import load as lload, generate as lgenerate, stream_generate
            self.model, self.processor = lload(model_path)
            self.generate = lgenerate if not self.stream else stream_generate
        else:
            from mlx_vlm import load as vload, generate as vgenerate, stream_generate
            from mlx_vlm.utils import load_config as vload_config
            from mlx_vlm.prompt_utils import apply_chat_template as vapply_chat_template
            self.model, self.processor = vload(model_path)
            self.model_cfg = vload_config(model_path)
            self.generate = vgenerate if not self.stream else stream_generate
            self.vapply_chat_template = vapply_chat_template

    def __call__(self, prompt, images=None, **kwds):
        messages = copy.deepcopy(self.messages_template)
        self.processor.apply_chat_template(messages, add_generation_prompt=True)
        if self.llm_type == 'llm':
            return self.generate(self.model,
                                 self.processor,
                                 prompt)
        else:
            if type(images) != list:
                images = [images]
            formatted_prompt = self.vapply_chat_template(
                self.processor, self.model_cfg, prompt, num_images=len(images)
            )

            return self.generate(self.model,
                                 self.processor,
                                 formatted_prompt,
                                 images)


if __name__ == "__main__":
    model_path = "/Users/brilian/Documents/aiot/Qwen2.5-14B-Instruct-4bit"
    mwrapper = MLXWrapper(model_path=model_path, llm_type='llm', verbose=True)
    prompt = "How to improve chinese skills? please give me a short answer"
    text = mwrapper(prompt)
    print('=' * 50)
    print('text: ', text)


    # model_path = "/Users/brilian/Documents/aiot/Qwen2.5-VL-7B-Instruct-4bit"
    # mwrapper = MLXWrapper(model_path=model_path, llm_type='vlm', verbose=True)
    # prompt = 'please return all 2d coordinate of pedestrian in xyxy with json format'
    # imagepath = '../examples/image1.jpg'
    # image = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB).tobytes()
    # text = mwrapper(prompt, imagepath)
    # print('=' * 50)
    # print('text: ', text)