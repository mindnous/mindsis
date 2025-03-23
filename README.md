# [MINDSIS]

> What does MINDSIS stand for? -> It is an abbrev. of **Mindnous AI Assistant**.

## What is MINDSIS?
- AI Assistant that could run on edge and cheap devices.
  - Support Arm64(Mac M series, rockchip devices - ONGOING), and desktop/amd64.
- Run on most popular frameworks (MLX - LLMLite - Ollama).
- [ONGOING] 

---

## MINDSIS Pipeline

<video autoplay loop playsinline>
  <source src="./mindsis.mp4" type="video/mp4">
</video>


## Next Version

![](./AI_ASSISTANT.png)

---


## Summary
- Realtime AI assistant with Speech-to-text (STT) + LLM + text-to-Speech (TTS) capability.
- Target device: Arm(RK-device, Mac, NVIDIA Jetson) and AMD64 devices.
- Supported Speech-Classify: [3dspeaker-CN-EN-16K-CAM++](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
- Supported Speech-to-text: [Whisper-base](https://huggingface.co/onnx-community/whisper-base).
- Supported Text-to-speech: any popular TTS on the community. Default: [vits-en-us-ryan-medium](https://huggingface.co/csukuangfj/vits-piper-en_US-ryan-medium).

## How to Use

- Install dependencies
  ```
  # be sure to install cmake, and portaudio beforehands
  pip3 install -r requirements.txt
  ```

- Run Demo script.
  ```
  python3 demo.py --llm_model "gemma3:12b" \
                    --llm_type "vlm" \
                    --target_platform "ollama_offline"
  ```

  * <details><summary>Other examples with demo.py.</summary>

    ```
    # MLX
    python3 demo.py --llm_model "[PATH_TO_LLM]/Qwen2.5-14B-Instruct-4bit" \
                    --llm_type "llm" \
                    --stt_modelenc model/encoder_model_fp16.onnx \
                    --stt_modeledec model/decoder_model_int8.onnx \
                    --target_platform "mlx"

    # OPENAI / OLLAMA SERVER / LITELLM
    python3 demo.py --llm_model "ollama/qwen2.5:latest" \
                    --llm_type "llm" \
                    --stt_modelenc model/encoder_model_fp16.onnx \
                    --stt_modeledec model/decoder_model_int8.onnx \
                    --target_platform "ollama"

    # OLLAMA-OFFLINE
    python3 demo.py --llm_model "qwen2.5:latest" \
                    --llm_type "llm" \
                    --stt_modelenc model/encoder_model_fp16.onnx \
                    --stt_modeledec model/decoder_model_int8.onnx \
                    --target_platform "ollama_offline"
    ```

    </details>
  * Scan QR-Code to access the URL, and enjoy!

<details> <summary>OLDER POST</summary>

- Use Voice Activity Detector (VAD)
  * Go to ```vad``` folder.
  * edit "sf.read" path inside **__main__** function, and fill it with your audio path.
  * Run vad.py, example:
    ```
    python3 vad.py
    ```

- Use Speech Identification
  * Go to ```speechclassify``` folder.
  * edit text inside **__main__** function at the "audio_file" variable and fill it with your audio path.
  * Run scls.py, example:
    ```
    python3 tts.py
    ```

- Use Speech to Text
  * Go to ```speech2text``` folder.
  * Run stt.py with model and audio folder path, example:
    ```
    python3 stt.py --encoder_model_path ../model/encoder_model_fp16.onnx --decoder_model_path ../model/decoder_model_int8.onnx --audio ../examples
    ```

- Use LLM/VLM
  * Go to ```llm``` folder.
  *   
  * Run llm.py with modelname/path and model_info / config that you used, <details><summary>see complete example</summary>

    ```
    print('GENERAL PARAMETER FOR INFERENCE')
    prompt = "please return all 2d coordinate of pedestrian in x1y1x2y2 with json format"
    imagepath = "/Users/brilian/Documents/aiot/mindsis/examples/image1.jpg"
    image = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB)

    print('INITIALIZATION')

    ### MLX 
    print('LLMWrapper[MLX] - LLM')
    # LLM type
    model_path = "/Users/brilian/Documents/aiot/Qwen2.5-14B-Instruct-4bit"
    model_info={'llm_type': 'llm'}
    llmwrap = LLMWrapper(model_path, model_info=model_info, model_type='mlx')
    for _ in range(5):
        response = llmwrap(messages=prompt)
        print(_, '=' * 50, '\nresponse: ', response)

    # VLM type
    print('LLMWrapper[MLX] - VLM')
    model_path = "/Users/brilian/Documents/aiot/Qwen2.5-VL-7B-Instruct-4bit"
    model_info={'llm_type': 'vlm'}
    llmwrap = LLMWrapper(model_path, model_info=model_info, model_type='mlx')
    for _ in range(5):
        response = llmwrap(messages=prompt, image_paths=[image])
        print(_, '=' * 50, '\nresponse: ', response)
    ###

    ### Litellm / openai / ollama server
    print('LLMWrapper[Litellm / openai / ollama server] - LLM')
    # litellm with ollama server
    modelname="ollama/qwen2.5:latest"
    model_info=dict(model_url="http://localhost:11434")
    llmwrap = LLMWrapper(modelname, model_info=model_info, model_type='ollama')
    for _ in range(5):
        response = llmwrap(messages=prompt)
        print(_, '=' * 50, '\nresponse: ', response)
    ###


    ### OLLAMA OFFLINE
    # Ollama offline - LLM type
    print('LLMWrapper[Ollama offline] - LLM')
    # modelname = "deepseek-r1:14b"
    modelname = "qwen2.5:latest"
    llmwrap = LLMWrapper(modelname, model_type='ollama_offline')
    for _ in range(5):
        response = llmwrap(messages=prompt)
        print(_, '=' * 50, '\nresponse: ', response)

    # Ollama offline - VLM type
    print('LLMWrapper[Ollama offline] - VLM')
    # modelname = "minicpm-v:8b-2.6-q4_K_M"
    modelname = "gemma3:12b"
    llmwrap = LLMWrapper(modelname, model_type='ollama_offline')
    for _ in range(5):
        response = llmwrap(messages=prompt, images=[image])
        print(_, '=' * 50, '\nresponse: ', response)
    ###
    ```

    </details>
  

- Use Text to Speech
  * Go to ```text2speech``` folder.
  * edit text inside **__main__** function at the bottom of the tts.py file,  then run bellow
  * Run tts.py, example:
    ```
    python3 tts.py
    ```
</details>


## PROGRESS

<details><summary>Complete Progress</summary>

- [ONGOING] Add MLC-LLM framework.
- [ONGOING] Add external interactions for productivity purpose.
- [2025/03/23] NOVAD + Add image upload for VLM.
- [2025/03/22] Integrate all pipeline on Gradio(NOVAD: not yet including VAD and Audio Classification).
- [2025/03/21] Add Large/Visual Language Model framework (support MLX - Ollama - LLMLite).
- [2025/03/19] Add Text to speech.
- [2025/03/18] Add Speech to text.
- [2025/03/17] Add Zero-shot Speech classification/identification.
- [2025/03/17] Add Voice Activity Detection (VAD).
</details>

---

# ACKNOWLEDGEMENT

- [sherpa-onnx](https://k2-fsa.github.io/sherpa/index.html)
- [3d Speaker](https://github.com/modelscope/3D-Speaker.git)
- [PINTO MODEL ZOO](https://k2-fsa.github.io/sherpa/index.html)
- [SILERO_VAD](https://huggingface.co/onnx-community/silero-vad/)
- Open-source community and packages.