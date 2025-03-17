# [MINDSIS]

> What is this? -> It is an AI Assistant created by mindnous.\
> What is MINDSIS? -> It is an abbrev. of **Mindnous AI Assistant**.

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

- Use Text to Speech
  * Go to ```text2speech``` folder.
  * edit text inside **__main__** function at the bottom of the tts.py file,  then run bellow
  * Run tts.py, example:
    ```
    python3 tts.py
    ```

## PROGRESS

- [ ] Voice Activity Detection (VAD).
- [x] Zero-shot Speech classification/identification.
- [ ] Large Language Model.
- [x] Speech to text.
- [x] Text to speech.

---

# ACKNOWLEDGEMENT

- [sherpa-onnx](https://k2-fsa.github.io/sherpa/index.html)
- [3d Speaker](https://github.com/modelscope/3D-Speaker.git)
- [PINTO MODEL ZOO](https://k2-fsa.github.io/sherpa/index.html)
- Open-source community and packages.