# MINDai-asSIStant

![](./AI_ASSISTANT.png)

## Summary
- Realtime AI assistant with Speech-to-text (STT) + LLM + text-to-Speech (TTS) capability.
- Tested device: Orange Pi-5 4GB, Mac Mini M4.
- Supported STT: Whisper-base.

## How to Use

- Use Speech to Text
  * Go to ```speech2text``` folder.
  * Run speech2text.py with model and audio folder path, example:
    ```
    python3 speech2text.py --encoder_model_path ../model/encoder_model_fp16.onnx --decoder_model_path ../model/decoder_model_int8.onnx --audio ../examples
    ```