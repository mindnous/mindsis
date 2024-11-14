# ROCKLLM

## Summary
- Realtime AI assistant with Speech-to-text (STT) + LLM + text-to-Speech (TTS) capability.
- Tested device: Orange Pi-5 4GB.
- Supported STT: Whisper-base.
- supported LLM: Qwen2.5 1.5B (due to hardware limitation on my site).
- Supported TTS: only need to install pyttsx3.
- Framework versions:
  - RKLLM: 0.9.8.
  - RKNN-Toolkit-lite: 2.3.0.

## How to Use
- Download the whisper model (for Speech-to-Text / STT):
```
sh download_models.sh
```
- Run the demo
```
python3 rkllm_text/rkllm_main.py --rkllm_model_path qwen25_1.5b.rkllm --target_platform rk3588
```
