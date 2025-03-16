# ref: https://huggingface.co/NeuML/kokoro-fp16-onnx
import json
import numpy as np
import onnxruntime as ort
import time
import soundfile as sf
from ttstokenizer import IPATokenizer

# This example assumes the files have been downloaded locally
with open("voices.json", "r", encoding="utf-8") as f:
    voices = json.load(f)

# Create model
model = ort.InferenceSession(
    "kokoro-quant-convinteger.onnx",
    providers=['CUDAExecutionProvider', 'CoreMLExecutionProvider', "CPUExecutionProvider"]
)

# Create tokenizer
tokenizer = IPATokenizer()

# Tokenize inputs

for _ in range(10):
    start = time.time()
    textref = """I am trying to my best here, can 
        somebody give me a good suggestion? thank 
        you xpeng to help massively with the tedious works."""
    inputs = tokenizer(textref)

    # Get speaker array
    speaker = np.array(voices["af"], dtype=np.float32)

    # Generate speech
    outputs = model.run(None, {
        "tokens": [[0, *inputs, 0]],
        "style": speaker[len(inputs)],
        "speed": np.ones(1, dtype=np.float32) * 1.0
    })

    # Write to file
    sf.write("out.wav", outputs[0], 24000)
    print(f"Time taken: {time.time() - start:.3f}s")
