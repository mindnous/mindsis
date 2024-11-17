import io
import base64
import gtts
import os

BPATH = os.path.dirname(os.path.abspath(__file__))
VPATH = os.path.join(BPATH, "vits-piper-en-ryan-medium/")

# ref: https://github.com/gradio-app/gradio/issues/1349#issuecomment-1490017507
def text_to_speech_gtts(text):
    print('[text_to_speech] text received: ', text[-1][1])
    
    # We can use file extension as mp3 and wav, both will work
    textpath = 'rktext2speech/speech.mp3'
    tts = gtts.gTTS(text[-1][1])
    tts.save(textpath)
    return textpath
    
def text_to_speech_offline(text):
    textpath = 'rktext2speech/speech.wav'
    print('text_to_speech_offline running...', BPATH)
    os.system(f'''python3 {BPATH}/offline-tts-play.py --vits-data-dir={VPATH}espeak-ng-data/ --vits-model={VPATH}en_US-ryan-medium.onnx --vits-tokens={VPATH}tokens.txt --output-filename={textpath} \"{text[-1][1]}\"''')
    return textpath

autoplay_audio = """ async () => {{
                    setTimeout(() => {{
                        document.querySelector('#speaker audio').play();
                    }}, {1000});
                }} """
