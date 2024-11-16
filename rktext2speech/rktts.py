import io
import base64
import gtts


# ref: https://github.com/gradio-app/gradio/issues/1349#issuecomment-1490017507
def text_to_speech(text):
    print('[text_to_speech] text received: ', text[-1][1])
    
    # We can use file extension as mp3 and wav, both will work
    textpath = 'rktext2speech/speech.mp3'
    tts = gtts.gTTS(text[-1][1])
    tts.save(textpath)
    return textpath

autoplay_audio = """ async () => {{
                    setTimeout(() => {{
                        document.querySelector('#speaker audio').play();
                    }}, {1000});
                }} """
