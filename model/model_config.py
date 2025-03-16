import pathlib
FILEPATH = pathlib.Path(__file__).parent.absolute()

## SPEECH-TO-TEXT CONFIG


## TEXT-TO-SPEECH CONFIG

# MODELPATH = f'{FILEPATH}/vits-melo-tts-zh_en/'
# MODELPATH = f'{FILEPATH}/vits-piper-en_US-lessac-medium/'
TTS_MODELPATH = f'{FILEPATH}/vits-piper-en_US-ryan-medium/'
TTS_OUTPUTPATH = f'{FILEPATH}/ttsout.wav'
TTS_THREADS = 1
TTS_MAX_NUM_SENTENCES = 1
TTS_PROVIDER = 'cpu'
TTS_SPEED = 1.0
TTS_SID = 0




