import os
import sys
import argparse
import time
import sherpa_onnx
import soundfile as sf
import pathlib
FILEPATH = pathlib.Path(__file__).parent.absolute()
sys.path.append(f'{FILEPATH}/../model/')
import model_config as cfg
# MODELPATH = f'{FILEPATH}/vits-piper-en_US-ryan-medium/'
MODELPATH = cfg.TTS_MODELPATH


def add_vits_args(parser):
    onnx_name = [i for i in os.listdir(MODELPATH) if i.endswith('.onnx')][0]
    parser.vits_model = f'{MODELPATH}/{onnx_name}'
    parser.vits_tokens = f'{MODELPATH}/tokens.txt'
    if os.path.exists(f'{MODELPATH}/lexicon.txt'):
        parser.vits_lexicon = f'{MODELPATH}/lexicon.txt'
        parser.vits_dict_dir = f'{MODELPATH}/dict/'
        parser.vits_data_dir = ""
    else:
        parser.vits_lexicon = ''
        parser.vits_dict_dir = ''
        parser.vits_data_dir = f"{MODELPATH}/espeak-ng-data/"


def get_args():
    class BaseClass:
        pass
    parser = BaseClass()
    add_vits_args(parser)
    return parser


def main(textref, loop=5):
    args = get_args()
    print(args)

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=args.vits_model,
                lexicon=args.vits_lexicon,
                data_dir=args.vits_data_dir,
                dict_dir=args.vits_dict_dir,
                tokens=args.vits_tokens,
            ),
            provider=cfg.TTS_PROVIDER,
            debug=False,
            num_threads=cfg.TTS_THREADS,
        ),
        rule_fsts='',
        max_num_sentences=cfg.TTS_MAX_NUM_SENTENCES,
    )
    if not tts_config.validate():
        raise ValueError("Please check your config")

    tts = sherpa_onnx.OfflineTts(tts_config)

    for _ in range(loop):
        start = time.time()
        audio = tts.generate(textref, sid=cfg.TTS_SID, speed=cfg.TTS_SPEED)
        end = time.time()

        if len(audio.samples) == 0:
            print("Error in generating audios. Please read previous error messages.")
            return

        elapsed_seconds = end - start
        audio_duration = len(audio.samples) / audio.sample_rate
        real_time_factor = elapsed_seconds / audio_duration

        sf.write(
            cfg.TTS_OUTPUTPATH,
            audio.samples,
            samplerate=audio.sample_rate,
            subtype="PCM_16",
        )
        print(f"Saved to {cfg.TTS_OUTPUTPATH}")
        print(f"The input is:\t'{textref}'")
        print(f"Elapsed seconds: {elapsed_seconds:.3f}")
        print(f"Audio duration in seconds: {audio_duration:.3f}")
        print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    textref = """I am trying to my best here, can 
        somebody give me a good suggestion? thank 
        you xpeng to help massively with the tedious works."""
    main(textref, loop=5)