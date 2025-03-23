import gradio as gr
import cv2
import base64
from pathlib import Path


def click_js():
    return """function audioRecord() {
    var xPathRes = document.evaluate ('//*[contains(@class, "record")]', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null); 
    xPathRes.singleNodeValue.click();}"""


def audio_action(btn):
    """Changes button text on click"""
    print(f'-> current button: {btn}; trigger the opposite [Speak | Stop]')
    if btn == 'Speak': return 'Stop'
    else: return 'Speak'


def check_btn_status(btn):
    """Checks for correct button text before invoking transcribe()"""
    print('-> check button return: ', btn)
    if btn != 'Speak': raise Exception('Recording...')
    return btn


# Record the user's input prompt        
def get_user_input(user_message, history):
    history = history + [[user_message, None]]
    return "", history


autoplay_audio = """ async () => {{
                    setTimeout(() => {{
                        document.querySelector('#speaker audio').play();
                    }}, {1000});
                }} """


import os
def upload_file(files):
    file_paths = [f.name for f in files]
    print('upload_file: ', file_paths)
    print('pathexist: ', [os.path.exists(p) for p in file_paths])
    return file_paths


def filepath_to_chat(file_obj, history):
    print('file_to_nparray', file_obj)
    # Read image using OpenCV
    img = cv2.imread(file_obj)
    # Convert from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Encode the numpy array to the specified image format
    success, encoded_img = cv2.imencode(f'.{format}', img)
    
    # Convert to base64 string
    b64_image = base64.b64encode(encoded_img).decode('utf-8')
    history += ["", f'<img src="data:image/png;base64,{b64_image}">']
    return history

def upload_file_to_chat(file_obj, history):
    files = upload_file(file_obj)

    for f in files:
        # Read image using OpenCV
        print('load file: ',f)
        img = cv2.imread(f)
        # Convert from BGR (OpenCV default) to RGB
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print('img: ', img.shape)
        # Encode the numpy array to the specified image format
        _, encoded_img = cv2.imencode('.png', img)
        
        # Convert to base64 string
        b64_image = base64.b64encode(encoded_img).decode('utf-8')
        history += [[f'<img src="data:image/png;base64,{b64_image}">', ""]]
    history += [[f"uploaded images: {len(files)}", ""]]
    return history
