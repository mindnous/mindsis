import gradio as gr


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
