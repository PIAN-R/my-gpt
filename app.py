import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import re
from langdetect import detect

dotenv.load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="DaviDan Intelligence",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
    }
    .chat-container {
        width: 66%;
        margin: auto;
        margin-bottom: 20px;
    }
    .user-message, .assistant-message {
        padding: 10px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 66%;
        word-wrap: break-word;
        white-space: pre-wrap;
        background: none;
        border: none;
    }
    .user-message {
        text-align: left;
        float: right;
        padding-left: 10px;
        padding-right: 10px;
    }
    .assistant-message {
        text-align: left;
        float: left;
    }
    .clear-fix {
        clear: both;
    }
    .stTextInput > div > div {
        width: 66%;
        margin: auto;
    }
    .stButton > div > button {
        width: 100%;
    }
    .dark-theme body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .dark-theme .user-message, .dark-theme .assistant-message {
        color: #fff;
    }
    .dark-theme .stTextInput > div > div {
        background-color: #444;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# Function to query and stream the response from the LLM
def stream_llm_response(client, model_params):
    response_message = ""

    for chunk in client.chat.completions.create(
        model=model_params.get("model", "gpt-4o-2024-05-13"),
        messages=st.session_state.messages,
        temperature=model_params.get("temperature", 0.3),
        max_tokens=4096,
        stream=True,
    ):
        delta_content = chunk.choices[0].delta.content
        response_message += delta_content if delta_content else ""
        yield delta_content if delta_content else ""

    return response_message

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

# Function to display chat messages
def display_chat_message(role, contents):
    for content in contents:
        if content["type"] == "text":
            css_class = "user-message" if role == "user" else "assistant-message"
            if "```" in content['text']:
                code_content = content['text'].split("```")[1]
                st.code(code_content, language='python')
            else:
                st.markdown(
                    f"<div class='chat-container {css_class}'>{content['text']}</div><div class='clear-fix'></div>",
                    unsafe_allow_html=True
                )
        elif content["type"] == "image_url":
            st.image(content["image_url"]["url"])

def main():
    # Initialize session state variables
    if "custom_response" not in st.session_state:
        st.session_state.custom_response = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Header ---
    st.markdown(
        "<h1 style='text-align: center; color: grey; font-family: \"Roboto\", sans-serif; font-size: 24px;'>"
        "<i>DaviDan Intelligence</i></h1>",
        unsafe_allow_html=True
    )

    # --- Side Bar ---
    with st.sidebar:
        default_openai_api_key = os.getenv("OPENAI_API_KEY", "")
        with st.expander("API Key"):
            openai_api_key = st.text_input(
                "Enter your API Key ",
                value=default_openai_api_key, type="password"
            )

        st.markdown("[Mesina Dan](https://www.instagram.com/mesina.dan?igshid=MWg0MnZ6Nm1hcml4)")
        st.markdown("[DaviDan Bakery](https://www.instagram.com/davidan.bakery?igshid=MTV4bWk0MXRiMTJyaw==)")

    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if not openai_api_key or "sk-" not in openai_api_key:
        st.write("#")
        st.warning("⬅️ Please introduce your API Key to continue...")
    else:
        client = OpenAI(api_key=openai_api_key)

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])

        # Side bar model options and inputs
        with st.sidebar:
            st.divider()

            model = st.selectbox("Select a model:", [
                "gpt-4o-2024-05-13",
                "gpt-4-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
            ], index=0)

            with st.expander(" Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            audio_response = st.checkbox("Audio response", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(" Reset conversation", on_click=reset_conversation)

            st.divider()

            # Image Upload
            if model in ["gpt-4o-2024-05-13", "gpt-4-turbo"]:
                st.write("### ** Add an image:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                        img = get_image_base64(raw_img)
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_type};base64,{img}"}
                                }]
                            }
                        )

                cols_img = st.columns(2)
                with cols_img[0]:
                    with st.expander(" Upload"):
                        st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="uploaded_img", on_change=add_image_to_messages)
                with cols_img[1]:
                    with st.expander(" Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input("Take a picture", key="camera_img", on_change=add_image_to_messages)

            # Audio Upload
            st.write("#")
            st.write("### ** Add an audio:**")

            audio_prompt = None
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None

            speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395")
            if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                st.session_state.prev_speech_hash = hash(speech_input)
                transcript = client.audio.transcriptions.create(model="whisper-1", file=("audio.wav", speech_input))
                audio_prompt = transcript.text

        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything...") or audio_prompt:
            st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt or audio_prompt}]})
            display_chat_message("user", [{"type": "text", "text": prompt or audio_prompt}])

            # Detect language of the prompt
            detected_lang = detect(prompt)

            # Check for specific prompt and customize response
            if re.search(r"(who created you[\s\S]*|Cine te-a creat[\s\S]*|Кто тебя создал[\s\S]*)", prompt, re.IGNORECASE):
                if detected_lang == "ro":
                    custom_response_text = "Am fost creat de DaviDan, o organizație de cercetare în domeniul inteligenței artificiale. DaviDan dezvoltă modele de limbaj de mari dimensiuni (LLM) și alte tehnologii avansate de inteligență artificială. Dacă ai mai multe întrebări sau ai nevoie de asistență suplimentară, nu ezita să întrebi!"
                elif detected_lang == "ru":
                    custom_response_text = "Я был создан компанией DaviDan, исследовательской организацией в области искусственного интеллекта. DaviDan разрабатывает крупномасштабные языковые модели (LLM) и другие передовые технологии искусственного интеллекта. Если у вас есть дополнительные вопросы или вам нужна дополнительная помощь, не стесняйтесь спрашивать!"
                else:
                    custom_response_text = "I was created by DaviDan, an artificial intelligence research organization. DaviDan develops LLM Large Language Models and other advanced AI models and technologies. If you have any more questions or need further assistance, feel free to ask!"

                custom_response = {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": custom_response_text
                    }]
                }
                st.session_state.custom_response = custom_response
                st.session_state.messages.append(custom_response)
                display_chat_message("assistant", custom_response["content"])

            elif re.search(r"translate.*romanian", prompt, re.IGNORECASE) and st.session_state.custom_response:
                translation = "Sigur, iată traducerea mesajului în română:\n\n" + st.session_state.custom_response["content"][0]["text"]
                st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": translation}]})
                display_chat_message("assistant", [{"type": "text", "text": translation}])

            elif re.search(r"translate.*russian", prompt, re.IGNORECASE) and st.session_state.custom_response:
                translation = "Конечно, вот перевод сообщения на русский язык:\n\n" + st.session_state.custom_response["content"][0]["text"]
                st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": translation}]})
                display_chat_message("assistant", [{"type": "text", "text": translation}])

            else:
                assistant_response = list(stream_llm_response(client, model_params))
                response_text = ''.join(assistant_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_text}]
                })
                display_chat_message("assistant", [{"type": "text", "text": response_text}])

            # --- Added Audio Response (optional) ---
            if audio_response:
                response = client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=st.session_state.messages[-1]["content"][0]["text"],
                )
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
