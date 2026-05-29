"""
Streamlit application for Assignment 3 multimodal exploration.
Supports text, image, audio, and video input modalities.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from vikas_app import MultimodalApp, TextModel, ImageModel, AudioModel

TEXT_MODEL_OPTIONS = {
    "openrouter": [
        "gpt-3.5-mini",
        "gpt-4o-mini",
        "gpt-4o-mini-vision",
        "gpt-3.5-turbo",
        "gpt-4o-mini-std",
    ],
    "google": [
        "text-bison-001",
        "chat-bison-001",
        "code-bison-001",
        "gpt-gecko-1",
        "text-bison-002",
    ],
    "huggingface": [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "EleutherAI/gpt-neo-125M",
        "distilgpt2",
    ],
}

IMAGE_MODEL_OPTIONS = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-base",
    "runwayml/stable-diffusion-2-1",
    "hakurei/waifu-diffusion",
    "stabilityai/stable-diffusion-2-1-base",
]

CAPTION_MODEL_OPTIONS = [
    "nlpconnect/vit-gpt2-image-captioning",
    "Salesforce/blip-image-captioning-base",
]

ASR_MODEL_OPTIONS = [
    "facebook/wav2vec2-base-960h",
    "openai/whisper-small",
    "openai/whisper-medium",
    "patrickvonplaten/wav2vec2-large-xlsr-53-english",
    "openai/whisper-tiny",
]

TTS_MODEL_OPTIONS = [
    "microsoft/speecht5_tts",
    "Avitas8485/speecht5_tts_fleur_en_us_v1",
]

st.set_page_config(
    page_title="Multi-Model AI Explorer",
    page_icon="🤖",
    layout="wide",
)

st.title("Multi-Model AI Explorer")
st.markdown(
    "Use text, image, audio, and video inputs to generate text, images, audio, or video outputs. "
    "This demo uses Hugging Face, OpenRouter, and local diffusers models."
)

if "app" not in st.session_state:
    st.session_state.app = MultimodalApp()

app = st.session_state.app

with st.sidebar:
    st.header("Options")
    provider = st.selectbox(
        "Text generation provider",
        ["openrouter", "google", "huggingface"],
        index=0,
    )
    model_options = TEXT_MODEL_OPTIONS.get(provider, [])
    model_name = st.selectbox(
        "Free text model",
        model_options,
        index=0,
    )
    asr_model_name = st.selectbox(
        "Speech recognition model",
        ASR_MODEL_OPTIONS,
        index=1,
    )
    app.text_model = TextModel(
        provider=provider,
        model_name=model_name,
    )
    if app.audio_model._asr_model_name != asr_model_name:
        app.audio_model._asr_model_name = asr_model_name
        app.audio_model._asr = None
    st.markdown("---")
    st.write("**Tips**")
    st.write("- Upload media files to test multimodal conversions")
    st.write("- Use GPU if available for image/video generation")
    st.write("- Text→Video uses generated image frames")

mode = st.radio(
    "Select modality",
    ["Text", "Image", "Audio", "Video"],
    index=0,
    horizontal=True,
)

if mode == "Text":
    st.subheader("Text Modality")
    task = st.selectbox("Task", ["Text → Text", "Text → Image", "Text → Audio", "Text → Video"])
    prompt = st.text_area("Enter text prompt", height=180)

    if st.button("Run", key="text_run"):
        if not prompt.strip():
            st.warning("Please enter a prompt before running.")
        else:
            final_prompt = prompt

            if task == "Text → Text":
                with st.spinner("Generating text..."):
                    output = app.text_to_text(final_prompt)
                st.markdown("### Generated Text")
                st.write(output)

            elif task == "Text → Image":
                with st.spinner("Generating image..."):
                    image = app.text_to_image(final_prompt)
                if image:
                    st.image(image, caption="Generated image", width=700)
                else:
                    st.error("Image generation failed.")

            elif task == "Text → Audio":
                with st.spinner("Generating audio..."):
                    audio_path = app.text_to_audio(final_prompt, output_path="text_to_audio.wav")
                if os.path.exists(audio_path):
                    st.audio(audio_path)
                    st.success(f"Saved audio to {audio_path}")
                else:
                    st.error(audio_path)

            elif task == "Text → Video":
                with st.spinner("Creating video from text..."):
                    video_path = app.text_to_video(final_prompt, output_path="text_to_video.mp4")
                if os.path.exists(video_path):
                    st.video(video_path)
                    st.success(f"Saved video to {video_path}")
                else:
                    st.error(video_path)

elif mode == "Image":
    st.subheader("Image Modality")
    task = st.selectbox("Task", ["Image → Text", "Image → Image"])
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width=700)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_image:
            image.save(tmp_image.name)
        tmp_image_path = tmp_image.name

        instructions = st.text_area(
            "Instructions",
            "Describe what you want the model to do with this image.",
            height=100,
        )

        if st.button("Run", key="image_run"):
            if task == "Image → Text":
                with st.spinner("Captioning image..."):
                    caption = app.image_to_text(tmp_image_path)
                st.markdown("### Image Caption")
                st.write(caption)
                if instructions.strip():
                    st.markdown("### Instructions")
                    st.write(instructions)

            elif task == "Image → Image":
                prompt = instructions.strip() or "A bright fantasy style remix"
                if prompt:
                    with st.spinner("Transforming image..."):
                        transformed = app.image_to_image(tmp_image_path, prompt)
                    if transformed:
                        st.image(transformed, caption="Transformed output", width=700)
                    else:
                        error_msg = app.get_image_error() or "Image-to-image transformation failed."
                        st.error(error_msg)
                else:
                    st.warning("Enter instructions for the image transformation.")

elif mode == "Audio":
    st.subheader("Audio Modality")
    task = st.selectbox("Task", ["Audio → Text", "Audio → Audio"])
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_file.getbuffer())
            tmp_audio.flush()
        tmp_audio_path = tmp_audio.name
        st.audio(tmp_audio_path)

        instructions = st.text_area(
            "Instructions",
            "Describe how you want the audio to be transcribed or converted.",
            height=100,
        )

        if st.button("Run", key="audio_run"):
            if task == "Audio → Text":
                with st.spinner("Transcribing audio..."):
                    transcript = app.audio_to_text(tmp_audio_path)
                st.markdown("### Transcription")
                st.write(transcript)
                if instructions.strip():
                    st.markdown("### Instructions")
                    st.write(instructions)

            elif task == "Audio → Audio":
                with st.spinner("Converting audio..."):
                    output_path = app.audio_to_audio(tmp_audio_path, output_path="audio_to_audio.wav")
                if os.path.exists(output_path):
                    st.audio(output_path)
                    st.success(f"Saved converted audio to {output_path}")
                    if instructions.strip():
                        st.markdown("### Instructions")
                        st.write(instructions)
                else:
                    st.error(output_path)

elif mode == "Video":
    st.subheader("Video Modality")
    task = st.selectbox("Task", ["Video → Text", "Video → Summary"])
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_video:
            tmp_video.write(uploaded_file.getbuffer())
            tmp_video.flush()
        tmp_video_path = tmp_video.name
        st.video(tmp_video_path)

        instructions = st.text_area(
            "Instructions",
            "Describe what you want to extract or summarize from the video.",
            height=100,
        )

        if st.button("Run", key="video_run"):
            if task == "Video → Text":
                with st.spinner("Extracting captions from video..."):
                    captions = app.video_to_text(tmp_video_path)
                st.markdown("### Video Captions")
                for index, caption in enumerate(captions, start=1):
                    st.write(f"Frame {index}: {caption}")
                if instructions.strip():
                    st.markdown("### Instructions")
                    st.write(instructions)

            elif task == "Video → Summary":
                with st.spinner("Summarizing video frames..."):
                    summary = app.video_to_summary(tmp_video_path)
                st.markdown("### Video Summary")
                st.write(summary)
                if instructions.strip():
                    st.markdown("### Instructions")
                    st.write(instructions)

st.markdown("---")
st.caption("Multimodal AI Explorer.")
