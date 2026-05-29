# Assignment 3 — Multi-Model AI Explorer

This assignment folder contains a multimodal AI demo built with an interactive Streamlit app and a backend orchestration module. The app supports text, image, audio, and video tasks, and lets you choose from multiple providers and ASR models.

## Folder Contents

- `vikas_app.py` — Multimodal backend implementation
- `streamlit_app.py` — Streamlit user interface
- `requirements.txt` — Python package dependencies
- `README.md` — Project documentation
- `Assignment.pdf` — Assignment brief

## Supported Modalities and Tasks

### Text
- `Text → Text`: text generation using OpenRouter, Google Gemini, or Hugging Face.
- `Text → Image`: image synthesis from a prompt using Stable Diffusion.
- `Text → Audio`: speech synthesis from text with a SpeechT5 TTS pipeline.
- `Text → Video`: short video generation by rendering multiple image frames from text prompts.

### Image
- `Image → Text`: image captioning using an image-to-text transformer.
- `Image → Image`: prompt-based image transformation with Stable Diffusion img2img.

### Audio
- `Audio → Text`: speech recognition/transcription using selectable ASR models.
- `Audio → Audio`: audio conversion by transcribing speech and re-synthesizing the result.

### Video
- `Video → Text`: video captioning by extracting frames and captioning each frame.
- `Video → Summary`: condensed text summary produced by summarizing frame captions.

## Model Options in the App

### Text generation providers
The sidebar allows switching providers and selecting models from each provider.

- OpenRouter models
  - `gpt-3.5-mini`
  - `gpt-4o-mini`
  - `gpt-4o-mini-vision`
  - `gpt-3.5-turbo`
  - `gpt-4o-mini-std`

- Google models
  - `text-bison-001`
  - `chat-bison-001`
  - `code-bison-001`
  - `gpt-gecko-1`
  - `text-bison-002`

- Hugging Face models
  - `google/flan-t5-small`
  - `google/flan-t5-base`
  - `google/flan-t5-large`
  - `EleutherAI/gpt-neo-125M`
  - `distilgpt2`

### ASR models (Speech recognition)
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-tiny`
- `facebook/wav2vec2-base-960h`
- `patrickvonplaten/wav2vec2-large-xlsr-53-english`

### Image and audio defaults
- Image generation: `runwayml/stable-diffusion-v1-5`
- Image captioning: `nlpconnect/vit-gpt2-image-captioning`
- Text-to-speech: `microsoft/speecht5_tts`

## Model Compatibility

### Best for CPU
- `google/flan-t5-small` and `google/flan-t5-base` for reliable Hugging Face text generation.
- `distilgpt2` for light, fast local text output.
- `openai/whisper-tiny` for faster, low-cost audio transcription.
- `microsoft/speecht5_tts` for moderate TTS on CPU.
- `runwayml/stable-diffusion-v1-5` with lower inference steps for smaller image workloads.

### Best for GPU
- `google/flan-t5-large` for larger text generation tasks.
- `runwayml/stable-diffusion-2-1` and `stabilityai/stable-diffusion-2-base` for better image quality.
- `hakurei/waifu-diffusion` for stylized image output.
- `openai/whisper-small` / `openai/whisper-medium` for higher-quality audio transcription.
- Video generation via `Text → Video`, which benefits from GPU acceleration.

### Provider notes
- OpenRouter models require a valid `OPENROUTER_API_KEY` and are suitable for hosted chat-style generation.
- Google Gemini models require `GOOGLE_API_KEY` and are useful for high-quality text generation without local model downloads.
- Hugging Face models work locally and are the fallback when API keys are unavailable.

## How the Code Works

### `streamlit_app.py`
- Builds the interactive UI and sidebar selectors.
- Creates a `MultimodalApp` instance stored in `st.session_state`.
- Lets the user select:
  - text provider and text generation model
  - speech recognition model
- Handles task execution for text, image, audio, and video modalities.

### `vikas_app.py`
- Implements `MultimodalApp` plus specialized classes:
  - `TextModel` for text generation and summarization
  - `ImageModel` for text-to-image, image captioning, and image-to-image
  - `AudioModel` for speech recognition, text-to-speech, and audio conversion
  - `VideoModel` for video frame extraction, captioning, and text-to-video generation
- Uses Hugging Face `pipeline()` for most tasks and `diffusers` for Stable Diffusion image work.
- Audio transcription now uses selectable ASR models and falls back to the file path input if required.

### Selection and fallback logic
- `TextModel.generate()` chooses the provider-specific generation method:
  - OpenRouter REST API with `requests`
  - Google Gemini REST API with `requests`
  - Hugging Face `transformers` pipeline locally
- Hugging Face text generation automatically selects `text2text-generation` for T5-style models and `text-generation` otherwise.
- `AudioModel.speech_to_text()` tries the audio file path first and falls back to direct audio array input if needed.
- `AudioModel.text_to_speech()` creates or loads speaker embeddings and writes the result to disk.

### Video workflow
- `VideoModel.video_to_text()` extracts frames with OpenCV and captions each frame.
- `VideoModel.text_to_video()` generates several image frames from prompt variants and stitches them into an MP4.

## Setup Instructions

1. Open a terminal in the assignment folder.

```powershell
cd c:\Users\vkspn\Python_Tutorials_Krish\Assignments\Assignment3_MultiModality_Model
```

2. Activate your Python environment.

```powershell
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies.

```powershell
pip install -r requirements.txt
```

4. Run the Streamlit app.

```powershell
streamlit run streamlit_app.py
```

5. Open `http://localhost:8501` in your browser.

## Using the App

- Pick a task under each modality tab.
- Upload the required image, audio, or video file when prompted.
- Select the desired provider and model in the sidebar before running.
- Review output directly in the browser.

## Practical Notes

- Image and video generation are resource intensive; prefer GPU if available.
- `Text → Audio` and `Audio → Audio` require the `soundfile` and `numpy` packages.
- `Audio → Text` may need `ffmpeg` if the audio file format is not WAV.
- `Video → Text` uses frame-by-frame captioning, so longer videos may take more time.
- If API keys are missing for OpenRouter or Google, the Hugging Face provider is the local fallback.

## Extending the Demo

- Add new model names in `streamlit_app.py` option lists.
- Update `TextModel.FREE_TEXT_MODELS` if you add more Hugging Face or provider-specific models.
- Add new image captioning or ASR models by updating backend defaults and pipeline initialization.
- Improve `VideoModel.text_to_video()` with a dedicated video-generation pipeline for faster results.

## Contact

Edit `vikas_app.py` and rerun `streamlit_app.py` to test new models and modalities.
