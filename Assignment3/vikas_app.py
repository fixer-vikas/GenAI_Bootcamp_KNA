"""
Multimodal AI backend module for Assignment 3.
Supports text, image, audio, video, and cross-modal conversions.
"""

import os
import re
import json
import base64
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

try:
    import requests
except ImportError:
    requests = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoConfig, pipeline
except ImportError:
    pipeline = None
    AutoConfig = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
except ImportError:
    StableDiffusionPipeline = None
    StableDiffusionImg2ImgPipeline = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import librosa
    import numpy as np
except ImportError:
    librosa = None
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

def set_api_keys() -> None:
    os.environ["OPENROUTER_API_KEY"] = "" --- insert your OpenRouter API key here if you have one ---
    os.environ["GOOGLE_API_KEY"] = "" --- insert your Google API key here if you have one ---
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "" --- insert your Hugging Face API token here if you have one ---

set_api_keys()

class ProviderKeys:
    OPENROUTER = os.getenv("OPENROUTER_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

class TextModel:
    """Text generation, summarization, and QA across multiple providers."""

    DEFAULT_TEMPERATURE = 0.7

    FREE_TEXT_MODELS = {
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

    def __init__(
        self,
        provider: str = "openrouter",
        model_name: str = "gpt-4o-mini",
        presentation_style: str = "Balanced",
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.presentation_style = presentation_style
        self._hf_text_pipeline = None
        self._hf_text_pipeline_task = None
        self._hf_text_pipeline_model_name = None
        self._summarizer = None
        self._qa_pipeline = None

    def _apply_presentation_style(self, prompt: str) -> str:
        style = self.presentation_style or "Balanced"
        if style.lower() == "balanced":
            return prompt
        return f"Write in a {style.lower()} style. {prompt}"

    def _is_seq2seq_model(self, model_name: str) -> bool:
        return any(
            key in model_name.lower()
            for key in ["t5", "flan", "mt5", "m2m", "mbart", "marian", "opus-mt"]
        )

    def _get_translation_model(self, target_language: str) -> str | None:
        mapping = {
            "hindi": "Helsinki-NLP/opus-mt-en-hi",
            "french": "Helsinki-NLP/opus-mt-en-fr",
            "spanish": "Helsinki-NLP/opus-mt-en-es",
            "german": "Helsinki-NLP/opus-mt-en-de",
            "italian": "Helsinki-NLP/opus-mt-en-it",
            "russian": "Helsinki-NLP/opus-mt-en-ru",
            "portuguese": "Helsinki-NLP/opus-mt-en-pt",
            "turkish": "Helsinki-NLP/opus-mt-en-tr",
        }
        return mapping.get(target_language.strip().lower())

    def generate(
        self,
        prompt: str,
        max_tokens: int = 120,
        presentation_style: str | None = None,
    ) -> str:
        temperature = self.DEFAULT_TEMPERATURE
        if presentation_style is not None:
            style_prompt = f"Write in a {presentation_style.lower()} style. {prompt}"
        else:
            style_prompt = self._apply_presentation_style(prompt)

        if self.provider == "openrouter" and ProviderKeys.OPENROUTER:
            return self._generate_openrouter(style_prompt, max_tokens, temperature)
        if self.provider == "google" and ProviderKeys.GOOGLE_API_KEY:
            return self._generate_google_gemini(style_prompt, temperature)
        return self._generate_huggingface(style_prompt, max_length=max_tokens, temperature=temperature)

    def _generate_openrouter(self, prompt: str, max_tokens: int, temperature: float | None = None) -> str:
        if not requests:
            return "requests library is not installed."

        url = "https://api.openrouter.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {ProviderKeys.OPENROUTER}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name or "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            return f"OpenRouter generation error: {exc}"

    def _generate_google_gemini(self, prompt: str, temperature: float | None = None) -> str:
        if not requests:
            return "requests library is not installed."

        key = ProviderKeys.GOOGLE_API_KEY
        model = self.model_name or "text-bison-001"
        url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText?key={key}"
        payload = {
            "prompt": {
                "text": prompt
            },
            "temperature": temperature,
            "maxOutputTokens": 200,
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["output"]
        except Exception as exc:
            return f"Google Gemini generation error: {exc}"

    def _generate_huggingface(self, prompt: str, max_length: int = 120, temperature: float | None = None) -> str:
        if pipeline is None:
            return "Transformers is required for Hugging Face generation."

        model_name = self.model_name or "EleutherAI/gpt-neo-125M"
        normalized_prompt = prompt.strip()
        is_translation_prompt = bool(re.match(r"(?i)^\s*translate\b", normalized_prompt))
        translation_model_name: str | None = None
        target_language = "Hindi"

        if is_translation_prompt:
            translate_to_match = re.match(
                r"(?i)^\s*translate\s+(?:to|into)\s+([a-zA-Z ]+?)\s*[:,\-]\s*(.+)$",
                normalized_prompt,
            )
            if translate_to_match:
                target_language = translate_to_match.group(1).strip().capitalize()
                normalized_prompt = translate_to_match.group(2).strip()
            else:
                colon_match = re.match(r"(?i)^\s*translate\s*[:\-]\s*(.+)$", normalized_prompt)
                if colon_match:
                    normalized_prompt = colon_match.group(1).strip()
                else:
                    inline_match = re.match(
                        r"(?i)^\s*translate\s+(.+?)\s+(?:to|into)\s+([a-zA-Z ]+)$",
                        normalized_prompt,
                    )
                    if inline_match:
                        normalized_prompt = inline_match.group(1).strip()
                        target_language = inline_match.group(2).strip().capitalize()
                    else:
                        normalized_prompt = re.sub(r"(?i)^translate[\s:,-]*", "", normalized_prompt).strip()

            normalized_prompt = f"Translate the following text to {target_language}:\n{normalized_prompt}"
            if not self._is_seq2seq_model(model_name):
                translation_model_name = self._get_translation_model(target_language)
                if translation_model_name:
                    model_name = translation_model_name

        is_seq2seq_model = self._is_seq2seq_model(model_name)
        task = "translation" if translation_model_name else "text2text-generation" if is_seq2seq_model else "text-generation"

        if (
            self._hf_text_pipeline is None
            or self._hf_text_pipeline_task != task
            or self._hf_text_pipeline_model_name != model_name
        ):
            try:
                self._hf_text_pipeline = pipeline(
                    task,
                    model=model_name,
                    device=0 if torch is not None and torch.cuda.is_available() else -1,
                )
            except Exception:
                self._hf_text_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if torch is not None and torch.cuda.is_available() else -1,
                )
                task = "text-generation"

            self._hf_text_pipeline_task = task
            self._hf_text_pipeline_model_name = model_name

        try:
            hf_args = {
                "max_length": max_length,
                "num_return_sequences": 1,
                "truncation": True,
            }
            if task != "translation" and temperature is not None:
                hf_args["temperature"] = temperature
                hf_args["do_sample"] = temperature > 0
            result = self._hf_text_pipeline(normalized_prompt, **hf_args)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                return result[0].get("generated_text", "") or result[0].get("translation_text", "")
            if isinstance(result, dict):
                return result.get("generated_text", "") or result.get("translation_text", "")
            return str(result)
        except Exception as exc:
            return f"Hugging Face text generation error: {exc}"

    def summarize(self, text: str) -> str:
        if pipeline is None:
            return "Transformers is required for summarization."
        if self._summarizer is None:
            self._summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch is not None and torch.cuda.is_available() else -1,
            )
        try:
            result = self._summarizer(text, max_length=130, min_length=30, do_sample=False)
            return result[0]["summary_text"]
        except Exception as exc:
            return f"Summarization error: {exc}"

    def answer_question(self, question: str, context: str) -> str:
        if pipeline is None:
            return "Transformers is required for QA."
        if self._qa_pipeline is None:
            self._qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if torch is not None and torch.cuda.is_available() else -1,
            )
        try:
            result = self._qa_pipeline(question=question, context=context)
            return result["answer"]
        except Exception as exc:
            return f"QA error: {exc}"


class ImageModel:
    """Image generation, captioning, and image-to-image transformations."""

    def __init__(
        self,
        image_model_name: str = "runwayml/stable-diffusion-v1-5",
        caption_model_name: str = "nlpconnect/vit-gpt2-image-captioning",
    ):
        self.image_model_name = image_model_name
        self.caption_model_name = caption_model_name
        self._captioner = None
        self._sd_pipe = None
        self._img2img_pipe = None
        self._last_error: Optional[str] = None

    def text_to_image(self, prompt: str, steps: int = 30) -> Optional["Image.Image"]:
        if StableDiffusionPipeline is None or Image is None or torch is None:
            print("Diffusers, PIL, and PyTorch are required for text-to-image generation.")
            return None
        if self._sd_pipe is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._sd_pipe = StableDiffusionPipeline.from_pretrained(
                self.image_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
        try:
            with torch.autocast("cuda") if torch.cuda.is_available() else nullcontext():
                result = self._sd_pipe(prompt, num_inference_steps=steps, guidance_scale=7.5)
            return result.images[0]
        except Exception as exc:
            print(f"Text-to-image error: {exc}")
            return None

    def image_to_text(self, image_path: str) -> str:
        if pipeline is None or Image is None:
            return "Transformers and PIL are required for image captioning."
        if self._captioner is None:
            self._captioner = pipeline(
                "image-to-text",
                model=self.caption_model_name,
                device=0 if torch is not None and torch.cuda.is_available() else -1,
            )
        try:
            image = Image.open(image_path)
            result = self._captioner(image)
            return result[0]["generated_text"]
        except Exception as exc:
            return f"Image captioning error: {exc}"

    def image_to_image(self, input_image_path: str, prompt: str, strength: float = 0.75, steps: int = 25) -> Optional["Image.Image"]:
        if StableDiffusionImg2ImgPipeline is None or Image is None or torch is None:
            print("Diffusers, PIL, and PyTorch are required for image-to-image generation.")
            return None
        if self._img2img_pipe is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.image_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
        try:
            init_image = Image.open(input_image_path).convert("RGB")
            init_image = init_image.resize((768, 768))
            with torch.autocast("cuda") if torch.cuda.is_available() else nullcontext():
                result = self._img2img_pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                )
            self._last_error = None
            return result.images[0]
        except Exception as exc:
            error_message = f"Image-to-image error: {exc}"
            print(error_message)
            self._last_error = error_message
            return None


class AudioModel:
    """Speech recognition, text-to-speech, and audio transformation."""

    def __init__(
        self,
        asr_model_name: str = "openai/whisper-small",
        tts_model_name: str = "microsoft/speecht5_tts",
    ):
        self._asr = None
        self._tts = None
        self._speaker_embeddings = None
        self._asr_model_name = asr_model_name
        self._tts_model_name = tts_model_name

    def _load_default_speaker_embeddings(self):
        if torch is None:
            return None
        if self._speaker_embeddings is not None:
            return self._speaker_embeddings

        embedding_dim = 512
        if AutoConfig is not None:
            try:
                config = AutoConfig.from_pretrained(self._tts_model_name)
                embedding_dim = getattr(config, "speaker_embedding_dim", embedding_dim) or embedding_dim
            except Exception:
                pass

        if load_dataset is not None:
            try:
                dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="train")
                if len(dataset) > 0:
                    example = dataset[0]
                    xvector = None
                    for key in ("xvector", "speaker_embeddings", "embeddings"):
                        if key in example:
                            xvector = example[key]
                            break
                    if xvector is not None:
                        embeddings = torch.tensor(xvector)
                        if embeddings.ndim == 1:
                            embeddings = embeddings.unsqueeze(0)
                        self._speaker_embeddings = embeddings
                        return self._speaker_embeddings
            except Exception as exc:
                print(f"Failed to load default speaker embeddings from xvector dataset: {exc}")

        try:
            self._speaker_embeddings = torch.zeros((1, embedding_dim), dtype=torch.float32)
            return self._speaker_embeddings
        except Exception as exc:
            print(f"Failed to create fallback speaker embeddings: {exc}")
            return None

    def _load_audio_array(self, audio_path: str):
        if np is None:
            return None, None
        if sf is not None:
            try:
                audio, sr = sf.read(audio_path, dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                return audio, int(sr)
            except Exception:
                pass
        if librosa is not None:
            try:
                audio, sr = librosa.load(audio_path, sr=None, mono=True)
                return audio.astype("float32"), int(sr)
            except Exception:
                pass
        return None, None

    def speech_to_text(self, audio_path: str) -> str:
        if pipeline is None:
            return "Transformers is required for speech recognition."
        if self._asr is None:
            self._asr = pipeline(
                "automatic-speech-recognition",
                model=self._asr_model_name,
                device=0 if torch is not None and torch.cuda.is_available() else -1,
            )
        try:
            audio_array, sample_rate = self._load_audio_array(audio_path)
            try:
                result = self._asr(audio_path)
            except Exception:
                if audio_array is not None:
                    result = self._asr(audio_array)
                else:
                    raise
            return result.get("text", "")
        except Exception as exc:
            msg = str(exc)
            if "ffmpeg" in msg.lower() or "avconv" in msg.lower():
                return (
                    "Speech recognition error: ffmpeg was not found. "
                    "Install ffmpeg or provide a WAV file for transcription."
                )
            return f"Speech recognition error: {exc}"

    def text_to_speech(self, text: str, output_path: str = "output_audio.wav") -> str:
        if pipeline is None:
            return "Transformers is required for text-to-speech."
        speaker_embeddings = self._load_default_speaker_embeddings()
        if speaker_embeddings is None:
            return (
                "Text-to-speech error: default speaker embeddings could not be loaded. "
                "Install the datasets library and ensure the xvector dataset is available."
            )
        if self._tts is None:
            self._tts = pipeline(
                "text-to-speech",
                model=self._tts_model_name,
                device=0 if torch is not None and torch.cuda.is_available() else -1,
            )
        try:
            try:
                result = self._tts(text, forward_params={"speaker_embeddings": speaker_embeddings})
            except Exception:
                result = self._tts(text)
            audio_array = None
            sample_rate = None
            if isinstance(result, dict):
                audio_array = result.get("audio") if result.get("audio") is not None else result.get("array")
                sample_rate = result.get("sampling_rate")
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                audio_array = result[0].get("audio") if result[0].get("audio") is not None else result[0].get("array")
                sample_rate = result[0].get("sampling_rate")
            else:
                return f"Text-to-speech error: unexpected pipeline output type {type(result).__name__}"

            if audio_array is None or sample_rate is None:
                return "Text-to-speech error: TTS pipeline returned no audio data."

            if np is not None:
                import soundfile as sf
                sf.write(output_path, audio_array, sample_rate)
                return output_path
            return "NumPy and soundfile are required to save audio."
        except Exception as exc:
            return f"Text-to-speech error: {exc}"

    def audio_to_audio(self, input_audio_path: str, output_audio_path: str = "converted_audio.wav") -> str:
        transcript = self.speech_to_text(input_audio_path)
        if transcript.startswith("Speech recognition error"):
            return transcript
        return self.text_to_speech(f"Rephrase the following speech clearly: {transcript}", output_audio_path)

    def analyze_audio(self, audio_path: str) -> Dict[str, Optional[float]]:
        if librosa is None or np is None:
            return {"error": "librosa and numpy are required for audio analysis."}
        try:
            y, sr = librosa.load(audio_path, sr=None)
            return {
                "sample_rate": sr,
                "duration_seconds": float(len(y) / sr),
                "mean_amplitude": float(np.mean(np.abs(y))),
                "rms_energy": float(np.mean(librosa.feature.rms(y=y))),
            }
        except Exception as exc:
            return {"error": f"Audio analysis error: {exc}"}


class VideoModel:
    """Video frame extraction, captioning, and text-driven video creation."""

    def __init__(self, image_model: Optional[ImageModel] = None):
        self.image_model = image_model or ImageModel()

    def extract_frames(self, video_path: str, frames: int = 4) -> List[str]:
        if cv2 is None:
            return []
        video = cv2.VideoCapture(video_path)
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [int(total * i / max(frames, 1)) for i in range(frames)]
        paths = []
        for idx in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video.read()
            if not ret:
                continue
            image_path = tempfile.mktemp(suffix=".png")
            cv2.imwrite(image_path, frame)
            paths.append(image_path)
        video.release()
        return paths

    def video_to_text(self, video_path: str, frames: int = 4) -> List[str]:
        captions = []
        frame_files = self.extract_frames(video_path, frames)
        for frame_path in frame_files:
            caption = self.image_model.image_to_text(frame_path)
            captions.append(caption)
            try:
                os.remove(frame_path)
            except OSError:
                pass
        return captions

    def video_info(self, video_path: str) -> Dict[str, Optional[float]]:
        if cv2 is None:
            return {"error": "opencv-python is required for video metadata."}
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total = video.get(cv2.CAP_PROP_FRAME_COUNT)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        duration = total / fps if fps else 0.0
        video.release()
        return {
            "fps": fps,
            "total_frames": int(total),
            "width": int(width),
            "height": int(height),
            "duration_seconds": float(duration),
        }

    def text_to_video(self, prompt: str, output_path: str = "generated_video.mp4", frames: int = 4, fps: int = 2) -> str:
        if cv2 is None or Image is None:
            return "opencv-python and PIL are required for text-to-video creation."
        frame_images = []
        for index in range(frames):
            frame_prompt = f"{prompt} for frame {index + 1}, cinematic, high detail"
            image = self.image_model.text_to_image(frame_prompt, steps=20)
            if image is None:
                return "Text-to-video generation failed at image creation."
            frame_images.append(image.convert("RGB"))
        frame_size = frame_images[0].size
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        for image in frame_images:
            frame = cv2.cvtColor(np.array(image.resize(frame_size)), cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()
        return output_path


class MultimodalApp:
    """High-level multimodal orchestration for the assignment."""

    def __init__(
        self,
        text_provider: str = "openrouter",
        text_model_name: str = "gpt-4o-mini",
        image_model_name: str = "runwayml/stable-diffusion-v1-5",
        caption_model_name: str = "nlpconnect/vit-gpt2-image-captioning",
        asr_model_name: str = "openai/whisper-small",
        tts_model_name: str = "microsoft/speecht5_tts",
    ):
        self.text_model = TextModel(provider=text_provider, model_name=text_model_name)
        self.image_model = ImageModel(
            image_model_name=image_model_name,
            caption_model_name=caption_model_name,
        )
        self.audio_model = AudioModel(
            asr_model_name=asr_model_name,
            tts_model_name=tts_model_name,
        )
        self.video_model = VideoModel(self.image_model)

    def text_to_text(self, text: str) -> str:
        return self.text_model.generate(text)

    def text_to_image(self, text: str) -> Optional["Image.Image"]:
        return self.image_model.text_to_image(text)

    def image_to_text(self, image_path: str) -> str:
        return self.image_model.image_to_text(image_path)

    def image_to_image(self, image_path: str, prompt: str) -> Optional["Image.Image"]:
        return self.image_model.image_to_image(image_path, prompt)

    def get_image_error(self) -> Optional[str]:
        return self.image_model._last_error

    def text_to_audio(self, text: str, output_path: str = "text_to_audio.wav") -> str:
        return self.audio_model.text_to_speech(text, output_path)

    def audio_to_text(self, audio_path: str) -> str:
        return self.audio_model.speech_to_text(audio_path)

    def audio_to_audio(self, audio_path: str, output_path: str = "audio_to_audio.wav", output_audio_path: str | None = None) -> str:
        if output_audio_path is not None:
            output_path = output_audio_path
        return self.audio_model.audio_to_audio(audio_path, output_path)

    def text_to_video(self, text: str, output_path: str = "text_to_video.mp4") -> str:
        return self.video_model.text_to_video(text, output_path)

    def video_to_text(self, video_path: str) -> List[str]:
        return self.video_model.video_to_text(video_path)

    def video_to_summary(self, video_path: str) -> str:
        captions = self.video_to_text(video_path)
        text = "\n".join(captions)
        return self.text_model.summarize(text)

    def video_info(self, video_path: str) -> Dict[str, Optional[float]]:
        return self.video_model.video_info(video_path)


# Null context manager for CPU runs when autocast is unavailable
class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
