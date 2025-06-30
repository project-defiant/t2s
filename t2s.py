# Implementation based on https://medium.com/@vndee.huynh/build-your-own-voice-assistant-and-run-it-locally-whisper-ollama-bark-c80e6f815cba
import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
import nltk
import torch
import warnings
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextToSpeechService class.
        Args:
            device (str, optional): The device to be used for the model, either "cuda" if a GPU is available or "cpu".
            Defaults to "cuda" if available, otherwise "cpu".
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given text using the specified voice preset.
        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.
        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]

        return self.model.generation_config.sample_rate, np.concatenate(pieces)


def main():
    console = Console()

    stt = whisper.load_model("base.en")
    tts = TextToSpeechService()

    template = """
    You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
    than 20 words.
    The conversation transcript is as follows:
    {history}
    And here is the user's follow-up: {input}
    Your response:
    """
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    chain = ConversationChain(
        prompt=PROMPT,
        verbose=False,
        memory=ConversationBufferMemory(ai_prefix="Assistant:"),
        llm=Ollama(),
    )

    def record_audio(stop_event, data_queue):
        """
        Captures audio data from the user's microphone and adds it to a queue for further processing.
        Args:
            stop_event (threading.Event): An event that, when set, signals the function to stop recording.
            data_queue (queue.Queue): A queue to which the recorded audio data will be added.
        Returns:
            None
        """

        def callback(indata, frames, time, status):
            if status:
                console.print(status)
            data_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=16000, dtype="int16", channels=1, callback=callback
        ):
            while not stop_event.is_set():
                time.sleep(0.1)

    def transcribe(audio_np: np.ndarray) -> str:
        """
        Transcribes the given audio data using the Whisper speech recognition model.
        Args:
            audio_np (numpy.ndarray): The audio data to be transcribed.
        Returns:
            str: The transcribed text.
        """
        result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
        text = result["text"].strip()
        return text

    def get_llm_response(text: str) -> str:
        """
        Generates a response to the given text using the Llama-2 language model.
        Args:
            text (str): The input text to be processed.
        Returns:
            str: The generated response.
        """
        response = chain.predict(input=text)
        if response.startswith("Assistant:"):
            response = response[len("Assistant:") :].strip()
        return response

    def play_audio(sample_rate, audio_array):
        """
        Plays the given audio data using the sounddevice library.
        Args:
            sample_rate (int): The sample rate of the audio data.
            audio_array (numpy.ndarray): The audio data to be played.
        Returns:
            None
        """
        sd.play(audio_array, sample_rate)
        sd.wait()

    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")


if __name__ == "__main__":
    main()
