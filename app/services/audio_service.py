import speech_recognition as sr
from speech_recognition import AudioData


class AudioService:
    def __init__(self, audio_model: str = 'whisper'):
        self.recognizer = sr.Recognizer()
        self.model = audio_model

    def transcribe(self, audio_path: str, start_time: float | None = None, end_time: float | None = None) -> str:
        if start_time and end_time:
            duration = end_time - start_time
            offset = start_time
        else:
            duration = offset = None
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source, duration, offset)
                return self._get_transcript(audio)
        except sr.UnknownValueError:
            print(f"No speech detected between {start_time:.2f}s and {end_time:.2f}s")
            return ""
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

    def _get_transcript(self, audio: AudioData) -> str:
        if self.model == 'google':
            return self.recognizer.recognize_google(audio)
        else:
            return self.recognizer.recognize_whisper(audio)
