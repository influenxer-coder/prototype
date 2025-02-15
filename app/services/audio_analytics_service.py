import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os


class AudioAnalyticsService:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def get_transcript(self, video_path: str, start_time: float, end_time: float) -> str:
        """Extract audio transcript for a specific time window."""
        with tempfile.TemporaryDirectory() as temp_dir:
            video_audio = AudioSegment.from_file(video_path)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment = video_audio[start_ms:end_ms]

            temp_audio = os.path.join(temp_dir, "segment.wav")
            segment.export(temp_audio, format="wav")

            try:
                with sr.AudioFile(temp_audio) as source:
                    audio = self.recognizer.record(source)
                    return self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                print(f"No speech detected between {start_time:.2f}s and {end_time:.2f}s")
                return ""
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {str(e)}")
                return ""