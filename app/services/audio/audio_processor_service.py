import os

import speech_recognition as sr
from speech_recognition import AudioData

import os
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import librosa
import requests
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, filtfilt
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip
import warnings

warnings.filterwarnings('ignore')



class AudioProcessorService:
    def __init__(self, audio_model: str = 'google', output_dir='audio_analysis'):
        self.recognizer = sr.Recognizer()
        self.model = audio_model
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create temp directory for downloaded videos and processed audio
        self.temp_dir = tempfile.mkdtemp()

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

    def extract_audio(self, video_path):
        """
        Extract audio from visual file.

        Args:
            video_path (str): Path to visual file

        Returns:
            str: Path to extracted audio file
        """
        # Define output audio path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.temp_dir, f"{base_name}_raw.wav")

        try:
            # Extract audio using moviepy
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(audio_path, logger=None)
            video.close()

            # Process audio to isolate speech
            processed_audio_path = self.isolate_speech(audio_path)

            return processed_audio_path

        except Exception as e:
            print(f"Error extracting audio from {audio_path}: {e}")
            return None

    def analyze_pitch(self, audio_path):
        """
        Analyze pitch characteristics of the audio.

        Args:
            audio_path (str): Path to audio file

        Returns:
            dict: Dictionary containing pitch metrics
        """
        # Load audio using parselmouth
        sound = parselmouth.Sound(audio_path)

        # Extract pitch using Praat algorithm
        pitch = call(sound, "To Pitch", 0.0, 75, 500)

        # Get pitch values
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames

        if len(pitch_values) == 0:
            return {
                'mean_pitch': 0,
                'min_pitch': 0,
                'max_pitch': 0,
                'pitch_range': 0,
                'pitch_std': 0,
                'pitch_variability': 0
            }

        # Calculate pitch metrics
        mean_pitch = float(np.mean(pitch_values))
        min_pitch = float(np.min(pitch_values))
        max_pitch = float(np.max(pitch_values))
        pitch_range = float(max_pitch - min_pitch)
        pitch_std = float(np.std(pitch_values))

        # Calculate pitch variability (coefficient of variation)
        pitch_variability = float(pitch_std / mean_pitch if mean_pitch > 0 else 0)

        return {
            'mean_pitch': mean_pitch,
            'min_pitch': min_pitch,
            'max_pitch': max_pitch,
            'pitch_range': pitch_range,
            'pitch_std': pitch_std,
            'pitch_variability': pitch_variability
        }

    def analyze_volume(self, audio_path):
        """
        Analyze volume characteristics of the audio.

        Args:
            audio_path (str): Path to audio file

        Returns:
            dict: Dictionary containing volume metrics
        """
        # Load audio using librosa
        y, sr = librosa.load(audio_path, sr=None)

        # Calculate volume (RMS energy)
        S, phase = librosa.magphase(librosa.stft(y))
        rms = librosa.feature.rms(S=S)

        # Get volume metrics
        mean_volume = float(np.mean(rms))
        max_volume = float(np.max(rms))
        volume_std = float(np.std(rms))

        # Calculate volume dynamics (std / mean)
        volume_dynamics = float(volume_std / mean_volume if mean_volume > 0 else 0)

        # Calculate silence ratio (proportion of low energy frames)
        silence_threshold = 0.1 * np.mean(rms)
        silence_ratio = float(np.sum(rms < silence_threshold) / len(rms[0]))

        return {
            'mean_volume': mean_volume,
            'max_volume': max_volume,
            'volume_std': volume_std,
            'volume_dynamics': volume_dynamics,
            'silence_ratio': silence_ratio
        }

    def analyze_speech_rate(self, audio_path):
        """
        Analyze speech rate and rhythm.

        Args:
            audio_path (str): Path to audio file

        Returns:
            dict: Dictionary containing speech rate metrics
        """
        # Load audio using librosa
        y, sr = librosa.load(audio_path, sr=None)

        # Detect speech onset frames
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

        # Calculate speech rate (onsets per second)
        duration = librosa.get_duration(y=y, sr=sr)
        speech_rate = float(len(onset_frames) / duration if duration > 0 else 0)

        # Calculate pause frequency
        # Using silence detection
        S = np.abs(librosa.stft(y))
        rms = librosa.feature.rms(S=S)[0]
        silence_threshold = 0.1 * np.mean(rms)
        is_silence = rms < silence_threshold

        # Calculate pause count and average pause duration
        pause_count = 0
        pause_durations = []
        current_pause = 0

        for i in range(len(is_silence)):
            if is_silence[i]:
                current_pause += 1
            elif current_pause > 0:
                pause_count += 1
                # Convert frames to seconds
                pause_duration = current_pause * librosa.frames_to_time(1, sr=sr, hop_length=512)
                pause_durations.append(pause_duration)
                current_pause = 0

        # Add the last pause if there is one
        if current_pause > 0:
            pause_count += 1
            pause_duration = current_pause * librosa.frames_to_time(1, sr=sr, hop_length=512)
            pause_durations.append(pause_duration)

        # Calculate average pause duration
        avg_pause_duration = float(np.mean(pause_durations) if pause_durations else 0)
        pause_frequency = float(pause_count / duration if duration > 0 else 0)

        return {
            'speech_rate': speech_rate,
            'pause_frequency': pause_frequency,
            'avg_pause_duration': avg_pause_duration
        }

    def analyze_voice_quality(self, audio_path):
        """
        Analyze voice quality metrics using Praat.

        Args:
            audio_path (str): Path to audio file

        Returns:
            dict: Dictionary containing voice quality metrics
        """
        # Load audio using parselmouth
        sound = parselmouth.Sound(audio_path)

        # Measure harmonicity (harmonics-to-noise ratio)
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = float(call(harmonicity, "Get mean", 0, 0))
        except:
            hnr = 0.0

        # Measure jitter (pitch perturbation)
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        except:
            jitter = 0.0

        # Measure shimmer (amplitude perturbation)
        try:
            shimmer = float(call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        except:
            shimmer = 0.0

        return {
            'harmonics_to_noise': hnr,
            'jitter': jitter,
            'shimmer': shimmer
        }

    def isolate_speech(self, audio_path):
        """
        Process audio to isolate speech from background music and noise.s

        Args:
            audio_path (str): Path to input audio file

        Returns:
            str: Path to processed audio file with isolated speech
        """
        # Define output path for processed audio
        base_name, ext = os.path.splitext(os.path.basename(audio_path))
        # Remove _raw suffix if present
        if base_name.endswith('_raw'):
            base_name = base_name[:-4]
        processed_path = os.path.join(self.temp_dir, f"{base_name}_speech_only{ext}")

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Apply vocal range bandpass filter
        nyquist = 0.5 * sr
        low = 80 / nyquist
        high = 3000 / nyquist
        b, a = butter(4, [low, high], btype='band')
        y_filtered = filtfilt(b, a, y)

        # Perform noise reduction
        y_reduced = nr.reduce_noise(
            y=y_filtered,
            sr=sr,
            prop_decrease=0.8,
            stationary=False
        )

        # Speech enhancement using harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y_reduced)

        # Voice Activity Detection
        S = np.abs(librosa.stft(y_harmonic)) ** 2
        energy = librosa.feature.rms(S=S)[0]

        # Threshold for speech detection
        threshold = 0.5 * np.mean(energy) + 0.1 * np.std(energy)
        speech_frames = energy > threshold

        # Create a mask for speech frames
        speech_mask = np.zeros_like(y_harmonic)
        frame_length = 2048
        hop_length = 512

        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                start = i * hop_length
                end = min(start + frame_length, len(y_harmonic))
                speech_mask[start:end] = 1.0

        # Apply smoothing to the mask
        smoothing_window = int(sr * 0.05)  # 50ms smoothing
        smoothed_mask = np.convolve(speech_mask, np.ones(smoothing_window) / smoothing_window, mode='same')
        smoothed_mask = np.minimum(smoothed_mask, 1.0)  # Cap at 1.0

        # Apply the mask to get speech-only audio
        y_speech = y_harmonic * smoothed_mask

        # Check if we have enough speech content
        speech_energy = np.sum(y_speech ** 2)
        original_energy = np.sum(y ** 2)
        speech_ratio = speech_energy / original_energy if original_energy > 0 else 0

        # If low speech content, use less aggressive filtering
        if speech_ratio < 0.1:
            y_speech = nr.reduce_noise(y=y, sr=sr)

        # Save the processed audio
        sf.write(processed_path, y_speech, sr)

        # Additional processing with pydub
        try:
            # Load processed audio
            sound = AudioSegment.from_file(processed_path)

            # Set silence threshold and duration
            silence_threshold = -40  # dB
            min_silence_len = 300  # ms

            # Split audio on silence
            chunks = self.split_on_silence(
                sound,
                min_silence_len=min_silence_len,
                silence_thresh=silence_threshold,
                keep_silence=100  # keep 100ms of silence
            )

            # Combine non-silent chunks
            if chunks:
                result = AudioSegment.empty()
                for chunk in chunks:
                    # Normalize volume
                    chunk = self.normalize_volume(chunk)
                    result += chunk

                # Export the final result
                result.export(processed_path, format="wav")

        except Exception as e:
            print(f"Warning: Additional audio processing failed - {e}")

        return processed_path

    def split_on_silence(self, audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100):
        """
        Split audio on silent parts.
        """
        # Initialize parameters
        not_silence_ranges = []
        silent = True

        # Convert silence threshold to amplitude value
        silence_thresh_amp = audio_segment.dBFS + silence_thresh

        # Get segment array data
        seg_array = np.array(audio_segment.get_array_of_samples())
        segment_len = len(seg_array)

        # Calculate RMS in chunks
        chunk_size = int(audio_segment.frame_rate * (min_silence_len / 1000.0))
        chunk_size = max(chunk_size, 1)  # Ensure at least 1 sample per chunk

        # Process chunks
        for i in range(0, segment_len, chunk_size):
            chunk_end = min(i + chunk_size, segment_len)
            chunk = seg_array[i:chunk_end]

            # Calculate RMS of chunk
            rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0

            # Check if chunk is silent
            is_silence = rms < silence_thresh_amp

            # Mark ranges of non-silence
            if is_silence:
                if not silent:
                    not_silence_ranges.append((start, i))
                    silent = True
            elif silent:
                start = i
                silent = False

        # Add the last non-silent range if needed
        if not silent:
            not_silence_ranges.append((start, segment_len))

        # Keep some silence around non-silent ranges
        keep_silence_samples = int(audio_segment.frame_rate * (keep_silence / 1000.0))

        # Extract non-silent chunks
        chunks = []
        for start_i, end_i in not_silence_ranges:
            # Adjust start and end with keep_silence
            start_i = max(0, start_i - keep_silence_samples)
            end_i = min(segment_len, end_i + keep_silence_samples)

            # Extract the chunk
            chunk = audio_segment[start_i:end_i]
            chunks.append(chunk)

        return chunks

    def normalize_volume(self, audio_segment, target_dBFS=-15.0):
        """
        Normalize the volume of an audio segment to a target dB level.
        """
        # Calculate current dBFS
        current_dBFS = audio_segment.dBFS

        # Only adjust if the audio isn't silent
        if np.isfinite(current_dBFS):
            # Calculate the change needed
            change_in_dBFS = target_dBFS - current_dBFS

            # Apply gain adjustment
            return audio_segment.apply_gain(change_in_dBFS)
        else:
            # If audio is essentially silent, return as is
            return audio_segment

    def extract_audio_features(self, video_path):
        # Extract audio
        audio_path = self.extract_audio(video_path)

        if not audio_path:
            print(f"Failed to extract audio from visual at {video_path}")
            return None

        # Analyze audio features
        pitch_features = self.analyze_pitch(audio_path)
        volume_features = self.analyze_volume(audio_path)
        speech_features = self.analyze_speech_rate(audio_path)
        voice_quality = self.analyze_voice_quality(audio_path)

        audio_features = {
            'pitch': pitch_features,
            'volume': volume_features,
            'speech_rate': speech_features,
            'voice_quality': voice_quality
        }

        return audio_features
