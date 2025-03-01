from pydub import AudioSegment


def extract_audio(video_file_path: str, audio_path: str) -> bool:
    """
    Extract audio from a video file
    Args:
        video_file_path: location of video file
        audio_path: path to save audio

    Returns:
        boolean: Successfully extracted audio
    """
    try:
        # Load the video file
        video = AudioSegment.from_file(video_file_path, format="mp4")

        # Export the audio
        video.export(audio_path, format="wav")
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False
