import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import whisper
import os

# Load Whisper model
# "base" is fast and accurate enough for meetings
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    """Convert audio file to text transcript"""
    print(f"Transcribing audio: {audio_path}")
    result = model.transcribe(audio_path)
    return result["text"]

def read_transcript(file_path):
    """Read transcript from text file"""
    with open(file_path, "r") as f:
        return f.read()

def process_input(file_path):
    """
    Handle both audio and text files
    Returns transcript as text regardless of input type
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    # Audio files
    if extension in [".mp3", ".mp4", ".wav", ".m4a", ".ogg"]:
        print("Audio file detected — transcribing with Whisper...")
        return transcribe_audio(file_path)
    
    # Text files
    elif extension in [".txt"]:
        print("Text file detected — reading directly...")
        return read_transcript(file_path)
    
    else:
        raise ValueError(f"Unsupported file type: {extension}")


# Test it
if __name__ == "__main__":
    # Test with audio file — change filename to yours
    audio_files = [f for f in os.listdir(".") 
                   if f.endswith(('.mp3', '.m4a', '.wav', '.ogg'))]
    
    if audio_files:
        print(f"Found audio file: {audio_files[0]}")
        transcript = process_input(audio_files[0])
        print("\\nTranscribed Text:")
        print("="*50)
        print(transcript)
        print("="*50)
        print(f"\\nTotal characters: {len(transcript)}")
    else:
        print("No audio files found — testing with sample_meeting.txt")
        transcript = process_input("sample_meeting.txt")
        print(transcript[:500])