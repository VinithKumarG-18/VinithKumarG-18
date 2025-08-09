import numpy as np
import torch
import whisper
from queue import Empty
import tempfile
import soundfile as sf


class SpeechToText:
    def __init__(self, flag_dict, model_name, voice_cancelled_queue, transcript_queue):
        self.flag_dict = flag_dict
        self.model_name = model_name
        self.voice_cancelled_queue = voice_cancelled_queue
        self.transcript_queue = transcript_queue
        
        # Load Whisper model
        try:
            self.model = whisper.load_model(model_name)
            print(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None

    def segment_detection(self):
        """Process voice segments and convert to text."""
        print("Speech-to-text thread started.")
        
        while True:
            try:
                # Get voice segment with timeout
                voice_segment = self.voice_cancelled_queue.get(timeout=1.0)
                
                # Check for shutdown signal
                if voice_segment is None:
                    print("Speech-to-text thread received shutdown signal.")
                    break
                
                # Validate segment
                if not isinstance(voice_segment, np.ndarray) or voice_segment.size == 0:
                    print("Invalid voice segment received, skipping.")
                    continue
                
                # Process with Whisper if available
                if self.model:
                    try:
                        # Ensure proper format for Whisper (float32, mono)
                        audio_data = voice_segment.astype(np.float32)
                        
                        # Normalize audio if needed
                        if np.max(np.abs(audio_data)) > 1.0:
                            audio_data = audio_data / np.max(np.abs(audio_data))
                        
                        # Transcribe using Whisper
                        result = self.model.transcribe(audio_data, language='en', fp16=False)
                        transcript = result['text'].strip()
                        
                        if transcript:
                            print(f"Transcribed: {transcript}")
                            # Put transcript in queue
                            try:
                                self.transcript_queue.put_nowait(transcript)
                            except:
                                print("Transcript queue is full, skipping transcript.")
                        else:
                            print("Empty transcript received.")
                            # Put None to indicate empty transcript
                            try:
                                self.transcript_queue.put_nowait(None)
                            except:
                                pass
                    
                    except Exception as e:
                        print(f"Error in speech-to-text processing: {e}")
                        # Put None to indicate transcription failed
                        try:
                            self.transcript_queue.put_nowait(None)
                        except:
                            pass
                else:
                    print("Whisper model not available, skipping transcription.")
                    try:
                        self.transcript_queue.put_nowait(None)
                    except:
                        pass
                        
            except Empty:
                # Timeout waiting for voice segment, continue
                continue
            except Exception as e:
                print(f"Error in speech-to-text thread: {e}")
                continue
        
        print("Speech-to-text thread ended.")