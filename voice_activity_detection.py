import numpy as np
import torch
from queue import Empty


class VoiceActivityDetection:
    def __init__(self, denoised_audio_queue, voice_detected_queue, silence_timeout, flag_dict):
        self.denoised_audio_queue = denoised_audio_queue
        self.voice_detected_queue = voice_detected_queue
        self.silence_timeout = silence_timeout
        self.flag_dict = flag_dict
        
        # Initialize Silero VAD
        try:
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                             model='silero_vad',
                                             force_reload=False,
                                             onnx=False)
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            
            self.vad_iterator = self.VADIterator(self.model)
        except Exception as e:
            print(f"Error initializing Silero VAD: {e}")
            self.model = None
            self.vad_iterator = None

    def process_stream(self):
        """Process audio stream for voice activity detection."""
        print("Voice activity detection thread started.")
        
        audio_buffer = []
        silence_start = None
        
        while True:
            try:
                # Get chunk from queue with timeout
                chunk = self.denoised_audio_queue.get(timeout=1.0)
                
                # Check for shutdown signal
                if chunk is None:
                    print("Voice activity detection thread received shutdown signal.")
                    break
                
                # Validate chunk
                if not isinstance(chunk, np.ndarray) or chunk.size == 0:
                    print("Invalid chunk received in VAD, skipping.")
                    continue
                
                # Convert to float32 if needed
                try:
                    chunk = chunk.astype(np.float32)
                except Exception as e:
                    print(f"Error converting chunk to float32: {e}")
                    continue
                
                # Add to buffer
                audio_buffer.extend(chunk)
                
                # Process when we have enough audio (about 1 second at 48kHz)
                if len(audio_buffer) >= 48000:
                    try:
                        # Convert to tensor
                        audio_tensor = torch.from_numpy(np.array(audio_buffer, dtype=np.float32))
                        
                        # Detect speech if VAD is available
                        if self.model and self.vad_iterator:
                            try:
                                speech_dict = self.vad_iterator(audio_tensor, return_seconds=True)
                                
                                if speech_dict:
                                    # Voice detected
                                    self.flag_dict["voice_detected"] = True
                                    silence_start = None
                                    
                                    # Put audio in voice detected queue
                                    try:
                                        self.voice_detected_queue.put_nowait(np.array(audio_buffer, dtype=np.float32))
                                    except:
                                        pass  # Queue full, skip
                                else:
                                    # No voice detected
                                    if self.flag_dict["voice_detected"]:
                                        if silence_start is None:
                                            silence_start = len(audio_buffer) / 48000.0  # Convert to seconds
                                        elif (len(audio_buffer) / 48000.0 - silence_start) > self.silence_timeout:
                                            self.flag_dict["voice_detected"] = False
                                            self.flag_dict["speech_ended"] = True
                            except Exception as e:
                                print(f"Error in VAD processing: {e}")
                                # Fallback: assume voice detected
                                try:
                                    self.voice_detected_queue.put_nowait(np.array(audio_buffer, dtype=np.float32))
                                except:
                                    pass
                        else:
                            # Fallback without VAD: just pass through
                            try:
                                self.voice_detected_queue.put_nowait(np.array(audio_buffer, dtype=np.float32))
                            except:
                                pass
                    
                    except Exception as e:
                        print(f"Error processing audio buffer in VAD: {e}")
                    
                    # Clear buffer (keep some overlap)
                    audio_buffer = audio_buffer[-4800:]  # Keep last 0.1 seconds
                    
            except Empty:
                # Timeout waiting for chunk, continue
                continue
            except Exception as e:
                print(f"Error in voice activity detection: {e}")
                continue
        
        print("Voice activity detection thread ended.")