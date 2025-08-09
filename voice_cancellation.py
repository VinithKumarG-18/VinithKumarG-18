import numpy as np
from queue import Empty


class VoiceCancellation:
    def __init__(self, voice_detected_queue, voice_cancelled_queue, flag_dict):
        self.voice_detected_queue = voice_detected_queue
        self.voice_cancelled_queue = voice_cancelled_queue
        self.flag_dict = flag_dict

    def get_loudest_voice(self):
        """Extract the loudest voice segment from detected voice chunks."""
        print("Voice cancellation thread started.")
        
        voice_chunks = []
        
        while True:
            try:
                # Get voice detected chunk with timeout
                voice_detected_chunk = self.voice_detected_queue.get(timeout=1.0)
                
                # Check for shutdown signal
                if voice_detected_chunk is None:
                    print("Voice cancellation thread received shutdown signal.")
                    break
                
                # Validate chunk
                if not isinstance(voice_detected_chunk, np.ndarray) or voice_detected_chunk.size == 0:
                    print("Invalid voice chunk received, skipping.")
                    continue
                
                # Calculate RMS (Root Mean Square) for volume
                try:
                    rms = np.sqrt(np.mean(voice_detected_chunk ** 2))
                    
                    # Store chunk with its RMS value
                    voice_chunks.append((voice_detected_chunk, rms))
                    
                    # Limit buffer size to prevent memory issues
                    if len(voice_chunks) > 100:
                        voice_chunks = voice_chunks[-50:]  # Keep only the latest 50 chunks
                    
                except Exception as e:
                    print(f"Error calculating RMS: {e}")
                    # Still add the chunk with default RMS
                    voice_chunks.append((voice_detected_chunk, 0.0))
                
                # Check if speech has ended
                if self.flag_dict.get("speech_ended", False):
                    try:
                        if voice_chunks:
                            # Find the chunk with highest RMS (loudest)
                            loudest_chunk = max(voice_chunks, key=lambda x: x[1])[0]
                            
                            # Put the loudest chunk in the output queue
                            try:
                                self.voice_cancelled_queue.put_nowait(loudest_chunk)
                                print(f"Loudest voice chunk selected (RMS: {max(voice_chunks, key=lambda x: x[1])[1]:.4f})")
                            except:
                                print("Voice cancelled queue is full, skipping loudest chunk.")
                        
                        # Clear chunks and reset flag
                        voice_chunks = []
                        self.flag_dict["speech_ended"] = False
                        
                    except Exception as e:
                        print(f"Error processing loudest voice: {e}")
                        # Clear chunks anyway to prevent memory buildup
                        voice_chunks = []
                        self.flag_dict["speech_ended"] = False
                        
            except Empty:
                # Timeout waiting for voice chunk, continue
                continue
            except Exception as e:
                print(f"Error in voice cancellation: {e}")
                continue
        
        print("Voice cancellation thread ended.")