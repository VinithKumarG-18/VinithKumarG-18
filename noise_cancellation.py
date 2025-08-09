import ctypes
import numpy as np
from queue import Empty


class NoiseCancellation:
    def __init__(self, raw_audio_queue, denoised_audio_queue, dll_path):
        self.raw_audio_queue = raw_audio_queue
        self.denoised_audio_queue = denoised_audio_queue
        
        try:
            # Load the RNNoise DLL
            self.lib = ctypes.CDLL(dll_path)
            self.lib.rnnoise_create.restype = ctypes.c_void_p
            self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
            self.lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            self.lib.rnnoise_process_frame.restype = ctypes.c_float
            
            # Create RNNoise state
            self.st = self.lib.rnnoise_create()
        except Exception as e:
            print(f"Error loading RNNoise DLL: {e}")
            self.lib = None
            self.st = None

    def process_frame(self):
        """Process audio frames for noise cancellation."""
        print("Noise cancellation thread started.")
        
        while True:
            try:
                # Get frame from queue with timeout
                frame = self.raw_audio_queue.get(timeout=1.0)
                
                # Check for shutdown signal
                if frame is None:
                    print("Noise cancellation thread received shutdown signal.")
                    break
                
                # Validate frame
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    print("Invalid frame received in noise cancellation, skipping.")
                    continue
                
                # Ensure correct frame size
                if frame.shape[0] != 480:
                    print(f"Warning: Frame size {frame.shape[0]} != 480, skipping.")
                    continue
                
                # Process with RNNoise if available
                if self.lib and self.st:
                    try:
                        # Convert to ctypes arrays
                        input_array = (ctypes.c_float * 480)(*frame)
                        output_array = (ctypes.c_float * 480)()
                        
                        # Process frame
                        vad_prob = self.lib.rnnoise_process_frame(self.st, output_array, input_array)
                        
                        # Convert back to numpy array
                        denoised_frame = np.array(output_array, dtype=np.float32)
                    except Exception as e:
                        print(f"Error in RNNoise processing: {e}")
                        denoised_frame = frame  # Fallback to original frame
                else:
                    # Fallback: just pass through without noise cancellation
                    denoised_frame = frame
                
                # Put processed frame in output queue
                try:
                    self.denoised_audio_queue.put_nowait(denoised_frame)
                except:
                    # Queue is full, skip this frame
                    pass
                    
            except Empty:
                # Timeout waiting for frame, continue
                continue
            except Exception as e:
                print(f"Error in noise cancellation: {e}")
                continue
        
        # Cleanup
        if self.lib and self.st:
            try:
                self.lib.rnnoise_destroy(self.st)
            except:
                pass
        
        print("Noise cancellation thread ended.")