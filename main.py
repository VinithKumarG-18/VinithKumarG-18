import base64
import json
import threading
from queue import Queue, Empty
import os
import time

import numpy as np
import torch
import torchaudio
from flask import Flask, Response, request, jsonify
from flask_sock import Sock

# --- SDK and Custom Module Imports ---
from piopiy import Action
# NOTE: Ensure these custom module files are in the same directory
from noise_cancellation import NoiseCancellation
from agent_response import AgentResponse
from text_to_speech import TextToSpeech
from voice_cancellation import VoiceCancellation
from voice_activity_detection import VoiceActivityDetection
from speech_to_text import SpeechToText

# --- Flask App Initialization ---
app = Flask(__name__)
sock = Sock(app)

# --- Global State for Managing Multiple Calls ---
active_calls = {}

# --- HARDCODED CONFIGURATION ---
# WARNING: Do not share this file publicly with your keys in it.
# All settings for the application are hardcoded in this section and at the bottom of the file.

# 1. Run `ngrok http 8080` in your terminal.
# 2. Copy the https:// forwarding URL (e.g., https://1a2b-3c4d.ngrok-free.app).
# 3. Change it to wss:// and paste it below.
WEBSOCKET_BASE_URL = "wss://3b4d0335faad.ngrok-free.app"  # <--- PASTE YOUR NGROK URL HERE

if "your-ngrok-id" in WEBSOCKET_BASE_URL:
    print("\n\n!!! WARNING: Please update the hardcoded WEBSOCKET_BASE_URL in the script. !!!\n\n")


# --- Audio Conversion Utility Functions (Unchanged) ---
def convert_piopiy_to_audio(base64_mulaw_string):
    """Decodes Piopiy's base64 μ-law audio and resamples it for our AI pipeline."""
    try:
        mulaw_bytes = base64.b64decode(base64_mulaw_string)
        if not mulaw_bytes:
            return np.array([], dtype=np.float32)

        mulaw_tensor = torch.from_numpy(np.frombuffer(mulaw_bytes, dtype=np.uint8))
        waveform = torchaudio.functional.mulaw_decoding(mulaw_tensor)
        resampled = torchaudio.functional.resample(waveform.unsqueeze(0), orig_freq=8000, new_freq=48000)
        return resampled.squeeze(0).numpy()
    except Exception as e:
        print(f"Error in convert_piopiy_to_audio: {e}")
        return np.array([], dtype=np.float32)


def convert_audio_to_piopiy(audio_data_float32, sample_rate=24000):
    """Resamples our generated audio and encodes it to μ-law for Piopiy."""
    try:
        audio_tensor = torch.from_numpy(audio_data_float32.astype(np.float32)).unsqueeze(0)
        resampled_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=8000)
        mulaw_encoded_tensor = torchaudio.functional.mulaw_encoding(resampled_tensor.squeeze(0))
        mulaw_bytes = mulaw_encoded_tensor.numpy().astype(np.uint8).tobytes()
        base64_mulaw_string = base64.b64encode(mulaw_bytes).decode('utf-8')
        return base64_mulaw_string
    except Exception as e:
        print(f"Error in convert_audio_to_piopiy: {e}")
        return None


# --- Per-Call VoiceAssistant AI Pipeline ---
class VoiceAssistant:
    """
    Manages the entire AI pipeline for a single phone call.
    Each call gets its own instance of this class.
    """

    def __init__(self, stt_model_name, silence_timeout, google_tts_key, dll_path, rms_threshold=0.01):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.frame_size = 480  # Frame size for RNNoise (10ms at 48kHz)
        self.shutdown_flag = threading.Event()  # Add shutdown flag for graceful termination

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flag_dict = {"speech_ended": False, "voice_detected": False, "loudest_voice_event": threading.Event(),
                          "is_recording": False, "device": self.device, "shutdown": self.shutdown_flag}

        # Each call gets its own set of queues
        self.raw_audio_queue = Queue()
        self.denoised_audio_queue = Queue()
        self.voice_detected_queue = Queue()
        self.voice_cancelled_queue = Queue()
        self.llm_response_queue = Queue()
        self.transcript_queue = Queue()
        self.synthesized_audio_queue = Queue()

        self.threads = []  # Keep track of threads for this instance

        print("Initializing AI Pipeline Modules for a new call...")
        # Initialize all processing modules
        self.noise_cancellation = NoiseCancellation(raw_audio_queue=self.raw_audio_queue,
                                                    denoised_audio_queue=self.denoised_audio_queue, dll_path=dll_path)
        self.voice_activity_detection = VoiceActivityDetection(denoised_audio_queue=self.denoised_audio_queue,
                                                               voice_detected_queue=self.voice_detected_queue,
                                                               silence_timeout=silence_timeout,
                                                               flag_dict=self.flag_dict)
        self.voice_cancellation = VoiceCancellation(voice_detected_queue=self.voice_detected_queue,
                                                    voice_cancelled_queue=self.voice_cancelled_queue,
                                                    flag_dict=self.flag_dict)
        self.speech_to_text = SpeechToText(flag_dict=self.flag_dict, model_name=stt_model_name,
                                           voice_cancelled_queue=self.voice_cancelled_queue,
                                           transcript_queue=self.transcript_queue)
        self.text_to_speech = TextToSpeech(llm_response_queue=self.llm_response_queue,
                                           synthesized_audio_queue=self.synthesized_audio_queue,
                                           google_tts_key=google_tts_key)
        self.agent_response = AgentResponse(llm_response_queue=self.llm_response_queue,
                                            transcript_queue=self.transcript_queue)
        print("AI Pipeline Modules Initialized.")

    def start(self):
        """Starts all the processing threads for this call's pipeline."""
        print("Starting AI Pipeline threads...")
        thread_targets = [
            self.noise_cancellation.process_frame,
            self.voice_activity_detection.process_stream,
            self.voice_cancellation.get_loudest_voice,
            self.speech_to_text.segment_detection,
            self.agent_response.process_and_respond,
            self.text_to_speech.speech_synthesize
        ]
        for target in thread_targets:
            thread = threading.Thread(target=target, daemon=True)
            self.threads.append(thread)
            thread.start()
        print("All AI Pipeline threads are running.")

    def process_incoming_audio(self, audio_data):
        """Chunks incoming audio and puts it into the processing queue."""
        if self.shutdown_flag.is_set():
            return
            
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        while len(self.audio_buffer) >= self.frame_size:
            frame = self.audio_buffer[:self.frame_size].astype(np.float32)
            try:
                self.raw_audio_queue.put_nowait(frame)
            except:
                # Queue is full or shutdown, skip this frame
                break
            self.audio_buffer = self.audio_buffer[self.frame_size:]

    def shutdown(self):
        """Gracefully stops all threads in the pipeline."""
        print("Shutting down VoiceAssistant instance...")
        self.shutdown_flag.set()
        
        # Put None in all queues to signal shutdown
        queues_to_terminate = [
            self.raw_audio_queue, self.denoised_audio_queue,
            self.voice_detected_queue, self.voice_cancelled_queue,
            self.llm_response_queue, self.transcript_queue, self.synthesized_audio_queue
        ]
        for q in queues_to_terminate:
            try:
                q.put_nowait(None)
            except:
                pass  # Queue might be full, that's okay

        # Wait for threads to finish (with timeout)
        for thread in self.threads:
            thread.join(timeout=2.0)

        print("VoiceAssistant instance shut down signal sent.")


# --- Flask Routes ---

@app.route("/voice-webhook", methods=["GET", "POST"])
def voice_webhook():
    """
    This is the initial endpoint Piopiy calls.
    It responds with JSON to start a WebSocket stream.
    """
    print("✅ Piopiy webhook received. Responding with JSON to start WebSocket stream.")
    action = Action()
    full_ws_url = f"{WEBSOCKET_BASE_URL}/media"
    print(f"   - Instructing Piopiy to connect to: {full_ws_url}")

    action.stream(full_ws_url, {"listen_mode": "caller"})
    return jsonify(action.PCMO())


@sock.route('/media')
def media(ws):
    """
    This WebSocket endpoint handles real-time audio for a single call.
    """
    call_sid = request.args.get('callsid', f"call_{int(time.time())}")  # Generate fallback call_sid
    stream_sid = request.args.get('streamsid', f"stream_{int(time.time())}")
    print(f"✅ WebSocket connection accepted for Call SID: {call_sid}")

    va = None
    try:
        # Create a dedicated VoiceAssistant instance for this specific call.
        va = VoiceAssistant(
            stt_model_name=app.config["STT_MODEL_NAME"],
            silence_timeout=app.config["VAD_SILENCE_TIMEOUT"],
            google_tts_key=app.config["GOOGLE_TTS_KEY"],
            dll_path=app.config["DLL_PATH"],
            rms_threshold=0.05
        )
        va.start()
        active_calls[call_sid] = va
    except Exception as e:
        print(f"CRITICAL: Failed to initialize Voice Assistant for {call_sid}: {e}")
        ws.close()
        return

    def send_audio_to_piopiy():
        """Thread to send synthesized audio from our TTS back to Piopiy."""
        while ws.connected and not va.shutdown_flag.is_set():
            try:
                base64_audio_from_tts = va.synthesized_audio_queue.get(timeout=0.1)
                if base64_audio_from_tts is None:  # Shutdown signal
                    break
                    
                audio_bytes_decoded = base64.b64decode(base64_audio_from_tts)
                audio_array = np.frombuffer(audio_bytes_decoded, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                piopiy_media_payload = convert_audio_to_piopiy(audio_float, sample_rate=24000)
                if piopiy_media_payload:
                    response_message = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": piopiy_media_payload}
                    }
                    ws.send(json.dumps(response_message))
            except Empty:
                continue
            except Exception as e:
                print(f"Error in send_audio thread for {call_sid}: {e}")
                break
        print(f"Sender thread for {call_sid} exiting.")

    sender_thread = threading.Thread(target=send_audio_to_piopiy)
    sender_thread.daemon = True
    sender_thread.start()

    try:
        while ws.connected and not va.shutdown_flag.is_set():
            try:
                message_str = ws.receive(timeout=1)
                if message_str is None:
                    continue
                    
                # Fix JSON parsing issue by handling potential multiple JSON objects
                messages = []
                decoder = json.JSONDecoder()
                idx = 0
                message_str = message_str.strip()
                
                while idx < len(message_str):
                    try:
                        obj, end_idx = decoder.raw_decode(message_str, idx)
                        messages.append(obj)
                        idx += end_idx
                        # Skip whitespace
                        while idx < len(message_str) and message_str[idx].isspace():
                            idx += 1
                    except json.JSONDecodeError:
                        break
                
                for message in messages:
                    event = message.get('event')
                    if event == 'media':
                        payload = message.get('media', {}).get('payload')
                        if payload:
                            audio_data = convert_piopiy_to_audio(payload)
                            if len(audio_data) > 0:
                                va.process_incoming_audio(audio_data)
                    elif event == 'start':
                        print(f"Piopiy media stream officially started for call: {call_sid}")
                    elif event == 'stop':
                        print(f"Piopiy media stream stopped for call: {call_sid}")
                        break
                        
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {call_sid}: {e}")
                # Continue processing, don't break on JSON errors
                continue
                
    except Exception as e:
        print(f"WebSocket error for {call_sid}: {e}")
    finally:
        print(f"WebSocket connection closing for {call_sid}.")
        if call_sid in active_calls:
            va_instance = active_calls.pop(call_sid)
            va_instance.shutdown()
            print(f"Cleaned up resources for Call SID: {call_sid}. Active calls remaining: {len(active_calls)}")


if __name__ == "__main__":
    # --- HARDCODED CONFIGURATION ---
    # Replace the placeholder values below with your actual keys and paths.

    app.config["STT_MODEL_NAME"] = "large-v3"
    app.config["VAD_SILENCE_TIMEOUT"] = 0.7

    # --- PASTE YOUR KEYS AND PATHS HERE ---
    app.config["GOOGLE_TTS_KEY"] = "AIzaSyDUcDLaA3f_ZziysIjGxWUSVaoJB2J7fZM"  # <--- PASTE YOUR GOOGLE API KEY

    # IMPORTANT: Use double backslashes (\\) or a raw string (r"...") for Windows paths.
    app.config[
        "DLL_PATH"] = r"C:\Users\Pc\PycharmProjects\PythonProject5\voice_assistant_twilio_integration\librnnoise-0.dll"  # <--- PASTE FULL DLL PATH

    print("Configuration has been hardcoded.")
    print(f"  - STT Model: {app.config['STT_MODEL_NAME']}")
    print(f"  - VAD Timeout: {app.config['VAD_SILENCE_TIMEOUT']}s")
    if not os.path.exists(app.config["DLL_PATH"]):
        print(f"WARNING: DLL file not found at hardcoded path: {app.config['DLL_PATH']}")

    print("\nStarting Flask server...")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)