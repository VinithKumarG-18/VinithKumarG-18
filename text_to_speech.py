import requests
import base64
import json
import time
from queue import Empty


class TextToSpeech:
    def __init__(self, llm_response_queue, synthesized_audio_queue, google_tts_key):
        self.llm_response_queue = llm_response_queue
        self.synthesized_audio_queue = synthesized_audio_queue
        self.google_tts_key = google_tts_key
        
        # Send initial welcome message
        welcome_message = "Welcome to Titan Customer Care Support. May I know your name please?"
        try:
            self.llm_response_queue.put_nowait(welcome_message)
        except:
            print("Could not queue welcome message.")

    def speech_synthesize(self):
        """Convert text responses to speech using Google TTS."""
        print("Text-to-speech thread started.")
        
        while True:
            try:
                # Get LLM response with timeout
                llm_response = self.llm_response_queue.get(timeout=1.0)
                
                # Check for shutdown signal
                if llm_response is None:
                    print("Text-to-speech thread received shutdown signal.")
                    break
                
                # Validate response
                if not isinstance(llm_response, str) or not llm_response.strip():
                    print("Received empty LLM response, skipping TTS.")
                    continue
                
                print(f"TTS: {llm_response}")
                
                # Synthesize speech
                try:
                    start_time = time.time()
                    base64_audio = self.synthesize_speech(llm_response)
                    
                    if base64_audio:
                        latency = time.time() - start_time
                        print(f"Google TTS Latency: {latency:.2f} seconds")
                        print(f"Base64 Audio Output (start): {base64_audio[:100]} ...")
                        
                        # Put synthesized audio in queue
                        try:
                            self.synthesized_audio_queue.put_nowait(base64_audio)
                        except:
                            print("Synthesized audio queue is full, skipping audio.")
                    else:
                        print("Failed to synthesize speech.")
                
                except Exception as e:
                    print(f"Error in speech synthesis: {e}")
                    
            except Empty:
                # Timeout waiting for LLM response, continue
                continue
            except Exception as e:
                print(f"Error in text-to-speech thread: {e}")
                continue
        
        print("Text-to-speech thread ended.")

    def synthesize_speech(self, text):
        """Synthesize speech using Google Text-to-Speech API."""
        try:
            # Validate and clean text
            if not text or not text.strip():
                print("Empty text provided for TTS synthesis.")
                return None
            
            # Clean and truncate text if too long
            text = text.strip()
            if len(text) > 5000:  # Google TTS has character limits
                text = text[:5000]
            
            # Prepare the request payload
            payload = {
                "input": {"text": text},
                "voice": {
                    "languageCode": "en-US",
                    "name": "en-US-Standard-D",
                    "ssmlGender": "MALE"
                },
                "audioConfig": {
                    "audioEncoding": "LINEAR16",
                    "sampleRateHertz": 24000
                }
            }
            
            # Make the API request
            url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.google_tts_key}"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                response_data = response.json()
                audio_content = response_data.get("audioContent")
                if audio_content:
                    return audio_content
                else:
                    print("No audio content in TTS response.")
                    return None
            else:
                print(f"Error during TTS: {response.status_code} {response.reason}")
                if response.status_code == 400:
                    print("Bad Request - check your API key and request format")
                    print(f"Response: {response.text}")
                elif response.status_code == 403:
                    print("Forbidden - check your API key permissions")
                elif response.status_code == 429:
                    print("Rate limit exceeded - too many requests")
                return None
                
        except requests.exceptions.Timeout:
            print("TTS request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"TTS request error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"TTS JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected TTS error: {e}")
            return None