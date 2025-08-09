import requests
from queue import Empty
import json


class AgentResponse:
    def __init__(self, llm_response_queue, transcript_queue):
        self.llm_response_queue = llm_response_queue
        self.transcript_queue = transcript_queue

    def process_and_respond(self):
        """Process transcripts and generate responses."""
        print("Agent response thread started.")
        
        while True:
            try:
                # Get transcript with timeout
                transcript = self.transcript_queue.get(timeout=1.0)
                
                # Check for shutdown signal
                if transcript is None:
                    print("Agent response thread received shutdown signal.")
                    break
                
                # Validate transcript
                if not isinstance(transcript, str) or not transcript.strip():
                    print("Received empty transcript, skipping API call.")
                    continue
                
                print(f"User said: {transcript}")
                
                # Generate response using a simple rule-based system for now
                # In a real implementation, you would call an LLM API here
                try:
                    response = self.generate_response(transcript)
                    
                    if response:
                        print(f"Agent response: {response}")
                        # Put response in LLM response queue
                        try:
                            self.llm_response_queue.put_nowait(response)
                        except:
                            print("LLM response queue is full, skipping response.")
                    else:
                        print("Generated empty response, skipping.")
                
                except Exception as e:
                    print(f"Error generating response: {e}")
                    # Generate fallback response
                    fallback_response = "I'm sorry, I didn't understand that. Could you please repeat?"
                    try:
                        self.llm_response_queue.put_nowait(fallback_response)
                    except:
                        pass
                        
            except Empty:
                # Timeout waiting for transcript, continue
                continue
            except Exception as e:
                print(f"Error in agent response thread: {e}")
                continue
        
        print("Agent response thread ended.")

    def generate_response(self, transcript):
        """Generate a response based on the transcript."""
        # Simple rule-based responses for demo
        transcript_lower = transcript.lower()
        
        if "hello" in transcript_lower or "hi" in transcript_lower:
            return "Hello! How can I help you today?"
        elif "name" in transcript_lower:
            return "I'm your virtual assistant. What's your name?"
        elif "help" in transcript_lower:
            return "I'm here to help! What do you need assistance with?"
        elif "thank" in transcript_lower:
            return "You're welcome! Is there anything else I can help you with?"
        elif "bye" in transcript_lower or "goodbye" in transcript_lower:
            return "Goodbye! Have a great day!"
        elif "weather" in transcript_lower:
            return "I don't have access to current weather information, but you can check a weather app for the latest forecast."
        elif "time" in transcript_lower:
            return "I don't have access to the current time, but you can check your device's clock."
        else:
            return "That's interesting! Tell me more about that."