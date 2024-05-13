from openai import OpenAI
import os

### Logging ###
import logging as log

class openai_chat:

    def __init__(self, openai_key, model, temperature=0.0):
        log.info(f"CLASS:openai_chat initialized with model: {model}")
        try:
            self.model_chat = model
            self.temperature = temperature
            self.messages = []
            self.openai_client = OpenAI(api_key=openai_key)
        except Exception as e: 
            self.openai_client = None
            log.error(f"Failed to initialize OpenAI client: {e}")
            print(f"Failed to initialize OpenAI client: Check log file for details.")


    def add_message(self, role, content):
        json_message = {
            "role": role, 
            "content": content
        }
        self.messages.append(json_message)


    def execute(self): 
        try:
            response = self.openai_client.chat.completions.create(
            model=self.model_chat,
            messages=self.messages,
            temperature=self.temperature,
        )
        except Exception as e:
            log.error(f"Error in openai_chat.execute: {e}")
            print(e)
        else:
            # self.add_message("assistant", response.choices[0].message.content)
            log.info(f"OpenAI response received and added to messages.")
            return response.choices[0].message.content


    def execute_stream(self):
        full_text = ""
        stream_response = []

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_chat,
                messages=self.messages,
                temperature=self.temperature,
                stream=True
            )
        except Exception as e:
            log.error(f"Error in openai_chat.execute_stream: {e}")
            print(e)
        else:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    stream_response.append(chunk_content)
                    for char in chunk_content:
                        print(char, end='', flush=True)
                        full_text += char
                else:
                    # End of stream detected
                    print("\n")
                    break
        
        log.info(f"OpenAI stream has completed and added to messages.")
        self.add_message("assistant", full_text)
        return full_text


    def display(self, response):
        print("-" * 67)
        print(response)
        print("-" * 67)


class openai_audio:

    def __init__(self, openai_key, model, temperature=0.0):
        log.info(f"CLASS:openai_chat initialized with model: {model}")
        try:
            self.model_audio = model
            self.temperature = temperature
            self.messages = []
            self.openai_client = OpenAI(api_key=openai_key)
        except Exception as e: 
            self.openai_client = None
            log.error(f"Failed to initialize OpenAI client: {e}")
            print(f"Failed to initialize OpenAI client: Check log file for details.")


    def speech_to_text(self, m4a_file_path):
        try:
            audio_prompt = "Transcribe the following audio file:"
            
            audio_file= open(m4a_file_path, "rb")
            transcript = self.openai_client.audio.transcriptions.create(
            model=self.model_audio, 
            file=audio_file,
            temperature=self.temperature,
            prompt=audio_prompt,
            response_format="text",
            language="en",
            )
        except Exception as e:
            log.error(f"Error in openai_chat.audio_to_text: {e}")
            print(e)
        else:
            # code to execute if no exception was raised
            return transcript

    def text_to_speech(self, text, file_name, response_format="aac", voice="alloy"):
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                input=text,
                voice=voice,
                response_format=response_format
            )
            response.stream_to_file(file_name)

        except Exception as e:
            log.error(f"Error in openai_chat.text_to_audio: {e}")
            print(e)
        else:
            # code to execute if no exception was raised
            return response

    def play_audio(self, audio_file):
        try:
            import pygame

            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # Keep the script running until the music stops
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        except Exception as e:
            log.error(f"Error in openai_chat.play_audio: {e}")
            print(e)
        else:
            # code to execute if no exception was raised
            return True

    def create_prompt_for_youtube_video(self, transcript):
        prompt = f"""
        You are an expert at creative summaries of videos.
        Your task is to generate a summary of the video transcript for youtube.
        The goal is to have the summary be engaging and informative and to make people want to watch the video.

        Summarize the video transcript in 1000 words and use AI and ChatGPT as much as it makes sense, 
        but make sure it is still readable by humans.
        also use first person pronouns like "I" and "me" to make it more personal.
        Transcript enclosed in triple backticks. 

        Transcript: ```{transcript}```
        """
        return prompt 





class openai_embeddings:

    def __init__(self, openai_key, model):
        self.model_embedding = model
        self.messages = []
        try:
            self.openai_client = OpenAI(api_key=openai_key)
            log.info(f"CLASS:openai_embeddings initialized with model: {model}")
        except Exception as e: 
            self.openai_client = None
            log.error(f"Failed to initialize OpenAI client: {e}")
            print(f"Failed to initialize OpenAI client: Check log file for details.")    


    def execute(self, text):
        try:
            embedding = self.openai_client.embeddings.create(
                input = text, 
                model=self.model_embedding
            )
        except Exception as e:
            log.error(f"Error in openai_embeddings.execute: {e}")
            print(e)
        else:
            return embedding

    
    def display(self, response):
        print("-" * 67)
        print("Dimension of text embedding: ", len(response.data[0].embedding))
        print("-" * 67)
        print(response.data[0].embedding)
        print("-" * 67)