import threading
import time
import pyttsx3
from gtts import gTTS
from playsound import playsound
import os
import pygame

class speaker():
    def __init__(self,):
        self.object = None
        self.direction = None
        self.distance = None
    
    def update(self, o, d):
        self.object = o
        self.direction = d
    
    def speak(self,):
        tts = gTTS(text=f"{self.object} on the {self.direction}", lang='en')
        audio_file = "temp.mp3"
        tts.save(audio_file)
        pygame.mixer.init()
        pygame.mixer.music.load("temp.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()

alarm = speaker()

thread = None

i = 0
j = 0
objects = ["truck", "car", "human", "truck"]
directions = ["front", "left", "right", "front"]

while j < len(objects):  # Run for the number of objects
    i += 1
    if thread is not None:
        time.sleep(1)
        print(i, thread.is_alive())
    # Check if the previous thread is finished before creating a new one
    if thread is None or not thread.is_alive():
        print(f"Alarm {i}: {objects[j]} on the {directions[j]}")
        if thread is not None:
            thread.join()
            print("JOINED")
        
        # Update alarm data
        alarm.update(objects[j], directions[j])
        print("UPDATED")
        
        # Start a new thread
        thread = threading.Thread(target=alarm.speak)
        thread.start()
        print("STARTING")
        
        j += 1  # Move to the next object