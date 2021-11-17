# ryder franklin
# chatbot 1 - supernatural

# imports
import datetime
import time

import ctypes
import subprocess
import time
import wolframalpha
import tkinter
import random
import datetime
import wikipedia
import pyjokes
import pyttsx3
import operator
import webbrowser
import requests
from bs4 import BeautifulSoup
import speech_recognition as sr
import json



# classes
class chatbotAI:
    def __init__(self, name):
        self.name = name


spnChatbot = chatbotAI("Chatty")


# functions

def greetUser():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        print("good morning! ")
    elif 12 <= hour < 18:
        print("good afternoon! ")
    else:
        print("good evening! ")

    print("I am a chatbot with knowledge about the tv show supernatural")

def main():
    greetUser()
    time.sleep(60)
    print("welcome! ")
    userInput = input("how are you? ")

main()