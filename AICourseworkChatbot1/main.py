# ryder franklin
# chatbot 1 - supernatural

# imports
import datetime
import time
import nltk
import aiml
import wikipedia
import pyjokes
import pyttsx3
import speech_recognition as sr
import json
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import fandom


# classes


class chatbotAI:
    def __init__(self, name):
        self.name = name


# setup chat bot

spnChatbot = chatbotAI("Dave")
engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)


def speak(audio):
    engine.say(audio)
    print(audio)
    engine.runAndWait()


def greetUser():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("good morning! ")
    elif 12 <= hour < 18:
        speak("good afternoon!")
    else:
        speak("good evening!")

    AIName = "dave version 1 point 0 "
    speak("i am the assistant")
    speak(AIName)


def takeCommand():
    r = sr.Recognizer()
    print(sr.Microphone.list_microphone_names())
    mic = sr.Microphone(device_index=3) # 3 for my home pc  # 2 for my laptop
    with mic as source:
        print("listening!")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("recognising")
        query = r.recognize_google(audio, language="en-UK")
        print("user said: ")
        print(query, "\n")
    except Exception as e:
        print(e)
        print("unable to recognise, sorry")
        return "none"
    return query


# main

def main():
    kern = aiml.Kernel()
    kern.setTextEncoding(None)

    apologyText = "Try again, I'll do better next time! "

    #kern.bootstrap(learnFiles="spnChatbot1-aiml.xml")
    greetUser()
    speak("Welcome to the spn chat bot!")
    takingQueries = True
    while takingQueries:
        print("Select which type of input you would like to use: [1] typing [2] voice \n")
        inputType: str = input("> ")
        if inputType == "1":
            speak("What would you like to ask me? ")
            query: str = input("> ")

        elif inputType == 2:
            speak("What would you like to ask me? ")
            query: str = takeCommand().lower()
            # responseAgent = 'aiml'
        answer = kern.respond(query)
        queryTokens = nltk.word_tokenize(query)

        d0 = query
        d1 = queryTokens[0][1]
        d2 = 'r2j'

        string = [d0, d1, d2]
        tfidf = TfidfVectorizer()
        result = tfidf.fit_transform(string)
        print('\nidf values:')
        for ele1, ele2 in zip(tfidf.get_feature_names(), tfidf.idf_):
            print(ele1, ':', ele2)

        print('\nWord indexes:')
        print(tfidf.vocabulary_)
        print('\ntf-idf value:')
        print(result)
        print('\ntf-idf values in matrix form:')
        print(result.toarray())
        '''
        if answer[0] == '#':
            params = answer[1:].split('$')
            command = int(params[0])
            if command == 0:
                speak(params[1])
                break
            elif command == 1:
                try:
                    # wikipedia facts here
                    wikiResults = wikipedia.summary(params[1], sentences=2)
                    time.sleep(5)
                    speak(wikiResults)
                except:
                    speak("Sorry, I don't know that one! There's a lot of supernatural to get through, you understand?")
                    speak(apologyText)
            elif command == 2:
                # tells a joke
                speak(pyjokes.get_joke())
            elif command == 3:
                # finish

            elif command == 98:
                speak("closing program down now")
                takingQueries = False
            elif command == 99:
                speak("Sorry, I didn't understand that one!")
                speak(apologyText)

        else:
            speak(answer) '''


main()
