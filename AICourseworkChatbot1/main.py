# ryder franklin
# chatbot 1 - supernatural

# imports
import datetime
import time
import nltk
from nltk.sem import Expression
from nltk.inference import ResolutionProver

try:
    if nltk.download('punkt'):
        print("punkt installed")
    if nltk.download('averaged_perception_tagger'):
        print("averaged_perception_tagger installed")
except:
    print("Error in package download!")

import aiml
import wikipedia
import pyjokes
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
import fandom

import pandas


# classes

class chatbotAI:
    def __init__(self, name):
        self.name = name


# setup chat bot

spnChatbot = chatbotAI("Dave")
engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)
fandom.set_wiki("supernatural")

# setup knowledge dataset
read_expr = Expression.fromstring
kb: list = []


# data = pandas.read_csv('knowledge.csv', header=None)
# [kb.append(read_expr(row)) for row in data[0]]


def speak(audio):
    print(audio)
    engine.say(audio)
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
    mic = sr.Microphone(device_index=2)  # 3 for my home pc  # 2 for my laptop
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


def checkKnowledge(query):
    queryTokens = nltk.word_tokenize(query)
    tagged: list = nltk.pos_tag(queryTokens)
    print(tagged)
    nouns: list = []
    for x in range(len(queryTokens)):
        if tagged[x][1] == 'NN':
            nouns.append(queryTokens[x])
            print("Noun found:", queryTokens[x])
    return nouns


def calculateTFIDF(query):
    # calculate number of times a word shows up in userinput
    # calculate number of times the same word shows up in the csv file
    # num1 * num2 = tfIDF value

    queryTokens = nltk.word_tokenize(query)
    # for item in range(len(queryTokens)):

    return 0


# main

def main():
    kern = aiml.Kernel()
    kern.setTextEncoding(None)
    kern.bootstrap(learnFiles="spnChatbot1-aiml.xml")
    apologyText: str = "Try again, I'll do better next time! "

    greetUser()
    speak("Welcome to the spn chat bot!")
    takingQueries: bool = True
    print("Select which type of input you would like to use: [1] typing [2] voice \n")
    inputType: str = input("> ")
    if inputType == "1" or inputType == "2":
        speak("What would you like to ask me? ")
        while takingQueries:
            if inputType == "1":
                try:
                    query: str = input("> ")
                except (KeyboardInterrupt, EOFError) as e:
                    print("Keyboard input error,", e, " shutting down!")
                    break
            elif inputType == "2":
                query: str = takeCommand().lower()
            else:
                speak("Invalid input detected, exiting!")
                break
            responseAgent = 'aiml'
            answer: str = kern.respond(query)
            nouns: list = checkKnowledge(query)
            if not nouns:
                print("no nouns received!")
            if answer[0] == '#':
                params: list[str] = answer[1:].split('$')
                command: int = int(params[0])
                if command == 0:
                    speak(params[1])
                    break
                elif command == 1:
                    try:
                        # wikipedia facts here
                        wikiResults: str = wikipedia.summary(params[1], sentences=2)
                        time.sleep(5)
                        speak(wikiResults)
                    except:
                        speak(
                            "Sorry, I don't know that one! There's a lot of supernatural to get through, "
                            "you understand?")
                        speak(apologyText)
                elif command == 2:
                    # tells a joke
                    speak(pyjokes.get_joke())
                elif command == 3:
                    # finish  # what was my command here again?
                    # fandom api
                    try:
                        print(query)
                        query = query.strip("check wiki ")
                        print(query)
                        checkPage: list = fandom.search(query, results=1)
                        page = fandom.page(pageid=checkPage[0][1])
                        speak(page.summary)
                    except (fandom.error.PageError):
                        print("Could not find a matching page!")
                    print("nice")

                    # Here are the processing of the new logical component:
                elif command == 31:  # if input pattern is "I know that * is *"
                    object, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + object + ')')
                    # >>> ADD SOME CODES HERE to make sure expr does not contradict
                    # with the KB before appending, otherwise show an error message.
                    kb.append(expr)
                    print('OK, I will remember that', object, 'is', subject)
                elif command == 32:  # if the input pattern is "check that * is *"
                    object, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + object + ')')
                    answer = ResolutionProver().prove(expr, kb, verbose=True)
                    if answer:
                        print('Correct.')
                    else:
                        print('It may not be true.')
                        # >> This is not an ideal answer.
                        # >> ADD SOME CODES HERE to find if expr is false, then give a
                        # definite response: either "Incorrect" or "Sorry I don't know."

                elif command == 98:
                    speak("closing program down now")
                    takingQueries = False
                elif command == 99:
                    speak("Sorry, I didn't understand that one!")
                    speak(apologyText)
            else:
                speak(answer)
    else:
        speak("This is not a valid response. Closing program.")


main()
