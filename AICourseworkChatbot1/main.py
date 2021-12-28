# ryder franklin
# chatbot 1 - supernatural

# imports
import datetime
import time
import re
from math import log

import nltk
from nltk.sem import Expression
from nltk.inference import ResolutionProver

if nltk.download('punkt'):
    print("nltk - punkt installed")
if nltk.download('averaged_perception_tagger'):
    print("nltk - averaged_perception_tagger installed")
if nltk.download('brown'):
    print("nltk - brown installed")

import aiml
import wikipedia
import pyjokes
import pyttsx3
import speech_recognition as sr
import fandom
from simpful import *
import pandas as pd


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
data = pd.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
'''
from nltk.chat.util import Chat, reflections
pairs = (
(r'I need (.*)',
( "Why do you need %1?",
"Would it really help you to get %1?",
"Are you sure you need %1?")),
(r'Why don\'t you (.*)',
( "Do you really think I don't %1?",
"Perhaps eventually I will %1.",
"Do you really want me to %1?")),
...
eliza_chatbot = Chat(pairs, reflections)
eliza_chatbot.converse() '''


def speak(audio):
    print(audio)
    engine.say(audio)
    engine.runAndWait()


def greetUser():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good morning! ")
    elif 12 <= hour < 18:
        speak("Good afternoon!")
    else:
        speak("Good evening!")

    AIName = "Dave version 1 point 0 "
    speak("I am the assistant")
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
        print("Unable to recognise, sorry!")
        return "none"
    return query


def checkSimilarWords(query):
    queryTokens = nltk.word_tokenize(query)
    tagged: list = nltk.pos_tag(queryTokens)

    '''text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
    for x in range(len(tagged)):
        print(text.similar(tagged[x]))'''
    return


def calculateIDF(itemDict, item, dataLine):
    # calculate number of times the same word shows up in the csv file
    dataCounter: int = 0
    for y in dataLine.split(","):
        if itemDict[item][1] in y:
            dataCounter += 1
            print("dataLine: ", dataLine, "itemDict[item][1]: ", itemDict[item][1])
    if dataCounter >= 1:
        knowledgeIDF = log(len(dataLine) / dataCounter)
        return knowledgeIDF
    else:
        return 0


def calculateTFIDFofInput(query):
    # calculate number of times a word shows up in userinput
    # calculate number of times the same word shows up in the csv file
    # num1 * num2 = tfIDF value
    queryTokens = nltk.word_tokenize(query)
    itemDict: dict = {}
    knowledgeIDF: dict = {}
    tfidf: dict = {}
    print(queryTokens)
    for item in queryTokens:
        itemCounter: int = 0
        for countItem in queryTokens:
            if countItem == item:
                itemCounter += 1
        itemDict[item] = (itemCounter, item)
        queryTF = itemCounter / len(query)
        dataset = pd.read_csv('knowledge.csv')
        for dataLine in dataset:
            knowledgeIDF[item] = (calculateIDF(itemDict, item, dataLine))
        tfidf[item] = (queryTF * knowledgeIDF[item])
        if tfidf[item] > 0:
            print("tfidf of item (", item, ") over 0: ", tfidf[item])
    return tfidf


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
                    query: str = input("> ").lower()
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
            checkSimilarWords(query)
            tfidf = calculateTFIDFofInput(query)
            # if tfidf > 0:

            if answer[0] == '#':
                params: list[str] = answer[1:].split('$')
                command: int = int(params[0])
                if command == 0:
                    speak(params[1])
                    break
                elif command == 2:
                    # tells a joke
                    speak(pyjokes.get_joke())
                elif command == 3:
                    # fandom wikipedia api
                    try:
                        query = query.strip("check wiki ")
                        print(query)
                        checkPage: list = fandom.search(query, results=1)
                        currentPage = fandom.page(pageid=checkPage[0][1])
                        plain_text = str(currentPage.plain_text)

                        plain_text = plain_text.strip(currentPage.title)
                        expr = re.compile(r'\n.\n(.*?)\n.\n(.*)\n')
                        # strips title and quotes from plain text so only sentences remain
                        plain_text = expr.sub('', plain_text)

                        totalLine: str = ""
                        sentenceCounter: int = 0
                        for index, line in enumerate(plain_text):
                            if sentenceCounter == 3:
                                break
                            totalLine += line
                            if len(totalLine) > 4:
                                if totalLine[-1] == ".":
                                    print(totalLine)
                                    if totalLine[0].isupper() and totalLine[-2] != "(b.":
                                        sentenceCounter += 1
                                        print(totalLine)
                                        totalLine = ""
                                    else:
                                        continue

                    except fandom.error.PageError:
                        print("Could not find a matching page!")

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
