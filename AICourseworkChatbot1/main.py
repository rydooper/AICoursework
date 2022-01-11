# ryder franklin
# chatbot 1 - supernatural

# imports
import datetime
import math
import random
import re
from csv import reader

import nltk
import numpy as np
from nltk import ResolutionProverCommand
from nltk.inference import ResolutionProver
from nltk.sem import Expression
from sklearn.metrics.pairwise import cosine_similarity

if nltk.download('punkt'):
    print("nltk - punkt installed")
if nltk.download('averaged_perception_tagger'):
    print("nltk - averaged_perception_tagger installed")
if nltk.download('brown'):
    print("nltk - brown installed")

import aiml
import pyjokes
import pyttsx3
import speech_recognition as sr
import fandom
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
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="spnChatbot1-aiml.xml")

# setup knowledge dataset
read_expr = Expression.fromstring
kb: list = []
kbData = pd.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in kbData[0]]
for knowledge in kb:
    if not ResolutionProver().prove(knowledge, kb):
        print("Error in kb!")


def speak(audio):
    """
    Speaks the given input audio.

    :param audio:
    :return: nothing
    """
    print(audio)
    engine.say(audio)
    engine.runAndWait()


def greetUser():
    """
    Greets the user based on the hour and speaks the AI name.

    :return: nothing
    """
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good morning! ")
    elif 12 <= hour < 18:
        speak("Good afternoon!")
    else:
        speak("Good evening!")

    AIName = "Dave version 1 point 1"
    speak("I am the assistant")
    speak(AIName)


def takeCommand():
    """
    Takes input from the users microphone and interprets using google translates API.

    :return: query or "none"
    """
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


def calculateCosineSimilarity(tfidfA, tfidfB):
    """
    Calculates the cosine similarity of query and knowledge bag.
    Returns nothing.

    :param tfidfB:
    :param tfidfA:
    :return: nothing
    """
    tempA = list(tfidfA.values())
    bagA = np.array(tempA)

    tempB = list(tfidfB.values())
    bagB = np.array(tempB)

    # calculate cosine similarity of tfidfs
    cosineSim = cosine_similarity(bagA.reshape(1, -1), bagB.reshape(1, -1))
    cos = float(cosineSim)
    return cos


def computeTF(wordDict, bagWords):
    """
    Calculates the TF.

    :param wordDict:
    :param bagWords:
    :return: tfDict.
    """
    tfDict = {}
    bagWordsCount = len(bagWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagWordsCount)
    return tfDict


def computeIDF(documents):
    """
    Calculates the IDF.

    :param documents:
    :return: idfDict.
    """
    N = len(documents[0])
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def computeTFIDF(tfBag, idfs):
    """
    Calculates the TFIDF.

    :param tfBag:
    :param idfs:
    :return: tfidf.
    """
    tfidf = {}
    for word, val in tfBag.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def tfidfAndCosineCheck(query):
    """
    Calculates tfidf and calls cosine similarity.
    Adapted from this github: https://github.com/mayank408/TFIDF/blob/master/TFIDF.ipynb

    :param query:
    :return: cos.
    """
    if "check wiki" in query:
        query = query.strip("check wiki ")

    with open("knowledge.csv", mode='r') as csv_file:
        allData = reader(csv_file)
        listData = []
        for lines in allData:
            listData += lines

    listData.remove('question')
    listData.remove('answer')
    knowledge = listData
    cos = []
    for index, cells in enumerate(listData):
        bagA = query.split(' ')
        bagB = knowledge[index].split(' ')
        uniqueWords = set(bagA).union(set(bagB))

        numWordsA = dict.fromkeys(uniqueWords, 0)
        for word in bagA:
            numWordsA[word] += 1

        numWordsB = dict.fromkeys(uniqueWords, 0)
        for word in bagB:
            numWordsB[word] += 1

        tfA = computeTF(numWordsA, bagA)
        tfB = computeTF(numWordsB, bagB)

        idfs = computeIDF([numWordsA, numWordsB])

        tfidfA = computeTFIDF(tfA, idfs)
        tfidfB = computeTFIDF(tfB, idfs)

        cosine = calculateCosineSimilarity(tfidfA, tfidfB)
        cos.append([float(cosine), listData[index]])
    return cos


def wikiCheck(query):
    """
    INCOMPLETE
    Checks wiki for the query and if found, outputs the first three sentences.
    :param query:
    :return: nothing.
    """
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

    return


# main
def main():
    """
    Main.

    :return: nothing.
    """

    apologyText: str = "Try again, I'll do better next time! "
    takingQueries: bool = True

    greetUser()
    speak("Welcome to the spn chat bot!")
    speak("Select which type of input you would like to use: [1] typing [2] voice \n")
    inputType: str = input("> ")
    if inputType == "1" or inputType == "2":
        speak("What would you like to ask me? ")
        while takingQueries:
            if inputType == "1":
                try:
                    query: str = input("> ").lower()
                except (KeyboardInterrupt, EOFError) as e:
                    textToSpeak = "Keyboard input error,", e, " shutting down!"
                    speak(textToSpeak)
                    break
            elif inputType == "2":
                query: str = takeCommand().lower()
            else:
                speak("Invalid input detected, exiting!")
                break
            responseAgent = 'aiml'
            answer: str = kern.respond(query)

            if answer[0] == '#':
                params: list[str] = answer[1:].split("$")
                command = int(params[0])

                if command == 0:
                    speak(params[1])
                    break

                elif command == 2:
                    # tells a joke from python jokes library
                    speak(pyjokes.get_joke())

                elif command == 3:
                    # fandom wikipedia api function called
                    wikiCheck(query)

                elif command == 31:
                    # LOGIC - "I know that * is *" statements
                    item, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + item + ')')
                    # >>> ADD SOME CODES HERE to make sure expr does not contradict
                    # with the KB before appending, otherwise show an error message.
                    if expr in kb:
                        print("already found in kb!")

                    contradicts = ResolutionProver().prove(expr, kb)
                    if contradicts:
                        print("Cannot add a contradicting statement.")
                    else:
                        kb.append(expr)
                        print('OK, I will remember that', item, 'is', subject)

                elif command == 32:
                    # LOGIC - "check that * is *" - statements
                    item, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + item + ')')
                    response = ResolutionProver().prove(expr, kb)
                    print(item, subject, expr, response)
                    print(kb)
                    if response:
                        print("You're correct!")
                    else:
                        print("First proven false.")
                        inverseAnswer = ResolutionProver().prove((-expr), kb)
                        # inverseAnswer = False
                        if inverseAnswer:
                            print("You're incorrect.")
                        else:
                            print("I am unclear if this is true or false. Try searching for an answer elsewhere.")

                elif command == 98:
                    # closes the program down
                    speak("closing program down now")
                    takingQueries = False

                elif command == 99:
                    # given if an error occurs
                    speak("Sorry, I didn't understand that one!")
                    speak(apologyText)

            else:
                # checks the knowledge.csv file and compares against knowledge given using tfidf and cosine
                cosineList = tfidfAndCosineCheck(query)
                for count, item in enumerate(cosineList):
                    if item[0] > 0.5:
                        QAnswer = cosineList[count + 1][1]
                        if "[" in QAnswer or "]" in QAnswer:
                            answerList = QAnswer.split(";")
                            randomChoice = str(random.choice(answerList))
                            QAnswer = randomChoice.strip("[]")
                        textToSpeak = "The answer is " + QAnswer
                        speak(textToSpeak)
    else:
        speak("This is not a valid response. Closing program.")


main()
