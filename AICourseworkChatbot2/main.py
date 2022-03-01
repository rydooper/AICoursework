# ryder franklin
# chatbot 2 - supernatural

# imports
import datetime
import math
import random
import re
from csv import reader
import numpy as np
from nltk.inference import ResolutionProver
from nltk.sem import Expression
from sklearn.metrics.pairwise import cosine_similarity
import aiml
import pyjokes
import pyttsx3
import speech_recognition as sr
import fandom
import pandas as pd
import simpful as sf


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
kern.bootstrap(learnFiles="spnChatbot2-aiml.xml")

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

    AIName = "Dave version 2 point 0"
    speak("I am the assistant")
    speak(AIName)


def getMicrophone():
    """
    Takes input from user to get microphone.

    :return: mic
    """

    r = sr.Recognizer()
    print(sr.Microphone.list_microphone_names())
    print("Starting from 0 as the first microphone, input the number of the microphone you wish to use:")
    chooseMic = int(input("> "))
    mic = sr.Microphone(device_index=chooseMic)  # 3 for my home pc  # 2 for my laptop
    return mic


def takeCommand(mic):
    """
    Takes input from the users microphone and interprets using google translates API.

    :return: query or "none"
    """
    r = sr.Recognizer()

    with mic as source:
        print("listening!")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("recognising!")
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

    # calculate cosine similarity of TFIDFs
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


def computeTFIDF(tfBag, IDFs):
    """
    Calculates the TFIDF.

    :param tfBag:
    :param IDFs:
    :return: tfidf.
    """
    tfidf = {}
    for word, val in tfBag.items():
        tfidf[word] = val * IDFs[word]
    return tfidf


def tfidfAndCosineCheck(query):
    """
    Calculates tfidf and calls cosine similarity.
    Adapted from this github: https://github.com/mayank408/TFIDF/blob/master/TFIDF.ipynb

    :param query:
    :return: cos.
    """

    with open("knowledge.csv", mode='r') as csv_file:
        allData = reader(csv_file)
        listData = []
        for lines in allData:
            listData += lines

    listData.remove('question')
    listData.remove('answer')
    allKnowledge = listData
    cos = []
    for index, cells in enumerate(listData):
        bagA = query.split(' ')
        bagB = allKnowledge[index].split(' ')
        uniqueWords = set(bagA).union(set(bagB))

        numWordsA = dict.fromkeys(uniqueWords, 0)
        for word in bagA:
            numWordsA[word] += 1

        numWordsB = dict.fromkeys(uniqueWords, 0)
        for word in bagB:
            numWordsB[word] += 1

        tfA = computeTF(numWordsA, bagA)
        tfB = computeTF(numWordsB, bagB)

        IDFs = computeIDF([numWordsA, numWordsB])

        tfidfA = computeTFIDF(tfA, IDFs)
        tfidfB = computeTFIDF(tfB, IDFs)

        cosine = calculateCosineSimilarity(tfidfA, tfidfB)
        cos.append([float(cosine), listData[index]])
    return cos


def wikiCheck(query):
    """
    Checks wiki for the query and if found, outputs the first three sentences.
    :param query:
    :return: nothing.
    """
    try:
        query = query.strip("check wiki ")
        query = query.strip("get summary ")
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
            # prints 3 sentence summary of page
            if sentenceCounter == 3:
                break
            totalLine += line
            if len(totalLine) > 4:
                if totalLine[-1] == ".":
                    if totalLine[-2] != "b" and totalLine[-2] != "d":
                        sentenceCounter += 1
                        speak(totalLine)
                        totalLine = ""
                else:
                    continue

    except Exception:
        speak("I couldn't find a matching page on the Supernatural wiki! Try this link and searching it manually: ")
        print("https://supernatural.fandom.com/wiki/Supernatural_Wiki")
    return


def fuzzyRating(inputType, mic):
    """
    Rates the given episode based upon user input using fuzzy logic.

    :return: nothing
    """
    episodeName = ""
    actingRating = ""
    plotRating = ""

    FS = sf.FuzzySystem(show_banner=False)
    TLV = sf.AutoTriangle(3, terms=['poor', 'average', 'good'], universe_of_discourse=[0, 10])
    FS.add_linguistic_variable("acting", TLV)
    FS.add_linguistic_variable("plot", TLV)

    lowRating = sf.TriangleFuzzySet(0, 0, 13, term="low")
    mediumRating = sf.TriangleFuzzySet(0, 13, 25, term="medium")
    highRating = sf.TriangleFuzzySet(13, 25, 25, term="high")
    FS.add_linguistic_variable("rating", sf.LinguisticVariable([lowRating, mediumRating, highRating],
                                                               universe_of_discourse=[0, 25]))

    FS.add_rules([
        "IF (acting IS poor) OR (plot IS poor) THEN (rating IS low)",
        "IF (acting IS average) THEN (rating IS medium)",
        "IF (acting IS good) OR (plot IS good) THEN (rating IS high)"
    ])
    speak("Input the name of the episode you wish to rate: ")
    if inputType == "1":
        episodeName = input("> ")
    elif inputType == "2":
        episodeName: str = takeCommand(mic).lower()
    speak("Input how you would rate the acting (from 0-25): ")
    if inputType == "1":
        actingRating = input("> ")
    elif inputType == "2":
        actingRating: str = takeCommand(mic).lower()
    speak("Input how would rate the plot (from 0-25): ")
    if inputType == "1":
        plotRating = input("> ")
    elif inputType == "2":
        plotRating: str = takeCommand(mic).lower()

    FS.set_variable("acting", actingRating)
    FS.set_variable("plot", plotRating)

    episodeFuzzRating = str(FS.inference())
    episode = [episodeName, actingRating, plotRating, episodeFuzzRating]

    numRating = re.findall('[0-9]+', episodeFuzzRating)
    textToSpeak = "Given your input, the episode " + episodeName + " was rated at " + str(numRating[0])
    speak(textToSpeak)
    return


# main
def main():
    """
    Main.

    :return: nothing.
    """
    takingQueries: bool = True

    mic = 0
    greetUser()
    speak("Welcome to the Supernatural chat bot!")
    speak("Select which type of input you would like to use: [1] typing [2] voice \n")
    inputType: str = input("> ")
    if inputType == "1" or inputType == "2":
        if inputType == "2":
            mic = getMicrophone()
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
                query: str = takeCommand(mic).lower()
            else:
                speak("Invalid input detected, exiting!")
                break
            responseAgent = 'aiml'
            answer: str = kern.respond(query)
            answered = False

            if answer[0] != "#":
                speak(answer)

            if answer[0] == "#":
                params: list[str] = answer[1:].split("$")
                command = int(params[0])

                if command == 0:
                    speak(params[1])
                    break

                elif command == 2:
                    # tells a joke from python jokes library
                    speak(pyjokes.get_joke())

                elif command == 3:
                    # fandom wikipedia api function
                    wikiCheck(query)

                elif command == 31:
                    # LOGIC - "I know that * is *" statements
                    item, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + item + ')')
                    contradicts = ResolutionProver().prove((-expr), kb)
                    if expr in kb:
                        speak("This fact is already within my knowledge set! Try something else.")
                    elif contradicts == True:
                        speak("I cannot add a contradicting statement!")
                    else:
                        kb.append(expr)
                        textToSpeak = "OK, I will remember that" + item + " is " + subject
                        speak(textToSpeak)

                elif command == 32:
                    # LOGIC - "check that * is *" - statements
                    item, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + item + ')')
                    response = ResolutionProver().prove(expr, kb)

                    if response:
                        speak("You're correct!")
                        textToSpeak = str(item + " is " + subject)
                        speak(textToSpeak)

                    else:
                        inverseAnswer = ResolutionProver().prove((-expr), kb)
                        if inverseAnswer:
                            speak("You're incorrect.")
                            textToSpeak = str(item + " is not " + subject)
                            speak(textToSpeak)
                        else:
                            speak("I am unclear if this is true or false. Try searching for an answer elsewhere.")

                elif command == 33:
                    # LOGIC - fuzzy logic to rate specific episodes of supernatural
                    fuzzyRating(inputType, mic)

                elif command == 98:
                    # closes the program down
                    speak("Closing program down now, goodbye!")
                    takingQueries = False

                elif command == 99:
                    # checks the knowledge.csv file and compares against knowledge given using tfidf and cosine
                    cosineList = tfidfAndCosineCheck(query)
                    maxItem = max(cosineList)
                    for count, item in enumerate(cosineList):
                        if maxItem[0] == 0.0:
                            break
                        if item == maxItem:
                            QAnswer = cosineList[count + 1][1]
                            if "[" in QAnswer or "]" in QAnswer:
                                # randomly picks an item if the answer is a list
                                answerList = QAnswer.split(";")
                                randomChoice = str(random.choice(answerList))
                                QAnswer = randomChoice.strip("[]")
                            textToSpeak = "The answer is " + QAnswer
                            speak(textToSpeak)
                            answered = True
                            break
                    if not answered:
                        # only occurs if no cosine value is over 0.5 (therefore no answer is in the knowledge.csv file)
                        speak("I'm afraid I can't answer that, try asking another chatbot instead!")

    else:
        # occurs if the user enters an invalid input type
        speak("This is not a valid response. Closing program.")


main()
