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