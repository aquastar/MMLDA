import random
import string
import nltk
from gensim import corpora


def PreprocessText(text, StopWordList):
    WordList = DelPunctuation(text)
    StemmeredWordList = Stemmer(WordList)
    FilteredWordList = FilterStopWords(StemmeredWordList, StopWordList)
    return FilteredWordList


def DelPunctuation(text):
    delset = string.punctuation
    newText = text.encode('utf8').translate(None, delset)
    WordList = [word for word in newText.split(" ") if word != '' and word != ' ']
    return WordList


def FilterStopWords(WordList, StopWordList):
    FilteredWordList = filter(lambda x: x.lower() not in StopWordList, WordList)
    return FilteredWordList


def Stemmer(WordList):
    stemmer = nltk.LancasterStemmer()
    StemmeredWordList = [stemmer.stem(w) for w in WordList]
    return StemmeredWordList


def ConstructDictionary(WordListSet):
    print "Begin to construct the dictionary"
    res = corpora.Dictionary(WordListSet)
    print "Total number of words is: ", len(res)
    return res


def Word2Id(WordList, Dictionary):
    IDList = []
    for word in WordList:
        print random.random()
        for k, v in Dictionary.items():
            if v == word:
                IDList.append(k)
    return IDList
