import os
import string


def LoadDataFromFile(path):
    fp = open(path, 'r')
    Docs = []
    for line in fp:
        ll = line.strip('\n').strip('\r')
        Docs.append(ll)
    fp.close()
    print "Done, load ", len(Docs), " docs from the file"
    return Docs


def LoadStopWords():
    path = os.getcwd()
    path += "/stopwords"
    fp = open(path, 'r')
    StopWordsList = [line.strip('\n') for line in fp]
    fp.close()
    return StopWordsList


def LoadDictionary():
    path = os.getcwd() + "/dictionary.txt"
    fp = open(path, 'r')
    Dictionary = dict()
    for line in fp:
        elements = line.strip('\n').split(" ")
        k = string.atoi(elements[0])
        v = elements[1]
        Dictionary[k] = v
    fp.close()
    return Dictionary
