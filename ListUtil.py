import string


def Normalize(list, smoother=0.0):
    sum = Sum(list)
    K = len(list)
    newlist = []
    if sum > 0:
        newlist = [float((item + smoother) / (sum + K * smoother)) for item in list]
    return newlist


def Sum(list):
    res = 0
    for item in list:
        res += item
    return res


def Initial(size, data=0):
    list = []
    for i in xrange(size):
        list.append(data)
    return list


def InitialMat(M, N, data=0):
    mat = []
    for i in xrange(M):
        row = Initial(N, data)
        mat.append(row)
    return mat


def InitialEmptyMat(rows):
    mat = []
    for i in xrange(rows):
        tmp = []
        mat.append(tmp)
    return mat


def toString(list):
    listStr = ""
    count = 0
    for ele in list:
        if type(ele) == int:
            eleStr = str(ele)
        elif type(ele) == float:
            eleStr = str("%.10f" % ele)
        elif type(ele) == str or type(ele) == unicode:
            eleStr = ele
        if count != len(list) - 1:
            eleStr += " "
        count += 1
        listStr += eleStr
    listStr += "\n"
    return listStr


def StringToFloatList(SS):
    res = [string.atof(item) for item in SS.split(" ")]
    return res


def AssignList(LL):
    newLL = []
    for ele in LL:
        newLL.append(ele)
    return newLL


def FindMax(LL):
    LL.sort()
    return LL[len(LL) - 1]
