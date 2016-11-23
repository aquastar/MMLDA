from LDAModel import LDAModel

if __name__ == '__main__':
    a = LDAModel(alpha=.01, beta=0.01, NumberOfIterations=1000, SaveStep=100, K=10)
    a.ModelInit('line_news')
    a.estimate()