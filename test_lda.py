from LDAModel import LDAModel

if __name__ == '__main__':
    a = LDAModel(alpha=.01, beta=0.01, NumberOfIterations=100, SaveStep=10, K=5)
    a.ModelInit('line_news')
    a.estimate()