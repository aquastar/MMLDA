from LDAModel import LDAModel

if __name__ == '__main__':
    a = LDAModel(alpha=10, beta=0.01, NumberOfIterations=1000, SaveStep=10, K=5, TopNum=10)
    a.ModelInit('line_news')
    a.estimate()
