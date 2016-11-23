from LDAModel import LDAModel

if __name__ == '__main__':
    a = LDAModel(alpha=10, beta=0.1, NumberOfIterations=100, SaveStep=20, K=5, TopNum=20)
    a.ModelInit('line_news')
    a.estimate()