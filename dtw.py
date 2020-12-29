import numpy as np
class DTW:
    def __init__(self):
        pass

    def dp(self,template,real):
        templateLen, featureNum = template.shape
        realLen, featureNum = real.shape
        distance = np.zeros((templateLen,realLen))
        for i in range(0,templateLen):
            for j in range(0,realLen):
                temp = np.power(template[i,:]-real[j,:],2)
                distance = np.sqrt(np.sum(temp)) / featureNum
        DP = np.zeros((enumerate + 1, realLen + 1))
        DP[0, :] = np.inf
        DP[:, 0] = np.inf
        DP[0, 0] = 0
        DP[1:, 1:] = distance

        # 寻找整个过程的最短匹配距离
        for i in range(templateLen):
            for j in range(realLen):
                dmin = min(DP[i, j], DP[i, j + 1], DP[i + 1, j])
                DP[i + 1, j + 1] += dmin

        cost = DP[templateLen, realLen]
        return cost