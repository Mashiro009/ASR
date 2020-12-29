from scipy.stats import multivariate_normal  # 生成多维概率分布的方法
import numpy as np
from GMM import GaussianMixtureModel

class HiddenMarkovModel:
    def __init__(self,numOfState=3):
        # state 数目
        self.numOfState = numOfState
        # π初始概率
        # 初始化进入第0个state的概率为1
        self.startPro = np.zeros(self.numOfState)
        self.startPro[0] += 1
        # transition 转移概率 a 初始化每个转移概率都为0.5
        self.transition = np.zeros(self.numOfState*2) + 0.5
        self.transition[self.numOfState*2 - 2] = 1    # 孤立词训练 最后一个state自转移概率为1
        self.transition[self.numOfState*2 - 1] = 0    # 孤立词训练 最后一个state转移到不发射state概率为0
        # GMM 混合高斯模型 b
        self.gmms = []
        for i in range(self.numOfState):
            self.gmms.append(GaussianMixtureModel())

        # ?
        # alpha表
        # beta表
        # gamma表
        # epsilon表
        # gamma k表
        # delta表 viterbi用
        pass

    def getTransitionIndex(self,i,j):
        return i + j

    def computeOneSpeech(self,X_Train):
        # 孤立词计算
        numFrame , featureDim = X_Train.shape
        # ##### 计算alpha表 #####
        alpha = np.zeros((numFrame,self.numOfState+1)) # 不管 non-emitting state
        for stateIndex in range(self.numOfState): # 初始化第一列 即 时间t=0
            gmm = self.gmms[stateIndex]
            alpha[0,stateIndex] = self.startPro[stateIndex] * gmm.getPro(X_Train[0])
        for t in range(1,numFrame): # 时间t
            for stateIndex in range(self.numOfState + 1):
                if stateIndex != self.numOfState: # 如果是会产生vector的state
                    gmm = self.gmms[stateIndex]

                    nowPro = gmm.getPro(X_Train[t])
                    preTranPro = 0
                    # 只有可能从前一个state或当前state转移过来
                    if stateIndex == 0:
                        preTranPro += alpha[t - 1, stateIndex] * self.transition[self.getTransitionIndex(stateIndex,stateIndex)]
                    else:
                        preTranPro += alpha[t - 1, stateIndex] * self.transition[
                            self.getTransitionIndex(stateIndex, stateIndex)]
                        preTranPro += alpha[t - 1, stateIndex-1] * self.transition[
                            self.getTransitionIndex(stateIndex-1, stateIndex)]
                    alpha[t,stateIndex] = nowPro * preTranPro
                else: # 如果是non-emitting state
                    # 只有可能从最后一个state转移过来
                    alpha[t,stateIndex] = alpha[t, stateIndex-1] * self.transition[
                            self.getTransitionIndex(stateIndex-1, stateIndex)]
        # ##### 计算alpha表 end #####

        # ##### 计算beta表 #####
        beta = np.zeros((numFrame, self.numOfState + 1))  # 加上 non-emitting state
        beta[numFrame-1,:] = 1 # 初始化最后一列 即 时间t=numFrame-1
        beta[numFrame-1,self.numOfState] = 0 # non-emitting beta为0
        for k in range(1, numFrame):  # 时间t=numFrame-1-k
            t = numFrame - 1 - k
            for stateIndex in range(self.numOfState ):
                if stateIndex != self.numOfState - 1:  # 如果是不是最后一个state
                    # 只有可能转移到下一个state或当前state
                    gmm = self.gmms[stateIndex + 1]
                    afterPro1 = gmm.getPro(X_Train[t+1])
                    gmm = self.gmms[stateIndex]
                    afterPro2 = gmm.getPro(X_Train[t + 1])
                    beta[t,stateIndex] = afterPro1 * self.transition[self.getTransitionIndex(stateIndex,stateIndex+1)] * beta[t+1,stateIndex+1]\
                                         + afterPro2 * self.transition[self.getTransitionIndex(stateIndex,stateIndex)] * beta[t+1,stateIndex]
                else:  # 如果是最后一个 state
                    # 那么可以转移到non-emitting state
                    afterPro1 = 0
                    gmm = self.gmms[stateIndex]
                    afterPro2 = gmm.getPro(X_Train[t + 1])
                    beta[t, stateIndex] = afterPro1 * self.transition[
                        self.getTransitionIndex(stateIndex, stateIndex + 1)] * beta[t, stateIndex + 1] \
                                          + afterPro2 * self.transition[
                                              self.getTransitionIndex(stateIndex, stateIndex)] * beta[t + 1, stateIndex]
        # ##### 计算beta表 end #####

        # P(O|λ)
        observationPro = np.sum(alpha*beta,axis=1)[numFrame - 1]

        # ##### 计算gamma表 #####
        gamma = alpha * beta / observationPro
        # ##### 计算gamma表 end #####

        # ##### 计算delta表  psi Ψ #####
        delta = np.zeros((numFrame, self.numOfState + 1))  # 不管 non-emitting state
        psi = np.zeros((numFrame, self.numOfState + 1))
        for stateIndex in range(self.numOfState):  # 初始化第一列 即 时间t=0
            gmm = self.gmms[stateIndex]
            delta[0, stateIndex] = self.startPro[stateIndex] * gmm.getPro(X_Train[0])
        for t in range(1, numFrame):  # 时间t
            for stateIndex in range(self.numOfState + 1):
                if stateIndex != self.numOfState:  # 如果是会产生vector的state
                    gmm = self.gmms[stateIndex]

                    nowPro = gmm.getPro(X_Train[t])
                    preTranPro = 0

                    # 只有可能从前一个state或当前state转移过来
                    if stateIndex == 0:
                        preTranPro += delta[t - 1, stateIndex] * self.transition[
                            self.getTransitionIndex(stateIndex, stateIndex)]
                        psi[t,stateIndex] = 0
                    else:
                        preTranPro1 = delta[t - 1, stateIndex] * self.transition[
                            self.getTransitionIndex(stateIndex, stateIndex)]
                        preTranPro2 = delta[t - 1, stateIndex - 1] * self.transition[
                            self.getTransitionIndex(stateIndex - 1, stateIndex)]
                        if preTranPro1 > preTranPro2:
                            preTranPro = preTranPro1
                            psi[t,stateIndex] = stateIndex
                        else:
                            preTranPro = preTranPro2
                            psi[t, stateIndex] = stateIndex - 1
                    delta[t, stateIndex] = nowPro * preTranPro
                else:  # 如果是non-emitting state
                    # 只有可能从最后一个state转移过来
                    delta[t, stateIndex] = delta[t, stateIndex - 1] * self.transition[
                        self.getTransitionIndex(stateIndex - 1, stateIndex)]
                    psi[t,stateIndex] = stateIndex - 1
        # ##### 计算delta表  psi Ψ end #####

        # ##### 计算epsilon #####
        epsilon = np.zeros((numFrame,self.numOfState+1,self.numOfState+1))
        for t in range(numFrame-1):
            for i in range(self.numOfState):
                for j in range(i,i+2):
                    gmm = self.gmms[stateIndex]
                    nowPro = gmm.getPro(X_Train[t+1])
                    epsilon[t,i,j] = alpha[t,i] * self.transition[self.getTransitionIndex(i,j)] * nowPro * beta[t,j] / observationPro
        # ##### 计算epsilon end #####

        # ##### M-step #####
        self.startPro = delta[0,:-1]

        for i in range(self.numOfState):
            for j in range(i,i+2):
                x = np.sum( epsilon[:-1,i,j] , axis=0 )
                y = np.sum( gamma[:-1,i] ,axis=0 )
                self.transition[self.getTransitionIndex(i,j)] = x / y

        gamma_k = np.zeros((numFrame,self.numOfState+1,self.gmms[0].numberOfCluster))
        for stateIndex in range(self.numOfState):
            for t in range(numFrame):
                PMatrix = self.gmms[stateIndex].getEachPro(X_Train[t])
                gamma_k[t,stateIndex] = PMatrix * gamma[t,stateIndex]
        for stateIndex in range(self.numOfState):
            x = np.sum( gamma_k[:,stateIndex,:], axis=0 )
            y = np.sum( np.sum( gamma_k[:,stateIndex,:],axis=0) , axis=0) # 不知道写的对不对
            self.gmms[stateIndex].weights = x / y
            z = np.zeros((self.gmms[stateIndex].numberOfCluster,featureDim))
            # z = np.sum( gamma_k[:,stateIndex,:] * X_Train ,axis= 0) # 不知道写的对不对
            for gIndex in range(self.gmms[stateIndex].numberOfCluster):
                z[gIndex] = np.sum(X_Train * gamma_k[:,stateIndex,gIndex] ,axis=0)
            self.gmms[stateIndex].means = z / x
            z1 = np.zeros((self.gmms[stateIndex].numberOfCluster,featureDim,featureDim))
            for gIndex in range(self.gmms[stateIndex].numberOfCluster):
                z1[gIndex] = np.sum(np.dot(
                    (gamma_k[:,stateIndex,gIndex] * (X_Train - self.gmms[stateIndex].means[gIndex]) ).T ,
                    (X_Train - self.gmms[stateIndex].means[gIndex]))
                    ,axis=0)
            self.gmms[stateIndex].covs = (z1/x) + self.gmms[stateIndex].regCovar
        # ##### M-step end #####





if __name__ == '__main__':
    HMM = HiddenMarkovModel()
