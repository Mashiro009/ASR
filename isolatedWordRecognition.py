from scipy.io import wavfile

from FeatureExtraction import FeatureExtraction
from source.timefeature import *
from source.HMM import HiddenMarkovModel
import numpy as np


class isolatedWordRecognition:
    def __init__(self,numOfState=3,gmmNumberOfCluster=5):
        self.HMMs = [HiddenMarkovModel(numOfState=numOfState,gmmNumberOfCluster=gmmNumberOfCluster)] * 10
        pass
    def connectedHMM(self,indexArr,X_Train):
        HMMNumber = len(indexArr)
        stateNumber = 0
        perHMMStateNum = 0
        hmms = []
        for item in indexArr:
            hmms.append(self.HMMs[item])
            stateNumber += self.HMMs[item].numOfState
            perHMMStateNum = self.HMMs[item].numOfState
        # 孤立词计算
        numFrame, featureDim = X_Train.shape
        # ##### 计算alpha表 #####
        alpha = np.zeros((numFrame, stateNumber + 1))  # 好几个hmm的state组合起来
        for stateIndex in range(stateNumber): # 初始化第一列 即 时间t=0
            nowHMMIndex = stateIndex / perHMMStateNum
            nowStateIndex = stateIndex % perHMMStateNum
            gmm = hmms[nowHMMIndex].gmms[nowStateIndex]
            alpha[0,stateIndex] = hmms[nowHMMIndex].startPro[nowStateIndex] * gmm.getPro(X_Train[0])
        for t in range(1, numFrame):  # 时间t
            for stateIndex in range(stateNumber):
                nowHMMIndex = stateIndex / perHMMStateNum
                nowStateIndex = stateIndex % perHMMStateNum
                gmm = hmms[nowHMMIndex].gmms[nowStateIndex]
                nowPro = gmm.getPro(X_Train[t])
                preTranPro = 0
                if stateIndex == 0:  # 如果是开头的第一个state 那么只能从自己转移过来
                    transitionPro = hmms[nowHMMIndex].transition[0 + 0]
                    preTranPro += alpha[t - 1, stateIndex] * transitionPro
                else: # 只有可能从前一个state或当前state转移过来
                    if nowStateIndex == 0:  # 如果是一个新的hmm的第一个state那么 转移就需要从上一个hmm来 或者从自己来
                        transitionPro = hmms[nowHMMIndex - 1].transition[
                            perHMMStateNum + perHMMStateNum - 1]  # 计算上一个hmm跳出的概率
                        preTranPro += alpha[t - 1, stateIndex - 1] * transitionPro
                        transitionPro = hmms[nowHMMIndex].transition[nowStateIndex + nowStateIndex]  # 计算从自己来的概率
                        preTranPro += alpha[t - 1, stateIndex] * transitionPro
                    else:
                        preTranPro += alpha[t - 1, stateIndex] * hmms[nowHMMIndex].transition[
                            nowStateIndex + nowStateIndex]
                        preTranPro += alpha[t - 1, stateIndex - 1] * hmms[nowHMMIndex].transition[
                            nowStateIndex - 1 + nowStateIndex]

        # ##### 计算alpha表 end #####

        # ##### 计算beta表 #####
        beta = np.zeros((numFrame, stateNumber + 1))  # 加上 non-emitting state
        beta[numFrame - 1, :] = 1  # 初始化最后一列 即 时间t=numFrame-1
        beta[numFrame - 1, stateNumber] = 0  # non-emitting beta为0
        for k in range(1, numFrame):  # 时间t=numFrame-1-k
            t = numFrame - 1 - k
            for stateIndex in range(stateNumber):
                nowHMMIndex = stateIndex / perHMMStateNum
                nowStateIndex = stateIndex % perHMMStateNum
                if stateIndex != stateNumber - 1:  # 如果是不是最后一个hmm的最后一个state
                    # 只有可能转移到下一个state或当前state
                    if nowStateIndex == perHMMStateNum - 1: # 如果是当前hmm的最后一个state 那么有可能转移到下一个hmm
                        gmm = hmms[nowHMMIndex+1].gmms[0]
                        afterPro1 = gmm.getPro(X_Train[t + 1])
                        gmm = hmms[nowHMMIndex].gmms[nowStateIndex]
                        afterPro2 = gmm.getPro(X_Train[t + 1])
                        beta[t, stateIndex] = afterPro1 * hmms[nowHMMIndex].transition[nowStateIndex+nowStateIndex+1] * beta[t + 1, stateIndex + 1] \
                                              + afterPro2 * hmms[nowHMMIndex].transition[nowStateIndex+nowStateIndex] * beta[
                                                  t + 1, stateIndex]
                    else:
                        gmm = hmms[nowHMMIndex].gmms[nowStateIndex+1]
                        afterPro1 = gmm.getPro(X_Train[t + 1])
                        gmm = hmms[nowHMMIndex].gmms[nowStateIndex]
                        afterPro2 = gmm.getPro(X_Train[t + 1])
                        beta[t, stateIndex] = afterPro1 * hmms[nowHMMIndex].transition[
                            nowStateIndex + nowStateIndex + 1] * beta[t + 1, stateIndex + 1] \
                                              + afterPro2 * hmms[nowHMMIndex].transition[
                                                  nowStateIndex + nowStateIndex] * beta[
                                                  t + 1, stateIndex]
                else:  # 如果是最后一个 state 只能转移到自己
                    afterPro1 = 0
                    gmm = hmms[nowHMMIndex].gmms[nowStateIndex]
                    afterPro2 = gmm.getPro(X_Train[t + 1])
                    beta[t, stateIndex] = afterPro2 * hmms[nowHMMIndex].transition[
                                              nowStateIndex + nowStateIndex] * beta[
                                              t + 1, stateIndex]
        # ##### 计算beta表 end #####

        # P(O|λ)
        observationPro = np.sum(alpha * beta, axis=1)[numFrame - 1]

        # ##### 计算gamma表 #####
        gamma = alpha * beta / observationPro
        # ##### 计算gamma表 end #####

        # ##### 计算delta表  psi Ψ #####
        delta = np.zeros((numFrame, stateNumber + 1))  # 不管 non-emitting state
        psi = np.zeros((numFrame, stateNumber + 1))
        for stateIndex in range(stateNumber):  # 初始化第一列 即 时间t=0
            nowHMMIndex = stateIndex / perHMMStateNum
            nowStateIndex = stateIndex % perHMMStateNum
            gmm = hmms[nowHMMIndex].gmms[nowStateIndex]
            delta[0, stateIndex] = hmms[nowHMMIndex].startPro[nowStateIndex] * gmm.getPro(X_Train[0])
        for t in range(1, numFrame):  # 时间t
            for stateIndex in range(stateNumber):
                nowHMMIndex = stateIndex / perHMMStateNum
                nowStateIndex = stateIndex % perHMMStateNum
                gmm = hmms[nowHMMIndex].gmms[nowStateIndex]
                nowPro = gmm.getPro(X_Train[t])
                preTranPro = 0
                # 只有可能从前一个state或当前state转移过来
                if stateIndex == 0:  # 如果是第一个hmm的第一个state
                    transitionPro = hmms[nowHMMIndex].transition[0 + 0]
                    preTranPro += delta[t - 1, stateIndex] * transitionPro
                    psi[t,stateIndex] = stateIndex
                else:
                    if nowStateIndex == 0:  # 如果是一个新的hmm的第一个state那么 转移就需要从上一个hmm来 或者从自己来
                        transitionPro = hmms[nowHMMIndex - 1].transition[
                            perHMMStateNum + perHMMStateNum - 1]  # 计算上一个hmm跳出的概率
                        preTranPro1 = delta[t - 1, stateIndex - 1] * transitionPro
                        transitionPro = hmms[nowHMMIndex].transition[nowStateIndex + nowStateIndex]  # 计算从自己来的概率
                        preTranPro2 = delta[t - 1, stateIndex] * transitionPro
                    else:
                        preTranPro1 = delta[t - 1, stateIndex - 1] * hmms[nowHMMIndex].transition[
                            nowStateIndex - 1 + nowStateIndex] # 上一个state
                        preTranPro2 = delta[t - 1, stateIndex] * hmms[nowHMMIndex].transition[
                            nowStateIndex + nowStateIndex] # 自己这个state
                    if preTranPro1 > preTranPro2:
                        preTranPro = preTranPro1
                        psi[t,stateIndex] = stateIndex - 1
                    else:
                        preTranPro = preTranPro2
                        psi[t, stateIndex] = stateIndex
                delta[t, stateIndex] = nowPro * preTranPro
        # ##### 计算delta表  psi Ψ end #####

        # ##### 计算epsilon #####
        epsilon = np.zeros((numFrame, stateNumber+1, stateNumber+1))
        for t in range(numFrame - 1):
            for i in range(stateNumber):
                for j in range(i, i + 2):
                    nowHMMIndex_i = i / perHMMStateNum
                    nowStateIndex_i = i % perHMMStateNum
                    nowHMMIndex_j = j / perHMMStateNum
                    nowStateIndex_j = j % perHMMStateNum
                    gmm = hmms[nowHMMIndex_j].gmms[nowStateIndex_j]
                    nowPro = gmm.getPro(X_Train[t + 1])
                    if nowStateIndex_i != nowHMMIndex_j:
                        transitionPro = hmms[nowStateIndex_i].transition[nowStateIndex_i+nowStateIndex_i+1]
                    else:
                        transitionPro = hmms[nowStateIndex_i].transition[nowStateIndex_i + nowStateIndex_j]
                    epsilon[t, i, j] = alpha[t, i] * transitionPro * nowPro * \
                                        beta[t, j] / observationPro
        # ##### 计算epsilon end #####

        # ##### M-step #####

        # ##### M-step end #####

        # wait = 0 # 用来打断点的



    def my_vad(self,x):
        """
        端点检测
        """
        Ini = 0.1  # 初始静默时间
        Ts = 0.01  # 窗的时长
        Tsh = 0.005  # 帧移时长
        Fs = 16000  # 采样频率
        counter1 = 0  # 以下四个参数用来寻找起始点和结束点
        counter2 = 0
        counter3 = 0
        counter4 = 0
        ZCRCountf = 0  # 用于存储过零率检测结果
        ZCRCountb = 0
        ZTh = 40  # 过零阈值
        w_sam = int(Ts * Fs)  # 窗口长度
        o_sam = int(Tsh * Fs)  # 帧移长度
        lengthX = len(x)
        segs = int((lengthX - w_sam) / o_sam) + 1
        sil = int((Ini - Ts) / Tsh) + 1
        win = np.hamming(w_sam)
        Limit = o_sam * (segs - 1) + 1
        FrmtIndex = [i for i in range(0, Limit, o_sam)]  # 每一帧的起始位置
        # 短时过零率
        ZCR_Vector = STZcr(x, w_sam, o_sam)
        # 能量
        Erg_Vector = STMn(x, w_sam, o_sam)
        IMN = np.mean(Erg_Vector[:sil])
        IMX = np.max(Erg_Vector)
        l1 = 0.03 * (IMX - IMN) + IMN
        l2 = 4 * IMN
        ITL = 100 * np.min((l1, l2))
        ITU = 10 * ITL
        IZC = np.mean(ZCR_Vector[:sil])
        stddev = np.std(ZCR_Vector[:sil])
        IZCT = np.min((ZTh, IZC + 2 * stddev))
        indexi = np.zeros(lengthX)
        indexj, indexk, indexl = indexi, indexi, indexi
        # 搜寻超过能量阈值上限的部分
        for i in range(len(Erg_Vector)):
            if Erg_Vector[i] > ITU:
                indexi[counter1] = i
                counter1 += 1
        ITUs = int(indexi[0])
        # 搜寻能量超过能量下限的部分
        for j in range(ITUs - 1, -1, -1):
            if Erg_Vector[j] < ITL:
                indexj[counter2] = j
                counter2 += 1
        start = int(indexj[0]) + 1
        Erg_Vectorf = np.flip(Erg_Vector, axis=0)
        # 重复上面过程相当于找结束帧
        for k in range(len(Erg_Vectorf)):
            if Erg_Vectorf[k] > ITU:
                indexi[counter3] = k
                counter3 += 1
        ITUs = int(indexk[0])
        for l in range(ITUs - 1, -1, -1):
            if Erg_Vectorf[l] < ITL:
                indexl[counter4] = l
                counter4 += 1
        finish = len(Erg_Vector) - int(indexl[0])  # 第一级判决结束帧
        # 从第一级判决起始帧开始进行第二判决（过零率）端点检测
        BackSearch = np.min((start, 25))
        for m in range(start, start - BackSearch, -1):
            rate = ZCR_Vector[m]
            if rate > IZCT:
                ZCRCountb += 1
                realstart = m
        if ZCRCountb > 3:
            start = realstart

        FwdSearch = np.min((len(Erg_Vector) - finish, 25))
        for n in range(finish, finish + FwdSearch):
            rate = ZCR_Vector[n]
            if rate > IZCT:
                ZCRCountf += 1
                realfinish = n
        if ZCRCountf > 3:
            finish = realfinish
        x_start = FrmtIndex[start]
        x_finish = FrmtIndex[finish - 1]  # ?????不知道咋改
        trimmed_X = x[x_start:x_finish]
        return trimmed_X

if __name__ == '__main__':
    IWR = isolatedWordRecognition(numOfState=3,gmmNumberOfCluster=5)
    FeatureExtraction = FeatureExtraction()
    # 制作模板集
    features = {}
    for i in range(10):
        fs,data = wavfile.read('p1/{}.wav'.format(i))
        speechIn1 = IWR.my_vad(data)
        fm = FeatureExtraction.getFeatureVector(data,fs)
        features['p1_{}'.format(i)] = fm
    for i in range(10):
        fs,data = wavfile.read('p2/{}.wav'.format(i))
        sspeechIn1 = IWR.my_vad(data)
        fm = FeatureExtraction.getFeatureVector(data,fs)
        features['p2_{}'.format(i)] = fm
    for i in range(10):
        fs,data = wavfile.read('p3/{}.wav'.format(i))
        speechIn1 = IWR.my_vad(data)
        fm = FeatureExtraction.getFeatureVector(data, fs)
        features['p3_{}'.format(i)] = fm
    train = features['p1_0']
    numFrame, featureDim = train.shape
    x1 = int(np.floor(numFrame / 3))
    x2 = int(np.floor(numFrame / 3 * 2))
    IWR.HMMs[0].gmms[0].initFromVectors(train[:x1])
    IWR.HMMs[0].gmms[1].initFromVectors(train[x1:x2])
    IWR.HMMs[0].gmms[2].initFromVectors(train[x2:])
    test = features['p2_0']
    numFrame, featureDim = test.shape
    IWR.HMMs[0].computeOneSpeech(train) # 代码好像能跑但是不work
    a = 1+2
    pass