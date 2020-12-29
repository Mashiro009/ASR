from scipy.stats import multivariate_normal  # 生成多维概率分布的方法
import numpy as np

class GaussianMixtureModel:
    def __init__(self,numberOfCluster = 1, maxIterTimes=100 , regCovar: float = 1e-06):
        self.means = None # 高斯簇的均值
        self.covs = None # 高斯簇的协方差矩阵
        self.weights = None # 高斯簇的权重
        self.numberOfCluster = numberOfCluster # 高斯簇的个数
        self.maxIterTimes = maxIterTimes # 优化时迭代的次数
        self.regCovar = regCovar # 防止出现奇异协方差矩阵 在对角线加1e-06防止对角线出现0
        pass

    def initFromVectors(self, X_Train):
        # 获取X_Train中的数量和维度
        numFrame , featureDim = X_Train.shape
        self.regCovar = self.regCovar * np.identity(featureDim)

        # 初始化参数
        # 生成numberOfCluster个处于最大值和最小值之间个featureDim维度的向量
        # 作为初始这些cluster的均值
        self.means = np.random.randint(X_Train.min() / 2, X_Train.max() /
                                        2, size=(self.numberOfCluster, featureDim))
        # 初始化协方差
        self.covs = np.zeros((self.numberOfCluster,featureDim,featureDim))
        for index in range(self.numberOfCluster):
            np.fill_diagonal(self.covs[index],1)

        # 初始化权重
        self.weights = np.ones(self.numberOfCluster) / self.numberOfCluster

        # 概率密度矩阵
        PMatrix = np.zeros((numFrame,self.numberOfCluster))

        for i in range(self.maxIterTimes): # 迭代maxIterTimes次
            # 求各个vector对每个cluster的概率
            for index in range(self.numberOfCluster):
                self.covs[index] += self.regCovar # 防止出现奇异矩阵
                gmm = multivariate_normal(mean=self.means[index],cov=self.covs[index])

                # ##### E-step 计算概率密度 #####
                # 计算X在各簇的概率密度
                PMatrix[:,index] = self.weights[index] * gmm.pdf(X_Train)

            # 计算各vector在该GMM中出现的总概率密度
            totalPro = PMatrix.sum(axis=1) # shape (numFrame,)
            # 如果vector在各簇的概率密度均为0 那么等分配
            totalPro[totalPro == 0] = 1 / self.numberOfCluster
            totalPro = np.expand_dims(totalPro,axis=1) # shape (numFrame,1)
            PMatrix /= totalPro
            # ##### E-step 计算概率密度 end #####


            # ##### M-step 更新参数　#####
            for index in range(self.numberOfCluster):
                # X_train 中 vector 出现在第index簇的频率
                frequency = np.sum(PMatrix[:,index],axis=0)
                # PMatrix[:index].shape (numFrame,)
                self.weights[index] = frequency / numFrame
                # reshape 中 -1为缺省值 reshape(-1,1)即不管第axis=0维有多少个，第axis=1维必须只有一个
                # 新均值
                self.means[index] = np.sum( X_Train * PMatrix[:index].reshape(-1,1) , axis=0 ) * (1 / frequency)
                self.covs[index] = np.dot( ( PMatrix[:,index].reshape(-1,1)  * (X_Train - self.means[index]) ).T ,
                                           (X_Train - self.means[index]) ) + self.regCovar

            # ##### M-step 更新参数　end #####
            # 循环continue

        # np.mean(X_Train,axis=0)
        # np.cov(X_Train.T) https://blog.csdn.net/jeffery0207/article/details/83032325
        pass

    def getPro(self,vector):
        """

        :param vector:
        :return: 返回该vector在此混合高斯分布中的概率密度
        """
        PMatrix = np.zeros((1, self.numberOfCluster))
        for gmmIndex in range(self.numberOfCluster):
            g = multivariate_normal(mean=self.means[gmmIndex], cov=self.covs[gmmIndex])
            PMatrix[:, gmmIndex] = self.weights[gmmIndex] * g.pdf(vector)
        return np.sum(PMatrix, axis=1)

    def getEachPro(self,vector):
        """

        :param vector:
        :return: 返回该vector在此混合高斯分布中分别在各簇的概率
        """
        PMatrix = np.zeros((1, self.numberOfCluster))
        for gmmIndex in range(self.numberOfCluster):
            g = multivariate_normal(mean=self.means[gmmIndex], cov=self.covs[gmmIndex])
            PMatrix[:, gmmIndex] = self.weights[gmmIndex] * g.pdf(vector)
        # 计算各vector在该GMM中出现的总概率密度
        totalPro = PMatrix.sum(axis=1)  # shape (numFrame,)
        # 如果vector在各簇的概率密度均为0 那么等分配
        totalPro[totalPro == 0] = 1 / self.numberOfCluster
        totalPro = np.expand_dims(totalPro, axis=1)  # shape (numFrame,1)
        PMatrix /= totalPro
        return PMatrix