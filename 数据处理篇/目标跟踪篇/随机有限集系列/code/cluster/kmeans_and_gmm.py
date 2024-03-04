# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot

import scipy.stats
    
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

############################################################
# 高斯混合模型EM算法
# x为样本矩阵，k为模型个数，times为模型迭代次数
############################################################


############################################################
# 数据预处理
# X为样本矩阵
# 将数据进行极差归一化处理
############################################################

def scale_data(x):
    for i in range(x.shape[1]):
        max_    = x[:, i].max()
        min_    = x[:, i].min()
        x[:, i] = (x[:, i] - min_) / (max_ - min_)
    return x



############################################################
# 初始化模型参数
# shape为样本矩阵x的维数（样本数，特征数）
# k为模型的个数
# mu, cov, alpha分别为模型的均值、协方差以及混合系数
############################################################

def init_params(shape, k):
    n, d  = shape
    mu    = np.random.rand(k, d)
    cov   = np.array([np.eye(d)] * k)
    alpha = np.array([1.0 / k] * k)
    return mu, cov, alpha



############################################################
# 第i个模型的高斯密度分布函数
# x 为样本矩阵，行数等于样本数，列数等于特征数
# mu_i, cov_i分别为第i个模型的均值、协方差参数
# 返回样本在该模型下的概率密度值
############################################################

def phi(x, mu_i, cov_i):
    norm = scipy.stats.multivariate_normal(mean=mu_i, cov=cov_i)
    return norm.pdf(x)



############################################################
# E步：计算每个模型对样本的响应度
# x 为样本矩阵，行数等于样本数，列数等于特征数
# mu为均值矩阵， cov为协方差矩阵
# alpha为各模型混合系数组成的一维矩阵
############################################################

def expectation(x, mu, cov, alpha):
    # 样本数，模型数
    n, k = x.shape[0], alpha.shape[0]

    # 计算各模型下所有样本出现的概率矩阵prob，行对应第i个样本，列对应第K个模型
    prob = np.zeros((n, k))
    for i in range(k):
        prob[:, i] = phi(x, mu[i], cov[i])
    prob = np.mat(prob)

    # 计算响应度矩阵gamma，行对应第i个样本，列对应第K个模型
    gamma = np.mat(np.zeros((n, k)))
    for i in range(k):
        gamma[:, i] = alpha[i] * prob[:, i]
    for j in range(n):
        gamma[j, :] /= np.sum(gamma[j, :])
    return gamma



############################################################
# M步：迭代模型参数
############################################################

def maximization(x, gamma):
    # 样本数，特征数
    n, d = x.shape
    # 模型数
    k = gamma.shape[1]

    # 初始化模型参数
    mu = np.zeros((k, d))
    cov = []
    alpha = np.zeros(k)

    # 更新每个模型的参数
    for i in range(k):
        # 第K个模型对所有样本的响应度之和
        ni       = np.sum(gamma[:, i])
        # 更新mu
        mu[i, :] = np.sum(np.multiply(x, gamma[:, i]), axis=0) / ni
        # 更新cov
        cov_i    = (x - mu[i]).T * np.multiply((x - mu[i]), gamma[:, i]) / ni
        cov.append(cov_i)
        # 更新alpha
        alpha[i] = ni / n
    cov = np.array(cov)
    return mu, cov, alpha

def gmm_em(k, times, data):
    # 载入数据集sample.csv
    # dataset = np.loadtxt('sample.data')
    x = data
    # 数据归一化处理
    x = scale_data(x)
    # 初始化模型参数
    mu, cov, alpha = init_params(x.shape, k)

    # 迭代模型参数
    for i in range(times):
        gamma          = expectation(x, mu, cov, alpha)
        mu, cov, alpha = maximization(x, gamma)

    # 求出当前模型参数下样本的响应矩阵
    gamma = expectation(x, mu, cov, alpha)
    # 样本矩阵中每一行最大值的列索引即为该样本的对应类别
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    # 将每个样本放入对应的类别中
    cluster = []
    for i in range(k):
        item = np.array([data[j] for j in range(x.shape[0]) if category[j] == i])
        cluster.append(item)
    
    return cluster

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index


if __name__ == '__main__':
    # 定义n个高斯分布
    mean = [2, 0] 
    cov = [[2, 0], 
            [0, 1]]
    data1 = np.random.multivariate_normal(mean, cov, 20) # 根据均值和协方差进行点云采样
    
    mean = [4, 4] 
    cov = [[1, 0], 
            [0, 2]]
    data2 = np.random.multivariate_normal(mean, cov, 15) # 根据均值和协方差进行点云采样
    
    print((data1, data2))
    x = np.concatenate([data1, data2], axis=0)

    k_means = K_Means(k=2) # k_means
    k_means.fit(x) # 执行k-means聚类算法
    print(k_means.centers_)
    
    pyplot.figure()
    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*', s=150)

    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))
            
    # pyplot.show()

    pyplot.figure()
    y_pred = KMeans(n_clusters=2).fit_predict(x)
    pyplot.scatter(x[:, 0], x[:, 1], c=y_pred)
    # pyplot.show()

    pyplot.figure()
    k, times = 2, 50
    cluster = gmm_em(k, times, x)
    color = ['rs', 'bo']
    for i in range(len(cluster)):
        pyplot.plot(cluster[i][:, 0], cluster[i][:, 1], color[i])
    pyplot.title("GMM Clustering")
    # pyplot.show()
    
    pyplot.figure()
    gmm = GaussianMixture(n_components=2)
    labels = gmm.fit_predict(x)
    pyplot.scatter(x[:, 0], x[:, 1], c=y_pred)
    pyplot.show()