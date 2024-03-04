import numpy as np 
from operator import attrgetter
from numpy.linalg import inv as inv_
from numpy.linalg import det as det_
from numpy.linalg import cholesky as chol

# 卡方分布表
# P  0.95	0.90	0.80	0.70	0.50	0.30	0.20	0.10	0.05	0.01	0.001
# ---------------------------------------------------------------------------------------
# 1  0.004	0.02	0.06	0.15	0.46	1.07	1.64	2.71	3.84	6.64	10.83
# 2  0.10	0.21	0.45	0.71	1.39	2.41	3.22	4.60	5.99	9.21	13.82
# 3  0.35	0.58	1.01	1.42	2.37	3.66	4.64	6.25	7.82	11.34	16.27
# 4  0.71	1.06	1.65	2.20	3.36	4.88	5.99	7.78	9.49	13.28	18.47
# 5  1.14	1.61	2.34	3.00	4.35	6.06	7.29	9.24	11.07	15.09	20.52
# 6  1.63	2.20	3.07	3.83	5.35	7.23	8.56	10.64	12.59	16.81	22.46
# 7  2.17	2.83	3.82	4.67	6.35	8.38	9.80	12.02	14.07	18.48	24.32
# 8  2.73	3.49	4.59	5.53	7.34	9.52	11.03	13.36	15.51	20.09	26.12
# 9  3.32	4.17	5.38	6.39	8.34	10.66	12.24	14.68	16.92	21.67	27.88
# 10 3.94	4.86	6.18	7.27	9.34	11.78	13.44	15.99	18.31	23.21	29.59


from math import pi

from scipy.special import gamma, multigammaln


def gamma_func(x):
    val = np.exp(-x) * x # TBD
    return val

class cluster:
    def __init__(self, z, Z, num) -> None:
        self.z = z
        self.Z = Z
        self.size = num


class ggiw_component():
    def __init__(self, X, P, v, V, alpha, beta, W) -> None:
        self.X = X
        self.P = P
        self.v = v
        self.V = V
        self.alpha = alpha
        self.beta = beta
        self.W = W
        
        self.scale = 1.5
                
    def getStatus(self):
        return self.X, self.P, self.W
    

class GGIW_Filter():
    def __init__(self) -> None:
        self.T = 0.1
        
        self.ggiw_comps = [ggiw_component(X=np.array([1,1,1,1]), P=np.eye(4)*10, 
                                         v=10, V=np.eye(2)*4, 
                                         alpha = 10, beta = 1,
                                         W=0.01)] # 随便初始化一个comp
        
        self.survival = 0.9     # 存活概率
        self.pd = 0.9    # 检测概率
        self.clutter_density = 0.001 # 杂波密度
        
        self.tau = 0.2 # 表征的是变化周期，数值越小，表示shape变化越快
        self.d = 2
        self.poisson_rate = 10 # 
        
        self.F = np.array([[1, 0, self.T, 0],
                            [0, 1, 0, self.T],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.eye(self.F.shape[0]) * 1.0
        self.R = np.eye(self.H.shape[0]) * 0.2
        
        self.scale = 0.9
        
    def proc(self, measSet):
        self.predict()
        self.update(measSet)
        self.prune()
    
    
    '''
    name: 
    description: Briefly describe the function of your function
    param {*} self
    return {*}
    '''
    def predict(self):
        print("__________ predict __________")
        for comp in self.ggiw_comps:
            comp.X = self.F @ comp.X
            comp.P = self.F @ comp.P @ self.F.T + self.Q
            comp.W = self.survival * comp.W # 权重更新
            
            tmpv = 2 * self.d + 2 + np.exp(-self.T / self.tau) * (comp.v - 2 * self.d - 2)
            comp.V = (tmpv-2*self.d-2) / (comp.v-2*self.d-2) * comp.V 
            comp.v = tmpv
            
            comp.alpha /= comp.scale
            comp.beta /= comp.scale
            
            # print("X: {} W: {}".format(comp.X[:2], comp.W))
            
    '''
    name: 
    description: Briefly describe the function of your function
    param {*} self
    param {*} measSet
    return {*}
    '''
    def update(self, measP):
        '''
        Ref: Gamma Gaussian inverse-Wishart Poisson multi-Bernoulli filter for extended target tracking
        (https://publications.lib.chalmers.se/records/fulltext/245112/local_245112.pdf)
        
        Ref: Implementation of the Gamma Gaussian Inverse Wishart Trajectory Probability Hypothesis Density Filter
        https://research.chalmers.se/publication/523776/file/523776_Fulltext.pdf
        '''
        
        newGMPHD_Comp = []
        print("last comp: [{}], meas num:[{}]".format(len(self.ggiw_comps), len(measP)))
        
        # case 1： 未关联到量测 (11a)
        for comp in self.ggiw_comps:
            newW = (1.0 - (1.0 - np.exp(-1.0 * comp.alpha / comp.beta)) * self.pd) * comp.W # 权重更新
            # print("W1:{}, W2:{}".format(comp.W , newW))
            comp.W = newW
            newGMPHD_Comp.append(comp)
        
        # case 2: 关联到量测
        # 这里一共会产生m*n个假设，其中m是量测个数，n是当前航迹个数
        newGMPHD_Comp2 = []
        for meas in measP:
            # print("cur z: {}".format(meas.z))
            newGMPHD_Comp3 = []
            for comp in newGMPHD_Comp:
                X, P, W, v, V, alpha, beta = comp.X, comp.P, comp.W, comp.v, comp.V, comp.alpha, comp.beta
                
                # print("x: {}".format(X))
                
                Xkk = V / (v - 2 * self.d - 2)

                K = P @ self.H.T
                S = self.H @ P @ self.H.T + (self.scale * Xkk + self.R) / meas.size # TBD
                K = P @ self.H.T @ inv_(S)     # 计算增益
                
                X_chol = chol(Xkk)
                S_chol = chol(S)
                
                res_ = (meas.z - self.H @ X).reshape([2,1])
                # print(res_.T, (self.H @ X).T)
                N = res_ @ res_.T
                Nk = X_chol @ inv_(S_chol) @ N @ inv_(S_chol).T @ X_chol.T
                
                tmp_V = V + Nk + meas.Z

                # likelihood
                p1 = ((2.0*pi) ** (meas.size * self.d / 2)) * (2 ** (0.5 * meas.size)) / (meas.size ** (0.5 * self.d))
                p2 = 1 # / ((self.clutter_density ** meas.size) * (((pi**meas.size) * meas.size * det_(S)) ** (self.d / 2)))
            
                tmp_v1 = v / 2
                tmp_v2 = (v + meas.size) / 2
                p3 = (det_(V) ** tmp_v1 ) / (det_(tmp_V) ** tmp_v2)
                
                # Ref: https://search.r-project.org/CRAN/refmans/CholWishart/html/lmvgamma.html
                p4 = multigammaln(int(tmp_v2), self.d) / multigammaln(int(tmp_v1), self.d) 
                
                # p5 = (det_(Xkk) ** (0.5 * meas.size)) / (((Xkk + self.R / meas.size) ** ((meas.size - 1)*0.5)) * (det_(S) ** 0.5))
                p5 = (det_(Xkk) ** (0.5 * meas.size)) / (det_(S) ** 0.5)
                p6 = (gamma(comp.alpha + meas.size) * (comp.beta ** comp.alpha)) / \
                        (gamma(comp.alpha) * ((comp.beta + 1) ** (comp.alpha + meas.size))) 
                
                # print("P1:{}, P2:{}, P3:{}".format(p1, p2, p3))
                # print(W)
                W = p1 * p2 * p3 * p4 * p5 * p6 * W
                # print(W)
                
                X = X + K @ (meas.z - self.H @ X)              # 更新状态
                # P = (np.eye(P.shape[0]) - K @ self.H) @ P   # 更新协方差
                P = P - K @ S @ K.T
                
                # 
                v = v + meas.size
                V = V + Nk + meas.Z
                
                #
                alpha = alpha + meas.size
                beta = beta + 1
                
                # print("residual:{}, likihoond:[{}]".format((measSet[:, i] - Z), self.gaussian_likelihood(Z, S, measSet[:, i])))
                newGMPHD_Comp3.append(ggiw_component(X, P, v, V, alpha, beta, W)) # 新comp

            # 归一化权重: 
            # 每个量测和n个comp会组合出n个新comp，这n个xincomp需要做权重归一化
            sum_w = np.sum([comp.W for comp in newGMPHD_Comp3])
            for comp in newGMPHD_Comp3:
                comp.W = comp.W / (sum_w + 0.0) # TBD 此处应该有个Kronecker delta，暂时先省略
                # print("X: {} W: {}".format(comp.X[:2], comp.W))

            newGMPHD_Comp2.extend(newGMPHD_Comp3)
        
        # 总共生成m*n+n个假设
        newGMPHD_Comp.extend(newGMPHD_Comp2)
        self.ggiw_comps = newGMPHD_Comp
        
        # print("component Number before prune: ", len(self.ggiw_comps))
        # for comp in self.ggiw_comps:
        #     print(comp.W) 
        # print("component Number before prune: ", len(self.ggiw_comps))
        
        
    '''
    name: 
    description: Briefly describe the function of your function
    param {*} self
    param {*} u
    param {*} rho
    param {*} x
    return {*}
    '''
    def gaussian_likelihood(self, u, rho, x):
        p1 = (2.0 * pi) ** (-1.0 * len(u) * 0.5) 
        p2 = np.linalg.det(rho) ** (-0.5)
        p3 = np.exp(-0.5 * ((x - u).T @ inv_(rho) @ (x - u)))
        return (p1 * p2 * p3)
    
    
    '''
    name: 
    description: Briefly describe the function of your function
    param {*} self
    return {*}
    '''
    def prune(self, truncthresh=1e-6, mergethresh=9.49, maxcomponents=10):
        # Truncation is easy
        # print("start components prune")
        weightsums = [np.sum(comp.W for comp in self.ggiw_comps)]   # diagnostic
        sourcegmm = [comp for comp in self.ggiw_comps if comp.W > truncthresh]        
        weightsums.append(np.sum(comp.W for comp in sourcegmm))
        origlen  = len(self.ggiw_comps)
        trunclen = len(sourcegmm)
        
        # print("origlen: %d, trunclen: %d" % (origlen, trunclen))
        
        # Iterate to build the new GMM
        newgmm = []
        while len(sourcegmm) > 0:
            windex = np.argmax(comp.W for comp in sourcegmm)
            weightiest = sourcegmm[windex] # 本次comp
            sourcegmm = sourcegmm[:windex] + sourcegmm[windex+1:] # 其余comp
            
            # 计算该comp与其他所有comp的“距离”
            distances = [float(np.dot(np.dot((comp.X - weightiest.X).T, np.linalg.inv(comp.P)), (comp.X - weightiest.X))) for comp in sourcegmm]
            
            print(distances)
            
            dosubsume = np.array([dist <= mergethresh for dist in distances])
            
            subsumed = [weightiest] # 当前comp作为新comp
            if any(dosubsume): # 其否需要对某些“过近”的comp进行合并
                # print(dosubsume)
                subsumed.extend(list(np.array(sourcegmm)[dosubsume]))   # 加入需合并的comp
                sourcegmm = list(np.array(sourcegmm)[~dosubsume])       # 从原列表中删除被合并的comp
                
            # create unified new component from subsumed ones
            aggweight = np.sum(comp.W for comp in subsumed)
            
            newW = aggweight
            normal_ = 1.0 / aggweight
            
            # comp融合
            newX = normal_ * np.sum(np.array([comp.W * comp.X for comp in subsumed]), axis=0)
            newP = normal_ * np.sum(np.array([comp.W * (comp.P + (weightiest.X - comp.X) * (weightiest.X - comp.X).T) \
                                                for comp in subsumed]), axis=0)
            
            newv = normal_ * np.sum(np.array([comp.W * comp.v for comp in subsumed]), axis=0)
            newV = normal_ * np.sum(np.array([comp.W * comp.V for comp in subsumed]), axis=0)
            
            newalpha = normal_ * np.sum(np.array([comp.W * comp.alpha for comp in subsumed]), axis=0)
            newbeta = normal_ * np.sum(np.array([comp.W * comp.beta for comp in subsumed]), axis=0)
            
            # 构造新comp
            newcomp = ggiw_component(newX, newP, newv, newV, newalpha, newbeta, newW)
            newgmm.append(newcomp)
        
        # 按照权重排序并取前maxcomponents个comp
        newgmm.sort(key=attrgetter('W'))
        newgmm.reverse()
        self.ggiw_comps = newgmm[:maxcomponents]
        
        # log
        weightsums.append(np.sum(comp.W for comp in newgmm))
        weightsums.append(np.sum(comp.W for comp in self.ggiw_comps))
        # print("prune(): %i -> %i -> %i -> %i" % (origlen, trunclen, len(newgmm), len(self.ggiw_comps)))
        # print("prune(): weightsums %g -> %g -> %g -> %g" % (weightsums[0], weightsums[1], weightsums[2], weightsums[3]))
        
        # pruning should not alter the total weightsum (which relates to total num items) - so we renormalise
        weightnorm = weightsums[0] / weightsums[3]
        
        for comp in self.ggiw_comps:
            comp.W *= weightnorm
            
        print("__________ Final __________")
        for comp in self.ggiw_comps:
            print("X: {} W: {}".format(comp.X[:2], comp.W))
            
    
    '''
    name: 
    description: Briefly describe the function of your function
    param {*} self
    param {*} gate
    return {*}
    '''
    def getComponents(self, gate=0.1):
        output_ = [comp for comp in self.ggiw_comps if comp.W > gate]
        return output_
    
    
    
    
    