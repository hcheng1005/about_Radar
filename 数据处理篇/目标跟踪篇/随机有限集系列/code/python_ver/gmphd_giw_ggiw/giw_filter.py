import numpy as np 
from operator import attrgetter
from numpy.linalg import inv as inv_
from numpy.linalg import det as det_
from numpy.linalg import cholesky as chol

from math import pi
from scipy.special import multigammaln

def gamma_func(x):
    val = np.exp(-x) * x # TBD
    return val

class cluster:
    def __init__(self, z, Z, num) -> None:
        self.z = z
        self.Z = Z
        self.size = num


class giw_component():
    def __init__(self, X, P, v, V, W) -> None:
        self.X = X
        self.P = P
        self.v = v
        self.V = V
        self.W = W
                
    def getStatus(self):
        return self.X, self.P, self.W
    

class GIW_Filter():
    def __init__(self) -> None:
        self.T = 0.1
        
        self.giw_comps = [giw_component(X=np.array([1,1,1,1]), P=np.eye(4)*10, v=10, V=np.eye(2)*4, W = 0.01)] # 随便初始化一个comp
        
        self.survival = 0.9     # 存活概率
        self.detection = 0.9    # 检测概率
        self.clutter_density = 0.001 # 杂波密度
        
        self.tau = 0.2 # 表征的是变化周期，数值越小，表示shape变化越快
        self.d = 2
        self.poisson_rate = 20 # 
        
        self.F = np.array([[1, 0, self.T, 0],
                            [0, 1, 0, self.T],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.eye(self.F.shape[0]) * 1.0
        self.R = np.eye(self.H.shape[0]) * 0.2
        
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
        for comp in self.giw_comps:
            # 
            comp.X = self.F @ comp.X
            comp.P = self.F @ comp.P @ self.F.T + self.Q
            comp.W = self.survival * comp.W # 权重更新
            
            tmpv = 2 * self.d + 2 + np.exp(-self.T / self.tau) * (comp.v - 2 * self.d - 2)
            comp.V = (tmpv - 2*self.d - 2) / (comp.v - 2 *self.d - 2) * comp.V 
            comp.v = tmpv
            
    '''
    name: 
    description: Briefly describe the function of your function
        Ref: Extended Target Tracking using a Gaussian-Mixture PHD filter 
        (https://www.researchgate.net/publication/258655341_Extended_Target_Tracking_using_a_Gaussian-Mixture_PHD_Filter)  
        
        Ref2: Anti-Clutter Gaussian Inverse Wishart PHD Filter for Extended Target Tracking
        (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6928846/pdf/sensors-19-05140.pdf)

    '''
    def update(self, measP):
        newGIW_Comp = []
        print("last comp: [{}], meas num:[{}]".format(len(self.giw_comps), len(measP)))
        
        # case 1： 未关联到量测 (11a)
        for comp in self.giw_comps:
            newW = (1.0 - (1.0 - np.exp(-1*self.poisson_rate))*self.detection) * comp.W # 权重更新 (Ref1: 11b)
            # print("W1:{}, W2:{}".format(comp.W , newW))
            comp.W = newW
            newGIW_Comp.append(comp)
        
        # case 2: 关联到量测
        # 这里一共会产生m*n个假设，其中m是量测个数，n是当前航迹个数
        newGMPHD_Comp2 = []
        for meas in measP:
            newGMPHD_Comp3 = []
            for comp in newGIW_Comp:
                X, P, W, v, V = comp.X, comp.P, comp.W, comp.v, comp.V

                K = P @ self.H.T
                S = self.H @ P @ self.H.T + V / meas.size # TBD
                K = P @ self.H.T @ inv_(S)     # 计算增益
            
                res_ = (meas.z - self.H @ X).reshape([2,1])
                
                N = res_ @ res_.T
                tmp_V = V + N + meas.Z

                # (Ref2: 5-6-7)
                p1 = np.exp(-self.poisson_rate) * (self.poisson_rate ** meas.size) * self.detection # 相比（5）少了分母lambda_c*ck 
                p2 = 1 / ((self.clutter_density ** meas.size) * (((pi**meas.size) * meas.size * det_(S)) ** (self.d / 2)))
                p3 = (det_(V) ** (v / 2)) / (det_(tmp_V) ** ( (v + meas.size) / 2))
                # p41 = gamma_func((v+meas.size)/2) / gamma_func(v/2)
                # p4 = gamma((v+meas.size)/2) / gamma(v/2) 
                
                # Ref: https://search.r-project.org/CRAN/refmans/CholWishart/html/lmvgamma.html
                p4 = multigammaln(int((v+meas.size)/2), self.d) / multigammaln(int(v/2), self.d) 
                
                # print("P1:{}, P2:{}, P3:{}".format(p1, p2, p3))
                
                W = p1 * p2 * p3 * p4 * W
                
                X = X + K @ (meas.z - self.H @ X)              # 更新状态
                # P = (np.eye(P.shape[0]) - K @ self.H) @ P   # 更新协方差
                P = P - K @ S @ K.T
                
                # SPD更新
                v = v + meas.size
                V = V + N + meas.Z
                
                # print("residual:{}, likihoond:[{}]".format((measSet[:, i] - Z), self.gaussian_likelihood(Z, S, measSet[:, i])))
                newGMPHD_Comp3.append(giw_component(X, P, v, V, W)) # 新comp

            # 归一化权重: 
            # 每个量测和n个comp会组合出n个新comp，这n个xincomp需要做权重归一化
            sum_w = np.sum([comp.W for comp in newGMPHD_Comp3])
            for comp in newGMPHD_Comp3:
                comp.W = comp.W / (sum_w + 0.0) # TBD 此处应该有个Kronecker delta，暂时先省略
                # print(newGMPHD_Comp3[idx].W)

            newGMPHD_Comp2.extend(newGMPHD_Comp3)
        
        # 总共生成m*n+n个假设
        newGIW_Comp.extend(newGMPHD_Comp2)
        self.giw_comps = newGIW_Comp
        
        print("component Number before prune: ", len(self.giw_comps))
        # for comp in self.giw_comps:
        #     print(comp.W) 
        # print("component Number before prune: ", len(self.giw_comps))
        
        
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
        print("start components prune")
        weightsums = [np.sum(comp.W for comp in self.giw_comps)]   # diagnostic
        sourcegmm = [comp for comp in self.giw_comps if comp.W > truncthresh]        
        weightsums.append(np.sum(comp.W for comp in sourcegmm))
        origlen  = len(self.giw_comps)
        trunclen = len(sourcegmm)
        
        print("origlen: %d, trunclen: %d" % (origlen, trunclen))
        
        # Iterate to build the new GMM
        newgmm = []
        while len(sourcegmm) > 0:
            windex = np.argmax(comp.W for comp in sourcegmm)
            weightiest = sourcegmm[windex] # 本次comp
            sourcegmm = sourcegmm[:windex] + sourcegmm[windex+1:] # 其余comp
            
            # 计算该comp与其他所有comp的“距离”
            distances = [float(np.dot(np.dot((comp.X - weightiest.X).T, np.linalg.inv(comp.P)), (comp.X - weightiest.X))) for comp in sourcegmm]
            dosubsume = np.array([dist <= mergethresh for dist in distances])
            
            subsumed = [weightiest] # 当前comp作为新comp
            if any(dosubsume):      # 其否需要对某些“过近”的comp进行合并
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
            
            # 构造新comp
            newcomp = giw_component(newX, newP, newv, newV, newW)
            newgmm.append(newcomp)
        
        # 按照权重排序并取前maxcomponents个comp
        newgmm.sort(key=attrgetter('W'))
        newgmm.reverse()
        self.giw_comps = newgmm[:maxcomponents]
        
        # log
        weightsums.append(np.sum(comp.W for comp in newgmm))
        weightsums.append(np.sum(comp.W for comp in self.giw_comps))
        print("prune(): %i -> %i -> %i -> %i" % (origlen, trunclen, len(newgmm), len(self.giw_comps)))
        print("prune(): weightsums %g -> %g -> %g -> %g" % (weightsums[0], weightsums[1], weightsums[2], weightsums[3]))
        
        # pruning should not alter the total weightsum (which relates to total num items) - so we renormalise
        weightnorm = weightsums[0] / weightsums[3]
        
        # print('final---------------')
        for comp in self.giw_comps:
            comp.W *= weightnorm
            # print(comp.W)
            
    
    '''
    name: 
    description: Briefly describe the function of your function
    param {*} self
    param {*} gate
    return {*}
    '''
    def getComponents(self, gate=0.1):
        output_ = [comp for comp in self.giw_comps if comp.W > gate]
        return output_
    
    
    
    
    