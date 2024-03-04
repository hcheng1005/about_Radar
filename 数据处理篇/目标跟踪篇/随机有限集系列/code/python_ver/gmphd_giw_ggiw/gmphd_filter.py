import numpy as np 
from operator import attrgetter

class Kalman_Filter():
    def __init__(self, X, P, T=0.1, F=None, H=None) -> None:
        self.X = X
        self.P = P
        
        if F is None:
            self.F = np.array([[1, 0, T, 0],
                    [0, 1, 0, T],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        else:
            self.F = F
            
        if H is None:
            self.H = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]])
        else:
            self.H = H

        self.Q = np.eye(self.F.shape[0]) * 1.0
        self.R = np.eye(self.H.shape[0]) * 1.0
        
        self.age = 1
        self.weight = 0.3

    
    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age = self.age + 1
    
    
    def update(self, Z):
        z = self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.X = self.X + K @ (Z - z)
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
    
    
    def updateStatusAndWeight(self, Z):
        z = self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.X = self.X + K @ (Z - z)
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
    
    def getStatus(self):
        return self.X, self.P
        

class gmphd_component():
    def __init__(self, X, P, W) -> None:
        self.X = X
        self.P = P
        self.W = W
        
    def getStatus(self):
        return self.X, self.P, self.W


class GMPHD_Filter():
    def __init__(self) -> None:
        self.T = 0.1
        
        self.gmm_comps = [gmphd_component(X=np.array([1,10,0,0]), P=np.eye(4)*10, W = 0.01), 
                                 gmphd_component(X=np.array([10,60,0,0]), P=np.eye(4)*10, W = 0.01),
                                 gmphd_component(X=np.array([70,50,0,0]), P=np.eye(4)*10, W = 0.01)]
        
        self.survival = 0.9     # 存活概率
        self.detection = 0.9    # 检测概率
        self.clutter = 0.001
        
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
        for comp in self.gmm_comps:
            comp.X = self.F @ comp.X
            comp.P = self.F @ comp.P @ self.F.T + self.Q
            comp.W = self.survival * comp.W # 权重更新
    
    
    '''
    name: 
    description: Briefly describe the function of your function
    param {*} self
    param {*} measSet
    return {*}
    '''
    def update(self, measSet):
        newGMPHD_Comp = []
        
        print("last comp: [{}], meas num:[{}]".format(len(self.gmm_comps), measSet.shape[1]))
        
        # case 1： 未关联到量测
        for comp in self.gmm_comps:
            X, P, W = comp.X, comp.P, comp.W
            W = (1.0 - self.detection) * W # 权重更新
            newGMPHD_Comp.append(gmphd_component(X, P, W))
        
        # case 2: 关联到量测
        # 这里一共会产生m*n个假设，其中m是量测个数，n是当前航迹个数
        newGMPHD_Comp2 = []
        for i in range(measSet.shape[1]):
            newGMPHD_Comp3 = []
            for comp in newGMPHD_Comp:
                X, P, W = comp.X, comp.P, comp.W
                
                Z = self.H @ X
                S = self.H @ P @ self.H.T + self.R      # 计算残差协方差
                K = P @ self.H.T @ np.linalg.inv(S)     # 计算增益
                
                X = X + K @ (measSet[:, i] - Z)             # 更新状态
                P = (np.eye(P.shape[0]) - K @ self.H) @ P   # 更新协方差
                W = self.detection * W * self.gaussian_likelihood(Z, S, measSet[:, i]) # 更新权重
                
                # print("residual:{}, likihoond:[{}]".format((measSet[:, i] - Z), self.gaussian_likelihood(Z, S, measSet[:, i])))
                newGMPHD_Comp3.append(gmphd_component(X, P, W))

            # 归一化权重
            sum_w = np.sum([comp.W for comp in newGMPHD_Comp3])
            for comp in newGMPHD_Comp3:
                comp.W = comp.W / (sum_w + self.clutter)
                # print(newGMPHD_Comp3[idx].W)

            newGMPHD_Comp2.extend(newGMPHD_Comp3)
        
        # 总共生成m*n+n个假设
        newGMPHD_Comp.extend(newGMPHD_Comp2)
        self.gmm_comps = newGMPHD_Comp
        
        print("component Number before prune: ", len(self.gmm_comps))
        # for comp in self.gmm_comps:
        #     print(comp.W) 
        # print("component Number before prune: ", len(self.gmm_comps))
        
        
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
        p1 = (2.0 * np.pi) ** (-1.0 * len(u) * 0.5) 
        p2 = np.linalg.det(rho) ** (-0.5)
        p3 = np.exp(-0.5 * ((x - u).T @ np.linalg.inv(rho) @ (x - u)))
        return (p1 * p2 * p3)
    
    
    '''
    name: 
    description: Briefly describe the function of your function
    param {*} self
    return {*}
    '''
    def prune(self, truncthresh=1e-6, mergethresh=0.01, maxcomponents=10):
        # Truncation is easy
        print("start components prune")
        weightsums = [np.sum(comp.W for comp in self.gmm_comps)]   # diagnostic
        sourcegmm = [comp for comp in self.gmm_comps if comp.W > truncthresh]
        weightsums.append(np.sum(comp.W for comp in sourcegmm))
        origlen  = len(self.gmm_comps)
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
            # 构造新comp
            newcomp = gmphd_component(newX, newP, newW)
            newgmm.append(newcomp)
        
        # 按照权重排序并取前maxcomponents个comp
        newgmm.sort(key=attrgetter('W'))
        newgmm.reverse()
        self.gmm_comps = newgmm[:maxcomponents]
        
        # log
        weightsums.append(np.sum(comp.W for comp in newgmm))
        weightsums.append(np.sum(comp.W for comp in self.gmm_comps))
        print("prune(): %i -> %i -> %i -> %i" % (origlen, trunclen, len(newgmm), len(self.gmm_comps)))
        print("prune(): weightsums %g -> %g -> %g -> %g" % (weightsums[0], weightsums[1], weightsums[2], weightsums[3]))
        
        # pruning should not alter the total weightsum (which relates to total num items) - so we renormalise
        weightnorm = weightsums[0] / weightsums[3]
        
        # print('final---------------')
        for comp in self.gmm_comps:
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
        sourcegmm = [comp for comp in self.gmm_comps if comp.W > gate]
        return sourcegmm
    
    
    
    
    