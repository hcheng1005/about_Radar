import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt

from ggiw_filter import cluster
from ggiw_filter import GGIW_Filter as Filter
# from giw_filter import Filter, cluster


def poissonSample(lamb):
	"Sample from a Poisson distribution. Algorithm is due to Donald Knuth - see wikipedia/Poisson_distribution"
	l = np.exp(-lamb)
	k = 0
	p = 1
	while True:
		k += 1
		p *= np.random.rand()
		if p <= l:
			break
	return k - 1


def display_comps(comps, ax):
    for comp in comps:
        # 中心位置
        xy = comp.X[:2]
        
        # 形状
        # cov_ = comp.V / np.maximum(1.0, comp.v - 2 * 2)
        cov_ = comp.V # 直接使用V能更准确的描述目标的扩展属性（长宽以及朝向）
        
        # print(x, y, cov_)
        lambda_, v = np.linalg.eig(cov_)    # 计算特征值lambda_和特征向量v
        sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值
        s = 3
        width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
        height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
        angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
        # print(width, height, angle)
        ell = mpl.patches.Ellipse(xy=xy, width=width, height=height, angle=angle, color='r')    # 绘制椭圆
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    
    
def display_meas(clusters, ax):
    for cluster in clusters:
        # print()
        x, y = cluster.z[:2]
        cov_ = cluster.Z
        
        # print(x, y, cov_)
        lambda_, v = np.linalg.eig(cov_)    # 计算特征值lambda_和特征向量v
        sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值
        s = 5.991
        width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
        height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
        angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
        # print(width, height, angle)
        ell = mpl.patches.Ellipse(xy=[x, y], width=width, height=height, angle=angle, color='g')    # 绘制椭圆
        ell.set_alpha(0.3)
        ax.add_artist(ell)
    
    
'''
name: 
description: Briefly describe the function of your function
param {*} smp_data
param {*} comps
param {*} ax
return {*}
'''
def visulization(all_meas, cluster, comps, ax):
    plt.cla()
    plt.axis([-20, 100, -20, 100])
    
    display_meas(cluster, ax)
    display_comps(comps, ax)

    # 格式转换
    all_meas = np.array(all_meas).transpose([1,0,2])
    all_meas = all_meas.reshape([2,-1])
    plt.scatter(all_meas[0, :], all_meas[1, :], s=2)
    plt.plot()
    plt.pause(0.5)
    

   
if __name__ == "__main__":
    np.random.seed(2024) 
    
    # 构造目标[x,y,vx,vy,l,w,theta]
    gt_obj_list = np.array([[1, 10, 8, 1], [20, 10, -1, 10], [60, 60, -5, 1]], dtype=np.float32).T
    cov = [[[2, 0.5],[0.5, 1]],[[1, -0.1],[-0.1, 2]],[[4, -0.3],[-0.3, 1]]]
    gt_num = 3

    # 生成真值和量测
    T_ = 0.1
    Ffun = np.array([[1, 0, T_, 0],
                    [0, 1, 0, T_],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    Hfun = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])

    R_ = np.array([[1,0],
                [0,1]]) * 0.2

    # 用于杂波建模
    clutterintensitytot = 10
    obsntype = "chirp"

    gt_data = gt_obj_list
    meas_data = Hfun @ gt_data

    # 定义gmphd滤波器
    my_giw = Filter()

    plt.rcParams["figure.figsize"] = (6.0, 6.0)
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # 执行仿真迭代
    sim_step = 100
    for i in range(sim_step):
        # 真值
        measP =[]
        all_meas = []
        for idx in range(gt_num):
            gt_obj_list[:,idx] = Ffun @ gt_obj_list[:,idx]
            # 椭圆采样
            meas_data = np.random.multivariate_normal(gt_obj_list[:2,idx].reshape([-1]), cov[idx], 10) # 根据均值和协方差进行点云采样
            meas_data = meas_data.T
            
            # 
            x, y = np.mean(meas_data, axis=1)
            cov_ = np.cov(meas_data)

            measP.append(cluster([x,y], cov_, meas_data.shape[1]))
            
            all_meas.append(meas_data)


        # GIW filter
        my_giw.proc(measP) # 滤波算法
        comps = my_giw.getComponents(gate=0.2) # 检出权重门限
        
        # 可视化        
        visulization(all_meas, measP, comps, ax)