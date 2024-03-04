import numpy as np 
import matplotlib.pyplot as plt
from gmphd_filter import GMPHD_Filter


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


# 构造目标[x,y,vx,vy,l,w,theta]
gt_num = 3
gt_obj_list = np.array([[1, 10, 7, 7],
                        [10, 60, 0.5, -5,],
                        [70, 50, -7, -0.5]]).T

np.random.seed(2024)   

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

clutterintensitytot = 10
span = (0, 100)
slopespan = (-2, 3)  # currently only used for clutter generation / inference
obsntype = "chirp"

gt_data = gt_obj_list
meas_data = Hfun @ gt_data

# 定义gmphd滤波器
my_GMM = GMPHD_Filter()

plt.figure

sim_step = 100
for i in range(sim_step):
    # 真值
    gt_obj_list = Ffun @ gt_obj_list
    gt_data = np.concatenate((gt_data, gt_obj_list))
    
    # 获取量测
    meas = Hfun @ gt_obj_list
    for ii in range(gt_num):
        meas[:, ii] = meas[:, ii] + np.random.multivariate_normal(np.zeros(2), R_)
    
    # 杂波
    numclutter = poissonSample(clutterintensitytot)
    print("clutter generating %i items" % numclutter)
    clutter = []
    for _ in range(numclutter):
        index = int(np.round(span[0] + np.random.rand() * (span[1] - span[0])))
        index2 = int(np.round(span[0] + np.random.rand() * (span[1] - span[0])))
        clutter.append([[index], [index2]])  # chirp-like
    clutter = np.asarray(clutter).T.squeeze()
    
    # 构造最终量测集
    measSet = np.concatenate((meas, clutter), axis=1)
    
    # gmphd filter
    my_GMM.proc(measSet)
    comps = my_GMM.getComponents()
    
    # 可视化
    # plt.clf()
    plt.axis([-20, 100, -20, 100])
    plt.plot(clutter[0, :], clutter[1, :], 'b*')
    plt.plot(meas[0, :], meas[1, :], 'y*')
    
    for comp in comps:
        print(comp.X)
        plt.plot(comp.X[0], comp.X[1], 'ro')

    plt.ion()
    plt.pause(0.2)
    