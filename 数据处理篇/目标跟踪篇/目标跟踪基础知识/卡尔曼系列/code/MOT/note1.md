# Single-Object Tracking in Clutter

[lecture](./lecture/Section%202%20Single-Object%20Tracking%20in%20Clutter/L2_handout_20190429.pdf)

---

![](images/20240305200448.png)

> 高斯、线性模型

## Nearest neighbour filtering
>  **Basic idea: Prune all hypotheses except the most probable one**

![](images/20240305195523.png)

![](images/20240305195419.png)

## Probabilistic data association filtering
> **各meas加权**

![](images/20240305195807.png)

![](images/20240305195933.png)

## Gaussian sum filtering
> **Basic idea: approximate the posterior as a Gaussian mixture with a few components.**

### Pruning 
> **剪枝： 删除小于给定阈值的假设分支。**

![](images/20240305201124.png)


### Merging
> **合并相似的假设分支。**

![](images/20240305201439.png)

### Capping
> **只保留前N个最大假设分支。**

![](images/20240305201545.png)

### Summary
> **以上几个步骤都是为了控制假设分支的数量。**


### Code
**Gaussian sum filtering算法比较经典，其中涉及到的假设合并、删除在后续其他算法中均有涉及，这部分代码应该熟读理解。**

1、[hypothesisReduction](./HA01/hypothesisReduction.m)

2、[Merging](./HA01/GaussianDensity.m)