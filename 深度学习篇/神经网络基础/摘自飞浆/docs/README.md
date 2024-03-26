# 目录

- [**面试宝典**](#面试宝典)
- [**深度学习**](#深度学习)
- [**CNN**](#cnn)
- [**计算机视觉**](#计算机视觉)
  - [**目标检测**](#目标检测)
  - [**目标分类**](#目标分类)
  - [**图像增强**](#图像增强)
- [**model\_tuning**](#model_tuning)

---

## **面试宝典**
  - [interview_questions](./tutorials/interview_questions/interview_questions.md)


## **深度学习**
  - **基础知识**
    - [神经元](./tutorials/deep_learning/basic_concepts/neuron.md)
    - [单层感知机](./tutorials/deep_learning/basic_concepts/single_layer_perceptron.md)
    - [多层感知机](./tutorials/deep_learning/basic_concepts/multilayer_perceptron.md)

  - **正则化**(包括什么是正则化？正则化如何帮助减少过度拟合？数据增强、L1 L2正则化介绍、L1和L2的贝叶斯推断分析法、Dropout、DropConnect、早停法等8个知识点)
      - [l1l2.md](./tutorials/deep_learning/model_tuning/regularization/l1l2.md)
      - [归一化基础知识点👍👍](./tutorials/deep_learning/normalization/basic_normalization.md)（包括什么是归一化、为什么要归一化、为什么归一化能提高求解最优解速度、归一化有哪些类型、不同归一化的使用条件、归一化和标准化的联系与区别等6个知识点）
      - [正则化](./tutorials/deep_learning/model_tuning/regularization/regularization.md)
      - [dropout.md](./tutorials/deep_learning/model_tuning/regularization/dropout.md)
      - [dropconnect.md](./tutorials/deep_learning/model_tuning/regularization/dropconnect.md)
      - [early_stop.md](./tutorials/deep_learning/model_tuning/regularization/early_stop.md)
      - [Layer_Normalization](./tutorials/deep_learning/normalization/Layer_Normalization.md)
      - [参数初始化👍](./tutorials/deep_learning/model_tuning/weight_initializer.md)（包括为什么不能全零初始化、常见的初始化方法等5个知识点）
        - [扩展阅读：一文搞懂深度网络初始化👍](https://cloud.tencent.com/developer/article/1587082)
        - [kaiming初始化的推导](https://zhuanlan.zhihu.com/p/305055975)
        - [Pytorch神经网络初始化kaiming分布](https://blog.csdn.net/winycg/article/details/86649832)
        - 
  - [激活函数👍](./tutorials/deep_learning/activation_functions/Activation_Function.md) （包括什么是激活函数、激活函数的作用、identity、step、sigmoid、tanh、relu、lrelu、prelu、rrelu、elu、selu、softsign、softplus、softmax、swish、hswish、激活函数的选择等21个知识点）

  - **优化策略**（包括什么是优化器、GD、SGD、BGD、鞍点、Momentum、NAG、Adagrad、AdaDelta、RMSProp、Adam、AdaMa、Nadam、AMSGrad、AdaBound、AdamW、RAdam、Lookahead等18个知识点）
    - [梯度下降、随机梯度下降👍](./tutorials/deep_learning/optimizers/gd.md)
    - [momentum👍](./tutorials/deep_learning/optimizers/momentum.md)
    - [adagrad](./tutorials/deep_learning/optimizers/adagrad.md)
    - [adam](./tutorials/deep_learning/optimizers/adam.md)
    - [adamax](./tutorials/deep_learning/optimizers/adamax.md)
    - [adamw](./tutorials/deep_learning/optimizers/adamw.md)
    - [adabound](./tutorials/deep_learning/optimizers/adabound.md)
    - [adadelta](./tutorials/deep_learning/optimizers/adadelta.md)
    - [amsgrad](./tutorials/deep_learning/optimizers/amsgrad.md)
    - [lookahead](./tutorials/deep_learning/optimizers/lookahead.md)
    - [nadam](./tutorials/deep_learning/optimizers/nadam.md)
    - [nag](./tutorials/deep_learning/optimizers/nag.md)
    - [radam](./tutorials/deep_learning/optimizers/radam.md)
    - [rmsprop](./tutorials/deep_learning/optimizers/rmsprop.md)
    - [梯度下降法、牛顿法和拟牛顿法](https://zhuanlan.zhihu.com/p/37524275)
    - [牛顿法与梯度下降法的讲解与Python代码实现](https://blog.csdn.net/qq_41133375/article/details/105337383)

  - **损失函数**
      - [Balanced_L1_Loss](./tutorials/deep_learning/loss_functions/Balanced_L1_Loss.md)
      - [交叉熵损失函数👍](./tutorials/deep_learning/loss_functions/CE_Loss.md)
      - [均方差损失（Mean Square Error，MSE）👍](./tutorials/deep_learning/loss_functions/MSE.md)
      - [CTC](./tutorials/deep_learning/loss_functions/CTC.md)
  
  - **模型调优**
    - [batch_size👍](./tutorials/deep_learning/model_tuning/batch_size.md)
    - [学习率👍](./tutorials/deep_learning/model_tuning/learning_rate.md)（包括什么是学习率、学习率对网络的影响以及不同的学习率率衰减方法，如：分段常数衰减等12个学习率衰减方法）
  

  - **距离度量方式**
      - [向量距离与相似度👍](./tutorials/deep_learning/distances/distances.md)

  - **评估方式**
      - [评估指标👍](./tutorials/deep_learning/metrics/evaluation_metric.md)
      - [mAP👍](./tutorials/deep_learning/metrics/mAP.md)

## **CNN**
- [CV_CNN.md](./tutorials/CNN/CV_CNN.md)
- [ParamsCounter.md](./tutorials/CNN/ParamsCounter.md)
- [池化👍](./tutorials/CNN/Pooling.md)（包括池化的基本概念、池化特点等2个知识点）
- **卷积算子**
  - [标准卷积👍](./tutorials/CNN/convolution_operator/Convolution.md)
  - [1*1卷积👍](./tutorials/CNN/convolution_operator/1_Convolution.md)
  - [3D卷积](./tutorials/CNN/convolution_operator/3D_Convolution.md)
  - [可变形卷积详解](./tutorials/CNN/convolution_operator/Deformable_Convolution.md)
  - [空洞卷积](./tutorials/CNN/convolution_operator/Dilated_Convolution.md)
  - [分组卷积](./tutorials/CNN/convolution_operator/Group_Convolution.md)
  - [可分离卷积](./tutorials/CNN/convolution_operator/Separable_Convolution.md)
  - [转置卷积](./tutorials/CNN/convolution_operator/Transpose_Convolution.md)


## **计算机视觉**

### **目标检测**
  - [边界框（bounding box）](./tutorials/computer_vision/object_detection/Bounding_Box_Anchor.md)
  - [IOU](./tutorials/computer_vision/object_detection/IOU.md)
  - [非极大值抑制NMS👍](./tutorials/computer_vision/object_detection/NMS.md)
  - [SoftNMS](./tutorials/computer_vision/object_detection/SoftNMS.md)
  - 
### **目标分类**
  - [AlexNet.md](./tutorials/computer_vision/classification/AlexNet.md)
  - [DarkNet.md](./tutorials/computer_vision/classification/DarkNet.md)
  - [GoogLeNet.md](./tutorials/computer_vision/classification/GoogLeNet.md)
  - [LeNet.md](./tutorials/computer_vision/classification/LeNet.md)
  - [Res2Net.md](./tutorials/computer_vision/classification/Res2Net.md)
  - [ResNeXt.md](./tutorials/computer_vision/classification/ResNeXt.md)
  - [ResNet.md](./tutorials/computer_vision/classification/ResNet.md)
  - [SwinTransformer.md](./tutorials/computer_vision/classification/SwinTransformer.md)
  - [VGG.md](./tutorials/computer_vision/classification/VGG.md)
  - [ViT.md](./tutorials/computer_vision/classification/ViT.md)

### **图像增强**
  - [ImageAugment.md](./tutorials/computer_vision/image_augmentation/ImageAugment.md)
  - [tta.md](./tutorials/computer_vision/image_augmentation/tta.md)

## **model_tuning**
  - **attention**
      - [attention_description.md](./tutorials/deep_learning/model_tuning/attention/attention_description.md)
      - [attention_varities.md](./tutorials/deep_learning/model_tuning/attention/attention_varities.md)
      - [classic_attention.md](./tutorials/deep_learning/model_tuning/attention/classic_attention.md)
      - [self_attention.md](./tutorials/deep_learning/model_tuning/attention/self_attention.md)
