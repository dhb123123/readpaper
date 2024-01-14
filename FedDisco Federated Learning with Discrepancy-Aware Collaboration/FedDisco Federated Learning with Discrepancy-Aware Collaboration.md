# FedDisco: Federated Learning with Discrepancy-Aware Collaboration

## 1.预知

### 1.1 标签偏好

​	**举个例子**，假设有一组人要对一部电影进行评价，并给出一些标签，比如"搞笑"、"感人"、"刺激"等。如果评价者中的大多数人来自相似的文化背景，可能会导致对于特定类型的幽默、情感或刺激的理解存在偏见。例如，某个文化中的笑话可能在另一个文化中不够有趣，导致对于"搞笑"这个标签的评价有偏差。

​	在过去的工作中，为了解决标签偏好的问题，研究人员通常考虑对局部模型进行正则化或对全局模型进行微调，以缓解模型过拟合的问题。然而，在调整模型权重方面，以前的方法可能只是简单地根据数据集大小来分配权重，而没有考虑到更复杂的情况，可能导致在模型集成中的不公平分配。

### 1.2 局部和全局类别分布之间的差异

​	考虑一个城市的犯罪数据集。全局类别分布可能是整个城市的犯罪类型的统计，而局部类别分布则是特定社区或街区的犯罪类型统计。如果某个社区的犯罪类型分布与整个城市的分布相比有显著差异，我们就可以说存在局部和全局类别分布之间的差异。$

### 1.3 类别分布异质性

​	**类别分布异质性**指的是在一个数据集或系统中，不同类别之间的分布不均匀或不一致的情况。这种异质性可能表现为某些类别的样本数量远远超过其他类别，或者某些类别在特定子集或区域中的分布与整体分布不同。

​	在另一个例子中，考虑一个电商网站的用户行为分析，其中有多个商品类别。如果在某个地理区域，某一类商品的购买频率远高于其他类别，那么就存在类别分布的异质性。这可能导致针对整体数据集的模型在这个地理区域的预测效果较差，因为它未能捕捉到类别分布异质性带来的影响。

举个例子，考虑一个医学图像分类的问题，其中目标是识别不同类型的病变（比如肿瘤）。如果数据集中某一种病变的样本数量远远多于其他病变类型，那么就存在类别分布的异质性。这可能导致模型在预测时偏向于出现频率较高的类别，而对于其他类别的表现可能较差。

​	**文中解释**：比如说一些客户端在类别1中有大量数据，而在类别2中的数据较少，而别的客户端则相反。

### 1.4 上限最小化

​	"上限最小化"通常是指在给定约束条件下寻找目标函数的最小值或最大值。这个约束条件可以是一个或多个条件，限制了变量的取值范围。

### 1.5 欧几里得距离

​	在机器学习领域，ℓ2距离常被用来比较两个数据点之间的相似程度，越小表示越相似，越大表示越不相似。

### 1.6 全局目标函数的优化界限

全局目标函数的优化界限通常指的是优化问题的约束条件或者限制条件，这些条件决定了优化问题中变量的取值范围。在优化问题中，这些界限可以是等式约束或不等式约束，限制了变量的取值范围，使得优化问题在满足这些条件的前提下寻找全局最优解。

举例来说，假设有一个优化问题要最小化一个目标函数 F(x)，同时存在如下约束条件：

1. **等式约束**：gx=0
2. **不等式约束**：hx<=0

这里，等式约束g*(*x*)=0 可能表示问题的某些限制条件必须被满足，而不等式约束 h(x)<=0则可能表示额外的限制条件，如资源的可用性或者其他限制。

全局优化问题就是在满足这些约束条件的前提下，找到能够最小化目标函数 F(x) 的解。在某些情况下，这些约束条件可能会在解空间中形成一些区域，被称为“可行域”，解需要在这个可行域内才能被接受。

因此，全局优化的界限指的是在考虑这些约束条件的情况下，变量可能取值的范围或者可行域，以便在这个范围内寻找目标函数的最优解。

### 1.7 全局目标函数优化过程中误差

​	在优化过程中，误差指的是在迭代中当前解（或参数）与最优解（或理想解）之间的差距或偏差。全局目标函数优化过程中的误差描述了优化算法在每次迭代中所得到的解与理想最优解之间的差异。

### 1.8 Lipschitz 平滑性

​	对于一个实值函数，如果它在定义域内的导数（斜率）的变化受到了一个有界的限制，那么这个函数就具有 Lipschitz 平滑性。

## 2 一些问题

### 2.1 类别异质性 和 聚合 ？

​	**$C_1$拥有很多类别A的数据，很少类别B的数据，而$C_2$则相反，那么在服务器中怎样聚合模型？**

比如：$C_1$有很多类别A的数据，$C_2$有很多类别B的数据，但是$C_1$数据集远大于$C_2$的数据集，在中央服务器中聚合的时候，之前是根据数据集的大小来设置权重聚合，那么得出的全局模型预测可能就更加偏向A,而对B的预测准确率就下降了，所以本文就考虑了类别异质性，既考虑了数据集的大小又考虑了类别异质性。

## 3 文章概述

### 3.1 摘要

​	之前的算法在服务器端聚合模型的时候都根据数据集的大小设置聚合的权重(这个数据集越大，模型聚合的权重就越大)但是并没有考虑类别的异质性，比如一个客户端拥有很多的类别A的数据，而另一个客户端拥有很多的类别B的数据，在服务器端聚合模型的时候，难以得到较好的鲁棒性。

![image-20231114143450344](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114143450344.png)

- 之前的全局模型聚合都是调整本地模型w，而不调整权重p。

### 3.2 引言

#### 3.2.1 问题

​	存在类别异质性(比如说一些客户端在类别1中有大量数据，而在类别2中的数据较少，而别的客户端则相反),这可能导致局部模型的优化趋向于不同的方向，在服务器聚合就难以得到较好的鲁棒性。

#### 3.2.2 之前的工作

​	1.在客户端调整模型

​	2.在服务端调整模型

#### 3.3.3 提出FedDisco

​	**这种新方法通过为数据集大小较大、差异值较小的客户端分配较大的聚合权重，利用每个客户端的数据集大小和差异来确定聚合权重。**(==核心==)

![image-20231111154618652](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231111154618652.png)

### 3.3 背景

#### 3.3.1 类别比例的计算

$D_{k.c}$=(y属于c这个类别的总数据量)/(该客户端拥有的总数据量)

![image-20231111162128676](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231111162128676.png)

### 3.4 经验观察

#### 3.4.1 数据来源

[CIFAR-10 and CIFAR-100 datasets (toronto.edu)](https://www.cs.toronto.edu/~kriz/cifar.html)

![image-20231111163605542](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231111163605542.png)

#### 3.4.2 实验b

![image-20231111170330905](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231111170330905.png)

**本实验的目的就是：当将较大权重分给较好的模型时，全局模型的效果能够达到最好。**

### 3.5 理论分析

​	大概意思是通过全局目标函数的优化界限，通过解决优化问题推出 权重$p_k$不仅与数据集大小n有关，也与差异性相关$d_k$。

- ####  假设1

![image-20231119154523899](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231119154523899.png)

​	Lipschitz 平滑性在优化理论和算法中具有重要意义。**具有 Lipschitz 连续性的函数更容易进行优化**，因为这种性质表明函数的变化是有限的，没有剧烈的波动或尖峰。在优化问题中，Lipschitz 平滑性是许多优化算法（比如梯度下降法）的收敛性和稳定性的一个重要保证，因为它限制了函数值的变化范围，使得优化过程更可控。

- #### 假设2

![image-20231119154721692](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231119154721692.png)

​	这种全局下界的存在性有助于确定优化问题的解空间，因为它确保了函数值不会无限地减小，有一个稳定的下限。这对于优化算法的收敛性分析以及问题解决的合理性和可行性非常重要。

- ####  假设3

![image-20231119154827208](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231119154827208.png)

这个假设是用来描述在联邦学习或分布式优化问题中，对于每个参与的客户端，其提供的随机梯度具有一定的性质。这些性质包括无偏性和有界的方差。

简单来说，这个假设的作用是为优化算法的分布式执行提供一种保证。具体来说：

1. **无偏性：** 表明客户端计算的随机梯度在期望意义上等于全局目标函数在相同点处的真实梯度。这确保了在随机梯度的平均值收敛到全局梯度时，每个客户端提供的贡献不会偏离真实的梯度方向。
2. **有界的方差：** 表明随机梯度的变化范围受到了限制。有界的方差意味着随机梯度的波动性是可控的，不会出现过大的波动。这对于优化算法的稳定性和收敛性至关重要，因为过大的随机梯度波动可能导致优化过程不稳定，影响算法的收敛性能。

综合来说，这个假设为优化算法提供了分布式执行的理论基础和保证，确保每个客户端提供的随机梯度都符合一定的性质，有利于整个优化过程的稳定和有效进行。

- ####  假设4

![image-20231119154942392](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231119154942392.png)

![image-20231119200502228](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231119200502228.png)

#### 3.5.1 全局目标函数的优化界限

![image-20231119193441682](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231119193441682.png)

### 3.6 FedDiso

**1.**首先全局模型的聚合：

![image-20231114142550538](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114142550538.png)

$p_k$为每个客户端的权重，$w_k$为为客户端的模型参数。

**2.**通过理论分析得出权重$p_k$与数据集n和差异性d的关系。

![image-20231114143020390](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114143020390.png)

**3.**

- 首先在第一轮中每个客户端计算本地的差异水平d，将其传给服务端。
- T轮后，每个客户端 计算好本地模型参数w，然后上传。服务器根据公式计算最终的全局模型。

![image-20231114142444650](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114142444650.png)

#### 3.6.1 框架	

- 算法流程

![image-20231114145809463](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114145809463.png)

- 本地差异性计算

  假设全局类别分布是均匀的，因此计算局部与全部的类别差异。此外，每个本地客户端都可以计算其差异，而无需额外的数据共享，防止了类别分布的信息泄露。

  ![image-20231114151033897](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114151033897.png)

  其中$D_{k,c}$是第k个客户端，类别为c的分布值(数量)，$T_c$为全局类别c的分布值。

  ![image-20231119205052405](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231119205052405.png)

  ![image-20231114151249252](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114151249252.png)

- 差异感知聚合权重确定

  ![image-20231114152743837](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114152743837.png)

  a,b是调节权重的两个超参数，这种Disco聚合可以为具有较大数据集大小和较小局部差异级别的客户端分配更大的权重，从而为客户端确定更具区别的聚合权重。

  ReLU是修正线性单元函数，即如果里面的值大于0，则保持不变，如果小于0，说明数据集过小，并且数据集类别差异性过大，将权重变为0。

- 全局模型聚合

  ![image-20231114153214484](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114153214484.png)

### 3.7 相关工作

​	联邦学习(FL)已经成为一个新兴的话题。然而，数据分布的异质性可能会显著降低FL的性能。研究这方面的工作主要分为两个方向:局部模型调整和全局模型调整。

- #### 局部模型调整

- #### 全局模型调整

### 3.8 实验

#### 3.8.1 实验设置

- 数据集

  ![image-20231114155815456](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114155815456.png)



[The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions - ViDIR Dataverse (harvard.edu)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

[CIFAR-10 and CIFAR-100 datasets (toronto.edu)](https://www.cs.toronto.edu/~kriz/cifar.html)

[CINIC-10 Is Not ImageNet or CIFAR-10 (ed.ac.uk)](https://datashare.ed.ac.uk/handle/10283/3192)

![image-20231114161659159](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114161659159.png)

[fashion-mnist/README.zh-CN.md at master · zalandoresearch/fashion-mnist (github.com)](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.zh-CN.md)

![image-20231114161603691](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114161603691.png)

- 联合场景

  考虑了两种数据异质性分布

  - **NIID-1**：

    - NIID-1 的数据分布遵循 Dirichlet 分布，记作 Dirβ，其中参数 β（默认为 0.5）与数据的异质性水平相关。
    - 在 NIID-1 中，考虑了 10 个客户端（或者说数据拥有者）。
    - Dirichlet 分布通常用于生成具有多元类别特征的数据集，其参数可以控制数据的不均匀性。在这里，参数 β 的变化可以引起数据异质性水平的变化，即数据分布的不均匀程度。

    ![image-20231114163632842](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231114163632842.png)

    HAM10000是一个严重不平衡的数据集。因此，我们选择了一个缓和的β=5.0。对于CIFAR-100，我们也选择β=5.0。**HAM10000是全局不平衡的，并且它在类0中具有最大数量的样本。**

    

  - **NIID-2**：

    - NIID-2 是一个更加异质性的设置，包含了 5 个偏向性客户端和 1 个无偏向性客户端。
    - 每个偏向性客户端包含来自于总类别数 C 的数据中的 C/5 个类别（总类别数的五分之一）。
    - 与之相对，无偏向性客户端包含所有 C 个类别的数据。
    - 这种设置模拟了在分布式学习中可能遇到的不同程度的数据异质性情况，其中一部分客户端拥有的数据更加偏向特定类别，而另一部分客户端则包含更加多样化的数据。

​		eg：对于**AG News**，我们将数据集划分为5个和50个客户端，分别用于完全和部分参与场景，其中80%的有偏见的客户端具有来自2个类别的数据样本，而其他20%的统一客户端具有来自4个类别的数据样本。

#### 3.8.2 实验结果

- FedDisco使用梯度下降方法(SGD)进行训练，可以看到使用FedDisco聚合方法后，准确率的平均值有所上升。

![image-20231115141909983](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231115141909983.png)

- 方法在聚合上使用Disco后，准确率也有所增加。

![image-20231115142040523](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231115142040523.png)

#### 3.8.3 部分客户场景

​	举个例子，假设有一个使用联邦学习的智能手机键盘应用程序，旨在改进预测输入文本的功能。在此场景下，部分客户端参与可能是这样实现的：

- 在每个 FL 轮中，该应用程序选择了一部分用户的设备来更新模型。
- 对于某些用户来说，在特定时间段内他们的设备可能处于非活动状态或未连接到网络，因此这部分用户的设备在这一轮的模型更新中不参与。
- 另外，某些用户可能由于隐私设置或其他限制，选择不参与数据共享或模型更新，因此也不被包括在特定的 FL 轮中。

因此，"部分客户端参与" 意味着在每个 FL 轮中，仅有部分客户端设备被选中参与模型训练或更新，而其他设备则可能在这一轮中暂时不参与。

- **场景设置：**每一轮只有部分客户端参与

  在CIFAR-10上进行了实验，其中有40个有偏见的客户和10个无偏见的客户。在每一轮中，我们随机抽取10名客户参加。

  ![image-20231115150008064](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231115150008064.png)

- **全局不平衡场景**：假设聚合的全局数据是不平衡的。

  1）全局类别分布是不均匀的，而测试数据集的分布是均匀的；

  $n_c=n_1p^{1-c/C-1}$

  - 实验设置：全局不平衡

  ![image-20231115152529858](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231115152529858.png)

  每个客户端按照NIID-1分布分配数据集，然后计算全局与客户端之间的差异并且进行FedDisco

  - 实验结果

  ![image-20231115151810811](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231115151810811.png)

  2） 全局类别分布和测试数据集的分布是相似的，并且都是非均匀的。
  
  ![image-20231115151601653](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231115151601653.png)

### 3.9 总结

​	本文重点研究了Fl中类别数据异构问题，传统的聚合模型方式都是基于数据集的大小，本文引入了差异值，提出FedDiso,==它为数据集大小较大，差异值较小的客户端分配较大的聚合权重==。

## 4. 参数表

![image-20231111154040492](D:\Desktop\研一\文献阅读\FedDisco Federated Learning with Discrepancy-Aware Collaboration\assets\image-20231111154040492.png)

​					C	一共有几个类别。