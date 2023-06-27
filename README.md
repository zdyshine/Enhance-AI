# Enhance-AI
开源的图像/视频修复增强算法    
arXiv:YYMM.numbervV

## 目录
  - [1. Image-enhance](#1-Image-enhance)
  - [2. Video-enhance](#2-Video-enhance)

    
###  1. <a name='Image-enhance'></a>Image-enhance

###  2. <a name='Video-enhance'></a>Video-enhance

#### 2.1 视频超分
* STDO-CVPR2023：
  * paper：https://arxiv.org/abs/2303.08331
  * code：https://github.com/coulsonlee/STDO-CVPR2023
  * 简介：该方法利用了空间-时间信息来准确地将视频划分为块，从而使块的数量和模型大小保持最小。此外，我们通过数据感知联合训练技术将我们的方法推进到单个过拟合模型中，这进一步降低了存储要求并几乎没有质量损失。我们在一个现成的移动电话上部署了我们的模型，方法能够在实时视频超分辨率的同时保持高视频质量。

* EAVSR (CVPRW 2023)：
  * paper：https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Wang_Benchmark_Dataset_and_Effective_Inter-Frame_Alignment_for_Real-World_Video_Super-Resolution_CVPRW_2023_paper.pdf
  * code：https://github.com/HITRainer/EAVSR
  * 简介：构建了一个真实世界×4 VSR数据集，其中分别使用智能手机的不同焦距镜头拍摄低分辨率和高分辨率视频。此外，提出了一种有效的真实世界VSR对齐方法，即EAVSR。
 
* GBR-WNN-2023：
  * paper：https://arxiv.org/pdf/2106.07190.pdf
  * code：https://github.com/YounggjuuChoi/GBR-WNN
  * 简介：提出了一种基于组的双向循环小波神经网络(GBR-WNN),有效地利用了时空信息。
 
* HiRN-2023：
  * paper：https://www.sciencedirect.com/science/article/abs/pii/S1568494623004404
  * code：https://github.com/YounggjuuChoi/HiRN
  * 简介：提出了基于特征演变的分层递归神经网络(HiRN)。所提出的HiRN是基于分层递归传播和基于时间小波注意力(TWA)模块的残差块骨干网络设计的。分层递归传播包含两个阶段，以结合基于低帧率的前向和后向方案的优势和多帧率的双向访问结构。
