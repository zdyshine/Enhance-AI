# Enhance-AI
开源的图像/视频修复增强算法    

## 目录
  - [1. Image-enhance](#1-Image-enhance)
  - [2. Video-enhance](#2-Video-enhance)
  - [3. Video-Interpolation](#3-Video-Interpolation)
  - [4. Diffusion-other](#4-Diffusion-other)
  - [5. BNN Restormer](#5-BNN-Restormer)

    
###  1. <a name='Image-enhance'></a>Image-enhance
#### 1.1 Real超分
* ResShift：
  * paper：-
  * code：https://github.com/zsyOAOA/ResShift
  * 简介：提出了一种新颖且高效的SR扩散模型，显著减少了扩散步骤的数量，从而消除了推理过程中的后加速需求及其相关的性能下降。此外，还开发了一个详细的噪声调度，以灵活地控制扩散过程中的平移速度和噪声强度
  * 测试效果: 速度很慢，且在工作场景效果差。

* GDSSR：
  * paper：https://ieeexplore.ieee.org/document/10041757
  * code：https://github.com/chiyich/GDSSR
  * 简介：使用了一个轻量级的全局退化提取器来提取全局退化特征，这可以方便地独立恢复更好的局部区域，并强制片间一致性。此外，我们还提出了一种局部和全局片段的联合训练方法，在训练过程中进行全局监督，这可以增强退化估计并恢复更加自然的结果。
  * 其他： 更加细致化的实现了以前的尝试即：两路输入，一路是全图，一路是局部图，用全图支路作为信息互补，监督局部图。

* DITN：
  * paper：https://arxiv.org/pdf/2308.02794.pdf
  * code：https://github.com/yongliuy/DITN
  * 简介：我们提出了一种适用于部署的Transformer单元，即UFONE(即UnFolding ONce is Enough),以解决这些问题。在每个UFONE中，我们引入了一个内部补丁Transformer层(ITL),以有效地从补丁中重构局部结构信息，以及一个空间感知层(SAL),以利用补丁之间的长距离依赖关系。基于UFONE,我们为SISR任务提出了一种适用于部署的内部补丁Transformer网络(DITN),它可以在训练和部署平台上实现低延迟和低内存使用的良好性能。此外，为了进一步提高我们在TensorRT上部署DITN的效率，我们还为层归一化提供了一种有效的替代方案，并为特定操作符提出了融合优化策略
  * 其他： 可尝试在此基础上进行训练

* StarSRGAN：
  * paper：https://arxiv.org/pdf/2307.16169.pdf
  * code：https://github.com/kynthesis/StarSRGAN
  * 简介：通过实施最近的技术，仍然有改进Real-ESRGAN超分辨率质量的空间。本研究论文介绍了StarSRGAN，这是一种为盲超分辨率任务设计的新颖GAN模型，它利用了五种不同的架构。
  * 其他：StarSRGAN Lite提供大约7.5倍更快的重建速度（从540p实时上采样到4K），但仍然可以保持近90%的图像质量，从而为未来的实时SR体验的开发提供了便利。

* FeMaSR(MM22 Oral)：
  * paper：https://arxiv.org/pdf/2202.13142.pdf
  * code：https://github.com/chaofengc/FeMaSR
  * 简介：提出了特征匹配SR（FeMaSR），它在一个更紧凑的特征空间中恢复了逼真的HR图像。与图像空间方法不同，我们的FeMaSR通过将扭曲的LR图像“特征”匹配到我们的预训练HR先验中的无失真HR对应物，并通过解码匹配特征来获得逼真的HR图像来恢复HR图像。具体来说，我们的HR先验包含一个离散特征代码本及其相关的解码器，这些是在HR图像上使用向量量化生成对抗网络（VQGAN）预训练的。
  * 其他: 基于高清图构建字典，然后嵌入到超分网络中进行图像恢复。

* VCISR(WACV 2024)：
  * paper：https://arxiv.org/pdf/2311.00996.pdf
  * code：https://github.com/Kiteretsu77/VCISR-official
  * 简介：图片超分，基于混合退化方式，但是对图片进行切块处理，然后添加编码噪声:MPEG-2, MPEG-4, H.264, and H.265。暂未开源
  * 其他: 待测试。
  
* PromptSR(arXiv,2023)：
  * paper：https://arxiv.org/pdf/2311.14282.pdf
  * code：https://github.com/zhengchen1999/PromptSR
  * 简介：设计一个文本图像生成管道，通过文本退化表示和退化模型将文本集成到 SR 数据集中。文本表示采用基于分箱方法的离散化方式来抽象地描述退化。

#### 1.2 通用超分
* WaveMixSR：
  * paper：https://arxiv.org/pdf/2307.00430v1.pdf
  * code：https://github.com/pranavphoenix/WaveMixSR
  * 简介：提出了一种新的神经网络——WaveMixSR,用于基于WaveMix架构的图像超分辨率，它使用2D离散小波变换进行空间标记混合。与基于Transformer的模型不同，WaveMixSR不会将图像展开为像素/块序列。相反，它利用卷积的内积偏置以及小波变换的无损标记混合性质来实现更高的性能，同时需要较少的资源和训练数据。

* DegAE (CVPR 2023)：
  * paper：https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_DegAE_A_New_Pretraining_Paradigm_for_Low-Level_Vision_CVPR_2023_paper.pdf
  * code：None
  * 简介：提出了一种新的预训练范式，称为降级自编码器(DegAE)。DegAE遵循设计借口任务以进行自我监督预训练的思想，并专门针对低层视觉进行了精心定制(针对Low-Level任务)。

* neucam (CVPR 2023)：
  * paper：https://arxiv.org/pdf/2304.12748.pdf
  * code：None
  * 简介：提出了一种新的隐式相机模型，将相机的物理成像过程表示为一个深度神经网络。计了一个隐式模糊生成器和一个隐式调色板来分别模拟相机成像过程的孔径和曝光度。我们的隐式相机模型与多个焦点堆叠和多个曝光支架监督下的隐式场景模型一起联合学习。
  * 其他：neural representations相比目前流行的大模型，可以作为一个研究重点
 
* SRFormer (CVPR 2023)：
  * paper：https://arxiv.org/pdf/2303.09735.pdf
  * code：https://github.com/HVision-NKU/SRFormer
  * 简介：提出了 SRFormer在享受大窗口自注意力好处的同时引入了更少的计算负担。本文SRFormer的核心是置换自注意力 (PSA)，它在通道和空间信息之间取得了适当的平衡以进行自注意力。所提出的PSA很简单，可以很容易地应用于现有的基于窗口自注意力的超分辨率网络中
  * 其他：在本文中，作者并不是直接缩减的空间尺度，而是先缩小通道维度，之后把通道维的缩小迁移到空间维的缩小。这一即使HW维度变小了，但是由于是通过permute置换得到的，因此并没有丢失空间信息。这一设计也最终帮助实现了在增大窗口大小的同时不带来复杂度上升。

#### 1.3 文本超分
* TextDiff(AAAI2024)：
  * paper：https://arxiv.org/pdf/2308.06743v1.pdf
  * code：https://github.com/lenubolim/textdiff
  * 简介：包含两个模块：文本增强模块（TEM）和掩膜引导残差扩散模块（MRD）。TEM生成一个初始去模糊的文本图像和一个编码文本空间位置的掩膜。MRD负责通过建模原始图像与初始去模糊图像之间的残差来有效锐化文本边缘。

#### 1.4 人脸修复
* PGDiff (Arxiv 2024)：
  * paper：None
  * code：https://github.com/pq-yang/PGDiff
  * 简介：-


###  2. <a name='Video-enhance'></a>Video-enhance

#### 2.1 Real视频超分
* EAVSR (CVPRW 2023)：
  * paper：https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Wang_Benchmark_Dataset_and_Effective_Inter-Frame_Alignment_for_Real-World_Video_Super-Resolution_CVPRW_2023_paper.pdf
  * code：https://github.com/HITRainer/EAVSR
  * 简介：构建了一个真实世界×4 VSR数据集，其中分别使用智能手机的不同焦距镜头拍摄低分辨率和高分辨率视频。此外，提出了一种有效的真实世界VSR对齐方法，即EAVSR。

* SRWD-CVPR2023：
  * paper：https://arxiv.org/pdf/2305.02660.pdf
  * code：None
  * 简介：提出了在合成训练数据集中合成真实世界降质的方法。所提出的合成真实世界降质(SRWD)包括模糊、噪声、下采样、像素分箱和图像和视频压缩伪影的组合。然后，我们提出了使用基于随机洗牌的策略来模拟这些降质在训练数据集上的模拟，并在所提出的更真实的合成训练数据集上训练单个端到端深度神经网络(DNN)。

* VQD-SR ：
  * paper：https://arxiv.org/pdf/2303.09826.pdf
  * code：None
  * 简介：提出了一种多尺度向量量化退化模型，用于动画视频超分辨率（VQD-SR）。该模型能够从全局结构中分解局部细节，并将现实世界动画视频中的退化先验知识转移至学习到的向量量化码本中，用于退化建模。为了提取先验知识，我们收集了丰富的内容实景动画低质量（RAL）视频数据集。
  * 其他：动画视频超分辨率（VQD-SR）


#### 2.2 通用视频超分
* STDO-CVPR2023：
  * paper：https://arxiv.org/abs/2303.08331
  * code：https://github.com/coulsonlee/STDO-CVPR2023
  * 简介：该方法利用了空间-时间信息来准确地将视频划分为块，从而使块的数量和模型大小保持最小。此外，我们通过数据感知联合训练技术将我们的方法推进到单个过拟合模型中，这进一步降低了存储要求并几乎没有质量损失。我们在一个现成的移动电话上部署了我们的模型，方法能够在实时视频超分辨率的同时保持高视频质量。
 
* GBR-WNN-2023：
  * paper：https://arxiv.org/pdf/2106.07190.pdf
  * code：https://github.com/YounggjuuChoi/GBR-WNN
  * 简介：提出了一种基于组的双向循环小波神经网络(GBR-WNN),有效地利用了时空信息。
 
* HiRN-2023：
  * paper：https://www.sciencedirect.com/science/article/abs/pii/S1568494623004404
  * code：https://github.com/YounggjuuChoi/HiRN
  * 简介：提出了基于特征演变的分层递归神经网络(HiRN)。所提出的HiRN是基于分层递归传播和基于时间小波注意力(TWA)模块的残差块骨干网络设计的。分层递归传播包含两个阶段，以结合基于低帧率的前向和后向方案的优势和多帧率的双向访问结构。

* FTVSR (ECCV 2022)：
  * paper：https://arxiv.org/pdf/2208.03012.pdf
  * code：https://github.com/researchmm/FTVSR
  * 简介：提出了一种新的频率变换器(FTVSR),用于压缩视频超分辨率，该变换器在空间-时间-频率联合域上执行自注意力。在细节的恢复上看起来比BasicVSR好，待实测。
    
* EgoVSR (ECCV 2022)：
  * paper：https://arxiv.org/pdf/2305.14708.pdf
  * code：https://github.com/chiyich/EGOVSR
  * 简介：由于捕获设备和场景的限制，自中心视频通常具有较低的视觉质量，主要由高压缩和严重的运动模糊引起，现有的针对第三人称视角视频的视频超分辨率(VSR)工作实际上并不适合处理由快速自我运动和物体运动引起的模糊伪影。为此，提出了一种专门针对自中心视频的VSR框架EgoVSR。在VSR框架中明确使用双分支去模糊网络(DB2Net)解决自中心视频中的运动模糊问题。
 
###  3. <a name='Video-Interpolation'></a>Video-Interpolation
* Exploring-Discontinuity-for-VFI (CVPR 2023)：
  * paper：https://openaccess.thecvf.com/content/CVPR2023/papers/Lee_Exploring_Discontinuity_for_Video_Frame_Interpolation_CVPR_2023_paper.pdf
  * code：https://github.com/pandatimo/Exploring-Discontinuity-for-VFI
  * 简介：提出了三项技术，可以使现有基于深度学习的视频流推断(VFI)架构对包含各种不自然物体和不连续运动的实际视频具有鲁棒性。
  * 测试说明：

###  4. <a name='Diffusion'></a>Diffusion
* ShadowDiffusion (CVPR 2023)：
  * paper：https://arxiv.org/pdf/2212.04711.pdf
  * code：https://github.com/GuoLanqing/ShadowDiffusion
  * 简介：提出一种统一的扩散框架，名为ShadowDiffusion，来解决阴影去除问题。该框架整合了图像和退化先验，以实现高效的阴影去除。ShadowDiffusion逐步优化估计的阴影掩码，作为扩散生成器的辅助任务，从而生成更准确、更鲁棒的无阴影图像。

* MCGdiff：
  * paper：https://arxiv.org/pdf/2308.07983.pdf
  * code：https://anonymous.4open.science/r/mcgdiff/README.md
  * 简介：一种求解具有SGM先验的贝叶斯线性高斯逆问题的新方法，可用于图像修复、超分辨率、去模糊和着色等，待尝试
* CFTL：
  * paper：https://openreview.net/pdf/a0c1eaf9320c504f4fb60a2e480793af66eb2b79.pdf
  * code：will
  * 简介：一种图像增强的新视角，(通道维度傅里叶变换)，与增强网络无缝集成涨点,对多种图像增强任务（如暗光图像增强、曝光校正、SDR2HDR 转换和水下图像增强）

###  5. <a name='BNN-Restormer'></a>BNN-Restormer
* BiSCI (NeurIPS 2023)：
  * paper：https://arxiv.org/pdf/2305.10299.pdf
  * code：https://github.com/caiyuanhao1998/BiSCI
  * 简介：二值化光谱重建算法。
* BBCU（ICLR2023）：
  * paper：https://arxiv.org/pdf/2210.00405.pdf
  * code：https://github.com/Zj-BinXia/BBCU
  * 简介：二值化图像恢复网络的基本二元卷积单元。

