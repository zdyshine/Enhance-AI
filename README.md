# Enhance-AI
开源的图像/视频修复增强算法    

## 目录
  - [1. Image-enhance](#1-Image-enhance)
  - [2. Video-enhance](#2-Video-enhance)
  - [3. Video-Interpolation](#3-Video-Interpolation)

    
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
