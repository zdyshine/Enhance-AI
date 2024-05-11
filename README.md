# Enhance-AI
开源的图像/视频修复增强算法    

### 图像修复/增强
#### 包括去噪，去雨，去雾等底层视觉
| time | paper | code | Abstract | task |
| --- | --- | --- | --- | --- |
| 2404 | [MambaUIE](https://arxiv.org/ftp/arxiv/papers/2404/2404.13884.pdf) | [code](https://github.com/1024AILab/MambaUIE) | 基于mamba, 2.8 FLOPs | 水下图增强 |
| 2404 | [FreqMamba](https://arxiv.org/pdf/2404.09476.pdf) | [code](none) | 基于mamba | 去雨 |
| 2404 | [HSIDMamba](https://arxiv.org/pdf/2404.09697.pdf) | [code](none) | 选择性空间模型 | 高光谱图片修复 |
| 2404 | [Mansformer](https://arxiv.org/pdf/2404.06135.pdf) | [code](none) | gated-dconv MLP | 去模糊 |
| 2404 | [CodeEnhance](https://arxiv.org/pdf/2404.05253.pdf) | [code](none) | 利用量化先验知识 | 暗光增强 |
| 2403 | [ALGNet](https://arxiv.org/pdf/2403.20106.pdf) | [code](https://github.com/Tombs98/ALGNet) | 高效指标高 | 去模糊 |
| 2403 | [VmambaIR](https://arxiv.org/pdf/2403.11423.pdf) | [code](https://github.com/AlphacatPlus/VmambaIR) | 基于mamba | 修复 |

#### 超分
| time | paper | code | Abstract | task | Other |
| --- | --- | --- | --- | --- | --- |
| 2404 | [PLKSR](https://arxiv.org/pdf/2404.11848.pdf) | [code](https://github.com/dslisleedh/PLKSR) | 大核，高效 | ISR | 待测试 |
| CVPR2024 | [AdaBM](https://arxiv.org/abs/2404.03296.pdf) | [code](https://github.com/Cheeun/AdaBM) | 适应量化位宽成功地降低了推理和存储成本 | ISR | 待测试 |
| 2404 | [AddSR](https://arxiv.org/pdf/2404.01717.pdf) | [code](https://github.com/NJU-PCALab/AddSR) | 蒸馏加速，感知-失真不平衡 | Real-ISR | 视觉对比上比之前的好一些，待测试 |
| 2404 | [DeeDSR](https://arxiv.org/pdf/2404.00661.pdf) | [code](https://github.com/bichunyang419/DeeDSR) | 蒸馏加速，感知-失真不平衡 | Real-ISR | 视觉对比上比之前的好一些，待测试 |
| 2404 | [Inf-DiT](https://arxiv.org/pdf/2405.04312) | [code](https://github.com/thudm/inf-dit) | DiT 结构，无限放大 | 尝试修改为Real-SR | 待实现 |


### 视频修复/增强
| time | paper | code | Abstract | task |
| --- | --- | --- | --- | --- |
| 2404 | [ColorMNet](https://arxiv.org/pdf/2404.06251) | [code](https://github.com/yyang181/colormnet) | 多个模块利用时间序列信息，高效| 视频上色 |
| 2404 | [RStab](https://arxiv.org/pdf/2404.12887.pdf) | [code](None) | 视频生成+VSR模型| 视频超分 |
| 2404 | [VideoGigaGAN](https://arxiv.org/pdf/2404.12388.pdf) | [code](https://videogigagan.github.io/) | 3D多帧视角来生成稳定的图像| 视频稳定 |
| 2404 | [CFDVSR](https://arxiv.org/pdf/2404.06036.pdf) | [code](https://github.com/House-Leo/CFDVSR) | 提出一个改进帧间对齐的模块，可插入到BasicVSR中 | 视频超分 |


## 目录
  - [1. Image-enhance](#1-Image-enhance)
  - [2. Video-enhance](#2-Video-enhance)
  - [3. Video-Interpolation](#3-Video-Interpolation)
  - [4. Diffusion-other](#4-Diffusion-other)
  - [5. BNN Restormer](#5-BNN-Restormer)
  - [6. All-In-One Restormer](#6-All-In-One-Restormer)

## 数据集
    https://github.com/IndigoPurple/ART    
    https://github.com/KAIST-VICLab/FMA-Net
    https://github.com/INVOKERer/AdaRevD
###  1. <a name='Image-enhance'></a>Image-enhance
#### 1.1 Real超分
* SED (CVPR 2024)：
  * paper：https://arxiv.org/pdf/2402.19387.pdf
  * code：https://github.com/lbc12345/SeD
  * 简介：新的判别器，可尝试。提出了简单有效的语义感知判别器（表示为 SeD），它鼓励 SR 网络通过引入图像语义作为条件来学习细粒度分布。
  * 
* APISR (CVPR 2024)：
  * paper：https://arxiv.org/pdf/2403.01598.pdf
  * code：https://github.com/Kiteretsu77/APISR
  * 简介：动漫超分，基于图像级，退化使用混合退化+视频压缩，训练使用多个混合损失，并针对动漫GT进行纹理增强处理。
* CAMixerSR (arXiv 202402)：
  * paper：https://arxiv.org/pdf/2402.19289.pdf
  * code：https://github.com/icandle/camixersr
  * 简介：一种集成内容感知加速框架和令牌混合器设计的新方法，通过为简单区域分配卷积但为复杂纹理分配窗口注意力来追求更高效的 SR 推理。

* SUPIR (arXiv 202401)：
  * paper：https://arxiv.org/pdf/2401.13627.pdf
  * code：https://supir.xpixel.group/
  * 简介：收集了一个包含2000万张高分辨率、高质量图像的数据集，用于模型训练，每张图像都添加了描述性文本注释。SUPIR提供了由文本提示引导的图像恢复功能，拓宽了其应用范围和潜力。此外，他们还引入了负质量提示，以进一步提高感知质量。他们还开发了一种修复引导采样方法，以抑制在基于生成的修复中遇到的保真度问题。
    
* CCSR(arXiv 202401)：
  * paper：https://arxiv.org/pdf/2401.00877.pdf
  * code：https://github.com/csslc/CCSR
  * 简介：内容一致超分辨率（CCSR）提出了一种非均匀时间步长的学习策略来训练紧凑的扩散网络，具有高效率和稳定性以再现图像主要结构，并微调变分自动编码器的预训练解码器（VAE）通过对抗性训练增强细节。可以显著降低基于扩散先验的随机共振的随机性。改善SR输出的内容一致性并加速图像生成过程.
    
* SeeSR：
  * paper：https://arxiv.org/pdf/2311.16518.pdf
  * code：https://github.com/cswry/seesr
  * 简介：训练了一个退化感知的提示提取器，即使在严重的退化下也能生成准确 soft 和 hard 语义提示。hard 语义提示指的是图像标签，旨在增强T2I模型的局部感知能力，而soft 语义提示则是为了补充hard 提示提供额外的表示信息。这些语义提示可以鼓励T2I模型生成详细且语义准确的结果。

* promptsr：
  * paper：https://arxiv.org/pdf/2311.14282.pdf
  * code：https://github.com/zhengchen1999/promptsr
  * 简介：设计了一个文本-图像生成管道，通过文本退化表示和退化模型将文本集成到SR数据集中。文本表示采用基于分箱方法的离散化方式来抽象地描述退化。这种表示方法还可以保持语言的灵活性。同时，我们提出了PromptSR来实现文本提示SR。PromptSR采用了扩散模型和预训练的语言模型（例如T5和CLIP）
    
* coser：
  * paper：https://arxiv.org/pdf/2311.16512.pdf
  * code：https://github.com/vinhyu/coser
  * 简介：该方法通过将图像外观和语言理解相结合，生成了一种认知嵌入，使SR模型能够理解低分辨率图像。为了进一步提高图像保真度，CoSeR提出了一种名为"All-in-Attention"的新条件注入方案，将所有条件信息整合到一个模块中。

* StableSR：
  * paper：https://arxiv.org/pdf/2305.07015.pdf
  * code：https://github.com/IceClear/StableSR
  * 简介：提出了一种新颖的方法，利用预训练的文本到图像扩散模型中包含的先验知识进行盲超分辨率（SR）。具体而言，通过使用我们的时间感知编码器，我们可以在不改变预训练的合成模型的情况下实现有希望的恢复结果，从而保留生成先验并最小化训练成本。为了弥补扩散模型固有的随机性导致的保真度损失，我们采用了一种可控特征包装模块，该模块允许用户通过在推理过程中简单地调整一个标量值来平衡质量和保真度。


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

* SinSR(arXiv,2023)：
  * paper：https://github.com/wyf0912/SinSR/blob/main/main.pdf
  * code：https://github.com/wyf0912/SinSR
  * 简介：一种单步基于扩散的图像超分辨率新网络，与之前的 SOTA 方法和教师模型相比，本方法只需一个采样步骤即可实现可比甚至更优越的性能，从而实现高达 10 倍的推理加速

* ITER(AAAI,2024)：
  * paper：-
  * code：https://github.com/chaofengc/ITER
  * 简介：一种单步基于扩散的图像超分辨率新网络，与之前的 SOTA 方法和教师模型相比，本方法只需一个采样步骤即可实现可比甚至更优越的性能，从而实现高达 10 倍的推理加速

#### 1.2 通用超分
* ATDSR：
  * paper：https://arxiv.org/pdf/2401.08209.pdf
  * code：https://github.com/LabShuHangGU/Adaptive-Token-Dictionary
  * 简介：向SR Transformer引入了一组辅助的自适应令牌字典，并建立了ATD -SR方法。引入的标记字典可以从训练数据中学习先验信息，并通过自适应细化步骤将学习到的先验信息适应特定的测试图像。细化策略不仅可以为所有输入标记提供全局信息，还可以将图像标记分组为类别。基于类别划分，我们进一步提出了一种基于类别的自注意力机制，旨在利用遥远但相似的标记来增强输入特征。细节还原好

* seemoredetails：
  * paper：https://arxiv.org/pdf/2402.03412.pdf
  * code：https://github.com/eduardzamfir/seemoredetails
  * 简介：我们引入了Seemo Re ，这是一种采用专家挖掘的高效 SR 模型。

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

* BFRffusion(arXiv 202312)：
  * paper：https://arxiv.org/pdf/2312.15736.pdf
  * code：https://github.com/chenxx89/BFRffusion
  * 简介：提出了一种BFRffusion算法，该算法能够有效地从低质量的人脸图像中提取特征，并利用预训练的Stable Diffusion算法的生成先验知识，恢复出真实可信的人脸细节。建立了一个具有平衡的种族、性别和年龄等属性的隐私保护人脸数据集PFHQ。
    
* PGDiff (Arxiv 2023)：
  * paper：None
  * code：https://github.com/pq-yang/PGDiff
  * 简介：-


###  2. <a name='Video-enhance'></a>Video-enhance

#### 2.1 Real视频超分
* FMA-Net(CVPR2024)：
  * paper：https://arxiv.org/pdf/2401.03707.pdf
  * code：https://github.com/KAIST-VICLab/FMA-Net
  * 简介：提出了一种新的框架，用于处理视频中的各种低级视觉任务，如去噪、物体去除、帧内插和超分辨率。

* Video Dynamics Prior ：
  * paper：https://arxiv.org/pdf/2312.07835.pdf
  * code：-
  * 简介：提出了一种新联合超分和去模糊的框架。
    
* Semantic-Lens-AAAI24 ：
  * paper：https://arxiv.org/pdf/2312.09909.pdf
  * code：https://github.com/Tang1705/Semantic-Lens-AAAI24
  * 简介：从退化的视频中提取语义先验来解决视频中复杂的运动交织问题，从而提高整体性能。
  * 
* Upscale-A-Video ：
  * paper：https://arxiv.org/pdf/2312.06640.pdf
  * code：https://github.com/sczhou/Upscale-A-Video
  * 简介：基于扩散模型的时序稳定的Real-World超分

* StableVSR：
  * paper：https://arxiv.org/pdf/2311.15908.pdf
  * code：https://github.com/claudiom4sir/stablevsr
  * 简介：通过引入时间条件模块（TCM），将用于单幅图像超分辨率的预训练DM转换为VSR方法来实现这一点。TCM采用时间纹理引导，其提供来自相邻帧的空间对齐和详细的纹理信息，以引导当前帧的生成过程朝向高质量和时间上一致的结果。此外，我们引入了逐帧双向采样策略，该策略鼓励使用来自过去和未来帧的信息，从而提高结果的感知质量和帧间的时间一致性

 * MGLD-VSR：
  * paper：https://arxiv.org/pdf/2312.00853.pdf
  * code：https://github.com/IanYeung/MGLD-VSR
  * 简介：利用预先训练的潜在扩散模型并结合具有运动引导损失的时间模块，该算法可以生成保持连贯和连续视觉流的高质量HR视频
  * 
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
* TMP：
  * paper：https://arxiv.org/pdf/2312.09909.pdf
  * code：https://github.com/xtudbxk/tmp
  * 简介：提出了一种有效的时间运动传播（TMP）方法，该方法利用运动场的连续性来实现连续帧之间的快速像素级对齐。具体来说，我们首先将先前帧的偏移传播到当前帧，然后在邻域中对其进行细化，这显著减少了匹配空间并加快了偏移估计过程。此外，为了增强对齐的鲁棒性，我们对扭曲的特征进行空间加权，其中具有更精确偏移的位置被赋予更高的重要性
    
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

* QuantSR (NeurIPS 2023 spotlight) ：
  * paper：https://openreview.net/pdf?id=3gamyee9Yh
  * code：https://github.com/htqin/QuantSR
  * 简介：二值化图像恢复网络的基本二元卷积单元。

###  6. <a name='All-In-One Restormer'></a>All-In-One Restormer
* NeRD-Rain (CVPR 2024)：
  * paper：-
  * code：https://github.com/cschenxiang/NeRD-Rain
  * 简介：尝试在其他框架上使用

* U-WADN (Arxiv 202401)：
  * paper：https://arxiv.org/pdf/2401.13221.pdf
  * code：https://github.com/xuyimin0926/U-WADN
  * 简介：提出了U-WADN框架，该框架包括两个主要组成部分:1.**宽度自适应骨干网（WAB）**：该组件包括多个具有不同宽度的嵌套子网络。其目的是为每个特定任务选择最合适的计算，旨在平衡运行时的准确性和计算效率。2.**宽度选择器（WS）**：对于不同的输入图像，WS自动选择最佳子网宽度。

