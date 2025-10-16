# AI推理加速技术论文分析报告
生成时间: 2025-10-16 10:21:43
分析论文数量: 28篇

## 论文技术简报

### 1. An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution

相关机构发布了An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution论文，使用模板感知动态卷积技术，解决了深度模板匹配与平面内位姿估计问题，达成了高效的性能提升。

### 2. BIOX-BRIDGE: MODEL BRIDGING FOR UNSUPERVISED CROSS-MODAL KNOWLEDGE TRANSFER ACROSS BIOSIGNALS

牛津大学发布了BIOX-BRIDGE论文，使用轻量级桥接网络（含对齐位置选择策略和原型网络），解决了无监督跨模态知识迁移中知识蒸馏导致的计算内存开销大问题，达成减少88-99%可训练参数且保持或提升迁移性能的效果

### 3. Budgeted Broadcast: An Activity-Dependent Pruning Rule for Neural Network Efficiency

Harvard University发布了Budgeted Broadcast论文，使用基于本地流量预算（长期激活率与扇出乘积）的活性依赖剪枝规则，解决了传统剪枝方法仅按损失影响移除参数的局限，在ASR的Transformers、人脸识别的ResNets等多种模型上相同稀疏度下提高编码熵、去相关性和准确率（有时超密集基线），并在电子显微镜图像上达成SOTA F1和PR-AUC。

### 4. ClustViT: Clustering-based Token Merging for Semantic Segmentation

发布了ClustViT论文，使用含可训练Cluster模块（基于分割掩码伪簇引导合并相似token）和Regenerator模块（恢复细节）的架构，解决了Vision Transformers在语义分割等密集预测任务中因二次注意力复杂度导致实用性受限的问题，达成了减少2.18倍GFLOPs、推理速度提升1.64倍且保持相当分割精度的效果。

### 5. CONSTRAINED ADAPTIVE REJECTION SAMPLING

University of California San Diego发布了CONSTRAINED ADAPTIVE REJECTION SAMPLING论文，使用Constrained Adaptive Rejection Sampling (CARS)技术（通过trie记录并减去违反约束延续的概率质量以自适应排除无效前缀），解决了现有约束生成中有效性与分布保真度、计算效率的矛盾（贪婪方法扭曲分布，拒绝采样效率低），达成了提高样本效率（减少每个有效样本的LM前向传递次数）、增强样本多样性且不扭曲分布的效果。

### 6. Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning

Max Planck Institute for Intelligent Systems与Emory University发布了Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning论文，使用系统研究LLM各层在检索、知识和推理中作用的技术，解决了现有研究对LLM深层作用的窄化评估问题，达成了揭示LLM深度利用的异质性和情境依赖性，明确浅层负责知识与检索、深层对推理至关重要且可通过蒸馏重塑的效果。

### 7. EXPLORING RESOLUTION-WISE SHARED ATTENTION IN HYBRID MAMBA-U-NETS FOR IMPROVED CROSS-CORPUS SPEECH ENHANCEMENT

Aalborg University发布了相关论文，使用RWSA-MambaUNet模型（融合Mamba、多头注意力的混合U-Net结构，采用分辨率级共享注意力），解决了跨语料库语音增强的泛化性能问题，达成了在两个域外测试集上实现最先进泛化性能，最小模型超越所有基线且参数和计算量显著降低的效果

### 8. FIDEDIFF: EFFICIENT DIFFUSION MODEL FOR HIGH-FIDELITY IMAGE MOTION DEBLURRING

Harvard University和Shanghai Jiao Tong University发布了FideDiff论文，使用单步扩散模型及一致性模型（通过将运动去模糊重构为扩散过程、匹配模糊轨迹的训练数据重建、集成Kernel ControlNet和自适应时间步预测增强），解决了扩散模型推理时间长和保真度低的问题，达成了在全参考指标上性能超越先前扩散方法并匹配其他最先进模型的效果

### 9. HISPEC: HIERARCHICAL SPECULATIVE DECODING FOR LLMS

The University of Texas at Austin发布了HISPEC论文，使用利用早退出模型进行低开销中间验证并重用关键资源的分层推测解码框架，解决了LLM推测解码中验证瓶颈及现有中间验证方法的训练开销大、内存增加和精度妥协问题，达成了平均吞吐量提升1.28倍、最高2.01倍且不影响精度的效果。

### 10. JaneEye: A 12-nm 2K-FPS 18.9-µJ/Frame Event-based Eye Tracking Accelerator

Leiden University和Delft University of Technology发布了JaneEye论文，使用带新型ConvJANET层的超轻量神经网络架构及12nm ASIC实现的基于事件的眼动追踪硬件加速器技术，解决了传统帧基眼动追踪难以满足XR高精度、低延迟、高能效需求的问题，达成2000 FPS、18.9 µJ/frame能效及2.45像素误差的效果。

### 11. LiLa-Net: Lightweight Latent LiDAR Autoencoder for 3D Point Cloud Reconstruction

发布了LiLa-Net: Lightweight Latent LiDAR Autoencoder for 3D Point Cloud Reconstruction论文，使用轻量级潜在LiDAR自动编码器（LiLa-Net）技术，解决了仅用LiDAR点云时通过减少编码器层和简化跳跃连接降低资源消耗同时实现准确3D点云重建的问题，达成了提升重建质量且不影响性能、具备强泛化能力能重建无关物体的效果。

### 12. Local Linear Attention: An Optimal Interpolation of Linear and Softmax Attention For Test-Time Regression

华盛顿大学发布了Local Linear Attention论文，使用LLA注意力机制及FlashLLA算法，解决了线性和Softmax注意力在测试时回归中的计算复杂度与表达能力问题，在测试时训练和上下文内学习中优于强基线并展现可扩展性。

### 13. Mamba Outpaces Reformer in Stock Prediction with Sentiments from Top Ten LLMs

美国北德克萨斯大学发布了Mamba在股票预测中超越Reformer的论文，使用结合十大LLM语义情感分数与分钟级股票数据的新框架及Mamba模型，解决了股票市场短期预测难（高波动性、新闻影响、非线性时间序列）的问题，达成了Mamba相比Reformer速度更快且预测更准确（LLaMA 3.3-70B时MSE最低0.137）的效果

### 14. Nav-EE: Navigation-Guided Early Exiting for Efficient Vision-Language Models in Autonomous Driving

香港城市大学发布了Nav-EE论文，使用导航引导的早期退出框架（离线预计算任务特定退出层并在线基于导航先验动态应用），解决了自动驾驶中视觉语言模型（VLMs）推理延迟高且早期退出泛化受限的问题，达成了准确率接近全推理的同时将延迟减少高达63.9%（实车集成中延迟从600ms降至300ms）的效果。

### 15. OPTIMAL STOPPING VS BEST-OF-N FOR INFERENCE TIME OPTIMIZATION

University of Michigan等机构发布了《OPTIMAL STOPPING VS BEST-OF-N FOR INFERENCE TIME OPTIMIZATION》论文，使用基于潘多拉魔盒问题的UCB风格算法及Bradley-Terry启发的自适应推理时间优化方法，解决了LLM多代生成时输出质量与推理成本的平衡问题，达成了与Best-of-N采样相同性能且平均减少15-35%生成次数的效果。

### 16. Pure-Pass: Fine-Grained, Adaptive Masking for Dynamic Token-Mixing Routing in Lightweight Image Super-Resolution

南京大学发布了Pure-Pass论文，使用Pure-Pass像素级掩蔽机制（通过固定颜色中心点分类像素实现细粒度、空间灵活且自适应的掩蔽），解决了现有轻量级图像超分辨率方法适应性差、粗粒度掩蔽及空间不灵活的问题，达成在节省类似计算量时，集成于ATD-light模型的PP-ATD-light重建质量和参数效率优于CAMixer-ATD-light的效果。

### 17. ReSSFormer: A Recursive Sparse Structured Transformer for Scalable and Long-Context Reasoning

Columbia University New York发布了ReSSFormer论文，使用集成递归推理记忆单元、自适应稀疏注意力模块和自组织编码器结构的递归稀疏结构化Transformer，解决了Transformer的长上下文推理、计算效率和结构泛化挑战，在可比计算量和参数预算下持续优于强基线，展现出可扩展性、效率和结构灵活性。

### 18. RETHINKING THE SHAPE CONVENTION OF AN MLP

MediaTek Research发布了“RETHINKING THE SHAPE CONVENTION OF AN MLP”论文，使用提出的wide-narrow-wide（Hourglass）MLP块（跳跃连接在扩展维度操作、残差通过窄瓶颈流动，且初始投影固定随机初始化），挑战了传统MLP的narrow-wide-narrow设计，在生成任务上相比传统设计实现了更优的性能-参数Pareto前沿。

### 19. RSAVQ: Riemannian Sensitivity-Aware Vector Quantization for Large Language Models

发布了RSAVQ论文，使用含Error Direction Sensitivity Guidance (EDSG)和Weight Channel Sensitivity Guidance (WCSG)的Riemannian Sensitivity-Aware Vector Quantization框架，解决了现有向量量化在大语言模型极低比特量化中面临的无约束方向误差和次优比特分配问题，在LLaMA-3 8B的2位量化中，比VPTQ和QuIP#等基线困惑度（PPL）提升0.4，零样本准确率提升1.5。

### 20. Self-Forcing++: Towards Minute-Scale High-Quality Video Generation

University of Central Florida发布了Self-Forcing++论文，使用利用教师模型知识通过自生成长视频采样片段指导学生模型的方法，解决了长视频生成中教师模型无法合成长视频导致的外推质量下降与误差累积问题，达成视频长度扩展至教师能力20倍、生成4分15秒视频（比基线长50多倍）且保真度和一致性显著提升的效果。

### 21. SHIFT-INVARIANT ATTRIBUTE SCORING FOR KOLMOGOROV-ARNOLD NETWORKS VIA SHAPLEY VALUE

新加坡国立大学发布了关于Kolmogorov-Arnold Networks（KANs）的论文，使用ShapKAN框架（基于Shapley值归因的平移不变节点重要性评估技术），解决了KAN网络剪枝中传统幅度方法对输入坐标偏移敏感的问题，达成了保留真实节点重要性并实现有效网络压缩、提升可解释性的效果。

### 22. Sparse Query Attention (SQA): A Computationally Efficient Attention Mechanism with Query Heads Reduction

研究团队发布了Sparse Query Attention (SQA)论文，通过减少查询头的稀疏查询注意力机制，解决了Transformer中MHA的二次计算复杂度及训练等场景的FLOPs瓶颈，在计算密集场景中实现吞吐量提升高达3倍且对模型质量影响极小。

### 23. SPUS: A Lightweight and Parameter-Efficient Foundation Model for PDEs

Los Alamos National Laboratory发布了SPUS: A Lightweight and Parameter-Efficient Foundation Model for PDEs论文，使用轻量级残差U-Net架构及自回归预训练策略，解决了现有PDE基础模型参数与计算开销高的问题，达成了在多样下游PDE任务上实现SOTA泛化且参数显著减少、微调数据需求低的效果。

### 24. SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs

武汉理工大学与中国人民大学发布的SSTAG论文，采用结构感知的自监督学习方法，解决了文本属性图中结构与文本信息融合不足的问题，显著提升了文本属性图的表示学习性能。

### 25. THE DISPARATE IMPACTS OF SPECULATIVE DECODING

University of Virginia发布了THE DISPARATE IMPACTS OF SPECULATIVE DECODING论文，使用针对投机解码的缓解策略，解决了其在不同任务上加速不均（尤其对欠拟合、代表性不足任务加速减少）的问题，达成了平均12%的公平性指标提升效果。

### 26. THE UNSEEN FRONTIER: PUSHING THE LIMITS OF LLM SPARSITY WITH SURROGATE-FREE ADMM

发布了《THE UNSEEN FRONTIER: PUSHING THE LIMITS OF LLM SPARSITY WITH SURROGATE-FREE ADMM》论文，使用ELSA技术（基于无代理ADMM的约束优化），解决了传统方法在大语言模型稀疏化中难以突破50-60%稀疏度且不严重降低精度的问题，达成了高达90%的极端稀疏度同时保持高模型保真度，在LLaMA-2-7B上90%稀疏度时困惑度比现有最佳方法低7.8倍，并推出可扩展至27B大模型的量化变体ELSA-L。

### 27. Ultra-Efficient Decoding for End-to-End Neural Compression and Reconstruction

爱荷华州立大学艾姆斯分校发布了《端到端神经压缩与重建的超高效解码》论文，使用在带矢量量化的自编码器中融入低秩表示的技术，解决了神经压缩中卷积基解码器重建时的高计算成本问题，达成了显著降低解码计算开销、消除解码器计算瓶颈并保持图像高保真度的效果。

### 28. VideoNSA: Native Sparse Attention Scales Video Understanding VIDEONSA: NATIVE SPARSE ATTENTION SCALES VIDEO UNDERSTANDING

纽约大学、普林斯顿大学发布了VideoNSA论文，使用硬件感知混合注意力（文本密集+视频Native Sparse Attention）技术，解决了多模态语言模型视频理解受上下文长度限制（错过关键帧、长时间尺度连贯性差）的问题，达成了长视频理解、时间推理和空间基准性能提升并可靠扩展到128K tokens的效果。

## 论文详细信息

### 1. An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution

**主要机构**: 
**作者数量**: 6人

**摘要**:


### 2. BIOX-BRIDGE: MODEL BRIDGING FOR UNSUPERVISED CROSS-MODAL KNOWLEDGE TRANSFER ACROSS BIOSIGNALS

**主要机构**: University of Oxford, Department of Engineering Science
**作者数量**: 4人

**摘要**:
Biosignals offer valuable insights into the physiological states of the human body. Although biosignal modalities differ in functionality, signal fidelity, sensor comfort, and cost, they are often intercorrelated, reflecting the holistic and interconnected nature of human physiology. This opens up the possibility of performing the same tasks using alternative biosignal modalities, thereby improving the accessibility, usability, and adaptability of health monitoring systems. However, the limited availability of large labeled datasets presents challenges for training models tailored to specific tasks and modalities of interest. Unsupervised cross-modal knowledge transfer offers a promising solution by leveraging knowledge from an existing modality to support model training for a new modality. Existing methods are typically based on knowledge distillation, which requires running a teacher model alongside student model training, resulting in high computational and memory overhead. This challenge is further exacerbated by the recent development of foundation models that demonstrate superior performance and generalization across tasks at the cost of large model sizes. To this end, we explore a new framework for unsupervised cross-modal knowledge transfer of biosignals by training a lightweight bridge network to align the intermediate representations and enable information flow between foundation models and across modalities. Specifically, we introduce an efficient strategy for selecting alignment positions where the bridge should be constructed, along with a flexible prototype network as the bridge architecture. Extensive experiments across multiple biosignal modalities, tasks, and datasets show that BioX-Bridge reduces the number of trainable parameters by 88-99% while maintaining or even improving transfer performance compared to state-of-the-art methods.

### 3. Budgeted Broadcast: An Activity-Dependent Pruning Rule for Neural Network Efficiency

**主要机构**: Harvard University
**作者数量**: 6人

**摘要**:
Most pruning methods remove parameters ranked by impact on loss (e.g., magnitude or gradient). We propose Budgeted Broadcast (BB), which gives each unit a local traffic budget-the product of its long-term on-rate 𝑎 𝑖 and fan-out 𝑘 𝑖. A constrained-entropy analysis shows that maximizing coding entropy under a global traffic budget yields a selectivity-audience balance, log 1-𝑎 𝑖 𝑎 𝑖 = 𝛽𝑘 𝑖. BB enforces this balance with simple local actuators that prune either fan-in (to lower activity) or fan-out (to reduce broadcast). In practice, BB increases coding entropy and decorrelation and improves accuracy at matched sparsity across Transformers for ASR, ResNets for face identification, and 3D U-Nets for synapse prediction, sometimes exceeding dense baselines. On electron microscopy images, it attains state-of-the-art F1 and PR-AUC under our evaluation protocol. BB is easy to integrate and suggests a path towards learning more diverse and efficient representations.

### 4. ClustViT: Clustering-based Token Merging for Semantic Segmentation

**主要机构**: 
**作者数量**: 3人

**摘要**:
Vision Transformers can achieve high accuracy and strong generalization across various contexts, but their practical applicability on real-world robotic systems is limited due to their quadratic attention complexity. Recent works have focused on dynamically merging tokens according to the image complexity. Token merging works well for classification but is less suited to dense prediction. We propose ClustViT, where we expand upon the Vision Transformer (ViT) backbone and address semantic segmentation. Within our architecture, a trainable Cluster module merges similar tokens along the network guided by pseudo-clusters from segmentation masks. Subsequently, a Regenerator module restores fine details for downstream heads. Our approach achieves up to 2.18× fewer GFLOPs and 1.64× faster inference on three different datasets, with comparable segmentation accuracy. Our code and models will be made publicly available.

### 5. CONSTRAINED ADAPTIVE REJECTION SAMPLING

**主要机构**: University of Warsaw, University of California San Diego
**作者数量**: 4人

**摘要**:
Language Models (LMs) are increasingly used in applications where generated outputs must satisfy strict semantic or syntactic constraints. Existing approaches to constrained generation fall along a spectrum: greedy constrained decoding methods enforce validity during decoding but distort the LM's distribution, while rejection sampling (RS) preserves fidelity but wastes computation by discarding invalid outputs. Both extremes are problematic in domains such as program fuzzing, where both validity and diversity of samples are essential. We present Constrained Adaptive Rejection Sampling (CARS), an approach that strictly improves the sample-efficiency of RS without distributional distortion. CARS begins with unconstrained LM sampling and adaptively rules out constraint-violating continuations by recording them in a trie and subtracting their probability mass from future draws. This adaptive pruning ensures that prefixes proven invalid are never revisited, acceptance rates improve monotonically, and the resulting samples exactly follow the constrained distribution. In experiments on a variety of domains-e.g., program fuzzing and molecular generation-CARS consistently achieves higher efficiency-measured in the number of LM forward passes per valid sample-while also producing stronger sample diversity than both GCD and methods that approximate the LM's distribution.

### 6. Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning

**主要机构**: Emory University, University of Surrey, Max Planck Institute for Intelligent Systems, Hong Kong Polytechic University, University of Tuebingen
**作者数量**: 5人

**摘要**:
Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers-yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.

### 7. EXPLORING RESOLUTION-WISE SHARED ATTENTION IN HYBRID MAMBA-U-NETS FOR IMPROVED CROSS-CORPUS SPEECH ENHANCEMENT

**主要机构**: Aalborg University, Department of Electronic Systems
**作者数量**: 4人

**摘要**:
Recent advances in speech enhancement have shown that models combining Mamba and attention mechanisms yield superior crosscorpus generalization performance. At the same time, integrating Mamba in a U-Net structure has yielded state-of-the-art enhancement performance, while reducing both model size and computational complexity. Inspired by these insights, we propose RWSA-MambaUNet, a novel and efficient hybrid model combining Mamba and multi-head attention in a U-Net structure for improved crosscorpus performance. Resolution-wise shared attention (RWSA) refers to layerwise attention-sharing across corresponding time-and frequency resolutions. Our best-performing RWSA-MambaUNet model achieves state-of-the-art generalization performance on two out-of-domain test sets. Notably, our smallest model surpasses all baselines on the out-of-domain DNS 2020 test set in terms of PESQ, SSNR, and ESTOI, and on the out-of-domain EARS-WHAM v2 test set in terms of SSNR, ESTOI, and SI-SDR, while using less than half the model parameters and a fraction of the FLOPs.

### 8. FIDEDIFF: EFFICIENT DIFFUSION MODEL FOR HIGH-FIDELITY IMAGE MOTION DEBLURRING

**主要机构**: Harvard University, Shanghai Jiao Tong University
**作者数量**: 6人

**摘要**:
Recent advancements in image motion deblurring, driven by CNNs and transformers, have made significant progress. Large-scale pre-trained diffusion models, which are rich in true-world modeling, have shown great promise for high-quality image restoration tasks such as deblurring, demonstrating stronger generative capabilities than CNN and transformer-based methods. However, challenges such as unbearable inference time and compromised fidelity still limit the full potential of the diffusion models. To address this, we introduce FideDiff, a novel single-step diffusion model designed for high-fidelity deblurring. We reformulate motion deblurring as a diffusion-like process where each timestep represents a progressively blurred image, and we train a consistency model that aligns all timesteps to the same clean image. By reconstructing training data with matched blur trajectories, the model learns temporal consistency, enabling accurate one-step deblurring. We further enhance model performance by integrating Kernel ControlNet for blur kernel estimation and introducing adaptive timestep prediction. Our model achieves superior performance on full-reference metrics, surpassing previous diffusionbased methods and matching the performance of other state-of-the-art models. FideDiff offers a new direction for applying pre-trained diffusion models to highfidelity image restoration tasks, establishing a robust baseline for further advancing diffusion models in real-world industrial applications. Our dataset and code will be available at https://github.com/xyLiu339/FideDiff.

### 9. HISPEC: HIERARCHICAL SPECULATIVE DECODING FOR LLMS

**主要机构**: The University of Texas at Austin, Department of Electrical and Computer Engineering
**作者数量**: 3人

**摘要**:
Speculative decoding accelerates LLM inference by using a smaller draft model to speculate tokens that a larger target model verifies. Verification is often the bottleneck (e.g. verification is 4× slower than token generation when a 3B model speculates for a 70B target model), but most prior works focus only on accelerating drafting. "Intermediate" verification reduces verification time by discarding inaccurate draft tokens early, but existing methods incur substantial training overheads in incorporating the intermediate verifier, increase the memory footprint to orchestrate the intermediate verification step, and compromise accuracy by relying on approximate heuristics. We propose Hierarchical Speculative Decoding (HiSpec), a framework for highthroughput speculative decoding that exploits early-exit (EE) models for lowoverhead intermediate verification. EE models allow tokens to exit early by skipping layer traversal and are explicitly trained so that hidden states at selected layers can be interpreted, making them uniquely suited for intermediate verification without drastically increasing compute and memory overheads. To improve resource-efficiency even further, we design a methodology that enables HiSpec to re-use key-value caches and hidden states between the draft, intermediate verifier, and target models. To maintain accuracy, HiSpec periodically validates the draft tokens accepted by the intermediate verifier against the target model. Our evaluations using various representative benchmarks and models show that HiSpec improves throughput by 1.28× on average and by up to 2.01× compared to the baseline single-layer speculation without compromising accuracy.

### 10. JaneEye: A 12-nm 2K-FPS 18.9-µJ/Frame Event-based Eye Tracking Accelerator

**主要机构**: Leiden University, Delft University of Technology, Leiden Institute of Advanced Computer Science (LIACS), Department of Microelectronics
**作者数量**: 4人

**摘要**:
Eye tracking has become a key technology for gaze-based interactions in Extended Reality (XR). However, conventional frame-based eye-tracking systems often fall short of XR's stringent requirements for high accuracy, low latency, and energy efficiency. Event cameras present a compelling alternative, offering ultra-high temporal resolution and low power consumption. In this paper, we present JaneEye, an energy-efficient eventbased eye-tracking hardware accelerator designed specifically for wearable devices, leveraging sparse, high-temporal-resolution event data. We introduce an ultra-lightweight neural network architecture featuring a novel ConvJANET layer, which simplifies the traditional ConvLSTM by retaining only the forget gate, thereby halving computational complexity without sacrificing temporal modeling capability. Our proposed model achieves high accuracy with a pixel error of 2.45 on the 3ET+ dataset, using only 17.6K parameters, with up to 1250 Hz event frame rate. To further enhance hardware efficiency, we employ custom linear approximations of activation functions (hardsigmoid and hardtanh) and fixed-point quantization. Through software-hardware co-design, our 12-nm ASIC implementation operates at 400 MHz, delivering an end-to-end latency of 0.5 ms (equivalent to 2000 Frames Per Second (FPS)) at an energy efficiency of 18.9 µJ/frame. JaneEye sets a new benchmark in low-power, highperformance eye-tracking solutions suitable for integration into next-generation XR wearables.

### 11. LiLa-Net: Lightweight Latent LiDAR Autoencoder for 3D Point Cloud Reconstruction

**主要机构**: 
**作者数量**: 5人

**摘要**:
This work proposed a 3D autoencoder architecture, named LiLa-Net, which encodes efficient features from real traffic environments, employing only the LiDAR's point clouds. For this purpose, we have real semi-autonomous vehicle, equipped with Velodyne LiDAR. The system leverage skip connections concept to improve the performance without using extensive resources as the state-of-the-art architectures. Key changes include reducing the number of encoder layers and simplifying the skip connections, while still producing an efficient and representative latent space which allows to accurately reconstruct the original point cloud. Furthermore, an effective balance has been achieved between the information carried by the skip connections and the latent encoding, leading to improved reconstruction quality without compromising performance. Finally, the model demonstrates strong generalization capabilities, successfully reconstructing objects unrelated to the original traffic environment.

### 12. Local Linear Attention: An Optimal Interpolation of Linear and Softmax Attention For Test-Time Regression

**主要机构**: University of Washington, Northwestern University
**作者数量**: 6人

**摘要**:
Transformer architectures have achieved remarkable success in various domains. While efficient alternatives to Softmax Attention have been widely studied, the search for more expressive mechanisms grounded in theoretical insight-even at greater computational cost-has been relatively underexplored. In this work, we bridge this gap by proposing Local Linear Attention (LLA), a novel attention mechanism derived from nonparametric statistics through the lens of test-time regression. First, we show that LLA offers theoretical advantages over Linear and Softmax Attention for associative memory via a bias-variance trade-off analysis. Next, we address its computational challenges and propose two memory-efficient primitives to tackle the Θ(n 2 d) and Θ(nd 2) complexity. We then introduce FlashLLA, a hardware-efficient, blockwise algorithm that enables scalable and parallel computation on modern accelerators. In addition, we implement and profile a customized inference kernel that significantly reduces memory overheads. Finally, we empirically validate the advantages and limitations of LLA on test-time regression, in-context regression, associative recall and state tracking tasks. Experiment results demonstrate that LLA effectively adapts to non-stationarity, outperforming strong baselines in test-time training and in-context learning, and exhibiting promising evidence for its scalability and applicability in large-scale models. Code is available at https://github.com/Yifei-Zuo/Flash-LLA.

### 13. Mamba Outpaces Reformer in Stock Prediction with Sentiments from Top Ten LLMs

**主要机构**: University of North Texas USA
**作者数量**: 2人

**摘要**:
The stock market is extremely difficult to predict in the short term due to high market volatility, changes caused by news, and the nonlinear nature of the financial time series. This research proposes a novel framework for improving minute-level prediction accuracy using semantic sentiment scores from ten different large language models (LLMs) combined with minute interval intraday stock price data. We systematically constructed a time-aligned dataset of AAPL news articles and 1-minute Apple Inc. (AAPL) stock prices for the dates of April 4 to May 2, 2025. The sentiment analysis was achieved using the DeepSeek-V3, GPT variants, LLaMA, Claude, Gemini, Qwen, and Mistral models through their APIs. Each article obtained sentiment scores from all ten LLMs, which were scaled to a [0, 1] range and combined with prices and technical indicators like RSI, ROC, and Bollinger Band Width. Two state-of-the-art such as Reformer and Mamba were trained separately on the dataset using the sentiment scores produced by each LLM as input. Hyper parameters were optimized by means of Optuna and were evaluated through a 3-day evaluation period. Reformer had mean squared error (MSE) or the evaluation metrics, and it should be noted that Mamba performed not only faster but also better than Reformer for every LLM across the 10 LLMs tested. Mamba performed best with LLaMA 3.3-70B, with the lowest error of 0.137. While Reformer could capture broader trends within the data, the model appeared to over smooth sudden changes by the LLMs. This study highlights the potential of integrating LLM-based semantic analysis paired with efficient temporal modeling to enhance real-time financial forecasting.

### 14. Nav-EE: Navigation-Guided Early Exiting for Efficient Vision-Language Models in Autonomous Driving

**主要机构**: Department of Computer Science, City University of Hong Kong
**作者数量**: 7人

**摘要**:
Vision-Language Models (VLMs) are increasingly applied in autonomous driving for unified perception and reasoning, but high inference latency hinders real-time deployment. Early-exit reduces latency by terminating inference at intermediate layers, yet its task-dependent nature limits generalization across diverse scenarios. We observe that this limitation aligns with autonomous driving: navigation systems can anticipate upcoming contexts (e.g., intersections, traffic lights), indicating which tasks will be required. We propose Nav-EE, a navigation-guided early-exit framework that precomputes task-specific exit layers offline and dynamically applies them online based on navigation priors. Experiments on CODA, Waymo, and BOSCH show that Nav-EE achieves accuracy comparable to full inference while reducing latency by up to 63.9%. Real-vehicle integration with Autoware Universe further demonstrates reduced inference latency (600 ms to 300 ms), supporting faster decision-making in complex scenarios. These results suggest that coupling navigation foresight with early-exit offers a viable path toward efficient deployment of large models in autonomous systems. Code and data are available at our anonymous repository: https://anonymous.4open.science/r/Nav-EE-BBC4 *Equal contribution

### 15. OPTIMAL STOPPING VS BEST-OF-N FOR INFERENCE TIME OPTIMIZATION

**主要机构**: University of Southern, University of Michigan
**作者数量**: 5人

**摘要**:
Large language model (LLM) generation often requires balancing output quality against inference cost, especially when using multiple generations. We introduce a new framework for inference-time optimization based on the classical Pandora's Box problem. Viewing each generation as opening a costly "box" with random reward, we develop algorithms that decide when to stop generating without knowing the underlying reward distribution. Our first contribution is a UCB-style Pandora's Box algorithm, which achieves performance that is provably close to Weitzman's algorithm, the optimal strategy when the distribution is known. We further adapt this method to practical LLM settings by addressing reward scaling across prompts via a Bradley-Terry inspired transformation. This leads to an adaptive inference-time optimization method that normalizes rewards and learns stopping thresholds on the fly. Experiments on the AlpacaFarm and HH-RLHF datasets, using multiple LLM-reward model pairs, show that our adaptive strategy can obtain the same performance as non-adaptive Best-of-N sampling while requiring 15-35% fewer generations on average. Our results establish a principled bridge between optimal stopping theory and inference-time scaling, providing both theoretical performance bounds and practical efficiency gains for LLM deployment.

### 16. Pure-Pass: Fine-Grained, Adaptive Masking for Dynamic Token-Mixing Routing in Lightweight Image Super-Resolution

**主要机构**: State Key Laboratory for Novel Software Technology Nanjing University
**作者数量**: 4人

**摘要**:
Image Super-Resolution (SR) aims to reconstruct highresolution images from low-resolution counterparts, but the computational complexity of deep learning-based methods often hinders practical deployment. CAMixer is the pioneering work to integrate the advantages of existing lightweight SR methods and proposes a content-aware mixer to route token mixers of varied complexities according to the difficulty of content recovery. However, several limitations remain, such as poor adaptability, coarse-grained masking and spatial inflexibility, among others. We propose Pure-Pass (PP), a pixel-level masking mechanism that identifies pure pixels and exempts them from expensive computations. PP utilizes fixed color center points to classify pixels into distinct categories, enabling fine-grained, spatially flexible masking while maintaining adaptive flexibility. Integrated into the state-of-the-art ATD-light model, PP-ATD-light achieves superior SR performance with minimal overhead, outperforming CAMixer-ATD-light in reconstruction quality and parameter efficiency when saving a similar amount of computation.

### 17. ReSSFormer: A Recursive Sparse Structured Transformer for Scalable and Long-Context Reasoning

**主要机构**: Hebei Institute of Communications Shijiazhuang, Columbia University New York
**作者数量**: 4人

**摘要**:
While Transformer architectures have demonstrated impressive scalability across domains, they continue to face challenges in longcontext reasoning, computational efficiency, and structural generalization-largely due to rigid layer stacking, dense attention, and reliance on positional encodings. We present ReSSFormer, a Recursive Sparse Structured Transformer that integrates three complementary innovations: Recurrent Reasoning & Memory Unit (R2MU) for iterative reasoning with bounded depth, Adaptive Sparse Attention Module (ASAM) for efficient and focused context selection, and Self-Organizing Encoder Structure (SOES) for position-free structure induction. ReSSFormer replaces conventional depth stacking with recurrent inference, substitutes full attention with tokenand expert-level sparsity, and models latent token topology directly from content. Across language modeling, multi-hop QA, and structure-sensitive tasks, ReSSFormer consistently outperforms strong baselines under comparable FLOPs and parameter budgets, highlighting its scalability, efficiency, and structural flexibility. CCS Concepts • Computing methodologies → Natural language generation.

### 18. RETHINKING THE SHAPE CONVENTION OF AN MLP

**主要机构**: MediaTek Research
**作者数量**: 4人

**摘要**:
Multi-layer perceptrons (MLPs) conventionally follow a narrow-wide-narrow design where skip connections operate at the input/output dimensions while processing occurs in expanded hidden spaces. We challenge this convention by proposing wide-narrow-wide (Hourglass) MLP blocks where skip connections operate at expanded dimensions while residual computation flows through narrow bottlenecks. This inversion leverages higher-dimensional spaces for incremental refinement while maintaining computational efficiency through parameter-matched designs. Implementing Hourglass MLPs requires an initial projection to lift input signals to expanded dimensions. We propose that this projection can remain fixed at random initialization throughout training, enabling efficient training and inference implementations. We evaluate both architectures on generative tasks over popular image datasets, characterizing performance-parameter Pareto frontiers through systematic architectural search. Results show that Hourglass architectures consistently achieve superior Pareto frontiers compared to conventional designs. As parameter budgets increase, optimal Hourglass configurations favor deeper networks with wider skip connections and narrower bottlenecks-a scaling pattern distinct from conventional MLPs. Our findings suggest reconsidering skip connection placement in modern architectures, with potential applications extending to Transformers and other residual networks.

### 19. RSAVQ: Riemannian Sensitivity-Aware Vector Quantization for Large Language Models

**主要机构**: 
**作者数量**: 5人

**摘要**:
Large language models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, their exponentially increasing parameters pose significant challenges for deployment on resourceconstrained devices. Vector Quantization (VQ) shows great promise for low-bit quantization (e.g., 2 to 4 bits), but existing work faces two key challenges: unconstrained direction error and suboptimal bit allocation. In this paper, we propose RSAVQ, a novel VQ framework to enhance extremely low-bit quantization for LLMs. RSAVQ introduces two geometry-driven innovations that effectively mitigate above limitations: (1) Error Direction Sensitivity Guidance (EDSG), which leverages the Fisher Information Matrix (FIM)-induced Riemannian metric to project quantization errors onto low-sensitivity directions in the parameter space. Specifically, this projection is performed along the negative natural gradient direction, which effectively suppresses error expansion. (2) Weight Channel Sensitivity Guidance (WCSG) , which constructs a channel-wise sensitivity metric via FIM curvature analysis to dynamically guide bit resource allocation. The approach facilitates a globally optimal quantization solution within prescribed bit constraints. Experiments demonstrate that RSAVQ outperforms existing methods for LLMs. For example, in 2-bit quantization of LLaMA-3 8B, RSAVQ leads baselines like VPTQ and QuIP# by 0.4 in perplexity (PPL) and 1.5 in zero-shot accuracy. This work offers a practical solution for constrained environments and a theoretical bridge between information geometry and the quantization of neural networks, advancing efficient deep learning.

### 20. Self-Forcing++: Towards Minute-Scale High-Quality Video Generation

**主要机构**: University of Central Florida
**作者数量**: 11人

**摘要**:
Diffusion models have revolutionized image and video generation, achieving unprecedented visual quality. However, their reliance on transformer architectures incurs prohibitively high computational costs, particularly when extending generation to long videos. Recent work has explored autoregressive formulations for long video generation, typically by distilling from short-horizon bidirectional teachers. Nevertheless, given that teacher models cannot synthesize long videos, the extrapolation of student models beyond their training horizon often leads to pronounced quality degradation, arising from the compounding of errors within the continuous latent space. In this paper, we propose a simple yet effective approach to mitigate quality degradation in long-horizon video generation without requiring supervision from long-video teachers or retraining on long video datasets. Our approach centers on exploiting the rich knowledge of teacher models to provide guidance for the student model through sampled segments drawn from self-generated long videos. Our method maintains temporal consistency while scaling video length by up to 20× beyond teacher's capability, avoiding common issues such as over-exposure and error-accumulation without recomputing overlapping frames like previous methods. When scaling up the computation, our method shows the capability of generating videos up to 4 minutes and 15 seconds, equivalent to 99.9% of the maximum span supported by our base model's position embedding and more than 50x longer than that of our baseline model. Experiments on standard benchmarks and our proposed improved benchmark demonstrate that our approach substantially outperforms baseline methods in both fidelity and consistency. Our long-horizon videos demo can be found at https://self-forcing-plus-plus.github.io/.

### 21. SHIFT-INVARIANT ATTRIBUTE SCORING FOR KOLMOGOROV-ARNOLD NETWORKS VIA SHAPLEY VALUE

**主要机构**: National University of Singapore
**作者数量**: 4人

**摘要**:
For many real-world applications, understanding feature-outcome relationships is as crucial as achieving high predictive accuracy. While traditional neural networks excel at prediction, their black-box nature obscures underlying functional relationships. Kolmogorov-Arnold Networks (KANs) address this by employing learnable spline-based activation functions on edges, enabling recovery of symbolic representations while maintaining competitive performance. However, KAN's architecture presents unique challenges for network pruning. Conventional magnitude-based methods become unreliable due to sensitivity to input coordinate shifts. We propose ShapKAN, a pruning framework using Shapley value attribution to assess node importance in a shift-invariant manner. Unlike magnitudebased approaches, ShapKAN quantifies each node's actual contribution, ensuring consistent importance rankings regardless of input parameterization. Extensive experiments on synthetic and real-world datasets demonstrate that ShapKAN preserves true node importance while enabling effective network compression. Our approach improves KAN's interpretability advantages, facilitating deployment in resource-constrained environments.

### 22. Sparse Query Attention (SQA): A Computationally Efficient Attention Mechanism with Query Heads Reduction

**主要机构**: 
**作者数量**: 1人

**摘要**:
The Transformer architecture, underpinned by the Multi-Head Attention (MHA) mechanism, has become the de facto standard for state-of-the-art models in artificial intelligence. However, the quadratic computational complexity of MHA with respect to sequence length presents a significant barrier to scaling, particularly for applications involving long contexts. Prevailing solutions, such as Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), have effectively addressed the memory bandwidth bottleneck that dominates autoregressive inference latency by sharing Key and Value projections. While highly successful, these methods do not reduce the fundamental number of floating-point operations (FLOPs) required for the attention score computation, which remains a critical bottleneck for training and full-sequence processing. This paper introduces Sparse Query Attention (SQA), a novel attention architecture that pursues an alternative and complementary optimization path. Instead of reducing Key/Value heads, SQA reduces the number of Query heads. This architectural modification directly decreases the computational complexity of the attention mechanism by a factor proportional to the reduction in query heads, thereby lowering the overall FLOPs. This work presents the theoretical foundation of SQA, its mathematical formulation, and a family of architectural variants. Empirical benchmarks on long sequences (32k-200k tokens) demonstrate that SQA can achieve significant throughput improvements of up to 3x in computation-bound scenarios such as model pre-training, fine-tuning, and encoder-based tasks, with only a minimal impact on model quality in preliminary smallscale experiments. SQA was discovered serendipitously during the development of the upcoming Reactive Transformer architecture, a context in which its computational advantages are maximized, suggesting its potential as a powerful tool for building more efficient and scalable models.

### 23. SPUS: A Lightweight and Parameter-Efficient Foundation Model for PDEs

**主要机构**: Space Remote Sensing and Data Science, Computing and Artificial Intelligence Division (CAI), Los Alamos National Laboratory
**作者数量**: 5人

**摘要**:
We introduce Small PDE U-Net Solver (SPUS), a compact and efficient foundation model (FM) designed as a unified neural operator for solving a wide range of partial differential equations (PDEs). Unlike existing state-of-the-art PDE FMs-primarily based on large complex transformer architectures with high computational and parameter overhead-SPUS leverages a lightweight residual U-Net-based architecture that has been largely underexplored as a foundation model architecture in this domain. To enable effective learning in this minimalist framework, we utilize a simple yet powerful auto-regressive pretraining strategy which closely replicates the behavior of numerical solvers to learn the underlying physics. SPUS is pretrained on a diverse set of fluid dynamics PDEs and evaluated across 6 challenging unseen downstream PDEs spanning various physical systems. Experimental results demonstrate that SPUS using residual U-Net based architecture achieves state-of-the-art generalization on these downstream tasks while requiring significantly fewer parameters and minimal fine-tuning data, highlighting its potential as a highly parameter-efficient FM for solving diverse PDE systems.

### 24. SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs

**主要机构**: Wuhan University of Technology, Renmin University of China, Institute of Information Engineering, CAS School of Cyberspace Security
**作者数量**: 13人

**摘要**:


### 25. THE DISPARATE IMPACTS OF SPECULATIVE DECODING

**主要机构**: Hofstra University, University of Virginia
**作者数量**: 5人

**摘要**:
The practice of speculative decoding, whereby inference is probabilistically supported by a smaller, cheaper, "drafter" model, has become a standard technique for systematically reducing the decoding time of large language models. This paper conducts an analysis of speculative decoding through the lens of its potential disparate speed-up rates across tasks. Crucially, the paper shows that speed-up gained from speculative decoding is not uniformly distributed across tasks, consistently diminishing for under-fit, and often underrepresented tasks. To better understand this phenomenon, we derive an analysis to quantify this observed "unfairness" and draw attention to the factors that motivate such disparate speed-ups to emerge. Further, guided by these insights, the paper proposes a mitigation strategy designed to reduce speed-up disparities and validates the approach across several model pairs, revealing on average a 12% improvement in our fairness metric.

### 26. THE UNSEEN FRONTIER: PUSHING THE LIMITS OF LLM SPARSITY WITH SURROGATE-FREE ADMM

**主要机构**: 
**作者数量**: 6人

**摘要**:
Neural network pruning is a promising technique to mitigate the excessive computational and memory requirements of large language models (LLMs). Despite its promise, however, progress in this area has diminished, as conventional methods are seemingly unable to surpass moderate sparsity levels (50-60%) without severely degrading model accuracy. This work breaks through the current impasse, presenting a principled and effective method called ELSA, which achieves extreme sparsity levels of up to 90% while retaining high model fidelity. This is done by identifying several limitations in current practice, all of which can be traced back to their reliance on a surrogate objective formulation. ELSA tackles this issue directly and effectively via standard and well-established constrained optimization techniques based on ADMM. Our extensive experiments across a wide range of models and scales show that ELSA achieves substantial improvements over existing methods; e.g., it achieves 7.8ˆless perplexity than the best existing method on LLaMA-2-7B at 90% sparsity. Furthermore, we present ELSA-L , a quantized variant that scales to extremely large models (27B), and establish its theoretical convergence guarantees. These results highlight meaningful progress in advancing the frontier of LLM sparsity, while promising that significant opportunities for further advancement may remain in directions that have so far attracted limited exploration.

### 27. Ultra-Efficient Decoding for End-to-End Neural Compression and Reconstruction

**主要机构**: Iowa State University Ames, Department of Electrical and Computer Engineering
**作者数量**: 2人

**摘要**:
Image compression and reconstruction are crucial for various digital applications. While contemporary neural compression methods achieve impressive compression rates, the adoption of such technology has been largely hindered by the complexity and large computational costs of the convolution-based decoders during data reconstruction. To address the decoder bottleneck in neural compression, we develop a new compression-reconstruction framework based on incorporating low-rank representation in an autoencoder with vector quantization. We demonstrated that performing a series of computationally efficient low-rank operations on the learned latent representation of images can efficiently reconstruct the data with high quality. Our approach dramatically reduces the computational overhead in the decoding phase of neural compression/reconstruction, essentially eliminating the decoder compute bottleneck while maintaining high fidelity of image outputs.

### 28. VideoNSA: Native Sparse Attention Scales Video Understanding VIDEONSA: NATIVE SPARSE ATTENTION SCALES VIDEO UNDERSTANDING

**主要机构**: New York University, University of California, Princeton University, Lambda
**作者数量**: 8人

**摘要**:
Video understanding in multimodal language models remains limited by context length: models often miss key transition frames and struggle to maintain coherence across long time scales. To address this, we adapt Native Sparse Attention (NSA) to video-language models. Our method, VideoNSA, adapts Qwen2.5-VL through end-to-end training on a 216K video instruction dataset. We employ a hardware-aware hybrid approach to attention, preserving dense attention for text, while employing NSA for video. Compared to token-compression and training-free sparse baselines, VideoNSA achieves improved performance on long-video understanding, temporal reasoning, and spatial benchmarks. Further ablation analysis reveals four key findings: (1) reliable scaling to 128K tokens; (2) an optimal global-local attention allocation at a fixed budget; (3) task-dependent branch usage patterns; and (4) the learnable combined sparse attention help induce dynamic attention sinks.
