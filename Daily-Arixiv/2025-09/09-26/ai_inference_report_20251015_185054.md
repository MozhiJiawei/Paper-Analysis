# AI推理加速技术论文分析报告
生成时间: 2025-10-15 18:50:54
分析论文数量: 88篇

## 论文技术简报

### 1. Achieving Fair Skin Lesion Detection through Skin Tone Normalization and Channel Pruning

伦敦大学学院发布了Achieving Fair Skin Lesion Detection through Skin Tone Normalization and Channel Pruning论文，使用基于ITA损失的肤色归一化与自适应元学习联合通道剪枝技术，解决了皮肤病变检测模型对多个敏感属性的偏见及公平性与准确率难以兼顾的问题，达成了同时提升模型对多个敏感属性的公平性且不显著降低准确率，剪枝后网络更小、推理时间更低的效果。

### 2. Agentic AI Reasoning for Mobile Edge General Intelligence: Fundamentals, Approaches, and Directions

发布了《Agentic AI Reasoning for Mobile Edge General Intelligence》论文，使用联合优化框架（结合自适应CoT提示与分布式MoE架构），解决了MEGI环境中LLM推理高计算需求与边缘设备资源有限的矛盾，达成了平衡推理质量与资源效率并验证其在资源受限MEGI环境部署可行性的效果。

### 3. An Enhanced Pyramid Feature Network Based on Long-Range Dependencies for Multi-Organ Medical Image Segmentation

研究团队发布了多器官医学图像分割论文，提出LamFormer网络（集成Linear Attention Mamba、Parallel Hierarchical Feature Aggregation模块及Reduced Transformer），解决了Transformer的高计算成本及局部细节提取不足问题，在七个复杂多样数据集上优于现有分割方法并平衡了模型性能与复杂度。

### 4. Asymmetric VAE for One-Step Video Super-Resolution Acceleration ASYMMETRIC VAE FOR ONE-STEP VIDEO SUPER-RESOLUTION ACCELERATION

上海交通大学发布了FastVSR论文，使用高压缩VAE（f16）、像素洗牌与通道复制及下界引导训练策略，解决了基于扩散的单步视频超分辨率模型推理效率优化问题，达成相比多步模型加速111.9倍、现有单步模型加速3.92倍的效果

### 5. ATTENTION SURGERY: AN EFFICIENT RECIPE TO LIN-EARIZE YOUR VIDEO DIFFUSION TRANSFORMER

高通发布了ATTENTION SURGERY论文，使用Attention Surgery框架（含混合注意力机制、轻量级蒸馏微调及成本感知块率策略），解决了Transformer视频扩散模型自注意力二次成本导致的长序列和高分辨率计算昂贵问题，达成了实现首个有竞争力的亚二次注意力视频扩散模型，注意力FLOPs降低40%，同时保持VBench和VBench-2.0生成质量的效果。

### 6. Each Complexity Deserves a Pruning Policy

上海交通大学发布了《Each Complexity Deserves a Pruning Policy》论文，使用Complexity-Adaptive Pruning（AutoPrune）框架，解决了大视觉语言模型中视觉token剪枝未根据样本和任务复杂度调整的问题，达成了在LLaVA-1.5-7B上剪枝89%视觉token、减少76.8%推理FLOPs并保持96.7%准确率，较PDrop提升9.1%的效果

### 7. A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving

中山大学与Kaon AI发布了NexusSched论文，使用结构感知在线性能模型驱动的预测性协同两层调度框架，解决了LLM服务两层架构因信息鸿沟导致的决策滞后及SLO违规/资源浪费问题，达成SLO达成率平均提升43%、长上下文和异构场景吞吐量最高3倍的效果。

### 8. A Second-Order Perspective on Pruning at Initialization and Knowledge Transfer

Politecnico di Torino发布了关于初始化剪枝与知识迁移的研究论文，使用初始化剪枝（Pruning-at-Initialization）技术对预训练视觉模型进行压缩，解决了下游任务未知时预训练模型剪枝依赖特定任务数据的问题，达成了用一个任务剪枝后在未见过任务上保留零样本性能，且微调后能同时提升原任务性能并恢复未见过任务性能的效果。

### 9. BALR-SAM: Boundary-Aware Low-Rank Adaptation of SAM for Resource-Efficient Medical Image Segmentation

上海交通大学发布了BALR-SAM论文，使用边界感知低秩适应框架（结合互补细节增强网络、低秩适配器及低秩张量注意力机制），解决了SAM在医学图像分割中领域适应不足及高效微调（资源低且高性能）的挑战，达成无需提示优于包括全微调MedSAM在内的多个SOTA方法、仅更新1.8%参数、内存减少75%且推理提速的效果。

### 10. Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability

印度理工学院孟买分校发布了提出UAT框架的论文，使用多臂老虎机框架自适应调整退出阈值的技术，解决了早期退出策略的过度自信及对分布偏移不鲁棒的问题，达成了1.70-2.10×加速且性能下降<2%的效果

### 11. CAOTE: KV Cache Selection for LLMs via Attention Output Error-Based Token Eviction

研究团队发布了CAOTE论文，使用基于注意力输出误差整合注意力分数与值向量的令牌驱逐标准（首个闭式结合值向量信息），解决了LLMs KV缓存令牌驱逐中仅用注意力分数缺乏输出贡献信息的问题，达成在LLAMA3和QWEN2.5模型上结合SOTA方法提升下游任务准确率的效果。

### 12. CasPoinTr: Point Cloud Completion with Cascaded Networks and Knowledge Distillation

复旦大学发布了CasPoinTr论文，使用级联网络与知识蒸馏技术，解决了点云补全中从高度不完整点云中预测整体形状及重建缺失区域的难题，达成了在ShapeNet-55不同难度设置下优于现有方法、形状恢复和细节保留更优的效果

### 13. 

东北大学发布了论文，使用电路蒸馏技术，解决了传统模型蒸馏仅关注行为模仿、将教师内部计算视为黑箱的问题，达成了优于标准蒸馏、通过调整少量学生模型参数成功转移算法能力的效果。

### 14. CLQ: CROSS-LAYER GUIDED ORTHOGONAL-BASED QUANTIZATION FOR DIFFUSION TRANSFORMERS

上海交通大学发布了CLQ论文，使用跨层引导的基于正交的量化技术，解决了扩散Transformer量化中的精度与效率平衡问题，达成了在降低计算存储成本的同时保持高性能的效果。

### 15. Compute-Optimal Quantization-Aware Training

École Polytechnique Fédérale de Lausanne (EPFL)发布了Compute-Optimal Quantization-Aware Training论文，使用基于tokens-per-parameter-byte统计的最优QAT/FP计算分配预测方法及cooldown与QAT融合技术，解决了量化感知训练中全精度与QAT阶段计算分配最优比例不明确的问题，达成显著计算节省并能在相同计算预算下训练更高质量的量化模型。

### 16. Context-Driven Performance Modeling for Causal Inference Operators on Neural Processing Units

南加州大学发布了Context-Driven Performance Modeling for Causal Inference Operators on Neural Processing Units论文，通过对多种因果推理算子（含标准二次注意力及子二次替代方案）在现代NPU上的全面性能分析，解决了长上下文大语言模型在资源受限边缘设备NPU部署的架构不匹配挑战，揭示了算子瓶颈（二次注意力内存受限、子二次模型计算受限）并为硬件感知模型协同设计提供关键见解以支持边缘长上下文AI推理。

### 17. D 2 CACHE: ACCELERATING DIFFUSION-BASED LLMS VIA DUAL ADAPTIVE CACHING

东南大学发布了D²Cache论文，使用双自适应缓存（d²Cache）框架，解决了扩散型大语言模型推理效率低的问题，达成了显著推理加速并提升生成质量的效果

### 18. DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

发布了DC-Gen论文，使用基于深度压缩潜空间的后训练框架（含嵌入对齐训练与LoRA微调），解决了文本到图像扩散模型高分辨率生成效率挑战及潜空间表示差距导致的微调不稳定问题，实现4K图像生成 latency 降低53×，结合NVFP4 SVDQuant在5090 GPU上3.5秒生成4K图像且总 latency 降低138×

### 19. DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder

研究团队发布了DC-VideoGen论文，使用深度压缩视频自编码器（含chunk-causal temporal设计）与AE-Adapt-V适应策略，解决了预训练视频扩散模型的推理效率问题，达成了推理延迟降低14.8倍且不损失质量、支持单GPU生成2160×3840视频的效果。

### 20. DocPruner: A STORAGE-EFFICIENT FRAMEWORK FOR MULTI-VECTOR VISUAL DOCUMENT RETRIEVAL VIA ADAPTIVE PATCH-LEVEL EMBEDDING PRUNING

香港科技大学、阿里云发布了DocPruner论文，使用自适应patch-level嵌入剪枝技术，解决了多向量视觉文档检索中存储开销过大的问题，达成50-60%存储量减少且检索性能几乎无下降的效果。

### 21. EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering

浙江大学发布了EasySteer框架相关论文，使用基于vLLM的统一框架与模块化可插拔接口技术，解决了现有LLM steering框架计算效率低、扩展性有限及功能受限的问题，达成5.5-11.4倍速度提升并在缓解过度思考、减少幻觉等应用中有效。

### 22. EfficientMIL: Efficient Linear-Complexity MIL Method for WSI Classification

Changhai Hospital Department of Radiology and North University of China发布了EfficientMIL论文，使用线性复杂度多实例学习（基于RNN序列模型替代二次复杂度注意力机制）及自适应补丁选择器（APS），解决了全切片图像分类中计算资源需求高的问题，在TCGA-Lung和CAMELYON16数据集上达成SOTA性能且计算资源需求显著低于基于注意力的方法。

### 23. EXGS: EXTREME 3D GAUSSIAN COMPRESSION WITH DIFFUSION PRIORS

香港科技大学与上海人工智能实验室发布了EXGS论文，使用扩散先验的3D高斯极端压缩技术，解决了3D高斯模型高效压缩难题，达成了高压缩率下的优质重建效果。

### 24. "Explainable Deep Learning for Cataract Detection in Retinal Images: A Dual-Eye and Knowledge Distillation Approach"

University College of Nabi Akram发布了"Explainable Deep Learning for Cataract Detection in Retinal Images: A Dual-Eye and Knowledge Distillation Approach"论文，使用双眼Siamese变体与知识蒸馏的可解释深度学习技术，解决了视网膜图像中白内障早期检测问题，达成了高准确率（Swin-Base Transformer达98.58%，蒸馏MobileNetV3达98.42%且计算成本低）效果。

### 25. Fast Thinking for Large Language Models

Zheqi Lv团队发布了Fast Thinking for Large Language Models论文，使用Latent Codebooks for Fast Thinking框架及GainRouter路由机制，解决了大语言模型推理依赖长显式推理轨迹导致的效率低问题，达成了多推理基准准确性竞争性或更优且大幅降低推理成本的效果

### 26. FLAME: A Serving System Optimized for Large-Scale Generative Recommendation with Efficiency

网易云音乐发布了FLAME论文，使用CPU-GPU异构硬件解耦、PDA内存优化、FKE融合内核引擎及DSO动态流编排技术，解决了生成式推荐模型计算量大的大规模在线部署难题，达成了吞吐量提升1.9x-6.3x、延迟降低1.7x及加速2.3x，支持大规模在线部署的效果

### 27. FlowLUT: Efficient Image Enhancement via Differentiable LUTs and Iterative Flow Matching

研究团队发布了FlowLUT论文，使用可微分3D LUTs与迭代流匹配技术，解决了图像增强中计算效率与表示能力的权衡问题，在三个基准数据集上证明了有效性。

### 28. Generalist Multi-Class Anomaly Detection via Distillation to Two Heterogeneous Student Networks

KAIST发布了Generalist Multi-Class Anomaly Detection via Distillation to Two Heterogeneous Student Networks论文，使用基于知识蒸馏的双异构学生网络（Encoder-Decoder与Encoder-Encoder）集成方法（共享DINOv2编码器并通过Noisy-OR目标联合学习），解决了现有异常检测方法在工业检测与语义异常检测间泛化能力差且对数据集和单类任务敏感的问题，达成了在多类和单类设置下均实现最先进准确率，如MVTec-AD图像级AUROC达99.7%、CIFAR-10达97.8%，优于先前通用模型及个别专业模型的效果。

### 29. Under review as a conference paper at ICLR 2026 GRACE-MOE: GROUPING AND REPLICATION WITH LOCALITY-AWARE ROUTING FOR EFFICIENT DISTRIBUTED MOE INFERENCE

中国科学技术大学发布了GRACE-MoE论文，使用基于专家亲和性分组与动态复制、局部感知路由的联合优化框架，解决了分布式SMoE推理中的通信开销大与计算负载不平衡问题，达成了端到端推理延迟显著降低、较最先进系统实现高达3.79×速度提升的效果

### 30. Hazy Pedestrian Trajectory Prediction via Physical Priors and Graph-Mamba

中山大学发布了Hazy Pedestrian Trajectory Prediction via Physical Priors and Graph-Mamba论文，使用结合大气散射物理先验、自适应Mamba变体和异质图注意力网络的深度学习模型，解决了雾霾天气下行人轨迹预测中物理信息退化和行人交互建模无效问题，在能见度<30m的浓雾场景下较SOTA模型将minADE/minFDE分别降低37.2%和41.5%。

### 31. HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models

中国科学院自动化研究所发布了HiViS论文，使用隐藏视觉标记的显式-隐式输入分解框架，解决了视觉-语言模型推测解码中视觉标记语义错位导致KV缓存偏差及大量视觉标记减慢解码效率的问题，达成压缩草稿器预填充序列长度至目标VLM输入的0.7%-1.3%、保持无损生成质量并实现高达2.65倍加速的效果

### 32. HIVTP: A Training-Free Method to Improve VLMs Efficiency via Hierarchical Visual Token Pruning Using Middle-Layer-Based Importance Score

南加州大学发布了HIVTP论文，使用基于中间层重要性分数的分层视觉token剪枝技术，解决了视觉语言模型因视觉token过多导致的推理效率低下问题，达成在不牺牲准确性的情况下减少LLaVA模型TTFT最高55.1%、提升吞吐量最高60.9%的效果

### 33. Hybrid ANN-SNN With Layer-Wise Surrogate Spike Encoding-Decoding Structure

会津大学发布了Hybrid ANN-SNN With Layer-Wise Surrogate Spike Encoding-Decoding Structure论文，使用基于位平面的spike编码函数的替代梯度实现端到端可微训练的逐层编解码SNN块整合技术，解决了现有混合ANN-SNN因spike编码函数不可微导致缺乏深层逐层合作的问题，达成与最先进纯ANN和SNN模型相当的精度并保留SNN的效率和时间表示优势的效果。

### 34. INFLLM-V2: DENSE-SPARSE SWITCHABLE ATTEN-TION FOR SEAMLESS SHORT-TO-LONG ADAPTATION

清华大学发布了INFLLM-V2论文，使用密集-稀疏可切换注意力框架，解决了长序列处理中标准Transformer自注意力的计算与内存瓶颈及现有稀疏注意力方法参数过多、破坏预训练-微调流程的问题，达成了比密集注意力快4倍且保留98.1%和99.7%性能，并开源了MiniCPM4.1模型的效果。

### 35. LEARNING KAN-BASED IMPLICIT NEURAL REPRESENTATIONS FOR DEFORMABLE IMAGE REGISTRATION

莫斯科国立大学发布了KAN-IDIR与RandKAN-IDIR论文，使用将KANs集成到基于INRs的可变形图像配准并提出随机基采样策略的技术，解决了INR方法的计算效率和学习稳定性问题，达成了在多模态医学图像配准中精度最高、计算开销小且学习稳定性优越的效果

### 36. Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding

University of Central发布了Learning to Parallel论文，使用Learn2PD框架（含轻量级自适应过滤器模型及End-of-Text Prediction），解决了扩散型大语言模型现有并行解码策略依赖固定输入无关启发式方法导致的速度-质量权衡不佳问题，达成在LLaDA基准上高达22.58×加速且无性能下降、结合KV-Cache可达57.51×的效果

### 37. LLaDA-MoE: A Sparse MoE Diffusion Language Model

中国人民大学发布了LLaDA-MoE论文，使用稀疏MoE架构，解决了扩散语言模型的计算开销问题，达成了7B参数容量推理仅激活1.4B参数、性能超同类模型且指令微调版与Qwen2.5-3B-Instruct能力相当的效果。

### 38. LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models

加州大学、伊利诺伊大学发布了LUQ论文，使用分层超低比特量化（选择性对更耐量化的层应用）技术，解决了多模态大语言模型部署资源需求大及多模态令牌对超低比特量化不耐受的问题，达成内存较4位版本减少40%和31%、性能下降小于10%（MME基准）的效果。

### 39. MAN: LATENT DIFFUSION ENHANCED MULTISTAGE ANTI-NOISE NETWORK FOR EFFICIENT AND HIGH-QUALITY LOW-DOSE CT IMAGE DENOISING

宁波诺丁汉大学发布了MAN论文，使用潜空间扩散增强的多级抗噪网络（通过压缩潜空间和感知优化自编码器实现快速确定性扩散），解决了扩散模型在低剂量CT去噪中计算成本高、推理时间长阻碍临床应用的问题，达成感知质量超CNN/GAN方法、重建保真度接近计算密集型扩散模型且推理速度快60倍以上的效果。

### 40. 

[主要机构]发布了[论文标题]论文，使用[主要技术创新点]技术，解决了[核心问题]问题，达成了[关键效果或性能提升]效果

### 41. Model Fusion with Multi-LoRA Inference for Tool-Enhanced Game Dialogue Agents

网易公司发布了多LoRA推理的模型融合用于工具增强游戏对话代理的论文，使用Qwen3-14B与LoRA微调、模型融合及多LoRA推理（基于vLLM实现三个LoRA适配器集成）技术，解决了构建符合角色设定、游戏世界观且支持工具调用的游戏对话AI的问题，达成了CPDC 2025 GPU赛道任务1和3第一名、任务2第二名的效果。

### 42. P r e p r i n t . U n d e r R e v i e w . MoE-PHDS: One MoE checkpoint for flexible runtime sparsity

苹果公司发布了MoE-PHDS论文，使用轻量级SFT方法MoE-PHDS，解决了稀疏混合专家模型需多个模型满足不同效率目标的问题，达成了允许推理时灵活调整稀疏度、匹配或超过oracle模型且跨稀疏度一致性提升22%的效果。

### 43. MotionVerse: A Unified Multimodal Framework for Motion Comprehension, Generation and Editing

研究团队发布了MotionVerse论文，使用残差量化运动tokenizer、Delay Parallel Modeling策略及双塔架构等技术，解决了人体运动（单/多人）理解、生成与编辑中的模态干扰及计算效率问题，达成了在多种运动相关任务上的优越性能。

### 44. 

[主要机构]发布了[论文标题]论文，使用[主要技术]技术，解决了[核心问题]问题，达成了[关键效果]效果

### 45. PATCH: LEARNABLE TILE-LEVEL HYBRID SPARSITY FOR LLMS

多伦多大学发布了PATCH论文，使用可学习的tile级混合稀疏框架（将权重矩阵分区为tile并通过可学习掩码分配密集或2:4稀疏模式），解决了大语言模型部署时内存和计算成本高及现有剪枝方法在准确性与硬件加速间的矛盾，达成了在0.5B到8B参数模型上实现0%-50%连续稀疏率、端到端加速1.18×-1.38×并提升准确率0.37%-2.96%的效果

### 46. PEARL: Peer-Enhanced Adaptive Radio via On-Device LLM

Nokia Technologies发布了PEARL论文，使用基于端侧LLM的协作跨层优化框架（利用发布者和订阅者状态指导Wi-Fi Aware参数选择，结合上下文感知奖励与轻量级变体），解决了设备到设备通信的协作跨层优化问题，达成了相比基线提升目标分数、低电量协作场景减少16%能量消耗且PEARL-Lite实现亚20ms推理的效果。

### 47. PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement

哈尔滨工业大学发布了PHASE-Net论文，使用融合Zero-FLOPs Axial Swapper模块、Adaptive Spatial Filter和Gated TCN的物理驱动谐波注意力系统，解决了远程光体积描记法（rPPG）在头部运动和光照变化下的精度下降及现有方法缺乏理论基础的问题，达成了state-of-the-art性能和强效率，提供了理论扎实且可部署的rPPG解决方案。

### 48. PROXY-GS: EFFICIENT 3D GAUSSIAN SPLATTING VIA PROXY MESH

香港科技大学、上海人工智能实验室发布了PROXY-GS论文，使用代理网格技术，解决了3D高斯溅射效率问题，达成了高效的3D高斯溅射效果

### 49. PROXYATTN: GUIDED SPARSE ATTENTION VIA REP-RESENTATIVE HEADS

百度发布了PROXYATTN论文，使用代表性头压缩与块感知动态预算估计的稀疏注意力算法，解决了大语言模型长文本任务中二次复杂度导致的效率问题及高稀疏率下性能下降，达成了10.3倍注意力加速和2.4倍预填充加速且无显著性能损失。

### 50. QUANTSPARSE: COMPREHENSIVELY COMPRESSING VIDEO DIFFUSION TRANSFORMER WITH MODEL QUANTIZATION AND ATTENTION SPARSIFICATION

ETH Zürich等机构发布了QUANTSPARSE论文，使用模型量化与注意力稀疏化技术，解决了视频扩散Transformer的压缩问题，达成了全面压缩的效果。

### 51. REASONING SCAFFOLDING: DISTILLING THE FLOW OF THOUGHT FROM LLMS

华为发布了REASONING SCAFFOLDING论文，使用Reasoning Scaffolding框架（通过抽象语义信号作为支架并多任务训练学生模型以传递推理的算法结构），解决了现有蒸馏方法使小模型仅模仿表面模式、缺乏逻辑鲁棒性的问题，达成了在推理基准上准确率和逻辑一致性显著优于现有蒸馏方法的效果。

### 52. ReSeFlow: Rectifying SE(3)-Equivariant Policy Learning Flows

研究团队发布了ReSeFlow论文，提出将整流流引入SE(3)等变扩散模型并采用SE(3)等变网络的技术，解决了SE(3)等变扩散模型推理时间成本高的问题，达成了仅需1步推理即优于基线100步推理的性能，在绘画任务上误差降低48.5%、旋转三角形任务上降低21.9%的效果。

### 53. RESTORECT: DEGRADED IMAGE RESTORATION VIA LATENT RECTIFIED FLOW & FEATURE DISTILLATION A PREPRINT

Purdue University发布了RestoRect论文，使用RestoRect（一种新型潜在整流流特征蒸馏方法，结合Retinex理论、可学习各向异性扩散约束和三角色彩空间极化，并引入特征层提取损失实现跨架构鲁棒知识迁移）技术，解决了现有图像恢复方法中高性能模型速度慢与快速模型效果差的权衡问题，达成了训练稳定性更好、收敛和推理更快且保持恢复质量，在15个图像恢复数据集（覆盖4项任务、8项指标）上取得优越结果的效果。

### 54. RETHINKING LARGE LANGUAGE MODEL DISTILLA-TION: A CONSTRAINED MARKOV DECISION PROCESS PERSPECTIVE

Huawei Noah's Ark Lab与UCL发布了大语言模型蒸馏研究论文，使用将其建模为约束强化学习问题的优化框架（最大化任务奖励同时约束与教师模型的 divergence），解决了现有方法依赖临时奖励加权的问题，达成了约束满足率和推理能力优于基线且保持任务性能竞争力的效果。

### 55. ROBUQ: PUSHING DITS TO W1.58A2 VIA ROBUST ACTIVATION QUANTIZATION

上海交通大学、清华大学发布了ROBUQ论文，使用鲁棒激活量化技术，解决了深度神经网络低比特量化问题，达成将DNNs推向W1.58A2的效果

### 56. ROLLING FORCING: AUTOREGRESSIVE LONG VIDEO DIFFUSION IN REAL TIME

南洋理工大学发布了ROLLING FORCING: AUTOREGRESSIVE LONG VIDEO DIFFUSION IN REAL TIME论文，使用ROLLING FORCING技术，解决了自回归长视频扩散模型实时生成的问题，达成了长视频实时生成的效果。

### 57. RServe: Overlapping Encoding and Prefill for Efficient LMM Inference

中山大学发布了RServe论文，使用单请求内重叠多模态编码与语言模型前向计算及跨微批令牌调度技术，解决了大型多模态模型推理中资源干扰、并行性未充分利用的问题，达成延迟降低达66%、吞吐量提升达109%的效果。

### 58. S 2 NN: Sub-bit Spiking Neural Networks

电子科技大学发布了S²NN: Sub-bit Spiking Neural Networks论文，使用Sub-bit权重表示、OS-Quant异常值感知子比特量化及MPFD膜电位特征蒸馏技术，解决了SNNs在资源受限部署中的存储与计算需求大的问题，在视觉和非视觉任务上性能与效率均优于现有量化SNNs，适用于边缘计算。

### 59. SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

Project团队发布了SANA-Video论文，使用Linear DiT和Constant-Memory KV Cache技术，解决了视频生成中效率低、内存占用高、难以生成高分辨率长视频的问题，达成了高效生成720×1280分辨率、分钟级时长视频，较现代小型扩散模型快16倍且训练成本仅为MovieGen 1%的效果

### 60. Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

微软研究院与清华大学发布了Scaling LLM Test-Time Compute with Mobile NPU on Smartphones论文，使用硬件感知分块量化与LUT替换复杂操作的并行测试时缩放技术，解决了移动设备部署LLM时小模型性能不足、大模型资源消耗多及NPU计算资源未充分利用的问题，达成混合精度GEMM加速19.0倍、Softmax加速2.2倍且小模型匹配大模型准确性的效果。

### 61. SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts via Token-Level LSH Matching

圣母大学发布了SemShareKV论文，使用token级LSH匹配结合RoPE实现语义相似提示的KV缓存共享技术，解决了LLM推理中KV缓存内存占用瓶颈（尤其语义相似但词汇不同的提示场景），达成5k tokens输入时加速6.25倍、GPU内存使用降低42%且质量损失可忽略的效果。

### 62. Sequential Token Merging: Revisiting Hidden States

复旦大学、上海人工智能实验室发布了《Sequential Token Merging: Revisiting Hidden States》论文，使用融合双向最近邻合并与隐藏状态保护的Sequential Token Merging（STM）技术，解决了Vision Mambas效率受图像分辨率二次令牌缩放限制的问题，达成了ViM-Ti 20%令牌减少精度仅降1.0%、ViM-S 40%减少降1.4%的同时实现最先进效率的效果。

### 63. Similarity-Aware Selective State-Space Modeling for Semantic Correspondence

Pohang University of Science and Technology (POSTECH)发布了相关论文，提出MambaMatcher方法，使用相似性感知的选择性状态空间模型（SSMs），解决了传统语义对应方法中复杂关联关系捕捉不足或4D相关图计算成本高的问题，在标准语义对应基准上达成了最先进性能。

### 64. SLA: BEYOND SPARSITY IN DIFFUSION TRANSFORM-ERS VIA FINE-TUNABLE SPARSE-LINEAR ATTENTION

清华大学发布了SLA论文，使用可训练的稀疏-线性注意力（SLA）技术，解决了扩散Transformer（DiT）模型在视频生成中的注意力延迟瓶颈，达成了减少95%注意力计算且不损失生成质量，在Wan2.1-1.3B上实现13.7×注意力计算加速和2.2×端到端视频生成加速的效果。

### 65. SPARSED: SPARSE ATTENTION FOR DIFFUSION LAN-GUAGE MODELS

新加坡国立大学、香港理工大学发布了SPARSED论文，使用预计算头特定稀疏模式并跨步骤复用、早期全注意力后期稀疏的SparseD稀疏注意力技术，解决了扩散语言模型（DLMs）推理延迟高的问题，达成了在64k上下文长度下无损加速、较FlashAttention提升1.50×速度的效果

### 66. SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving

华为云与香港中文大学发布了SparseServe论文，使用分层HBM-DRAM管理（含碎片感知KV缓存传输、工作集感知批大小控制、层分段预填充）技术，解决了动态稀疏注意力下KV缓存导致的HBM容量瓶颈问题，达成了9.26×更低平均TTFT延迟和3.14×更高令牌生成吞吐量的效果

### 67. SPEC-RL: ACCELERATING ON-POLICY REINFORCEMENT LEARNING VIA SPECULATIVE ROLLOUTS

清华大学发布了SPEC-RL论文，使用结合推测式解码的draft-and-verify机制重用轨迹段作为推测前缀的框架，解决了强化学习rollout阶段计算昂贵、存在冗余的问题，达成了减少rollout时间2-3倍且不影响策略质量的效果

### 68. ACCELERATING LARGE REASONING MODEL VIA SPECULATIVE EXIT

腾讯发布了《ACCELERATING LARGE REASONING MODEL VIA SPECULATIVE EXIT》论文，使用SpecExit框架（轻量级草稿模型直接预测未来token和早期退出信号，无探测开销），解决了大型推理模型因“过度思考”导致生成冗长、延迟高的问题，达成平均生成长度减少66%、端到端延迟加速2.5倍且不影响准确性的效果。

### 69. Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding

Seoul National University发布了Speculative Verification论文，使用引入辅助模型估计draft与目标模型分布对齐并利用信息增益优化验证决策的技术，解决了推测解码中推测准确率低导致被拒绝token开销大、尤其在大batch size时效果受限的问题，达成了最高提升推测解码性能2倍、大batch场景平均加速1.4倍的效果

### 70. Streamline pathology foundation model by cross-magnification distillation

俄亥俄州立大学韦克斯纳医学中心发布了Streamline pathology foundation model by cross-magnification distillation论文，使用跨放大倍数蒸馏技术，解决了病理基础模型因参数量大、高放大倍数处理需求导致的临床部署计算障碍问题，达成了30倍处理加速（8.8 WSIs/分钟）且诊断准确率与大型模型相差不到1%的效果。

### 71. Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack

南洋理工大学发布了“Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack”论文，使用双级优化及SCAR方法（隐式微分算法与预优化触发注入函数），解决了知识蒸馏中教师模型休眠后门在学生模型中通过KD激活的威胁，达成了有效实施蒸馏条件后门攻击并抵抗现有后门检测的效果。

### 72. TEQUILA: TRAPPING-FREE TERNARY QUANTIZA-TION FOR LARGE LANGUAGE MODELS

McGill University发布了TEQUILA论文，使用将死区被困权重重新用作动态偏置的无陷阱量化优化技术，解决了三元量化中权重死区陷阱导致模型容量和优化受损的问题，在ARC基准上较SOTA提升>4%、接近全精度性能（差距<1%）并实现3.0×推理加速。

### 73. TEXTURE VECTOR-QUANTIZATION AND RECONSTRUC-TION AWARE PREDICTION FOR GENERATIVE SUPER-RESOLUTION

电子科技大学发布了TEXTURE VECTOR-QUANTIZATION AND RECONSTRUC-TION AWARE PREDICTION FOR GENERATIVE SUPER-RESOLUTION论文，使用纹理矢量量化和重建感知预测策略，解决了现有矢量量化模型量化误差大及未考虑最终重建误差的问题，达成了生成逼真超分辨率结果且计算成本低的效果。

### 74. The Hidden Costs of Translation Accuracy: Distillation, Quantization, and Environmental Impact

加州大学发布了关于翻译准确性隐藏成本的论文，使用蒸馏和量化等模型压缩策略，解决了大语言模型翻译任务中计算与环境成本的问题，达成了在保持翻译质量的同时减少71-78%推理时间、63-65%碳排放的效果。

### 75. Tiny-QMoE

相关机构发布了Tiny-QMoE论文，使用8位量化与字典压缩结合的重新设计量化压缩方案，解决了边缘设备（如移动设备）的内存限制及网络请求延迟问题，达成了使原本超出内存需求的LLAMA 3.2模型能在受限设备（如2060 6GB VRAM）上运行的效果

### 76. Token Merging via Spatiotemporal Information Mining for Surgical Video Understanding

华中科技大学与香港科技大学发布了《Token Merging via Spatiotemporal Information Mining for Surgical Video Understanding》论文，使用解耦时空信息挖掘的token merging（STIM-TM）技术（通过时间显著性加权合并连续帧空间对应token、空间时间稳定性分析优先合并静态token的解耦策略），解决了手术视频理解中Vision Transformer因处理大量时空tokens导致的高计算成本问题（现有方法未充分考虑视频时空结构与信息分布异质性），达成了超65% GFLOPs减少同时保持竞争精度、支持长序列手术视频高效训练的效果。

### 77. TOWARDS A COMPREHENSIVE SCALING LAW OF MIXTURE-OF-EXPERTS

澳门大学发布了TOWARDS A COMPREHENSIVE SCALING LAW OF MIXTURE-OF-EXPERTS论文，使用系统分解MoE设置并通过446个控制实验构建综合联合MoE缩放定律的技术，解决了现有密集模型缩放定律不适用于MoE模型的问题（因影响因素多样、耦合复杂及性能影响非单调），达成了推导出理论和实际最优配置，G和S最优设置独立于模型架构和数据大小，随N增长Na/N最优激活参数比更稀疏，为MoE模型设计和训练提供准确指导的效果。

### 78. Towards Efficient CoT Distillation: Self-Guided Rationale Selector for Better Performance with Fewer Rationales

哈尔滨工业大学发布了关于高效CoT蒸馏的论文，使用模型导向的推理依据选择蒸馏（MoRSD）技术及Rationale Difficulty（RD）指标，解决了现有CoT蒸馏中推理依据质量低、传递噪声信息的问题，在7个数据集3个任务上平均提升4.6%，实现用更少推理依据获得更好性能。

### 79. Towards Trustworthy Lexical Simplification: Exploring Safety and Efficiency with Small LLMs

巴塞罗那庞培法布拉大学发布了关于可信词汇简化的论文，使用小型LLM的高效框架（含知识蒸馏、上下文学习及基于输出概率的过滤策略），解决了大LLM在词汇简化中面临的隐私资源限制及对弱势群体的安全正确性问题，达成了建立高效安全LS基准、抑制有害简化同时保留有益简化的效果。

### 80. TRAINING-FREE TOKEN PRUNING VIA ZEROTH-ORDER GRADIENT ESTIMATION IN VISION-LANGUAGE MODELS

Inha University和Sungkyunkwan University发布了TRAINING-FREE TOKEN PRUNING VIA ZEROTH-ORDER GRADIENT ESTIMATION IN VISION-LANGUAGE MODELS论文，使用基于零阶梯度估计的无训练标记修剪框架（ZOO-Prune），解决了视觉语言模型中冗余视觉标记导致的高推理成本及现有标记修剪方法的局限性，达成了修剪高达94.4%标记同时保持准确性、端到端推理速度提升2.30×的效果。

### 81. Training Agents Inside of Scalable World Models

研究团队发布了Training Agents Inside of Scalable World Models论文，使用Dreamer 4可扩展智能体（通过强化学习在快速准确的世界模型中训练，结合shortcut forcing objective和高效transformer架构实现实时推理，并从少量数据学习动作条件以从无标签视频提取知识）的技术，解决了复杂环境中世界模型无法准确预测物体交互及Minecraft中仅从离线数据获取钻石（需20,000+动作序列）的问题，达成了在Minecraft中准确预测物体交互大幅超越先前模型，成为首个仅从离线数据获取钻石智能体的效果。

### 82. TREE REWARD-ALIGNED SEARCH FOR TREASURE IN MASKED DIFFUSION LANGUAGE MODELS

复旦大学、中国科学技术大学发布了TREE REWARD-ALIGNED SEARCH FOR TREASURE IN MASKED DIFFUSION LANGUAGE MODELS论文，使用TREASURE（含UNMASKBRANCH分支策略与RE-SUBSTITUTESCORE修剪规则）技术，解决了掩码扩散语言模型树搜索中分支高度相关及奖励评估高方差问题，达成了在困惑度、语言可接受性等指标上的SOTA结果，尤其在低NFE场景下性能显著提升。

### 83. Tunable-Generalization Diffusion Powered by Self-Supervised Contextual Sub-Data for Low-Dose CT Reconstruction

研究团队发布了SuperDiff论文，使用自监督上下文子数据驱动的可调泛化扩散技术（含上下文子数据相似性自适应感知、知识蒸馏与潜在扩散模型结合及像素级自校正融合），解决了低剂量CT重建中依赖配对数据、泛化性差及跨剂量扩展困难的问题，达成了在数据集和真实数据上重建与泛化性能一致优于现有SOTA方法的效果。

### 84. TY-RIST: Tactical YOLO Tricks for Real-time Infrared Small Target Detection

卡尔斯鲁厄理工学院发布了TY-RIST论文，使用基于YOLO的战术性改进技巧，解决了红外小目标实时检测问题，提升了实时性与检测精度。

### 85. ULTRAUNET: REAL-TIME ULTRASOUND TONGUE SEGMENTATION FOR DIVERSE LINGUISTIC AND IMAGING CONDITIONS *

Biomedical Engineering Department发布了UltraUNet论文，使用轻量级编码器-解码器架构（含轻量级Squeeze-and-Excitation块、Group Normalization、summation-based skip connections及超声特定增强技术），解决了实时超声舌头轮廓分割的低信噪比、成像条件多变及计算需求挑战，达成250帧/秒实时处理速度与高分割精度（单数据集Dice=0.855、MSD=0.993px，跨数据集平均Dice 0.734/0.761）。

### 86. VAMamba: An Efficient Visual Adaptive Mamba for Image Restoration

中山大学发布了VAMamba论文，使用QCLAM和GPS-SS2D的Visual Adaptive Mamba框架，解决了现有基于Mamba的图像恢复方法受固定扫描模式和低效特征利用限制、无法适应多样退化的问题，达成了在多种恢复任务中恢复质量和效率均优于现有方法并建立新基准的效果。

### 87. VID-LLM: A COMPACT VIDEO-BASED 3D MULTIMODAL LLM WITH RECONSTRUCTION-REASONING SYNERGY

武汉大学、深圳大学发布了VID-LLM论文，使用基于视频的3D多模态大语言模型及重建-推理协同机制，解决视频驱动的3D多模态理解与推理问题，达成模型紧凑且高效的3D多模态理解与推理效果。

### 88. YOLO26: KEY ARCHITECTURAL ENHANCEMENTS AND PERFORMANCE BENCHMARKING FOR REAL-TIME OBJECT DETECTION

康奈尔大学发布了YOLO26论文，使用移除DFL、端到端无NMS推理、集成ProgLoss与STAL及MuSGD优化器等架构增强技术，解决了实时边缘设备目标检测的效率、准确性及部署准备问题，达成了作为YOLO家族最新最先进成员、支持多任务且在边缘设备性能优于前代YOLO和Transformer检测器的效果。

## 论文详细信息

### 1. Achieving Fair Skin Lesion Detection through Skin Tone Normalization and Channel Pruning

**主要机构**: Dept of Medical Physics & Biomedical Engineering, University College London
**作者数量**: 2人

**摘要**:
Recent works have shown that deep learning based skin lesion image classification models trained on unbalanced dataset can exhibit bias toward protected demographic attributes such as race, age, and gender. Current bias mitigation methods usually either achieve high level of fairness with the degradation of accuracy, or only improve the model fairness on a single attribute. Additionally usually most bias mitigation strtageies are either pre hoc through data processing or post hoc through fairness evaluation, instead of being integrated into the model learning itself. To solve these existing drawbacks, we propose a new Individual Typology Angle (ITA) Loss-based skin tone normalization and data augmentation method that directly feeds into an adaptable meta learning-based joint channel pruning framework. In skin tone normalization, ITA is used to estimate skin tone type and adjust automatically to target tones for dataset balancing. In the joint channel pruning framework, two nested optimization loops are used to find critical channels. The inner optimization loop finds and prunes the local critical channels by weighted soft nearest neighbor loss, and the outer optimization loop updates the weight of each attribute using group wise variance loss on meta-set. Experiments conducted in the ISIC2019 dataset validate the effectiveness of our method in simultaneously improving the fairness of the model on multiple sensitive attributes without significant degradation of accuracy. Finally, although the pruning mechanism adds some computational cost during training phase, usually training is done offline. More importantly, the pruned network becomes smaller in size and hence has lower compute time at inference stage thus making it easier to deploy in low resource clinical settings.

### 2. Agentic AI Reasoning for Mobile Edge General Intelligence: Fundamentals, Approaches, and Directions

**主要机构**: 
**作者数量**: 8人

**摘要**:
The rapid advancement of large language models (LLMs) has enabled an emergence of agentic artificial intelligence (AI) with powerful reasoning and autonomous decision-making capabilities. This integration with edge computing has led to the development of Mobile Edge General Intelligence (MEGI), which brings real-time, privacy-preserving reasoning to the network edge. However, deploying LLM-based agentic AI reasoning in MEGI environments poses significant challenges due to the high computational demands of reasoning and the limited resources of edge devices. To address these challenges, we propose a joint optimization framework for efficient LLM reasoning deployment in MEGI. First, we review methods that enhance LLM reasoning capabilities, such as Chain-of-Thought (CoT) prompting, Supervised Fine-Tuning (SFT), and Mixture of Experts (MoE). Next, we present a distributed framework that addresses two correlated aspects: reasoning enhancement through adaptive CoT prompting and scalable deployment through distributed MoE architecture. The framework dynamically activates expert networks and adjusts reasoning depth based on task complexity and device capabilities. We further conduct experimental evaluations in mobile edge environments. Experimental results demonstrate the framework's effectiveness in balancing reasoning quality with resource efficiency, validating the practical viability of deploying sophisticated LLM reasoning capabilities in resource-constrained MEGI environments.

### 3. An Enhanced Pyramid Feature Network Based on Long-Range Dependencies for Multi-Organ Medical Image Segmentation

**主要机构**: 
**作者数量**: 7人

**摘要**:
In the field of multi-organ medical image segmentation, recent methods frequently employ Transformers to capture long-range dependencies from image features. However, these methods overlook the high computational cost of Transformers and their deficiencies in extracting local detailed information. To address high computational costs and inadequate local detail information, we reassess the design of feature extraction modules and propose a new deep-learning network called LamFormer for fine-grained segmentation tasks across multiple organs. LamFormer is a novel U-shaped network that employs Linear Attention Mamba (LAM) in an enhanced pyramid encoder to capture multi-scale long-range dependencies. We construct the Parallel Hierarchical Feature Aggregation (PHFA) module to aggregate features from different layers of the encoder, narrowing the semantic gap among features while filtering information. Finally, we design the Reduced Transformer (RT), which utilizes a distinct computational approach to globally model up-sampled features. RRT enhances the extraction of detailed local information and improves the network's capability to capture long-range dependencies. LamFormer outperforms existing segmentation methods on seven complex and diverse datasets, demonstrating exceptional performance. Moreover, the proposed network achieves a balance between model performance and model complexity. The code has been made available on GitHub: https://github.com/kec1212/LamFormer.

### 4. Asymmetric VAE for One-Step Video Super-Resolution Acceleration ASYMMETRIC VAE FOR ONE-STEP VIDEO SUPER-RESOLUTION ACCELERATION

**主要机构**: Shanghai Jiao Tong University, South China University of Technology
**作者数量**: 4人

**摘要**:
Diffusion models have significant advantages in the field of real-world video super-resolution and have demonstrated strong performance in past research. In recent diffusion-based video super-resolution (VSR) models, the number of sampling steps has been reduced to just one, yet there remains significant room for further optimization in inference efficiency. In this paper, we propose FastVSR, which achieves substantial reductions in computational cost by implementing a high compression VAE (spatial compression ratio of 16, denoted as f16). We design the structure of the f16 VAE and introduce a stable training framework. We employ pixel shuffle and channel replication to achieve additional upsampling. Furthermore, we propose a lower-bound-guided training strategy, which introduces a simpler training objective as a lower bound for the VAE's performance. It makes the training process more stable and easier to converge. Experimental results show that FastVSR achieves speedups of 111.9 times compared to multistep models and 3.92 times compared to existing one-step models. We will release code and models at https://github.com/JianzeLi-114/FastVSR.

### 5. ATTENTION SURGERY: AN EFFICIENT RECIPE TO LIN-EARIZE YOUR VIDEO DIFFUSION TRANSFORMER

**主要机构**: AI Research is an initiative of Qualcomm Technologies, Inc
**作者数量**: 4人

**摘要**:
Transformer-based video diffusion models (VDMs) deliver state-of-the-art video generation quality but are constrained by the quadratic cost of self-attention, making long sequences and high resolutions computationally expensive. While linear attention offers sub-quadratic complexity, prior attempts fail to match the expressiveness of softmax attention without costly retraining. We introduce Attention Surgery, an efficient framework for linearizing or hybridizing attention in pretrained VDMs without training from scratch. Inspired by recent advances in language models, our method combines a novel hybrid attention mechanism-mixing softmax and linear tokens-with a lightweight distillation and fine-tuning pipeline requiring only a few GPU-days. Additionally, we incorporate a cost-aware blockrate strategy to balance expressiveness and efficiency across layers. Applied to Wan2.1 1.3B, a state-of-the-art DiT-based VDM, Attention Surgery achieves the first competitive sub-quadratic attention video diffusion models, reducing attention cost by up to 40% in terms of FLOPs, while maintaining generation quality as measured on the standard VBench and VBench-2.0 benchmarks.

### 6. Each Complexity Deserves a Pruning Policy

**主要机构**: Shanghai Jiao Tong University, AutoLab, School of Artificial Intelligence, State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS)
**作者数量**: 8人

**摘要**:
The established redundancy in visual tokens within large vision-language models (LVLMs) allows for pruning to effectively reduce their substantial computational demands. Empirical evidence from previous works indicates that visual tokens in later decoder stages receive less attention than shallow layers. Then, previous methods typically employ heuristics layer-specific pruning strategies where, although the number of tokens removed may differ across decoder layers, the overall pruning schedule is fixed and applied uniformly to all input samples and tasks, failing to align token elimination with the model's holistic reasoning trajectory. Cognitive science indicates that human visual processing often begins with broad exploration to accumulate evidence before narrowing focus as the target becomes distinct. Our experiments reveal an analogous pattern in LVLMs. This observation strongly suggests that neither a fixed pruning schedule nor a heuristics layer-wise strategy can optimally accommodate the diverse complexities inherent in different inputs. To overcome this limitation, we introduce Complexity-Adaptive Pruning (AutoPrune), which is a training-free, plug-and-play framework that tailors pruning policies to varying sample and task complexities. Specifically, AutoPrune quantifies the mutual information between visual and textual tokens, and then projects this signal to a budget-constrained logistic retention curve. Each such logistic curve, defined by its unique shape, is shown to effectively correspond with the specific complexity of different tasks, and can easily guarantee adherence to a pre-defined computational constraints. We evaluate AutoPrune not only on standard vision-language tasks but also on Vision-Language-Action (VLA) models for autonomous driving. Notably, when applied to LLaVA-1.5-7B, our method prunes 89% of visual tokens and reduces inference FLOPs by 76.8%, but still retaining 96.7% of the original accuracy averaged over all tasks. This corresponds to a 9.1% improvement over the recent work PDrop (CVPR'2025), demonstrating the effectivenes. Code is available at https://github.com/AutoLab-SAI-SJTU/AutoPrune.

### 7. A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving

**主要机构**: Guangdong Polytechnic Normal University, Kaon AI San Francisco, Sun Yat-sen University Guangdong
**作者数量**: 10人

**摘要**:
LLM inference serving typically scales out with a two-tier architecture: a cluster router distributes requests to multiple inference engines, each of which then in turn performs its own internal scheduling. However, this commonly used paradigm suffers from critical, systemic inefficiency caused by the information gaps across two layers. At the cluster-layer, the router mainly relies on lagging, coarse-grained metrics, such as average latency and queue length to make decisions, resulting in "decision lag" that leads to suboptimal request routing. At the engine-layer, static heuristic scheduling policies cannot effectively handle the dynamic workloads, leading a poor balance between latency and throughput. Besides, these gaps may cause SLO violations and resource waste, especially in heterogeneous cloud environments. To bridge such gaps, we propose NexusSched, a crosslayer framework that shifts LLM serving system from reactive load balancing to predictive orchestration. The core of NexusSched lies in a structurally-informed online performance model that provides accurate, forward-looking perstep latency and capacity estimations. This model empowers two key components. At the engine-layer, LENS performs SLO-aware, adaptive scheduling, dynamically optimizing batching to meet SLOs under real-time loads. At the clusterlayer, PRISM uses predictive signals to perform state-driven routing, maximizing cluster-wide performance and SLO attainment. Performance evaluations show that NexusSched improves SLO attainment by 43% on average and achieves up to 3× throughput speedup in long-context and heterogeneous scenarios. Besides, we also deploy NexusSched on FlowGPT's clusters to demonstrate its advantages in production environment.

### 8. A Second-Order Perspective on Pruning at Initialization and Knowledge Transfer

**主要机构**: Politecnico di Torino
**作者数量**: 3人

**摘要**:
The widespread availability of pre-trained vision models has enabled numerous deep learning applications through their transferable representations. However, their computational and storage costs often limit practical deployment. Pruning-at-Initialization has emerged as a promising approach to compress models before training, enabling efficient task-specific adaptation. While conventional wisdom suggests that effective pruning requires task-specific data, this creates a challenge when downstream tasks are unknown in advance. In this paper, we investigate how data influences the pruning of pre-trained vision models. Surprisingly, pruning on one task retains the model's zero-shot performance also on unseen tasks. Furthermore, fine-tuning these pruned models not only improves performance on original seen tasks but can recover held-out tasks' performance. We attribute this phenomenon to the favorable loss landscapes induced by extensive pre-training on large-scale datasets.

### 9. BALR-SAM: Boundary-Aware Low-Rank Adaptation of SAM for Resource-Efficient Medical Image Segmentation

**主要机构**: Shanghai Jiao Tong University
**作者数量**: 7人

**摘要**:
Vision foundation models like the Segment Anything Model (SAM), pretrained on large-scale natural image datasets, often struggle in medical image segmentation due to a lack of domain-specific adaptation. In clinical practice, fine-tuning such models efficiently for medical downstream tasks with minimal resource demands, while maintaining strong performance, is challenging. To address these issues, we propose BALR-SAM, a boundary-aware low-rank adaptation framework that enhances SAM for medical imaging. It combines three tailored components: (1) a Complementary Detail Enhancement Network (CDEN) using depthwise separable convolutions and multi-scale fusion to capture boundary-sensitive features essential for accurate segmentation; (2) lowrank adapters integrated into SAM's Vision Transformer blocks to optimize feature representation and attention for medical contexts, while simultaneously significantly reducing the parameter space; and (3) a lowrank tensor attention mechanism in the mask decoder, cutting memory usage by 75% and boosting inference speed. Experiments on standard medical segmentation datasets show that BALR-SAM, without requiring prompts, outperforms several state-of-the-art (SOTA) methods, including fully fine-tuned MedSAM, while updating just 1.8% (11.7M) of its parameters. Our code will be open-sourced upon acceptance.

### 10. Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability

**主要机构**: Indian Institute of Technology Bombay Powai, Department of Industrial Engineering and Operations Research
**作者数量**: 4人

**摘要**:
Early-Exit Deep Neural Networks enable adaptive inference by allowing prediction at intermediary layers, significantly reducing computational costs and latency. Most of the early exit strategies greedily exit a sample at an intermediary layer if the confidence in class prediction exceeds a predefined threshold that is set using a static validation set. This is problematic as the model might be overconfident in a wrong class. Also, they are not robust to distribution shifts encountered in deployment, which can undermine model trustworthiness and accuracy. To address these challenges, we propose UAT that adapts the threshold for exit decisions using a Multi-Armed Bandit framework, enabling online, unsupervised adjustment of exit decisions. UAT makes decisions based on a new reward function that assesses predictive certainty and its reliability to balance computational efficiency and prediction quality while penalizing unnecessary late exits. We provide guarantees on risk achieved by UAT and validate its performance on diverse tasks spanning vision-language understanding, text generation, and classification. Our framework demonstrates consistent improvements in speedup (1.70-2.10×) with a minimal performance drop (< 2%) as compared to full model performance. Our source code is available at https://github.com/Div290/UAT.

### 11. CAOTE: KV Cache Selection for LLMs via Attention Output Error-Based Token Eviction

**主要机构**: 
**作者数量**: 8人

**摘要**:
While long context support of large language models has extended their abilities, it also incurs challenges in memory and compute which becomes crucial bottlenecks in resource-restricted devices. Token eviction, a widely adopted post-training methodology designed to alleviate the bottlenecks by evicting less important tokens from the cache, typically uses attention scores as proxy metrics for token importance. However, one major limitation of attention score as a token-wise importance metrics is that it lacks the information about contribution of tokens to the attention output. In this paper, we propose a simple eviction criterion based on the contribution of cached tokens to attention outputs. Our method, CAOTE, optimizes for error due to token eviction, by seamlessly integrating attention scores and value vectors. This is the first method to use information from the value tokens on top of attention-based eviction scores in closed-form. Additionally, CAOTE can act as a meta-heuristic method with flexible usage with any token eviction method. We show that CAOTE, when combined with state-of-the-art attention score-based methods, always improves accuracies on the downstream task for LLAMA3 and QWEN2.5 model families, indicating the importance of leveraging information from values during token eviction process.

### 12. CasPoinTr: Point Cloud Completion with Cascaded Networks and Knowledge Distillation

**主要机构**: Fudan University, Institute of Science and Technology for Brain- Inspired Intelligence (ISTBI)
**作者数量**: 4人

**摘要**:
Point clouds collected from real-world environments are often incomplete due to factors such as limited sensor resolution, single viewpoints, occlusions, and noise. These challenges make point cloud completion essential for various applications. A key difficulty in this task is predicting the overall shape and reconstructing missing regions from highly incomplete point clouds. To address this, we introduce CasPoinTr, a novel point cloud completion framework using cascaded networks and knowledge distillation. CasPoinTr decomposes the completion task into two synergistic stages: Shape Reconstruction, which generates auxiliary information, and Fused Completion, which leverages this information alongside knowledge distillation to generate the final output. Through knowledge distillation, a teacher model trained on denser point clouds transfers incomplete-complete associative knowledge to the student model, enhancing its ability to estimate the overall shape and predict missing regions. Together, the cascaded networks and knowledge distillation enhance the model's ability to capture global shape context while refining local details, effectively bridging the gap between incomplete inputs and complete targets. Experiments on ShapeNet-55 under different difficulty settings demonstrate that CasPoinTr outperforms existing methods in shape recovery and detail preservation, highlighting the effectiveness of our cascaded structure and distillation strategy.

### 13. 

**主要机构**: Khoury College of Computer Sciences Northeastern University Boston
**作者数量**: 3人

**摘要**:
Model distillation typically focuses on behavioral mimicry, where a student model is trained to replicate a teacher's output while treating its internal computations as a black box. In this work we propose an alternative approach: Distilling the underlying computational mechanisms implemented by a teacher model. Specifically, we propose circuit distillation, which introduces an objective to align internal representations between analogous circuit components in teacher and student models. We propose a method to match "functionally correspondent" circuit components and introduce a loss reflecting similarities between the representations that these induce. We evaluate circuit distillation on entity tracking and theory of mind (ToM) tasks using models from the Llama3 family. Our results demonstrate that circuit distillation outperforms standard distillation, successfully transferring algorithmic capabilities by adjusting only a small, targeted subset of student model parameters. This work establishes the feasibility of transferring mechanisms, which may in turn allow for efficient distillation of targeted teacher capabilities via interpretable and controllable internal student mechanisms.

### 14. CLQ: CROSS-LAYER GUIDED ORTHOGONAL-BASED QUANTIZATION FOR DIFFUSION TRANSFORMERS

**主要机构**: Shanghai Jiao Tong University
**作者数量**: 4人

**摘要**:


### 15. Compute-Optimal Quantization-Aware Training

**主要机构**: École Polytechnique Fédérale de Lausanne (EPFL)
**作者数量**: 4人

**摘要**:
Quantization-aware training (QAT) is a leading technique for improving the accuracy of quantized neural networks. Previous work has shown that decomposing training into a full-precision (FP) phase followed by a QAT phase yields superior accuracy compared to QAT alone. However, the optimal allocation of compute between the FP and QAT phases remains unclear. We conduct extensive experiments with various compute budgets, QAT bit widths, and model sizes from 86.0M to 2.2B to investigate how different QAT durations impact final performance. We demonstrate that, contrary to previous findings, the loss-optimal ratio of QAT to FP training increases with the total amount of compute. Moreover, the optimal fraction can be accurately predicted for a wide range of model sizes and quantization widths using the tokens-perparameter-byte statistic. From experimental data, we derive a loss scaling law that predicts both optimal QAT ratios and final model performance across different QAT/FP compute allocation strategies and QAT bit widths. We use the scaling law to make further predictions, which we verify experimentally, including which QAT bit width is optimal under a given memory constraint and how QAT accuracy with different bit widths compares to full-precision model accuracy. Additionally, we propose a novel cooldown and QAT fusion approach that performs learning rate decay jointly with quantization-aware training, eliminating redundant full-precision model updates and achieving significant compute savings. These findings provide practical insights into efficient QAT planning and enable the training of higher-quality quantized models with the same compute budget.

### 16. Context-Driven Performance Modeling for Causal Inference Operators on Neural Processing Units

**主要机构**: University of Southern California
**作者数量**: 4人

**摘要**:
The proliferation of large language models (LLMs) has driven demand for long-context inference on resourceconstrained edge devices. However, deploying these models on Neural Processing Units (NPUs) presents significant challenges due to the architectural mismatch: quadratic complexity of standard attention mechanisms conflicts with memory and compute patterns of edge accelerators. This paper presents a comprehensive performance analysis of various causal inference operators on a modern NPU. We benchmark standard quadratic attention against several sub-quadratic alternatives, including structured state-space and linear attention models. Our analysis reveals that while sub-quadratic methods offer superior scalability, they introduce distinct computational bottlenecks on the NPU's specialized execution units. We identify that quadratic attention becomes severely memory-bound, suffering from cache inefficiency and pipeline stalls exceeding 95% at long contexts. In contrast, subquadratic models can become compute-bound on programmable vector cores. These findings provide critical insights for the codesign of hardware-aware models and optimization strategies to enable on-device AI inference with long-contexts.

### 17. D 2 CACHE: ACCELERATING DIFFUSION-BASED LLMS VIA DUAL ADAPTIVE CACHING

**主要机构**: Key Laboratory of New Generation Artificial Intelligence Technology and Its Interdisciplinary Applications (Southeast University, Qiyuan Tech
**作者数量**: 7人

**摘要**:
Diffusion-based large language models (dLLMs), despite their promising performance, still suffer from inferior inference efficiency. This is because dLLMs rely on bidirectional attention and cannot directly benefit from the standard key-value (KV) cache as autoregressive models (ARMs) do. To tackle this issue, we introduce Dual aDaptive Cache (d 2 Cache), which is a training-free approximate KV cache framework for accelerating dLLM inference. d 2 Cache features a twostage fine-grained selection strategy to identify tokens and adaptively update their KV states at each decoding step, while caching the KV states of the remaining tokens for reuse. Furthermore, d 2 Cache naturally offers a more reliable decoding alternative, which can enable quasi left-to-right generation and mitigate premature overconfidence in tokens at the end of the sequence. Extensive experimental results on two representative dLLMs (i.e., LLaDA and Dream) demonstrate that d 2 Cache not only achieves substantial inference speedups, but also yields consistent improvements in generation quality. The code is available at https://github.com/Kamichanw/d2Cache.

### 18. DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

**主要机构**: 
**作者数量**: 14人

**摘要**:
Existing text-to-image diffusion models excel at generating high-quality images, but face significant efficiency challenges when scaled to high resolutions, like 4K image generation. While previous research accelerates diffusion models in various aspects, it seldom handles the inherent redundancy within the latent space. To bridge this gap, this paper introduces DC-Gen, a general framework that accelerates text-to-image diffusion models by leveraging a deeply compressed latent space. Rather than a costly training-from-scratch approach, DC-Gen uses an efficient post-training pipeline to preserve the quality of the base model. A key challenge in this paradigm is the representation gap between the base model's latent space and a deeply compressed latent space, which can lead to instability during direct fine-tuning. To overcome this, DC-Gen first bridges the representation gap with a lightweight embedding alignment training. Once the latent embeddings are aligned, only a small amount of LoRA fine-tuning is needed to unlock the base model's inherent generation quality. We verify DC-Gen's effectiveness on SANA and FLUX.1-Krea. The resulting DC-Gen-SANA and DC-Gen-FLUX models achieve quality comparable to their base models but with a significant speedup. Specifically, DC-Gen-FLUX reduces the latency of 4K image generation by 53× on the NVIDIA H100 GPU. When combined with NVFP4 SVDQuant, DC-Gen-FLUX generates a 4K image in just 3.5 seconds on a single NVIDIA 5090 GPU, achieving a total latency reduction of 138× compared to the base FLUX.1-Krea model.

### 19. DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder

**主要机构**: 
**作者数量**: 16人

**摘要**:
We introduce DC-VideoGen, a post-training acceleration framework for efficient video generation. DC-VideoGen can be applied to any pre-trained video diffusion model, improving efficiency by adapting it to a deep compression latent space with lightweight fine-tuning. The framework builds on two key innovations: (i) a Deep Compression Video Autoencoder with a novel chunk-causal temporal design that achieves 32×/64× spatial and 4× temporal compression while preserving reconstruction quality and generalization to longer videos; and (ii) AE-Adapt-V, a robust adaptation strategy that enables rapid and stable transfer of pre-trained models into the new latent space. Adapting the pre-trained Wan-2.1-14B model with DC-VideoGen requires only 10 GPU days on the NVIDIA H100 GPU. The accelerated models achieve up to 14.8× lower inference latency than their base counterparts without compromising quality, and further enable 2160×3840 video generation on a single GPU. Table 2 Step Wan-2.1-T2V-1.3B DC-VideoGen-Wan-2.1-T2V-1.3B 480×832 1.49 0.24

### 20. DocPruner: A STORAGE-EFFICIENT FRAMEWORK FOR MULTI-VECTOR VISUAL DOCUMENT RETRIEVAL VIA ADAPTIVE PATCH-LEVEL EMBEDDING PRUNING

**主要机构**: Hong Kong University of Science and Technology, Alibaba Cloud Computing, Hong Kong University of Science and Technology (Guangzhou)
**作者数量**: 6人

**摘要**:
Visual Document Retrieval (VDR), the task of retrieving visually-rich document pages using queries that combine visual and textual cues, is crucial for numerous real-world applications. Recent state-of-the-art methods leverage Large Vision-Language Models (LVLMs) in a multi-vector paradigm, representing each document as patch-level embeddings to capture fine-grained details. While highly effective, this approach introduces a critical challenge: prohibitive storage overhead, as storing hundreds of vectors per page makes large-scale deployment costly and impractical. To address this, we introduce DocPruner, the first framework to employ adaptive patch-level embedding pruning for VDR to effectively reduce the storage overhead. DocPruner leverages the intra-document patch attention distribution to dynamically identify and discard redundant embeddings for each document. This adaptive mechanism enables a significant 50-60% reduction in storage for leading multi-vector VDR models with negligible degradation in document retrieval performance. Extensive experiments across more than ten representative datasets validate that DocPruner offers a robust, flexible, and effective solution for building storage-efficient, large-scale VDR systems.

### 21. EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering

**主要机构**: Zhejiang University
**作者数量**: 8人

**摘要**:
Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 5.5-11.4× speedup over existing frameworks. Extensive experiments demonstrate its effectiveness in overthinking mitigation, hallucination reduction, and other key applications. EasySteer transforms steering from research technique to production-ready capability, establishing critical infrastructure for deployable, controllable language models.

### 22. EfficientMIL: Efficient Linear-Complexity MIL Method for WSI Classification

**主要机构**: North University of China, Department of Radiology, Changhai Hospital, Chinese Academy of Sciences, Shanghai Advanced Research Institute
**作者数量**: 8人

**摘要**:
HIGHLIGHTS • EfficientMIL introduces linear-complexity multiple instance learning for whole slide image classification, replacing quadratic-complexity attention mechanisms with efficient RNNbased sequence models • Adaptive patches selector (APS) intelligently identifies informative patches using relevance, diversity, and uncertainty criteria, significantly outperforming conventional selection strategies • EfficientMIL achieves state-of-the-art performance on TCGA-Lung and CAMELYON16 datasets while requiring substantially lower computational resources than attention-based methods

### 23. EXGS: EXTREME 3D GAUSSIAN COMPRESSION WITH DIFFUSION PRIORS

**主要机构**: Hong Kong University of Science and Technology, Shanghai Artificial Intelligence Laboratory, Northwestern Polytechnical University
**作者数量**: 12人

**摘要**:


### 24. "Explainable Deep Learning for Cataract Detection in Retinal Images: A Dual-Eye and Knowledge Distillation Approach"

**主要机构**: University College of Nabi Akram, Department of Computer Engineering, Department of Computer Science, Northern Illinois University
**作者数量**: 3人

**摘要**:
Cataract remains a leading cause of visual impairment worldwide, and early detection from retinal imaging is critical for timely intervention. We present a deep learning pipeline for cataract classification using the Ocular Disease Recognition dataset, containing left and right fundus photographs from 5000 patients. We evaluated CNNs, transformers, lightweight architectures, and knowledge-distilled models. The top-performing model, Swin-Base Transformer, achieved 98.58% accuracy and an F1-score of 0.9836. A distilled MobileNetV3, trained with Swin-Base knowledge, reached 98.42% accuracy and a 0.9787 F1-score with greatly reduced computational cost. The proposed dual-eye Siamese variant of the distilled MobileNet, integrating information from both eyes, achieved an accuracy of 98.21%. Explainability analysis using Grad-CAM demonstrated that the CNNs concentrated on medically significant features, such as lens opacity and central blur. These results show that accurate, interpretable cataract detection is achievable even with lightweight models, supporting potential clinical integration in resource-limited settings.

### 25. Fast Thinking for Large Language Models

**主要机构**: Zheqi Lv Juncheng Li Siliang Tang Yueting Zhuang Hongyang He
**作者数量**: 5人

**摘要**:
Reasoning-oriented Large Language Models (LLMs) often rely on generating explicit tokens step by step, and their effectiveness typically hinges on large-scale supervised fine-tuning or reinforcement learning. While Chain-of-Thought (CoT) techniques substantially enhance performance on complex reasoning tasks, they remain inefficient, requiring long reasoning traces that increase latency and token usage. In this work, we introduce Latent Codebooks for Fast Thinking, a framework that uses concise CoT sketches only during training to learn a codebook of discrete strategy priors. At inference, the model conditions on a handful of continuous thinking vectors distilled from the codebook in a single pass, enabling strategy-level guidance without producing explicit reasoning tokens. To complement this design, we propose GainRouter, a lightweight routing mechanism that adaptively switches between fast codebook-guided inference and slow explicit reasoning, thereby suppressing overthinking and reducing unnecessary token generation. Experiments across multiple reasoning benchmarks show that our approach achieves competitive or superior accuracy while substantially lowering inference cost, offering a practical path toward efficient and controllable reasoning in large language models.

### 26. FLAME: A Serving System Optimized for Large-Scale Generative Recommendation with Efficiency

**主要机构**: Netease Cloud Music
**作者数量**: 9人

**摘要**:
Generative recommendation (GR) models possess greater scaling power compared to traditional deep learning recommendation models (DLRMs), yet they also impose a tremendous increase in computational burden. Measured in FLOPs, a typical GR model's workload sits in 10 9 ∼ 10 11 range, roughly four orders of magnitude higher than traditional DLRMs. Delivering accurate results in a few tens of milliseconds while processing billions of such requests per day puts extreme demands on the performance of the online serving system. Therefore, for industry practitioners, the alluring gains of GR models are tempered by the formidable challenge of online deployment at scale in production services. In this work, we introduce a comprehensive solution of online serving system tailored For Largescale Generative Recommendation with Efficiency (FLAME). Specifically, we leveraging CPU-GPU heterogeneous hardware to decouple feature pre-processing and model computation. We encapsulated several memory optimization features as the Proximal Data Accelerator (PDA) module to make full use of limited bandwidth and storage resources, which achieves a 1.9x throughput gain and a 1.7x latency reduction. We implement the Fused Kernel Engine (FKE) module based on the functionality and interface of NVIDIA TensorRT to boost model computation, delivering a speedup ratio of 4.6x-6.1x, throughput gain ratio of 4.7x-6.3x one step further. In addition, we design the Dynamic Stream Orchestrator (DSO) module to coordinate concurrent requests, enhancing the system throughput performance with 1.3x improvement in throughput and 2.3x speed-up under non-uniform distribution of upstream candidates. Comprehensive evaluations demonstrate that our FLAME effectively supports large-scale online deployment of GR models and achieves remarkable improvements in system performance.

### 27. FlowLUT: Efficient Image Enhancement via Differentiable LUTs and Iterative Flow Matching

**主要机构**: 
**作者数量**: 6人

**摘要**:
Deep learning-based image enhancement methods face a fundamental trade-off between computational efficiency and representational capacity. For example, although a conventional three-dimensional Look-Up Table (3D LUT) can process a degraded image in real time, it lacks representational flexibility and depends solely on a fixed prior. To address this problem, we introduce FlowLUT, a novel end-to-end model that integrates the efficiency of LUTs, multiple priors, and the parameterindependent characteristic of flow-matched reconstructed images. Specifically, firstly, the input image is transformed in color space by a collection of differentiable 3D LUTs (containing a large number of 3D LUTs with different priors). Next, a lightweight fusion prediction network runs on multiple 3D LUTs, with O(1) complexity for scene-adaptive color correction. Furthermore, to address the inherent representation limitations of LUTs, we design an innovative iterative flow matching method to restore local structural details and eliminate artifacts. Finally, the entire model is jointly optimized under a composite loss function enforcing perceptual and structural fidelity. Extensive experimental results demonstrate the effectiveness of our method on three benchmarks.

### 28. Generalist Multi-Class Anomaly Detection via Distillation to Two Heterogeneous Student Networks

**主要机构**: School of Computing, KAIST
**作者数量**: 3人

**摘要**:
Anomaly detection (AD) plays an important role in various real-world applications. Recent advancements in AD, however, are often biased towards industrial inspection, struggle to generalize to broader tasks like semantic anomaly detection and vice versa. Although recent methods have attempted to address general anomaly detection, their performance remains sensitive to dataset-specific settings and single-class tasks. In this paper, we propose a novel dual-model ensemble approach based on knowledge distillation (KD) to bridge this gap. Our framework consists of a teacher and two student models: an Encoder-Decoder model, specialized in detecting patch-level minor defects for industrial AD and an Encoder-Encoder model, optimized for semantic AD. Both models leverage a shared pre-trained encoder (DINOv2) to extract high-quality feature representations. The dual models are jointly learned using the Noisy-OR objective, and the final anomaly score is obtained using the joint probability via local and semantic anomaly scores derived from the respective models. We evaluate our method on eight public benchmarks under both single-class and multi-class settings: MVTec-AD, MVTec-LOCO, VisA and Real-IAD for industrial inspection and CIFAR-10/100, FMNIST and View for semantic anomaly detection. The proposed method achieved state-of-the-art accuracies in both domains, in multi-class as well as single-class settings, demonstrating generalization across multiple domains of anomaly detection. Our model achieved an image-level AUROC of 99.7% on MVTec-AD and 97.8% on CIFAR-10, which is significantly better than the prior general AD models in multi-class settings and even higher than the best specialist models on individual benchmarks.

### 29. Under review as a conference paper at ICLR 2026 GRACE-MOE: GROUPING AND REPLICATION WITH LOCALITY-AWARE ROUTING FOR EFFICIENT DISTRIBUTED MOE INFERENCE

**主要机构**: University of Science and Technology of China
**作者数量**: 6人

**摘要**:
Sparse Mixture of Experts (SMoE) performs conditional computation by selectively activating a subset of experts, thereby enabling scalable parameter growth in large language models (LLMs). However, the expanded parameter scale exceeds the memory capacity of a single device, necessitating distributed deployment for inference. This setup introduces two critical challenges: (1) Communication Issue: Transferring features to devices with activated experts leads to significant communication overhead. (2) Computational Load Issue: Skewed expert activation overloads certain GPUs, resulting in load imbalance across devices. Among these, communication overhead is identified as the main bottleneck in SMoE inference. Nevertheless, reducing communication between devices may exacerbate computational load imbalance, leading to device idleness and resource waste. Therefore, we present GRACE-MoE, short for Grouping and Replication with Locality-Aware Routing for SMoE inference. GRACE-MoE is a co-optimization framework that jointly reduces communication overhead and alleviates computational load imbalance. Specifically, the framework comprises two key phases: ① Grouping & Replication: This phase groups experts based on their affinity to reduce cross-device communication. Additionally, dynamic replication is applied to address load skew, improving computational load balance across GPUs. ② Routing: This phase employs a locality-aware routing strategy with load prediction. It prioritizes local replicas to minimize communication overhead and balances requests across remote replicas when necessary. Experiments on diverse models and multi-node, multi-GPU environments demonstrate that GRACE-MoE efficiently reduces end-to-end inference latency, achieving up to 3.79× speedup over stateof-the-art systems. Code for GRACE-MoE will be released upon acceptance.

### 30. Hazy Pedestrian Trajectory Prediction via Physical Priors and Graph-Mamba

**主要机构**: Shandong Jiaotong University, Sun Yat-sen University, Shandong Normal University
**作者数量**: 7人

**摘要**:
To address the issues of physical information degradation and ineffective pedestrian interaction modeling in pedestrian trajectory prediction under hazy weather conditions, we propose a deep learning model that combines physical priors of atmospheric scattering with topological modeling of pedestrian relationships. Specifically, we first construct a differentiable atmospheric scattering model that decouples haze concentration from light degradation through a network with physical parameter estimation, enabling the learning of haze-mitigated feature representations. Second, we design an adaptive scanning state space model for feature extraction. Our adaptive Mamba variant achieves a 78% inference speed increase over native Mamba while preserving long-range dependency modeling. Finally, to efficiently model pedestrian relationships, we develop a heterogeneous graph attention network, using graph matrices to model multi-granularity interactions between pedestrians and groups, combined with a spatio-temporal fusion module to capture the collaborative evolution patterns of pedestrian movements. Furthermore, we constructed a new pedestrian trajectory prediction dataset based on ETH/UCY to evaluate the effectiveness of the proposed method. Experiments show that our method reduces the minADE / minFDE metrics by 37.2% and 41.5%, respectively, compared to the SOTA models in dense haze scenarios (visibility < 30m), providing a new modeling paradigm for reliable perception in intelligent transportation systems in adverse environments.

### 31. HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models

**主要机构**: DL, Institute of Automation, Chinese Academy of Sciences
**作者数量**: 4人

**摘要**:
Speculative decoding is an effective approach for accelerating inference in Large Language models (LLMs), but its adaptation to Vision-Language models (VLMs) remains challenging for additional visual tokens in multimodal inputs. First, owing to the fact that the drafter and the target VLM may derived from different families, the semantic representations of visual tokens in the target VLM are misaligned with those in the drafter, introducing bias into the KV-cache during the prefill stage. Second, the large number of visual tokens substantially slows down the drafter's selfattention during the decoding stage. We propose Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models (HiViS), an explicit-implicit input decomposition framework that alleviates the above inefficiency. All visual tokens are removed from the drafter's input, retaining only textual tokens as explicit inputs, while directly reusing the target VLM's corresponding last-layer hidden states as implicit visual information without additional processing. To train the drafter efficiently, we introduces multi-step self-feedback training strategy with dynamic data selection and sequential embedding supervision to simulate reasoning during training. Our approach compresses the prefill sequence length of the drafter to only 0.7%-1.3% of the target VLM's input, while maintaining lossless generation quality. Extensive experiments across diverse models and tasks demonstrate up to 2.65× speedup, confirming the effectiveness of HiViS in accelerating VLM inference.

### 32. HIVTP: A Training-Free Method to Improve VLMs Efficiency via Hierarchical Visual Token Pruning Using Middle-Layer-Based Importance Score

**主要机构**: University of Southern California
**作者数量**: 5人

**摘要**:
Vision-Language Models (VLMs) have shown strong capabilities on diverse multimodal tasks. However, the large number of visual tokens output by the vision encoder severely hinders inference efficiency, and prior studies have shown that many of these tokens are not important and can therefore be safely pruned. In this work, we propose (HIVTP), a training-free method to improve VLMs efficiency via hierarchical visual token pruning using a novel middle-layer-based importance score. Specifically, we utilize attention maps extracted from the middle layers of the vision encoder, which better reflect fine-grained and objectlevel attention, to estimate visual token importance. Based on this, we propose a hierarchical visual token pruning method to retain both globally and locally important visual tokens. Specifically, we reshape the 1-D visual token sequence output by the vision encoder into a 2-D spatial layout. In the global retaining stage, we divide the image into regions and retain tokens with higher importance scores in each region; in the local retaining stage, we then divide the image into small windows and retain the most important token in each local window. Experimental results show that our proposed method, HIVTP, can reduce the time-to-first-token (TTFT) of LLaVA-v1.5-7B and LLaVA-Next-7B by up to 50.0% and 55.1%, respectively, and improve the token generation throughput by up to 60.9% and 47.3%, without sacrificing accuracy, and even achieving improvements on certain benchmarks. Compared with prior works, HIVTP achieves better accuracy while offering higher inference efficiency. Code: https://github.com/Blacktower27/HIVTP.

### 33. Hybrid ANN-SNN With Layer-Wise Surrogate Spike Encoding-Decoding Structure

**主要机构**: The University of Aizu, College of Engineering and Computer Science, College of Communication and Information Technology, Information and Network Management Center, Department of Computer Science and Engineering
**作者数量**: 5人

**摘要**:
Spiking Neural Networks (SNNs) have gained significant traction in both computational neuroscience and artificial intelligence for their potential in energy-efficient computing. In contrast, artificial neural networks (ANNs) excel at gradient-based optimization and high accuracy. This contrast has consequently led to a growing subfield of hybrid ANN-SNN research. However, existing hybrid approaches often rely on either a strict separation between ANN and SNN components or employ SNN-only encoders followed by ANN classifiers due to the constraints of non-differentiability of spike encoding functions, causing prior hybrid architectures to lack deep layer-wise cooperation during backpropagation. To address this gap, we propose a novel hybrid ANN-SNN framework that integrates layer-wise encode-decode SNN blocks within conventional ANN pipelines. Central to our method is the use of surrogate gradients for a bit-plane-based spike encoding function, enabling end-to-end differentiable training across ANN and SNN layers. This design achieves competitive accuracy with state-of-the-art pure ANN and SNN models while retaining the potential efficiency and temporal representation benefits of spiking computation. To the best of our knowledge, this is the first implementation of a surrogate gradient for bit plane coding specifically and spike encoder interface in general to be utilized in the context of hybrid ANN-SNN, successfully leading to a new class of hybrid models that pave new directions for future research. Source code for our experiments is publicly available at https://github.com/luutn2002/has-8.

### 34. INFLLM-V2: DENSE-SPARSE SWITCHABLE ATTEN-TION FOR SEAMLESS SHORT-TO-LONG ADAPTATION

**主要机构**: Tsinghua University, Harbin Institute of Technology
**作者数量**: 13人

**摘要**:
Long-sequence processing is a critical capability for modern large language models. However, the self-attention mechanism in the standard Transformer architecture faces severe computational and memory bottlenecks when processing long sequences. While trainable sparse attention methods offer a promising solution, existing approaches such as NSA introduce excessive extra parameters and disrupt the conventional pretrain-on-short, finetune-on-long workflow, resulting in slow convergence and difficulty in acceleration. To overcome these limitations, we introduce dense-sparse switchable attention framework, termed as InfLLM-V2. InfLLM-V2 is a trainable sparse attention that seamlessly adapts models from short to long sequences. Specifically, InfLLM-V2 reuses dense attention parameters through parameter-free architecture modification, maintaining consistency between short and long sequence processing. Additionally, InfLLM-V2 ensures computational efficiency across all sequence lengths, by using dense attention for short inputs and smoothly transitioning to sparse attention for long sequences. To achieve practical acceleration, we further introduce an efficient implementation of InfLLM-V2 that significantly reduces the computational overhead. Our experiments on long-context understanding and chain-of-thought reasoning demonstrate that InfLLM-V2 is 4× faster than dense attention while retaining 98.1% and 99.7% of the performance, respectively. Based on the InfLLM-V2 framework, we have trained and open-sourced MiniCPM4.1 1 , a hybrid reasoning model, providing a reproducible implementation for the research community.

### 35. LEARNING KAN-BASED IMPLICIT NEURAL REPRESENTATIONS FOR DEFORMABLE IMAGE REGISTRATION

**主要机构**: Lomonosov Moscow State University
**作者数量**: 3人

**摘要**:
Deformable image registration (DIR) is a cornerstone of medical image analysis, enabling spatial alignment for tasks like comparative studies and multi-modal fusion. While learning-based methods (e.g., CNNs, transformers) offer fast inference, they often require large training datasets and struggle to match the precision of classical iterative approaches on some organ types and imaging modalities. Implicit neural representations (INRs) have emerged as a promising alternative, parameterizing deformations as continuous mappings from coordinates to displacement vectors. However, this comes at the cost of requiring instance-specific optimization, making computational efficiency and seeddependent learning stability critical factors for these methods. In this work, we propose KAN-IDIR and RandKAN-IDIR, the first integration of Kolmogorov-Arnold Networks (KANs) into deformable image registration with implicit neural representations (INRs). The learnable activation functions of KANs and their inherent suitability for approximating physical systems make them ideal for modeling deformation fields. Our proposed randomized basis sampling strategy reduces the required number of basis functions in KAN while maintaining registration quality, thereby significantly lowering computational costs. We evaluated our approach on three diverse datasets (lung CT, brain MRI, cardiac MRI) and compared it with competing instance-specific learning-based approaches, datasettrained deep learning models, and classical registration approaches. KAN-IDIR and RandKAN-IDIR achieved the highest accuracy among INR-based methods across all evaluated modalities and anatomies, with minimal computational overhead and superior learning stability across multiple random seeds. Additionally, we discovered that our RandKAN-IDIR model with randomized basis sampling slightly outperforms the model with learnable basis function indices, while eliminating its additional training-time complexity. Source code is available at https://github.com/anac0der/KAN-IDIR.

### 36. Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding

**主要机构**: University of Central
**作者数量**: 7人

**摘要**:
Autoregressive decoding in large language models (LLMs) requires O(n) sequential steps for n tokens, fundamentally limiting inference throughput. Recent diffusion-based LLMs (dLLMs) enable parallel token generation through iterative denoising. However, current parallel decoding strategies rely on fixed, inputagnostic heuristics (e.g., confidence thresholds), which fail to adapt to input-specific characteristics, resulting in suboptimal speed-quality trade-offs across diverse NLP tasks. In this work, we explore a more flexible and dynamic approach to parallel decoding. We propose Learning to Parallel Decode (Learn2PD), a framework that trains a lightweight and adaptive filter model to predict, for each token position, whether the current prediction matches the final output. This learned filter approximates an oracle parallel decoding strategy that unmasks tokens only when correctly predicted. Importantly, the filter model is learned in a post-training manner, requiring only a small amount of computation to optimize it (minute-level GPU time). Additionally, we introduce End-of-Text Prediction (EoTP) to detect decoding completion at the end of sequence, avoiding redundant decoding of padding tokens. Experiments on the LLaDA [Nie et al., 2025] benchmark demonstrate that our method achieves up to 22.58× speedup without any performance drop, and up to 57.51× when combined with KV-Cache.

### 37. LLaDA-MoE: A Sparse MoE Diffusion Language Model

**主要机构**: Renmin University of China
**作者数量**: 26人

**摘要**:
We introduce LLaDA-MoE, a large language diffusion model with the Mixture-of-Experts (MoE) architecture, trained from scratch on approximately 20T tokens. LLaDA-MoE achieves competitive performance with significantly reduced computational overhead by maintaining a 7B-parameter capacity while activating only 1.4B parameters during inference. Our empirical evaluation reveals that LLaDA-MoE achieves state-of-the-art performance among diffusion language models with larger parameters, surpassing previous diffusion language models LLaDA, LLaDA 1.5, and Dream across multiple benchmarks. The instruct-tuned model LLaDA-MoE-7B-A1B-Instruct demonstrates capabilities comparable to Qwen2.5-3B-Instruct in knowledge understanding, code generation, mathematical reasoning, agent and alignment tasks, despite using fewer active parameters. Our results show that integrating a sparse MoE architecture into the training objective of masked diffusion language models still brings out MoE's strengths under efficient inference with few active parameters, and opens ample room for further exploration of diffusion language models. LLaDA-MoE models are available at Huggingface 1 .

### 38. LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models

**主要机构**: University of Illinois, University of California, HP Inc
**作者数量**: 7人

**摘要**:
Large Language Models (LLMs) with multimodal capabilities have revolutionized visionlanguage tasks, but their deployment often requires huge memory and computational resources. While post-training quantization (PTQ) has successfully compressed language models to as low as 1-bit precision without significant performance loss, its effectiveness for multimodal LLMs (MLLMs) remains relatively unexplored. In this paper, we present the first study on ultra-low bit (<4-bit) quantization for multimodal LLMs. Our analysis reveals that multimodal tokens and intermediate layer activations produced by them exhibit significantly higher statistical variance and entropy compared to text tokens, making them less tolerant to ultra-low bit quantization. However, the activation distributions of multimodal tokens varies significantly over different layers, with some layers having lower entropy activation distributions. We empirically show that such layers in these models can better tolerate ultra-low bit quantization. Building on these insights, we propose a novel strategy for MLLM quantization, LUQ: Layerwise Ultra-Low Bit Quantization, which selectively applies ultra-low bit quantization to layers that are more resilient to it. Additionally, we also show that using a mix of multimodal tokens (image and text) for PTQ boosts VQA performance in the ultra-low bit regime. We evaluate our method on LLaVA-1.5 and Qwen-2.5-VL across 9 popular VQA benchmarks. The resulting LUQ models use 40% and 31% less memory than their 4-bit counterparts, respectively, while exhibiting a performance degradation of less than 10% on the MME benchmark.

### 39. MAN: LATENT DIFFUSION ENHANCED MULTISTAGE ANTI-NOISE NETWORK FOR EFFICIENT AND HIGH-QUALITY LOW-DOSE CT IMAGE DENOISING

**主要机构**: School of Computer Science, University of Nottingham Ningbo China
**作者数量**: 4人

**摘要**:
While diffusion models have set a new benchmark for quality in Low-Dose Computed Tomography (LDCT) denoising, their clinical adoption is critically hindered by extreme computational costs, with inference times often exceeding thousands of seconds per scan. To overcome this barrier, we introduce MAN, a Latent Diffusion Enhanced Multistage Anti-Noise Network for Efficient and High-Quality Low-Dose CT Image Denoising task. Our method operates in a compressed latent space via a perceptually-optimized autoencoder, enabling an attention-based conditional U-Net to perform the fast, deterministic conditional denoising diffusion process with drastically reduced overhead. On the LDCT and Projection dataset, our model achieves superior perceptual quality, surpassing CNN/GAN-based methods while rivaling the reconstruction fidelity of computationally heavy diffusion models like DDPM and Dn-Dp. Most critically, in the inference stage, our model is over 60x faster than representative pixel space diffusion denoisers, while remaining competitive on PSNR/SSIM scores. By bridging the gap between high fidelity and clinical viability, our work demonstrates a practical path forward for advanced generative models in medical imaging.

### 40. 

**主要机构**: 
**作者数量**: 0人

**摘要**:


### 41. Model Fusion with Multi-LoRA Inference for Tool-Enhanced Game Dialogue Agents

**主要机构**: Netease Inc, Interactive Entertainment Group
**作者数量**: 6人

**摘要**:
This paper presents the opdainlp team's solution for the GPU track of the CPDC 2025 challenge. The challenge consists of three tasks, aiming to build an in-game conversational AI that adheres to character personas, aligns with the game's worldview, and supports function calling. Considering both effectiveness and resource/time constraints during inference, we synthesized data for some of the tasks based on the datasets provided by the competition organizers. We employed Qwen3-14B with LoRA fine-tuning and model fusion, and utilized a base model integrated with multiple LoRA adapters during inference. Specifically, in the competition, we used three distinct LoRA adapters to handle tool calling, response generation with tool call results, and response generation without tool call results, respectively. MultiLoRA inference was implemented using vLLM. Our solution achieved the first place in Task 1 and Task 3, and the second place in Task 2 of the GPU track.

### 42. P r e p r i n t . U n d e r R e v i e w . MoE-PHDS: One MoE checkpoint for flexible runtime sparsity

**主要机构**: University of California, Apple, Inc
**作者数量**: 7人

**摘要**:
Sparse Mixtures of Experts (MoEs) are typically trained to operate at a fixed sparsity level, e.g. k in a top-k gating function. This global sparsity level determines an operating point on the accuracy/latency curve; currently, meeting multiple efficiency targets means training and maintaining multiple models. This practice complicates serving, increases training and maintenance costs, and limits flexibility in meeting diverse latency, efficiency, and energy requirements. We show that pretrained MoEs are more robust to runtime sparsity shifts than commonly assumed, and introduce MoE-PHDS (Post Hoc Declared Sparsity), a lightweight SFT method that turns a single checkpoint into a global sparsity control surface. PHDS mixes training across sparsity levels and anchors with a short curriculum at high sparsity, requiring no architectural changes. The result is predictable accuracy/latency tradeoffs from one model: practitioners can "dial k" at inference time without swapping checkpoints, changing architecture, or relying on tokenlevel heuristics. Experiments on OLMoE-1B-7B-0125, Qwen1.5-MoE-A2.7B, and proprietary models fit on multiple operating points show that PHDS matches or exceeds well-specified oracle models, improves cross-sparsity agreement by up to 22% vs. well-specified oracle models, and enables simplified, flexible runtime MoE deployment by making global sparsity a first-class serving primitive.

### 43. MotionVerse: A Unified Multimodal Framework for Motion Comprehension, Generation and Editing

**主要机构**: 
**作者数量**: 5人

**摘要**:
This paper proposes MotionVerse, a unified framework that harnesses the capabilities of Large Language Models (LLMs) to comprehend, generate, and edit human motion in both single-person and multi-person scenarios. To efficiently represent motion data, we employ a motion tokenizer with residual quantization, which converts continuous motion sequences into multi-stream discrete tokens. Furthermore, we introduce a Delay Parallel Modeling strategy, which temporally staggers the encoding of residual token streams. This design enables LLMs to effectively capture inter-stream dependencies while maintaining computational efficiency comparable to single-stream modeling. Moreover, to alleviate modality interference between motion and language, we design a dual-tower architecture with modality-specific parameters, ensuring stable integration of motion information for both comprehension and generation tasks. Comprehensive ablation studies demonstrate the effectiveness of each component in MotionVerse, and extensive experiments showcase its superior performance across a wide range of motion-relevant tasks.

### 44. 

**主要机构**: 
**作者数量**: 0人

**摘要**:


### 45. PATCH: LEARNABLE TILE-LEVEL HYBRID SPARSITY FOR LLMS

**主要机构**: University of Toronto, Department of Computer Science
**作者数量**: 3人

**摘要**:
Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy-acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 8B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18×-1.38× end-to-end speedup over dense baselines while improving accuracy by 0.37%-2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM.

### 46. PEARL: Peer-Enhanced Adaptive Radio via On-Device LLM

**主要机构**: Nokia Technologies
**作者数量**: 3人

**摘要**:
We present PEARL (Peer-Enhanced Adaptive Radio via On-Device LLM), a framework for cooperative cross-layer optimization in device-to-device (D2D) communication. Building on our previous work on single-device on-device LLMs [Lee et al., 2025], PEARL extends the paradigm by leveraging both publisher and subscriber states to guide Wi-Fi Aware (WA) parameter selection. A contextaware reward, which normalizes latency by application tolerances and modulates energy by device battery states, provides richer supervision for KL-based finetuning. We study two lightweight variants: PEARL (Head + Low-Rank Adaptation (LoRA) [Hu et al., 2022]) achieves the best overall performance, while PEARL-Lite (Head-only) delivers sub-20 ms inference at near-identical objective scores. Across synthetic scenarios grounded in real measurements, PEARL improves objective scores over heuristic and compact model baselines and reduces energy by up to 16% in cooperative low-battery cases. These results demonstrate that peer-aware context, reward-aligned training, and head-based efficiency make LLMs practical for always-on, on-device cross-layer control. Code, real-world demo, and dataset are available at https://github.com/abman23/pearl.

### 47. PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement

**主要机构**: Harbin Institute of Technology, Hefei University of Technology, Macao Polytechnic University, Great Bay University, University of Science and Technology
**作者数量**: 9人

**摘要**:
Remote photoplethysmography (rPPG) measurement enables non-contact physiological monitoring but suffers from accuracy degradation under head motion and illumination changes. Existing deep learning methods are mostly heuristic and lack theoretical grounding, limiting robustness and interpretability. In this work, we propose a physics-informed rPPG paradigm derived from the Navier-Stokes equations of hemodynamics, showing that the pulse signal follows a second-order dynamical system whose discrete solution naturally leads to a causal convolution, justifying the use of a Temporal Convolutional Network (TCN). Based on this principle, we design the PHASE-Net, a lightweight model with three key components: 1) Zero-FLOPs Axial Swapper module to swap or transpose a few spatial channels to mix distant facial regions, boosting cross-region feature interaction without changing temporal order; 2) Adaptive Spatial Filter to learn a soft spatial mask per frame to highlight signal-rich areas and suppress noise for cleaner feature maps; and 3) Gated TCN, a causal dilated TCN with gating that models long-range temporal dynamics for accurate pulse recovery. Extensive experiments demonstrate that PHASE-Net achieves state-of-the-art performance and strong efficiency, offering a theoretically grounded and deployment-ready rPPG solution.

### 48. PROXY-GS: EFFICIENT 3D GAUSSIAN SPLATTING VIA PROXY MESH

**主要机构**: Hong Kong University of Science and Technology, Shanghai Artificial Intelligence Laboratory, Sichuan University, Northwestern Polytechnical University
**作者数量**: 9人

**摘要**:


### 49. PROXYATTN: GUIDED SPARSE ATTENTION VIA REP-RESENTATIVE HEADS

**主要机构**: Harbin Institute of Technology, Baidu Inc
**作者数量**: 7人

**摘要**:
The quadratic complexity of attention mechanisms limits the efficiency of Large Language Models (LLMs) on long-text tasks. Recently, methods that dynamically estimate block importance have enabled efficient block sparse attention, leading to significant acceleration in long-text pre-filling of LLMs. However, their coarse-grained estimation inevitably leads to performance degradation at high sparsity rates. In this work, we propose ProxyAttn, a training-free sparse attention algorithm that achieves more precise block estimation by compressing the dimension of attention heads. Based on our observation of the similarity among multiple attention heads, we use the scores of pooled representative heads to approximate the scores for all heads. To account for the varying sparsity among heads, we also propose a block-aware dynamic budget estimation method. By combining the scores from representative proxy heads with multi-head dynamic budgets, we achieve a more fine-grained block importance evaluation at low computational cost. Experiments on a variety of mainstream models and extensive benchmarks confirm the underlying similarity among attention heads. Leveraging a fine-grained estimation, the proposed method achieves substantial gains in performance and efficiency compared to existing methods. More precisely, ProxyAttn can achieve up to 10.3x attention acceleration and 2.4x prefilling acceleration without significant performance loss. Our code is available at https://github.com/wyxstriker/ProxyAttn.

### 50. QUANTSPARSE: COMPREHENSIVELY COMPRESSING VIDEO DIFFUSION TRANSFORMER WITH MODEL QUANTIZATION AND ATTENTION SPARSIFICATION

**主要机构**: ETH Zürich, Institute of Computing Technology, University of Chinese Academy of Sciences, Chinese Academy of Sciences
**作者数量**: 11人

**摘要**:


### 51. REASONING SCAFFOLDING: DISTILLING THE FLOW OF THOUGHT FROM LLMS

**主要机构**: HUAWEI Hong Kong Research Center, HUAWEI Noah's Ark Lab, Southeast University, The Chinese University of Hong
**作者数量**: 9人

**摘要**:
The prevailing approach to distilling reasoning from Large Language Models (LLMs)-behavioral cloning from textual rationales-is fundamentally limited. It teaches Small Language Models (SLMs) to mimic surface-level patterns rather than the underlying algorithmic structure of thought, resulting in a critical lack of logical robustness. We argue that instead of cloning text, distillation should transfer this algorithmic structure directly. We introduce Reasoning Scaffolding, a framework that reframes reasoning as a structured generation process. Our method first abstracts the teacher's thought process into a sequence of discrete, interpretable semantic signals (e.g., Contrast, Addition) that act as a scaffold. The student model is then trained via a multi-task objective to both (1) predict the next semantic signal, anticipating the reasoning flow, and (2) generate the corresponding step, conditioned on that signal. This multi-task scheme acts as a powerful regularizer, compelling the student to internalize the computational patterns of coherent reasoning. On a suite of challenging reasoning benchmarks, our method significantly outperforms state-of-the-art distillation in both accuracy and logical consistency, providing a path towards creating smaller models that are genuine reasoners, not just fluent mimics 1 .

### 52. ReSeFlow: Rectifying SE(3)-Equivariant Policy Learning Flows

**主要机构**: 
**作者数量**: 5人

**摘要**:
Robotic manipulation in unstructured environments requires the generation of robust and long-horizon trajectory-level policy with conditions of perceptual observations and benefits from the advantages of SE(3)-equivariant diffusion models that are data-efficient. However, these models suffer from the inference time costs. Inspired by the inference efficiency of rectified flows, we introduce the rectification to the SE(3)-diffusion models and propose the ReSeFlow, i.e., Rectifying SE(3)-Equivariant Policy Learning Flows, providing fast, geodesic-consistent, least-computational policy generation. Crucially, both components employ SE(3)-equivariant networks to preserve rotational and translational symmetry, enabling robust generalization under rigid-body motions. With the verification on the simulated benchmarks, we find that the proposed ReSeFlow with only one inference step can achieve better performance with lower geodesic distance than the baseline methods, achieving up to a 48.5% error reduction on the painting task and a 21.9% reduction on the rotating triangle task compared to the baseline's 100-step inference. This method takes advantages of both SE(3) equivariance and rectified flow and puts it forward for the real-world application of generative policy learning models with the data and inference efficiency.

### 53. RESTORECT: DEGRADED IMAGE RESTORATION VIA LATENT RECTIFIED FLOW & FEATURE DISTILLATION A PREPRINT

**主要机构**: Purdue University, Institute for Cancer Research *Equal Contribution, Department of Computer Science
**作者数量**: 4人

**摘要**:
Current approaches for restoration of degraded images face a critical trade-off: high-performance models are too slow for practical use, while fast models produce poor results. Knowledge distillation transfers teacher knowledge to students, but existing static feature matching methods cannot capture how modern transformer architectures dynamically generate features. We propose 'RestoRect', a novel Latent Rectified Flow Feature Distillation method for restoring degraded images. We apply rectified flow to reformulate feature distillation as a generative process where students learn to synthesize teacher-quality features through learnable trajectories in latent space. Our framework combines Retinex theory for physics-based decomposition with learnable anisotropic diffusion constraints, and trigonometric color space polarization. We introduce a Feature Layer Extraction loss for robust knowledge transfer between different network architectures through cross-normalized transformer feature alignment with percentile-based outlier detection. RestoRect achieves better training stability, and faster convergence and inference while preserving restoration quality. We demonstrate superior results across 15 image restoration datasets, covering 4 tasks, on 8 metrics.

### 54. RETHINKING LARGE LANGUAGE MODEL DISTILLA-TION: A CONSTRAINED MARKOV DECISION PROCESS PERSPECTIVE

**主要机构**: Huawei Noah's Ark Lab UCL Centre for Artificial Intelligence, Huawei Noah's Ark Lab
**作者数量**: 5人

**摘要**:
We introduce a novel approach to large language model (LLM) distillation by formulating it as a constrained reinforcement learning problem. While recent work has begun exploring the integration of task-specific rewards into distillation processes, existing methods typically rely on ad-hoc reward weighting. We propose a principled optimization framework that maximizes task-specific rewards while constraining the divergence from the teacher model to remain below a specified threshold. Our approach adapts constrained state augmented reinforcement learning to the distillation setting, introducing a modified reward function that maintains theoretical guarantees of constraint satisfaction without requiring state augmentation or teacher model access during deployment and without the computational overhead of the dual Lagrangian methods. Through extensive experiments on mathematical reasoning tasks, we demonstrate that our method achieves better constraint satisfaction rates and better reasoning compared to the soft Lagrangian relaxation baselines while maintaining competitive task performance. Our framework provides a theoretically grounded and practically efficient solution for reward-aware distillation in resource-constrained settings.

### 55. ROBUQ: PUSHING DITS TO W1.58A2 VIA ROBUST ACTIVATION QUANTIZATION

**主要机构**: Shanghai Jiao Tong University, Tsinghua University
**作者数量**: 7人

**摘要**:


### 56. ROLLING FORCING: AUTOREGRESSIVE LONG VIDEO DIFFUSION IN REAL TIME

**主要机构**: Nanyang Technological University, ARC Lab
**作者数量**: 5人

**摘要**:
A skeleton wearing a flower hat and sunglasses dances in the wild at sunset. A longboarder accelerating downhill, carving through turns. A handheld camera following a dog running through a park.

### 57. RServe: Overlapping Encoding and Prefill for Efficient LMM Inference

**主要机构**: Sun Yat-sen University Guangzhou, CSE, Xianwei Zhang, Tianming Xu, Nong Xiao
**作者数量**: 9人

**摘要**:
Large multimodal models (LMMs) typically employ an encoding module to transform multimodal data inputs into embeddings, which are then fed to language models for further processing. However, efficiently serving LMMs remains highly challenging due to the inherent complexity of their inference pipelines. Traditional serving engines co-locate the encoding module and the language model, leading to significant resource interference and tight data dependency. Recent studies have alleviated this issue by disaggregating the encoding module from the model, following a design style of prefill-decode disaggregation. Nevertheless, these approaches fail to fully exploit parallelism both within individual requests (intra-request) and across multiple requests (inter-request). To overcome the limitation, we propose RServe, an LMM inference system that efficiently orchestrates intra-and interrequest pipelines. RServe is designed to reduce low latency and maximize parallelism at both intra-and inter-request granularities. Built on the disaggregated architecture of the encoding module and language model, RServe adopts a finegrained scheduling method that overlaps multimodal encoding with the forward computation of the language model within a single request. For inter-request pipeline, RServe leverages schedulable tokens and token budgets to balance computational loads across micro-batches. Combined with chunked prefill, this enables a novel scheduling strategy that coordinates the execution of intra-and inter-request pipelines. Experimental evaluations on representative LMMs show that RServe achieves substantial latency reduction of up to 66% while improving throughput by up to 109%, significantly outperforming existing serving approaches. CCS Concepts: • Computing methodologies → Distributed computing methodologies; • Computer systems organization → Cloud computing.

### 58. S 2 NN: Sub-bit Spiking Neural Networks

**主要机构**: University of Electronic Science and Technology of China
**作者数量**: 11人

**摘要**:
Spiking Neural Networks (SNNs) offer an energy-efficient paradigm for machine intelligence, but their continued scaling poses challenges for resource-limited deployment. Despite recent advances in binary SNNs, the storage and computational demands remain substantial for large-scale networks. To further explore the compression and acceleration potential of SNNs, we propose Sub-bit Spiking Neural Networks (S 2 NNs) that represent weights with less than one bit. Specifically, we first establish an S 2 NN baseline by leveraging the clustering patterns of kernels in well-trained binary SNNs. This baseline is highly efficient but suffers from outlier-induced codeword selection bias during training. To mitigate this issue, we propose an outlier-aware sub-bit weight quantization (OS-Quant) method, which optimizes codeword selection by identifying and adaptively scaling outliers. Furthermore, we propose a membrane potential-based feature distillation (MPFD) method, improving the performance of highly compressed S 2 NN via more precise guidance from a teacher model. Extensive results on vision and non-vision tasks reveal that S 2 NN outperforms existing quantized SNNs in both performance and efficiency, making it promising for edge computing applications.

### 59. SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

**主要机构**: Equal contribution. Project
**作者数量**: 21人

**摘要**:
We introduce SANA-Video, a small diffusion model that can efficiently generate videos up to 720×1280 resolution and minute-length duration. SANA-Video synthesizes high-resolution, high-quality and long videos with strong text-video alignment at a remarkably fast speed, deployable on RTX 5090 GPU. Two core designs ensure our efficient, effective and long video generation: (1) Linear DiT: We leverage linear attention as the core operation, which is more efficient than vanilla attention given the large number of tokens processed in video generation. (2) Constant-Memory KV Cache for Block Linear Attention: we design block-wise autoregressive approach for long video generation by employing a constant-memory state, derived from the cumulative properties of linear attention. This KV cache provides the Linear DiT with global context at a fixed memory cost, eliminating the need for a traditional KV cache and enabling efficient, minute-long video generation. In addition, we explore effective data filters and model training strategies, narrowing the training cost to 12 days on 64 H100 GPUs, which is only 1% of the cost of MovieGen. Given its low cost, SANA-Video achieves competitive performance compared to modern state-of-the-art small diffusion models (e.g., Wan 2.1-1.3B and SkyReel-V2-1.3B) while being 16× faster in measured latency. Moreover, SANA-Video can be deployed on RTX 5090 GPUs with NVFP4 precision, accelerating the inference speed of generating a 5-second 720p video from 71s to 29s (2.4× speedup). In summary, SANA-Video enables low-cost, high-quality video generation.

### 60. Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

**主要机构**: Microsoft Research, Tsinghua University, University of Science and Technology of China, Tsinghua Univeristy, Institute for AI Industry Research (AIR)
**作者数量**: 8人

**摘要**:
Deploying Large Language Models (LLMs) on mobile devices faces the challenge of insufficient performance in smaller models and excessive resource consumption in larger ones. This paper highlights that mobile Neural Processing Units (NPUs) have underutilized computational resources, particularly their matrix multiplication units, during typical LLM inference. To leverage this wasted compute capacity, we propose applying parallel test-time scaling techniques on mobile NPUs to enhance the performance of smaller LLMs. However, this approach confronts inherent NPU challenges, including inadequate hardware support for fine-grained quantization and low efficiency in general-purpose computations. To overcome these, we introduce two key techniques: a hardwareaware tile quantization scheme that aligns group quantization with NPU memory access patterns, and efficient LUTbased replacements for complex operations such as Softmax and dequantization. We design and implement an end-to-end inference system that leverages the NPU's compute capability to support test-time scaling on Qualcomm Snapdragon platforms. Experiments show our approach brings significant speedups: up to 19.0× for mixed-precision GEMM and 2.2× for Softmax. More importantly, we demonstrate that smaller models using test-time scaling can match or exceed the accuracy of larger models, achieving a new performance-cost Pareto frontier.

### 61. SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts via Token-Level LSH Matching

**主要机构**: University of Notre
**作者数量**: 2人

**摘要**:
As large language models (LLMs) continue to scale, the memory footprint of key-value (KV) caches during inference has become a significant bottleneck. Existing approaches primarily focus on compressing KV caches within a single prompt or reusing shared prefixes or frequently ocurred text segments across prompts. However, such strategies are limited in scenarios where prompts are semantically similar but lexically different, which frequently occurs in tasks such as multi-document summarization and conversational agents. We propose SemShareKV, a KV cache sharing and compression framework that accelerates LLM inference by reusing KVCache in semantically similar prompts. Instead of relying on exact token matches, SemShareKV applies fuzzy token matching using locality-sensitive hashing (LSH) on token embeddings and incorporates Rotary Position Embedding (RoPE) to better preserve positional information. By selectively reusing relevant key-value pairs from a reference prompt's cache, SemShareKV reduces redundant computation while maintaining output quality. Experiments on diverse summarization datasets show up to 6.25× speedup and 42% lower GPU memory usage with 5k tokens input, with negligible quality degradation. These results highlight the potential of semantic-aware cache sharing for efficient LLM inference.

### 62. Sequential Token Merging: Revisiting Hidden States

**主要机构**: Fudan University, College of Future Information Technology, Shanghai Artificial Intelligence Laboratory, Independent Researcher
**作者数量**: 7人

**摘要**:
Vision Mambas (ViMs) achieve remarkable success with sub-quadratic complexity, but their efficiency remains constrained by quadratic token scaling with image resolution. While existing methods address token redundancy, they overlook ViMs' intrinsic Limited Directional Sequential Dependence (LDSD)-a critical information flow mechanism revealed in our analysis. We further identify Mamba's selective scan enables gradual information aggregation in hidden states. Based on these insights, we propose Sequential Token Merging (STM), featuring: 1) Bidirectional nearest neighbor merging to preserve sequential dependencies through symmetric spatial aggregation, and 2) Hidden states protection to stabilize the hidden states around the class token. STM strategically leverages Mamba's layerwise loss convergence to convert temporal forgetfulness into stability. Experiments demonstrate STM's superiority: 1.0% accuracy drop for ViM-Ti at 20% token reduction, and only 1.4% degradation for ViM-S at 40% reduction. Our method achieves state-of-the-art efficiency with minimal complexity, while providing new insights into state-space model dynamics. Codes will be released soon.

### 63. Similarity-Aware Selective State-Space Modeling for Semantic Correspondence

**主要机构**: Pohang University of Science and Technology (POSTECH)
**作者数量**: 2人

**摘要**:
Establishing semantic correspondences between images is a fundamental yet challenging task in computer vision. Traditional feature-metric methods enhance visual features but may miss complex inter-correlation relationships, while recent correlation-metric approaches are hindered by high computational costs due to processing 4D correlation maps. We introduce MambaMatcher, a novel method that overcomes these limitations by efficiently modeling high-dimensional correlations using selective statespace models (SSMs). By implementing a similarity-aware selective scan mechanism adapted from Mamba's linearcomplexity algorithm, MambaMatcher refines the 4D correlation map effectively without compromising feature map resolution or receptive field. Experiments on standard semantic correspondence benchmarks demonstrate that Mam-baMatcher achieves state-of-the-art performance.

### 64. SLA: BEYOND SPARSITY IN DIFFUSION TRANSFORM-ERS VIA FINE-TUNABLE SPARSE-LINEAR ATTENTION

**主要机构**: Tsinghua University
**作者数量**: 13人

**摘要**:
In Diffusion Transformer (DiT) models, particularly for video generation, attention latency is a major bottleneck due to the long sequence length and the quadratic complexity. Interestingly, we find that attention weights can be decoupled into two matrices: a small fraction of large weights with high rank and the remaining weights with very low rank. This naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second. Based on this finding, we propose SLA (Sparse-Linear Attention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. SLA classifies attention weights into critical, marginal, and negligible, applying O(N 2) attention to critical weights, O(N) attention to marginal weights, and skipping negligible ones. SLA combines these computations into a single GPU kernel and supports both forward and backward passes. With only a few fine-tuning steps using SLA, DiT models achieve a 20× reduction in attention computation, resulting in significant acceleration without loss of generation quality. Experiments show that SLA reduces attention computation by 95% without degrading end-to-end generation quality, outperforming baseline methods. In addition, we implement an efficient GPU kernel for SLA, which yields a 13.7× speedup in attention computation and a 2.2× end-to-end speedup in video generation on Wan2.1-1.3B. The code will be available at https://github.com/thu-ml/SLA.

### 65. SPARSED: SPARSE ATTENTION FOR DIFFUSION LAN-GUAGE MODELS

**主要机构**: National University of Singapore, The Hong Kong Polytechnic University
**作者数量**: 5人

**摘要**:
While diffusion language models (DLMs) offer a promising alternative to autoregressive models (ARs), existing open-source DLMs suffer from high inference latency. This bottleneck is mainly due to the attention's quadratic complexity with respect to context length in computing all query-key pairs. Intuitively, to reduce this complexity, a natural strategy is to restrict attention to sparse patterns that retain only the most relevant connections. Such approaches are well-established in ARs, where attention follows fixed and clearly defined sparse patterns. However, in DLMs, we observe distinct sparsity behaviors: (1) attention patterns vary across heads, (2) attention patterns in each head remain highly similar across denoising steps, and (3) early denoising steps are critical for generation. These findings render sparse attention methods designed for ARs largely incompatible with DLMs, as they fail to capture head-specific structures and risk degrading generation when applied in early denoising steps. To address these challenges, we propose SparseD, a novel sparse attention method for DLMs. Leveraging the observations, SparseD only requires pre-computing head-specific sparse patterns one time, and reuses them across all steps. This prevents recomputing sparse patterns at each denoising step. Meanwhile, SparseD uses full attention in the early steps, then switches to sparse attention later to maintain generation quality. Together, these establish SparseD as a practical and efficient solution for deploying DLMs in long-context applications. Experimental results demonstrate that SparseD achieves lossless acceleration, delivering up to 1.50× speedup over FlashAttention at a 64k context length with 1,024 denoising steps. Code is available at https://github.com/INV-WZQ/SparseD.

### 66. SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving

**主要机构**: Huawei Cloud, The Chinese University of Hong Kong
**作者数量**: 4人

**摘要**:
Serving long-context LLMs is costly because attention computation grows linearly with context length. Dynamic sparse attention algorithms (DSAs) mitigate this by attending only to the key-value (KV) cache of critical tokens. However, with DSAs, the main performance bottleneck shifts from HBM bandwidth to HBM capacity: KV caches for unselected tokens must remain in HBM for low-latency decoding, constraining parallel batch size and stalling further throughput gains. Offloading these underutilized KV caches to DRAM could free HBM capacity, allowing larger parallel batch sizes. Yet, achieving such hierarchical HBM-DRAM storage raises new challenges, including fragmented KV cache access, HBM cache contention, and high HBM demands of hybrid batching, that remain unresolved in prior work. This paper proposes SparseServe, an LLM serving system that unlocks the parallel potential of DSAs through efficient hierarchical HBM-DRAM management. SparseServe introduces three key innovations to address the challenges mentioned above: (1) fragmentation-aware KV cache transfer, which accelerates HBM-DRAM data movement through GPU-direct loading (FlashH2D) and CPU-assisted saving (FlashD2H); (2) working-set-aware batch size control that adjusts batch sizes based on real-time working set estimation to minimize HBM cache thrashing; (3) layer-segmented prefill that bounds HBM use during prefill to a single layer, enabling efficient execution even for long prompts. Extensive experimental results demonstrate that SparseServe achieves up to 9.26× lower mean time-to-first-token (TTFT) latency and up to 3.14× higher token generation throughput compared to state-of-the-art LLM serving systems.

### 67. SPEC-RL: ACCELERATING ON-POLICY REINFORCEMENT LEARNING VIA SPECULATIVE ROLLOUTS

**主要机构**: Xiamen University, Shopee Pte. Ltd, Tsinghua University, School of Informatics, Institute for AI Industry Research (AIR), LLM Team
**作者数量**: 8人

**摘要**:
Large Language Models (LLMs) increasingly rely on reinforcement learning with verifiable rewards (RLVR) to elicit reliable chain-of-thought reasoning. However, the training process remains bottlenecked by the computationally expensive rollout stage. Existing acceleration methods-such as parallelization, objective-and data-driven modifications, and replay buffers-either incur diminishing returns, introduce bias, or overlook redundancy across iterations. We identify that rollouts from consecutive training epochs frequently share a large portion of overlapping segments, wasting computation. To address this, we propose SPEC-RL, a novel framework that integrates SPECulative decoding with the RL rollout process. SPEC-RL reuses prior trajectory segments as speculative prefixes and extends them via a draft-and-verify mechanism, avoiding redundant generation while ensuring policy consistency. Experiments on diverse math reasoning and generalization benchmarks, including GSM8K, MATH-500, OlympiadBench, MMLU-STEM, and others, demonstrate that SPEC-RL reduces rollout time by 2-3× without compromising policy quality. As a purely rollout-stage enhancement, SPEC-RL integrates seamlessly with mainstream algorithms (e.g., PPO, GRPO, DAPO), offering a general and practical path to scale RLVR for large reasoning models. Our code is available at: https://github.com/ShopeeLLM/Spec-RL.

### 68. ACCELERATING LARGE REASONING MODEL VIA SPECULATIVE EXIT

**主要机构**: 
**作者数量**: 10人

**摘要**:
Despite their strong performance on reasoning tasks, large reasoning models (LRMs) often suffer from overthinking, producing unnecessarily long outputs and incurring high end-to-end latency, a significant limitation to their real-world deployment. To address overthinking, early-exit mechanisms have been proposed to terminate reasoning before typical completion, showing that this approach can effectively shorten generation length with minimal impact on accuracy. However, their reliance on probing mechanisms introduces a detection overhead that limits their end-to-end latency gains and compromises their generalizability across diverse problems. Inspired by the use of hidden states in speculative decoding, we propose SpecExit, a novel framework that predicts both future tokens and an early-exit signal directly from a lightweight draft model without probing overhead. Our method offers significant improvements, reducing average generation length by 66% and achieving a 2.5x speedup in end-to-end latency compared to the speculative decoding baseline, without compromising accuracy. Our method leverages the inherent signals from hidden states to provide effective early-exit signals, suggesting broader use of hidden states for efficient reasoning. Our code is available at: https://github.com/Tencent/AngelSlim.

### 69. Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding

**主要机构**: Seoul National University, University of Seoul, Hanyang University
**作者数量**: 6人

**摘要**:
LLMs have low GPU efficiency and high latency due to autoregressive decoding. Speculative decoding (SD) mitigates this using a small draft model to speculatively generate multiple tokens, which are then verified in parallel by a target model. However, when speculation accuracy is low, the overhead from rejected tokens can offset the benefits, limiting SD's effectiveness, especially at large batch sizes. To address this, we propose Speculative Verification (SV), an efficient augmentation to SD that dynamically predicts speculation accuracy and adapts the verification length to maximize throughput. SV introduces a companion model-a small auxiliary model similar in size to the draft model-to estimate the alignment between draft and target model distributions. By maximizing the information gain from quantifying this alignment, SV refines verification decisions, reducing wasted computation on rejected tokens and improving decoding efficiency. Moreover, SV requires no modifications to the draft or target models and is compatible with existing SD variants. We extensively evaluated SV on publicly available LLMs across three NLP tasks using nine combinations of draft, companion, and target models, including 13B-72B target models and three types of variations: base (no finetuning), instructiontuned, and task fine-tuned. Across all experiments and batch sizes (4-80), SV consistently outperforms both SD and standard decoding with the target model. It improves SD performance by up to 2×, with an average speedup of 1.4× in large-batch settings (batch sizes 32-80). These results demonstrate SV's robustness, scalability, and practical utility for efficient LLM inference.

### 70. Streamline pathology foundation model by cross-magnification distillation

**主要机构**: College of Medicine, Department of Pathology, The Ohio State University Wexner Medical Center
**作者数量**: 6人

**摘要**:
Foundation models (FM) have transformed computational pathology but remain computationally prohibitive for clinical deployment due to their massive parameter counts and high-magnification processing requirements. Here, we introduce XMAG, a lightweight FM developed through crossmagnification distillation that transfers knowledge from state-of-the-art 20x magnification teacher to an efficient 5x magnification student architecture. XMAG employs a compact backbone and operates entirely at 5x, requiring 11.3 times fewer patches per whole slide image (WSI) compared to existing approaches. Our novel distillation framework incorporates dual-level knowledge transfer, aligning both global image representations and local spatial features across magnification levels through carefully designed projection heads and spatial token mapping. We trained XMAG on 3.49 million images curated from publicly available datasets and evaluated performance across six clinically relevant histopathology analysis tasks spanning multiple cancer types. XMAG achieved diagnostic accuracy within 1% of substantially larger foundation models while delivering 30-fold processing acceleration, reaching 8.8 WSIs per minute processing speed. Our cross-institutional validation confirmed robust generalization. Further, we developed an end-to-end training strategy to further boost our model's performance to approach the larger FMs' performance. These results establish cross-magnification distillation as a viable approach for deploying FM capabilities in resource-constrained clinical environments, potentially enabling real-time pathology AI integration.

### 71. Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack

**主要机构**: State Key Laboratory of Blockchain and Data Security, Nanyang Technological University, Tech Zone (Binjiang) Institute of Blockchain and Data Security
**作者数量**: 8人

**摘要**:
Knowledge distillation (KD) is a vital technique for deploying deep neural networks (DNNs) on resource-constrained devices by transferring knowledge from large teacher models to lightweight student models. While teacher models from third-party platforms may undergo security verification (e.g., backdoor detection), we uncover a novel and critical threat: distillation-conditional backdoor attacks (DCBAs). DCBA injects dormant and undetectable backdoors into teacher models, which become activated in student models via the KD process, even with clean distillation datasets. While the direct extension of existing methods is ineffective for DCBA, we implement this attack by formulating it as a bilevel optimization problem and proposing a simple yet effective method (i.e., SCAR). Specifically, the inner optimization simulates the KD process by optimizing a surrogate student model, while the outer optimization leverages outputs from this surrogate to optimize the teacher model for implanting the conditional backdoor. Our SCAR addresses this complex optimization utilizing an implicit differentiation algorithm with a pre-optimized trigger injection function. Extensive experiments across diverse datasets, model architectures, and KD techniques validate the effectiveness of our SCAR and its resistance against existing backdoor detection, highlighting a significant yet previously overlooked vulnerability in the KD process. Our code is available at https://github.com/WhitolfChen/SCAR.

### 72. TEQUILA: TRAPPING-FREE TERNARY QUANTIZA-TION FOR LARGE LANGUAGE MODELS

**主要机构**: McGill University, City University
**作者数量**: 12人

**摘要**:
Quantization techniques are essential for the deployment of Large Language Models (LLMs) on edge devices. However, prevailing methods often rely on mixedprecision multiplication that lacks efficient hardware support, making it not feasible. Ternary weight quantization addresses this by constraining weights to {-1, 0, 1}, replacing expensive multiplications with hardware-efficient additions. However, such aggressive compression leads to significant accuracy degradation, even after costly quantization-aware training with massive data. We identify the core issue as deadzone trapping: a large number of weights are trapped at the deadzone boundary. This occurs because these weights receive only noisy, uninformative gradients, preventing stable escape from the deadzone and severely impeding model capacity and optimization. To address this issue, we propose Tequila, a trapping-free quantization optimization method that reactivates deadzone-trapped weights by repurposing them as dynamic biases. This allows the repurposed weights to provide a continuous signal in the forward pass and, critically, receive direct, meaningful gradient signals during backpropagation, thereby enhancing model capacity and optimization with nearly zero inference overhead. Extensive evaluations demonstrate that Tequila outperforms state-of-the-art (SOTA) ternary quantization methods across five benchmarks. Specifically, on the ARC benchmark, it achieves > 4% accuracy gain over the SOTA baseline, nearly matching full-precision performance (within < 1% gap) with an 3.0× inference speedup. Consequently, Tequila offers a highly practical and efficient implementation for the deployment of advanced LLMs in resource-constrained environments. The code is available at https://github.com/Tencent/AngelSlim.

### 73. TEXTURE VECTOR-QUANTIZATION AND RECONSTRUC-TION AWARE PREDICTION FOR GENERATIVE SUPER-RESOLUTION

**主要机构**: University of Electronic Science and Technology of China
**作者数量**: 6人

**摘要**:
Vector-quantized based models have recently demonstrated strong potential for visual prior modeling. However, existing VQ-based methods simply encode visual features with nearest codebook items and train index predictor with code-level supervision. Due to the richness of visual signal, VQ encoding often leads to large quantization error. Furthermore, training predictor with code-level supervision can not take the final reconstruction errors into consideration, result in sub-optimal prior modeling accuracy. In this paper we address the above two issues and propose a Texture Vector-Quantization and a Reconstruction Aware Prediction strategy. The texture vector-quantization strategy leverages the task character of superresolution and only introduce codebook to model the prior of missing textures. While the reconstruction aware prediction strategy makes use of the straightthrough estimator to directly train index predictor with image-level supervision. Our proposed generative SR model (TVQ&RAP) is able to deliver photo-realistic SR results with small computational cost. di se nt an gl in g Vanilla Codebook: Modeling a complex feature space containing both structures and textures.

### 74. The Hidden Costs of Translation Accuracy: Distillation, Quantization, and Environmental Impact

**主要机构**: University of California, Research Spark Hub Inc
**作者数量**: 2人

**摘要**:
The rapid expansion of large language models (LLMs) has heightened concerns about their computational and environmental costs. This study investigates the trade-offs between translation quality and efficiency by comparing full-scale, distilled, and quantized models using machine translation as a case study. We evaluated performance on the Flores+ benchmark and through human judgments of conversational translations in French, Hindi, and Kannada. Our analysis revealed that the full 3.3B FP32 model, while achieving the highest BLEU scores, incurred the largest environmental footprint (≈ 0.007-0.008 kg CO 2 per run). The distilled 600M FP32 model reduced inference time by 71-78% and carbon emissions by 63-65% compared with the full model, with only minimal reductions in BLEU scores. Human evaluations further showed that even aggressive quantization (INT4) preserved high levels of accuracy and fluency, with differences between models generally minor. These findings demonstrate that model compression strategies can substantially reduce computational demands and environmental impact while maintaining competitive translation quality, though trade-offs are more pronounced in low-resource settings. We argue for evaluation frameworks that integrate efficiency and sustainability alongside accuracy as central dimensions of progress in NLP.

### 75. Tiny-QMoE

**主要机构**: 
**作者数量**: 1人

**摘要**:
The QMoE model [1] provides a practical approach for compression of massive Mixture-of-Experts (MoE) models. QMoE offers a solution geared towards memory limitations that often reach terabyte scales, and it has the advantage of working with high sparsity models which implicitly lend themselves to compression techniques. QMoE also has the advantage of only taking MoE models into account and does not evaluate its use with non mixture of expert systems. Although this prior attempt focuses on the limitations of large servers with the latest NVIDIA hardware which in the case of the H100 and V100 which have 80 GB of HBM (High Bandwidth Memory), what is not being considered is a significantly more constrained environment, such as in the case of mobile devices which may have in the case of the iPhone anywhere from 4 to 8 GB of unified memory which also needs to be shared with the operating system and additional processes. Although edge devices such as phones and laptops are becoming increasingly more computationally powerful, they are still not close to the level of advanced server machines such as NVIDIA. An additional constraint that we must consider is that of latency. The communication time of sending a request to an LLM server and then getting it back is an additional waiting time that can be removed. We may also want to use LLM technology in environments where there is no reliable network connection. As will be discussed later, while the latency of the internet is removed, the latency of the decompression is incurred. In this paper, we present a solution to this highly constrained memory problem. This takes the form of a re-imagined quantization compression schema and execution such that models which would have normally exceeded the memory requirements of say a 2060 with 6 GB of VRAM. Specifically, Tiny-QMoE works on a variety of LLAMA 3.2 models and does so by quantizing the models into 8-bit versions and then taking said models and storing them in a dictionary based compression format.

### 76. Token Merging via Spatiotemporal Information Mining for Surgical Video Understanding

**主要机构**: Department of Electronic and Computer Engineering, Huazhong University of Science and Technology, The Hong Kong University of Science and Technology, School of Electronic Information and Communica- tions
**作者数量**: 9人

**摘要**:
Vision Transformer models have shown impressive effectiveness in the surgical video understanding tasks through long-range dependency modeling. However, current methods suffer from prohibitive computational costs due to processing massive spatiotemporal tokens across video frames. While prior work on token merging has advanced model efficiency, they fail to adequately consider the inherent spatiotemporal structure of video data and overlook the heterogeneous nature of information distribution, leading to suboptimal performance. In this paper, we propose a spatiotemporal information mining token merging (STIM-TM) method, representing the first dedicated approach for surgical video understanding. STIM-TM introduces a decoupled strategy that reduces token redundancy along temporal and spatial dimensions independently. Specifically, the temporal component merges spatially corresponding tokens from consecutive frames using saliency weighting, preserving critical sequential information and maintaining continuity. Meanwhile, the spatial component prioritizes merging static tokens through temporal stability analysis, protecting dynamic regions containing essential surgical information. Operating in a training-free manner, STIM-TM achieves significant efficiency gains with over 65% GFLOPs reduction while preserving competitive accuracy across comprehensive surgical video tasks. Our method also supports efficient training of long-sequence surgical videos, addressing computational bottlenecks in surgical applications. Code is available at https://github.com/xjiangmed/STIM-TM.

### 77. TOWARDS A COMPREHENSIVE SCALING LAW OF MIXTURE-OF-EXPERTS

**主要机构**: University of Macau
**作者数量**: 14人

**摘要**:
Mixture-of-Experts (MoE) models have become the consensus approach for enabling parameter-efficient scaling and cost-effective deployment in large language models. However, existing scaling laws for dense models are inapplicable to MoE models, which stems from three critical challenges: the multiplicity of influencing factors, their intricate coupling relationships and the non-monotonic nature of their performance impacts. They collectively necessitate a fine-grained investigation into MoE-specific scaling laws. In this work, we perform a systematic decomposition of MoE settings, identifying five key factors that influence model performance from both size and structural perspectives (data size (D), total model size (N), activated model size (N a), number of active experts (G) and the ratio of shared experts (S)). Specifically, we design 446 controlled experiments to characterize their marginal effects, ultimately constructing a comprehensive and precise joint MoE scaling law that considers all essential factors. Furthermore, we derive the theoretically optimal and practically efficiency-aware optimal configurations for G, S and N a /N with detailed analyses. Our results demonstrate that the optimal settings for G and S are independent of both the model architecture and data size. With the scaling of N , the optimal activation parameter ratio of N a /N becomes sparser. Our proposed MoE scaling law could function as an accurate and insightful guidance to facilitate future MoE model design and training.

### 78. Towards Efficient CoT Distillation: Self-Guided Rationale Selector for Better Performance with Fewer Rationales

**主要机构**: Harbin Institute of Technology, Pengcheng Laboratory
**作者数量**: 6人

**摘要**:
Chain-of-thought (CoT) distillation aims to enhance small language models' (SLMs) reasoning by transferring multi-step reasoning capability from the larger teacher models. However, existing work underestimates rationale quality, focusing primarily on data quantity, which may transfer noisy or incorrect information to the student model. To address the above issues, we proposed Model-Oriented Rationale Selection Distillation (MoRSD), which can discern and select high quality rationales for distillation to improve performance further. We further propose a Rationale Difficulty (RD) metric to measure the ability of the student model to generate the correct answer under a given rationale. Compared to the baseline, we achieved 4.6% average improvement on seven datasets over three tasks, using fewer rationales by controlling their accuracy, diversity, and difficulty. Our results reveal that a small portion of the high quality rationales can enhance the reasoning ability of student models than the entire dataset. Our method promises to be a possible solution for efficient CoT distillation. Our code will be released in https://github.com/Leon221220/MoRSD.

### 79. Towards Trustworthy Lexical Simplification: Exploring Safety and Efficiency with Small LLMs

**主要机构**: Universitat Pompeu Fabra Barcelona, Department of Engineering
**作者数量**: 3人

**摘要**:
Despite their strong performance, large language models (LLMs) face challenges in realworld application of lexical simplification (LS), particularly in privacy-sensitive and resourceconstrained environments. Moreover, since vulnerable user groups (e.g., people with disabilities) are one of the key target groups of this technology, it is crucial to ensure the safety and correctness of the output of LS systems. To address these issues, we propose an efficient framework for LS systems that utilizes small LLMs deployable in local environments. Within this framework, we explore knowledge distillation with synthesized data and in-context learning as baselines. Our experiments in five languages evaluate model outputs both automatically and manually. Our manual analysis reveals that while knowledge distillation boosts automatic metric scores, it also introduces a safety trade-off by increasing harmful simplifications. Importantly, we find that the model's output probability is a useful signal for detecting harmful simplifications. Leveraging this, we propose a filtering strategy that suppresses harmful simplifications while largely preserving beneficial ones. This work establishes a benchmark for efficient and safe LS with small LLMs. It highlights the key trade-offs between performance, efficiency, and safety, and demonstrates a promising approach for safe real-world deployment. 1

### 80. TRAINING-FREE TOKEN PRUNING VIA ZEROTH-ORDER GRADIENT ESTIMATION IN VISION-LANGUAGE MODELS

**主要机构**: Inha University, Sungkyunkwan University
**作者数量**: 6人

**摘要**:
Large Vision-Language Models (VLMs) enable strong multimodal reasoning but incur heavy inference costs from redundant visual tokens. Token pruning alleviates this issue, yet existing approaches face limitations. Attention-based methods rely on raw attention scores, which are often unstable across layers and heads and can lead to redundant selections. Diversity-based methods improve robustness by selecting tokens far apart in feature space but risk dropping regions needed for accurate prediction. We propose ZOO-Prune, a training-free framework built on a simple intuition: tokens with higher sensitivity are more likely to influence the model's output, and they should also capture complementary visual cues rather than overlapping information. To achieve this, we estimate token sensitivity using zeroth-order perturbations at the projection layer, a shallow and computationally light component of the model. This approach measures how small random perturbations affect the projection outputs, allowing us to approximate each token's influence through lightweight forward passes without backpropagation. Extensive experiments across multiple VLMs and benchmarks show that ZOO-Prune consistently outperforms prior methods, pruning up to 94.4% of tokens while maintaining accuracy and significantly improving efficiency, achieving up to 2.30× faster end-to-end inference over the baseline.

### 81. Training Agents Inside of Scalable World Models

**主要机构**: 
**作者数量**: 3人

**摘要**:
World models learn general knowledge from videos and simulate experience for training behaviors in imagination, offering a path towards intelligent agents. However, previous world models have been unable to accurately predict object interactions in complex environments. We introduce Dreamer 4, a scalable agent that learns to solve control tasks by reinforcement learning inside of a fast and accurate world model. In the complex video game Minecraft, the world model accurately predicts object interactions and game mechanics, outperforming previous world models by a large margin. The world model achieves real-time interactive inference on a single GPU through a shortcut forcing objective and an efficient transformer architecture. Moreover, the world model learns general action conditioning from only a small amount of data, allowing it to extract the majority of its knowledge from diverse unlabeled videos. We propose the challenge of obtaining diamonds in Minecraft from only offline data, aligning with practical applications such as robotics where learning from environment interaction can be unsafe and slow. This task requires choosing sequences of over 20,000 mouse and keyboard actions from raw pixels. By learning behaviors in imagination, Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction. Our work provides a scalable recipe for imagination training, marking a step towards intelligent agents.

### 82. TREE REWARD-ALIGNED SEARCH FOR TREASURE IN MASKED DIFFUSION LANGUAGE MODELS

**主要机构**: Fudan University, University of Science and Technology of China
**作者数量**: 4人

**摘要**:
Tree search has recently emerged as a powerful framework for aligning generative models with task-specific rewards at test time. Applying tree search to Masked Diffusion Language Models, however, introduces two key challenges: (i) parallel unmasking yields highly correlated branches, limiting exploration, and (ii) reward evaluation via sampled completions produces high-variance estimates, making pruning unstable. We propose TREASURE, a tree-search test-time alignment method that addresses these issues. It introduces (i) UNMASKBRANCH, a branching strategy based on first-hitting unmasking that diversifies both token content and reveal order with a single model call per parent node, and (ii) RE-SUBSTITUTESCORE, a pruning rule that uses deterministic resubstitution to score partially masked sequences with low-variance proxy completions. Theoretically, we quantify branching efficiency gains in NFEs (number of function evaluations), show that the scoring rule approximates the true reward with error bounded by predictive uncertainty, and prove improvements with larger tree widths. Empirically, TREASURE achieves state-of-the-art results on perplexity, linguistic acceptability, and control of sentiment and toxicity, outperforming prior methods under matched compute budgets, with especially strong gains in low-NFE regimes.

### 83. Tunable-Generalization Diffusion Powered by Self-Supervised Contextual Sub-Data for Low-Dose CT Reconstruction

**主要机构**: 
**作者数量**: 5人

**摘要**:
Current models based on deep learning for low-dose CT denoising rely heavily on paired data and generalize poorly. Even the more concerned diffusion models need to learn the distribution of clean data for reconstruction, which is difficult to satisfy in medical clinical applications. At the same time, self-supervised-based methods face the challenge of significant degradation of generalizability of models pre-trained for the current dose to expand to other doses. To address these issues, this paper proposes a novel method of tunable-generalization diffusion powered by self-supervised contextual sub-data for low-dose CT reconstruction, named SuperDiff. Firstly, a contextual subdata similarity adaptive sensing strategy is designed for denoising centered on the LDCT projection domain, which provides an initial prior for the subsequent progress. Subsequently, the initial prior is used to combine knowledge distillation with a deep combination of latent diffusion models for optimizing image details. The pretrained model is used for inference reconstruction, and the pixel-level self-correcting fusion technique is proposed for fine-grained reconstruction of the image domain to enhance the image fidelity, using the initial prior and the LDCT image as a guide. In addition, the technique is flexibly applied to the generalization of upper and lower doses or even unseen doses. Dual-domain strategy cascade for self-supervised LDCT denoising, SuperDiff requires only LDCT projection domain data for training and testing. Full qualitative and quantitative evaluations on both datasets and real data show that SuperDiff consistently outperforms existing state-of-the-art methods in terms of reconstruction and generalization performance.

### 84. TY-RIST: Tactical YOLO Tricks for Real-time Infrared Small Target Detection

**主要机构**: Middle East Technical University, Karlsruhe Institute of Technology
**作者数量**: 6人

**摘要**:


### 85. ULTRAUNET: REAL-TIME ULTRASOUND TONGUE SEGMENTATION FOR DIVERSE LINGUISTIC AND IMAGING CONDITIONS *

**主要机构**: Biomedical Engineering Department, Department of Biomedical Engineering, Department of Chinese and Bilingual Studies, English and Communication Department, Research Institute for Smart Ageing Hong Kong Polytechnic University Hong Kong, Hong Kong Polytechnic University Hong Kong
**作者数量**: 6人

**摘要**:
Ultrasound tongue imaging (UTI) provides a non-invasive, cost-effective modality for investigating speech articulation, speech motor control, and speech-related disorders. However, real-time tongue contour segmentation remains a significant challenge due to the inherently low signal-to-noise ratio, variability in imaging conditions, and computational demands of real-time performance. In this study, we proposed UltraUNet, a lightweight and efficient encoder-decoder architecture specifically optimized for real-time segmentation of tongue contours in ultrasound images. UltraUNet introduces several domain-informed innovations, including lightweight Squeeze-and-Excitation blocks for channel-wise feature recalibration in deeper layers, Group Normalization for enhanced stability in small-batch training, and summation-based skip connections to minimize memory and computational overhead. These architectural refinements enabled UltraUNet to achieve a high segmentation accuracy while maintaining an exceptional processing speed of 250 frames per second, making it suitable for real-time clinical workflows. UltraUNet integrates ultrasound-specific augmentation techniques, including denoising and blur simulation using point spread function. Additionally, we annotated UTI images from 8 different datasets with various imaging conditions. Comprehensive evaluations demonstrated the model's robustness and precision, with superior segmentation metrics on singledataset testing (Dice = 0.855, MSD = 0.993px) compared to established architectures. Furthermore, cross-dataset testing on 7 unseen datasets with 1 train dataset revealed UltraUNet's generalization capabilities and high accuracy, achieving average Dice Scores of 0.734 and 0.761, respectively, in Experiments 1 and 2. The proposed framework offers a competitive solution for time-critical applications in speech research, speech motor disorder analysis, and clinical diagnostics, with realtime performance in tongue functional analysis in diverse medical and research settings. * This is a preprint submitted to arXiv.

### 86. VAMamba: An Efficient Visual Adaptive Mamba for Image Restoration

**主要机构**: School of Transportation and Logistics Engineering, Sun Yat-sen University, Shandong Normal University
**作者数量**: 4人

**摘要**:
Recent Mamba-based image restoration methods have achieved promising results but remain limited by fixed scanning patterns and inefficient feature utilization. Conventional Mamba architectures rely on predetermined paths that cannot adapt to diverse degradations, constraining both restoration performance and computational efficiency. To overcome these limitations, we propose VAMamba, a Visual Adaptive Mamba framework with two key innovations. First, QCLAM (Queue-based Cache Low-rank Adaptive Memory) enhances feature learning through a FIFO cache that stores historical representations. Similarity between current LoRA-adapted and cached features guides intelligent fusion, enabling dynamic reuse while effectively controlling memory growth. Second, GPS-SS2D (Greedy Path Scan SS2D) introduces adaptive scanning. A Vision Transformer generates score maps to estimate pixel importance, and a greedy strategy determines optimal forward and backward scanning paths. These learned trajectories replace rigid patterns, enabling SS2D to perform targeted feature extraction. The integration of QCLAM and GPS-SS2D allows VAMamba to adaptively focus on degraded regions while maintaining high computational efficiency. Extensive experiments across diverse restoration tasks demonstrate that VAMamba consistently outperforms existing approaches in both restoration quality and efficiency, establishing new benchmarks for adaptive image restoration. Our code is available at https://github.com/WaterHQH/VAMamba.

### 87. VID-LLM: A COMPACT VIDEO-BASED 3D MULTIMODAL LLM WITH RECONSTRUCTION-REASONING SYNERGY

**主要机构**: School of Geodesy and Geomatics, Wuhan University, Shenzhen University, School of Architecture and Urban Planning
**作者数量**: 6人

**摘要**:
The room is a small office with desks arranged along the walls, a sofa in the center, and several chairs. It has a compact layout where most of the workspace is around the edges… Describe the writing desk on the right side of the room in detail. The desk is long and rectangular, placed against the wall with monitors, books, and … Give the coordinates of the chair right of the sofa.

### 88. YOLO26: KEY ARCHITECTURAL ENHANCEMENTS AND PERFORMANCE BENCHMARKING FOR REAL-TIME OBJECT DETECTION

**主要机构**: Cornell University, Biological & Environmental Engineering, Department of Biological and Agricultural Engineering, Kansas State University
**作者数量**: 4人

**摘要**:
This study presents a comprehensive analysis of Ultralytics YOLO26, highlighting its key architectural enhancements and performance benchmarking for real-time edge object detection. YOLO26, released in September 2025, stands as the newest and most advanced member of the YOLO family, purpose-built to deliver efficiency, accuracy, and deployment readiness on edge and low-power devices. The paper sequentially details YOLO26's architectural innovations, including the removal of Distribution Focal Loss (DFL), adoption of end-to-end NMS-free inference, integration of ProgLoss and Small-Target-Aware Label Assignment (STAL), and the introduction of the MuSGD optimizer for stable convergence. Beyond architecture, the study positions YOLO26 as a multi-task framework, supporting object detection, instance segmentation, pose/keypoints estimation, oriented detection, and classification. We present performance benchmarks of YOLO26 on edge devices such as NVIDIA Jetson Nano and Orin, comparing its results with YOLOv8, YOLOv11, YOLOv12, YOLOv13, and transformer-based detectors. This paper further explores real-time deployment pathways, flexible export options (ONNX, TensorRT, CoreML, TFLite), and quantization for INT8/FP16. Practical use cases of YOLO26 across robotics, manufacturing, and IoT are highlighted to demonstrate cross-industry adaptability. Finally, insights on deployment efficiency and broader implications are discussed, with future directions for YOLO26 and the YOLO lineage outlined.
