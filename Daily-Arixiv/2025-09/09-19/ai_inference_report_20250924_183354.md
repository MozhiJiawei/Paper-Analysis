# AI推理加速技术论文分析报告
生成时间: 2025-09-24 18:33:54
分析论文数量: 46篇

## 论文技术简报

### 1. A study on Deep Convolutional Neural Networks, transfer learning, and Mnet model for Cervical Cancer Detection

University of Southern Queensland发布了宫颈癌检测研究论文，使用轻量级CNN模型S-Net结合迁移学习与XAI技术，解决了现有SOTA CNN计算资源需求大、推理慢及决策不透明问题，达成了99.99%准确率且显著提升计算效率与推理速度，同时实现模型可解释性

### 2. A Unified AI Approach for Continuous Monitoring of Human Health and Diseases from Intensive Care Unit to Home with Physiological Foundation Models (UNIPHY+)

Emory Healthcare Atlanta发布了UNIPHY+论文，使用统一生理基础模型框架（含多模态学习、特征融合调优和知识蒸馏等创新策略），解决了跨护理场景（从ICU到家庭）的持续健康与疾病监测问题，达成了可泛化、可扩展、个性化的生理AI以支持临床决策和长期健康监测的效果。

### 3. Breaking Token Into Concepts: Exploring Extreme Compression in Token Representation Via Compositional Shared Semantics

Indian Institute of Technology Kharagpur发布了相关论文，使用Aggregate Semantic Grouping (ASG)技术并结合Product Quantization (PQ)，解决了标准语言模型中令牌单一整体嵌入限制捕捉词义多面性的问题，达成了嵌入参数0.4-0.5%的极端压缩同时保持>95%任务性能的效果，且适用于生成任务、跨语言迁移和特定领域。

### 4. CSDformer: A Conversion Method for Fully Spike-Driven Transformer

西湖大学发布了CSDformer论文，使用NReLU替代softmax、时间分解技术及延迟积分发放神经元的转换方法，解决了现有脉冲Transformer训练成本高或硬件不友好的问题，达成ImageNet 7时间步76.36%准确率并减少75%计算资源、提升训练速度2-3倍的效果

### 5. DA-MAMBA: DIALOGUE-AWARE SELECTIVE STATE-SPACE MODEL FOR MULTIMODAL ENGAGEMENT ESTIMATION

中国科学院深圳先进技术研究院发布了DA-MAMBA论文，使用基于Mamba的对话感知选择性状态空间模型（含对话感知编码器及模态组、伙伴组融合机制），解决了对话场景中的多模态人类参与度估计问题，达成了在三个基准数据集上超越SOTA、提升CCC指标，同时减少训练时间和峰值内存、支持更长序列处理及实时部署的效果

### 6. Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling

University of Electronic, Science and Technology发布了Dendritic Resonate-and-Fire Neuron论文，使用多树突和胞体架构及胞体自适应阈值机制的D-RF模型，解决了RF神经元在复杂时序任务中有效记忆容量有限及能量效率与训练速度权衡问题，达成了保持竞争力准确率同时确保稀疏脉冲且不影响训练计算效率的效果

### 7. Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs

中兴通讯发布了Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs论文，使用基于异构GPU的P-D解耦推理系统（含异构兼容传输模块及并行策略与实例分配联合优化算法），解决了多厂商异构GPU的混合推理及数据兼容性问题，达成了有效支持混合推理并通过联合优化算法获得最优部署方案的效果。

### 8. DOMAIN ADAPTIVE OBJECT DETECTION FOR SPACE APPLICATIONS WITH REAL-TIME CONSTRAINTS

卢森堡大学发布了相关论文，使用监督域适应（SDA）结合域不变特征学习、CNN域判别器及域无关回归头的不变风险最小化技术，解决了合成数据训练的空间目标检测模型在真实数据上因域差距导致性能下降的问题，达成仅用250张标注真实图像平均精度（AP）提升高达20个百分点的效果。

### 9. Dynamic Expert Specialization: Towards Catastrophic Forgetting-Free Multi-Domain MoE Adaptation

香港科技大学发布了《Dynamic Expert Specialization: Towards Catastrophic Forgetting-Free Multi-Domain MoE Adaptation》论文，使用动态专家专业化框架DES-MoE（含自适应路由器、实时专家-域相关映射及三阶段自适应微调计划），解决了MoE模型多域适应中的灾难性遗忘问题，达成了匹配单域ESFT性能、训练统一模型，当域从2扩展到6时遗忘减少89%且收敛快68%的效果。

### 10. EG-MLA: Embedding-Gated Multi-head Latent Attention for Scalable and Efficient LLMs

Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)发布了EG-MLA论文，使用嵌入门控多头潜在注意力（引入token-specific embedding gating机制），解决了大型语言模型(LLMs)推理中KV缓存大小限制问题，达成相比MHA减少91.6% KV缓存且性能损失可忽略、相比MLA提升任务准确率并额外节省59.9%内存的效果。

### 11. EPICACHE: EPISODIC KV CACHE MANAGEMENT FOR LONG CONVERSATIONAL QUESTION ANSWERING

Hanyang University发布了EPICACHE论文，使用情节性KV缓存管理框架（含块级预填充、对话历史聚类为情节的特定驱逐及自适应层预算分配），解决了长对话问答中KV缓存峰值内存无界和多轮对话准确性下降的问题，达成准确率提高多达40%、4-6倍压缩下保持接近全KV准确率、延迟和内存减少高达2.4倍和3.5倍的效果。

### 12. Equip Pre-ranking with Target Attention by Residual Quantization

阿里巴巴天猫集团发布了Equip Pre-ranking with Target Attention by Residual Quantization论文，使用残差量化近似目标注意力架构的TARQ框架，解决了工业推荐系统预排序阶段效率与效果的冲突，达成了显著提升排序性能并在淘宝部署带来实质性业务改进的效果

### 13. Evaluating the Energy Efficiency of NPU-Accelerated Machine Learning Inference on Embedded Microcontrollers

ARM与Alif Semiconductor发布了评估NPU加速嵌入式微控制器机器学习推理能效的论文，使用ARM Cortex-M55核心结合Ethos-U55 NPU及严格的能量测量方法，解决了嵌入式MCU部署ML模型受能源、延迟和内存限制的问题，达成中等至大型网络延迟提升7×-125×+、净能量减少143×及支持CPU-only无法运行模型的效果。

### 14. Expert-as-a-Service: Towards Efficient, Scalable, and Robust Large-scale MoE Serving

新加坡国立大学与南洋理工大学发布了Expert-as-a-Service (EaaS)论文，使用将MoE模块分解为独立无状态服务并结合高性能无CPU点对点通信库的技术，解决了MoE模型因动态稀疏专家使用导致传统服务系统不稳定的问题，达成了高效、可扩展、鲁棒的大规模部署，硬件故障下吞吐量仅降<2%且节省37.5%计算资源的效果。

### 15. EYE GAZE TELLS YOU WHERE TO COMPUTE: GAZE-DRIVEN EFFICIENT VLMS

Leiden University发布了GazeVLM论文，使用利用人眼注视作为自然监督信号的训练-free框架，解决了Vision-Language Models（VLMs）因视觉令牌冗余导致的推理效率低问题，达成了减少视觉令牌达93.1%、总令牌59.6%及FLOPs 50%，同时保持更好的回答质量的效果

### 16. FG-ATTN: LEVERAGING FINE-GRAINED SPARSITY IN DIFFUSION TRANSFORMERS

研究团队发布了FG-ATTN论文，提出FG-Attn细粒度稀疏注意力机制及asynchronous-gather load操作，解决了扩散Transformer视频生成中注意力层计算量大的瓶颈，在H100 GPU上实现5秒480p视频平均1.55倍（最高1.65倍）、720p视频平均1.41倍（最高1.49倍）加速。

### 17. Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics

四川大学发布了Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics论文，使用RPLIF方法（通过尖峰触发的阈值动态将不应期整合到LIF神经元），解决了现有SNN神经元模型忽略不应期导致的过度兴奋及异常信号干扰问题，达成了在Cifar10-DVS(82.40%)、N-Caltech101(83.35%)上达到SOTA性能，DVS128 Gesture(97.22%)低延迟表现优异且计算高效的效果。

### 18. ISCS: Parameter-Guided Channel Ordering and Grouping for Learned Image Compression

Futurewei Technologies, Inc.与Santa Clara University发布了ISCS: Parameter-Guided Channel Ordering and Grouping for Learned Image Compression论文，使用参数统计（权重方差、偏置大小、成对相关性）估计通道重要性并构建Invariant Salient Channel Space (ISCS)结构的技术，解决了现有学习型图像压缩方法依赖昂贵数据集特定消融测试、孤立分析通道忽略相互依赖的问题，达成了在多个LIC架构上减少比特率和计算量同时保持重建质量的效果。

### 19. MCP: A Control-Theoretic Orchestration Framework for Synergistic Efficiency and Interpretability in Multimodal Large Language Models

东北大学发布了MCP框架论文，使用基于模型-控制器-任务适配（MCP）的三层协作框架（通过解耦大模型功能模块并结合强化学习动态路由与任务适配机制），解决了大型模型在复杂任务中的计算效率低和可解释性不足问题，达成跨模态基准任务性能提升15-30%、推理效率提升40%及90%人工可解释性评分的效果。

### 20. MOA-OFF: ADAPTIVE HETEROGENEOUS MODALITY-AWARE OFFLOADING WITH EDGE-CLOUD COLLABORATION FOR EFFICIENT MULTIMODAL LLM INFERENCE

中国科学院计算技术研究所发布了MoA-Off论文，使用自适应异构模态感知卸载框架（含轻量级异构模态感知模块和自适应边缘-云协作卸载策略），解决了MLLM推理在资源受限环境中的计算与延迟负担挑战，达成了延迟减少30%以上、资源开销减少30%-65%且保持准确性的效果。

### 21. MOES ARE STRONGER THAN YOU THINK: HYPER-PARALLEL INFERENCE SCALING WITH ROE

加州大学圣地亚哥分校发布了“MOES ARE STRONGER THAN YOU THINK: HYPER-PARALLEL INFERENCE SCALING WITH ROE”论文，使用RoE（Roster of Experts）无训练超并行推理缩放技术（通过动态专家集成、控制路由随机性及高效批处理/KV缓存），解决了大语言模型推理时token级预测质量提升与计算成本控制的问题，达成7B MoE模型匹配10.5B MoE模型性能且推理计算减少30%的效果。

### 22. MRADNET: A COMPACT RADAR OBJECT DETECTOR WITH METAFORMER

渥太华大学发布了mRadNet论文，使用MetaFormer块的U-net架构及轻量级设计策略（含可分离卷积、注意力令牌混合器），解决了雷达目标检测模型在实时嵌入式系统中紧凑性和效率需求被忽视的问题，在CRUW数据集上以最少参数和FLOPs提升了最先进性能。

### 23. OS-DiffVSR: Towards One-step Latent Diffusion Model for High-detailed Real-world Video Super-Resolution

华为诺亚方舟实验室发布了OS-DiffVSR论文，使用相邻帧对抗训练范式与多帧融合机制，解决了扩散基视频超分辨率（VSR）方法中视频质量与推理效率的权衡问题，达成了比需几十步采样的现有扩散基VSR方法质量更优的效果。

### 24. PGSTalker: Real-Time Audio-Driven Talking Head Generation via 3D Gaussian Splatting with Pixel-Aware Density Control

新疆大学发布了PGSTalker论文，使用3D高斯溅射与像素感知密度控制并融合音频和空间特征的技术，解决了音频驱动说话头生成中高斯变形预测准确性问题，达成了在渲染质量、唇同步精度和推理速度上优于现有NeRF和3DGS方法，并具备强泛化能力与实际部署潜力的效果。

### 25. Preference Distillation via Value based Reinforcement Learning

KAIST发布了Preference Distillation via Value based Reinforcement Learning论文，使用Teacher Value-based Knowledge Distillation (TVKD)技术，通过引入教师模型价值函数的辅助奖励并满足基于势的奖励塑造，解决了DPO二进制监督对小模型不足及现有蒸馏方法忽略奖励建模的问题，达成了跨各种基准和模型大小一致提升性能的效果。

### 26. PRISM: PRECISION-RECALL INFORMED DATA-FREE KNOWLEDGE DISTILLATION VIA GENERATIVE DIFFUSION

电子科技大学发布了PRISM论文，使用能量引导分布对齐和多样化提示工程技术，解决了数据无关知识蒸馏中合成大规模图像的模式崩溃及精确率-召回率挑战，达成了在大规模图像数据集上表现优越且模型具有强领域泛化能力的效果。

### 27. PruneCD: Contrasting Pruned Self Model to Improve Decoding Factuality

POSTECH发布了PruneCD论文，使用通过层剪枝构建业余模型的对比解码技术，解决了大语言模型的幻觉问题，达成了一致提升事实性且推理开销极小的效果

### 28. PTQTP: POST-TRAINING QUANTIZATION TO TRIT-PLANES FOR LARGE LANGUAGE MODELS

加州大学发布了PTQTP论文，使用三元权重后训练量化框架（将权重分解为结构化{-1,0,1}trit-planes，2×1.58位表示）实现无乘法推理，解决了大语言模型极低比特量化中计算效率与表达能力的权衡问题，达成数学推理保留率82.4%（竞品0%）且量化仅需1小时的效果。

### 29. 

阿里巴巴达摩院发布了Qwen3-Omni论文，使用Thinker-Talker Mixture-of-Experts (MoE)架构，解决了多模态模型在文本、图像、音频、视频多种模态上保持SOTA性能且不劣于单模态模型的问题，达成了匹配同尺寸单模态模型性能、音频任务突出，在36项音视频基准中开源SOTA 32项、总体SOTA 22项并超越Gemini-2.5-Pro等闭源模型的效果。

### 30. QWHA: QUANTIZATION-AWARE WALSH-HADAMARD ADAPTATION FOR PARAMETER-EFFICIENT FINE-TUNING ON LARGE LANGUAGE MODELS

首尔国立大学发布了QWHA论文，使用QWHA方法（采用Walsh-Hadamard变换作为变换核及新的适配器初始化方案），解决了现有量化感知参数高效微调方法在量化模型中表示能力有限、量化误差减少无效及计算开销增加的问题，达成了有效减轻量化误差、降低计算成本，在低比特量化精度上优于基线并实现显著训练加速的效果。

### 31. R-Net: A Reliable and Resource-Efficient CNN for Colorectal Cancer Detection with XAI Integration

University of Southern Queensland发布了R-Net论文，使用可靠且资源高效的CNN结合可解释AI（XAI）技术，解决了结直肠癌检测问题，达成了可靠且资源高效的检测效果。

### 32. RCTDistill: Cross-Modal Knowledge Distillation Framework for Radar-Camera 3D Object Detection with Temporal Fusion

现代汽车公司发布了RCTDistill论文，使用基于时间融合的跨模态知识蒸馏框架（含RAKD、TKD、RDKD模块），解决雷达-相机3D目标检测中物体运动不确定性及传感器误差问题，达成nuScenes和VoD数据集SOTA雷达-相机融合性能及26.2 FPS推理速度

### 33. SAEC: SCENE-AWARE ENHANCED EDGE-CLOUD COLLABORATIVE INDUSTRIAL VISION INSPECTION WITH MULTIMODAL LLM

中科院计算所发布了SAEC论文，使用场景感知增强的边缘云协同框架（结合高效多模态LLM微调、轻量级场景复杂度估计及自适应边缘云调度），解决了工业视觉检测中高精度需求与多模态LLM高计算成本、轻量级边缘模型复杂案例失效的权衡问题，达成了在MVTec AD和KSDD2数据集上准确率达85.11%和82.72%（显著超越Qwen、LLaVA），运行时减少22.4%，能耗降低40%-74%的效果

### 34. SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning

苏州大学发布了SCAN论文，使用Self-Denoising Monte Carlo Annotation框架，解决了过程奖励模型人工标注成本高、Monte Carlo合成数据噪声大的问题，在ProcessBench中F1分数提升39.2（从19.9到59.1），轻量级模型以6%推理成本超越PRM800K等基线。

### 35. SecureFixAgent: A Hybrid LLM Agent for Automated Python Static Vulnerability Repair

The George计算机科学系发布了SecureFixAgent论文，使用混合修复框架结合Bandit与轻量级本地LLMs的迭代detect-repair-validate循环及LoRA微调技术，解决了静态分析工具高误报、缺乏修复能力及LLM幻觉和自验证不足的问题，达成了减少误报10.8%、修复准确率提升13.51%及开发者解释质量评价4.5/5的效果。

### 36. ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching

Harvard University与University of Washington发布了ShadowServe论文，使用SmartNIC加速的无干扰前缀缓存系统（分离控制平面与数据平面并卸载至SmartNIC，结合分块流水线和最小复制内存管理），解决了网络带宽有限时KV缓存获取瓶颈及解压与模型计算的干扰问题，在低带宽场景下实现TPOT降低2.2倍、TTFT减少1.38倍，吞吐量提升1.35倍。

### 37. Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

Snowflake AI Research发布了Shift Parallelism论文，通过将Sequence Parallelism适配推理并与Tensor Parallelism动态切换，解决了LLM推理中Tensor Parallelism低延迟但吞吐量受限、Data Parallelism高吞吐量但延迟高且无法结合的问题，达成交互式工作负载响应快1.51倍、批处理工作负载吞吐量高50%的效果（相比TP-only）

### 38. SISMA: Semantic Face Image Synthesis with Mamba

帕多瓦大学与锡耶纳大学发布了SISMA: Semantic Face Image Synthesis with Mamba论文，使用基于Mamba的SISMA架构，解决了扩散模型在人脸语义图像合成中因注意力层二次复杂度导致的计算昂贵问题，达成了生成高质量样本、FID分数更优且速度达最先进架构三倍的效果

### 39. SlowFast-SCI: Slow-Fast Deep Unfolding Learning for Spectral Compressive Imaging

哈佛大学与上海交通大学发布了SlowFast-SCI论文，使用双速深度展开学习框架（慢学习预训练蒸馏与快学习测试时自监督自适应），解决了现有深度展开方法难以快速适应新光学配置、分布外性能差及计算量大的问题，达成参数和计算量减少70%以上、分布外数据PSNR提升达5.79dB、适应速度快4倍的效果。

### 40. Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding

高通AI研究发布了Spiffy论文，使用基于自动推测（利用dLLM自身分布）和有向草稿图的无损推测解码技术，解决了现有扩散型LLM（dLLMs）生成速度低的问题，达成了2.8-3.1倍推理加速、结合并行解码方法总提速达7.9倍且无损保留输出分布的效果

### 41. STENCIL: SUBJECT-DRIVEN GENERATION WITH CONTEXT GUIDANCE

南洋理工大学发布了STENCIL论文，使用主题驱动生成与上下文引导技术，解决了主题生成中上下文一致性不足的问题，达成了更精准的主题控制与上下文融合的生成效果

### 42. The Role of Vocabularies in Learning Sparse Representations for Ranking

研究团队发布了探究词汇表在SPLADE模型中作用的论文，通过构建100K词汇表BERT模型（ESPLADE预训练与随机初始化）并结合修剪优化，解决了词汇表对检索效率与有效性影响不明确的问题，实现了在BM25计算预算下比32K普通SPLADE更有效，且ESPLADE模型在相似检索成本下效果优于随机词汇表模型。

### 43. Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few

北京邮电大学发布了相关论文，使用Contract-and-Broadcast Self-Attention (CBSA)机制，解决了Transformer注意力机制的可解释性不足和二次复杂度问题，在多个视觉任务上达成了相当甚至更优的性能。

### 44. Visual Detector Compression via Location-Aware Discriminant Analysis

Bowling Green State University和University of Alabama at Birmingham发布了基于位置感知判别分析的视觉检测器压缩论文，使用主动检测判别压缩方法（利用定位信息），解决了现有压缩方法对检测模型关注不足、未充分利用定位信息及被动依赖预训练模型导致难移除无用组件的问题，达成了压缩模型在大幅降低复杂度的同时甚至超过原始模型性能的效果。

### 45. ViTCAE: ViT-based Class-conditioned Autoencoder

NC State University发布了ViTCAE论文，使用将Class token重构为生成关键并结合基于意见动力学的自适应注意力机制（含收敛感知温度调度器和头冻结机制）的技术，解决了ViT自编码器中Class token未充分利用及静态注意力机制导致的生成控制与优化效率受限问题，达成了显著提高计算效率且不损失保真度的效果

### 46. When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using Small VLMs

Indian Institute of Technology Jodhpur发布了相关论文，使用Label-Free Model Parity Alignment（MPA）框架，解决了小视觉语言模型（S-VLMs）效率高但性能与大模型差距显著且传统知识蒸馏依赖标注数据的问题，达成了在TextVQA、ST-VQA、ChartQA和OKVQA四个VQA基准上提升S-VLM性能、缩小差距同时保持计算效率的效果

## 论文详细信息

### 1. A study on Deep Convolutional Neural Networks, transfer learning, and Mnet model for Cervical Cancer Detection

**主要机构**: Department of Computer Science and Engineering Daffodil, University of Southern Queensland, School of Mathematics, Physics and Computing Toowoomba Campus, International University
**作者数量**: 9人

**摘要**:
Cervical cancer remains one of the most common cancers affecting women worldwide, particularly in low-resource settings. Early and accurate detection through Pap smear analysis is critical to improving patient outcomes and reducing mortality. Although state-of-the-art (SOTA) Convolutional Neural Networks (CNNs) have revolutionized disease diagnosis, most SOTA CNNs are designed for large-scale object detection and classification tasks. As a result, they require substantial computational resources, extended training time, and large datasets. In this study, a lightweight CNN model, S-Net (Simple Net), is developed specifically for cervical cancer detection and classification using Pap smear images to address these limitations. Alongside S-Net, six SOTA CNNs were evaluated using transfer learning, including multi-path (DenseNet201, ResNet152), depth-based (Serasnet152), width-based multiconnection (Xception), depth-wise separable convolutions (MobileNetV2), and spatial exploitation-based (VGG19). All models, including S-Net, achieved comparable accuracy, with S-Net reaching 99.99%. However, S-Net significantly outperforms the SOTA CNNs in terms of computational efficiency and inference time, making it a more practical choice for real-time and resource-constrained applications. A major limitation in CNN-based medical diagnosis remains the lack of transparency in the decision-making process. To address this, Explainable AI (XAI) techniques, such as SHAP, LIME, and Grad-CAM, were employed to visualize and interpret the key image regions influencing model predictions. The novelty of this study lies in the development of a highly accurate yet computationally lightweight model (S-Net) caPable of rapid inference while maintaining interpretability through XAI integration. Furthermore, this work analyzes the behavior of SOTA CNNs, investigates the effects of negative transfer learning on Pap smear images, and examines pixel intensity patterns in correctly and incorrectly classified samples.

### 2. A Unified AI Approach for Continuous Monitoring of Human Health and Diseases from Intensive Care Unit to Home with Physiological Foundation Models (UNIPHY+)

**主要机构**: Emory Healthcare Atlanta, Jocelyn Grun- well † CHOA, NKDHS Inc Irvine, Emory University Atlanta, Michael Fundora † CHOA, Naveen Muthu Emory University Atlanta, Tony Pan Emory University Atlanta
**作者数量**: 29人

**摘要**:
We present UNIPHY+, a unified physiological foundation model (physioFM) framework designed to enable continuous human health and diseases monitoring across care settings using ubiquitously obtainable physiological data. We propose novel strategies for incorporating contextual information during pretraining, fine-tuning, and lightweight model personalization via multi-modal learning, feature fusion-tuning, and knowledge distillation. We advocate testing UNIPHY+ with a broad set of use cases from intensive care to ambulatory monitoring in order to demonstrate that UNIPHY+ can empower generalizable, scalable, and personalized physiological AI to support both clinical decision-making and long-term health monitoring. CCS Concepts • Applied computing → Health informatics; • Computing methodologies → Machine learning.

### 3. Breaking Token Into Concepts: Exploring Extreme Compression in Token Representation Via Compositional Shared Semantics

**主要机构**: Indian Institute of Technology Kharagpur
**作者数量**: 1人

**摘要**:
Standard language models employ unique, monolithic embeddings for each token, potentially limiting their ability to capture the multifaceted nature of word meanings. We investigate whether tokens can be more effectively represented through a compositional structure that accumulates diverse semantic facets. To explore this, we propose Aggregate Semantic Grouping (ASG), a novel approach leveraging Product Quantization (PQ). We apply ASG to standard transformer architectures (mBERT, XLM-R, mT5) and evaluate this representational scheme across diverse tasks (NLI, NER, QA), as well as a biomedical domain-specific benchmark (BC5CDR) using BioBERT. Our findings demonstrate that representing tokens compositionally via ASG achieves extreme compression in embedding parameters (0.4-0.5%) while maintaining >95% task performance relative to the base model, even in generative tasks and extends to both cross lingual transfer and domain-specific settings. These results validate the principle that tokens can be effectively modeled as combinations of shared semantic building blocks. ASG offers a simple yet concrete method for achieving this, showcasing how compositional representations can capture linguistic richness while enabling compact yet semantically rich models.

### 4. CSDformer: A Conversion Method for Fully Spike-Driven Transformer

**主要机构**: CenBRAIN Neurotech, Westlake Institute for Optoelectronics, Westlake University, Zhejiang Key Laboratory, School of Engineering, Fabrication and Characterization
**作者数量**: 6人

**摘要**:
Spike-based transformer is a novel architecture aiming to enhance the performance of spiking neural networks while mitigating the energy overhead inherent to transformers. However, methods for generating these models suffer from critical limitations: excessive training costs introduced by direct training methods, or unavoidably hardware-unfriendly operations in existing conversion methods. In this paper, we propose CSDformer, a novel conversion method for fully spike-driven transformers. We tailor a conversion-oriented transformer-based architecture and propose a new function NReLU to replace softmax in self-attention. Subsequently, this model is quantized and trained, and converted into a fully spike-driven model with temporal decomposition technique. Also, we propose delayed Integrate-and-Fire neurons to reduce conversion errors and improve the performance of spiking models. We evaluate CSDformer on ImageNet, CIFAR-10 and CIFAR-100 datasets and achieve 76.36% top-1 accuracy under 7 time-steps on ImageNet, demonstrating superiority over state-of-the-art models. Furthermore, CSDformer eliminates the need for training SNNs, thereby reducing training costs (reducing computational resource by 75% and accelerating training speed by 2-3×). To the best of our knowledge, this is the first fully spike-driven transformer-based model developed via conversion method, achieving high performance under ultra-low latency, while dramatically reducing both computational complexity and training overhead.

### 5. DA-MAMBA: DIALOGUE-AWARE SELECTIVE STATE-SPACE MODEL FOR MULTIMODAL ENGAGEMENT ESTIMATION

**主要机构**: School of Information Engineering, Chinese Academy of Sciences, Shenzhen Institute of Advanced Technology, Beijing Institute of Graphic Communication, Communication University of China
**作者数量**: 6人

**摘要**:
Human engagement estimation in conversational scenarios is essential for applications such as adaptive tutoring, remote healthcare assessment, and socially aware human-computer interaction. Engagement is a dynamic, multimodal signal conveyed by facial expressions, speech, gestures, and behavioral cues over time. In this work we introduce DA-Mamba, a dialogue-aware multimodal architecture that replaces attention-heavy dialogue encoders with Mamba-based selective state-space processing to achieve linear time and memory complexity while retaining expressive cross-modal reasoning. We design a Mamba dialogue-aware selective state-space model composed of three core modules: a Dialogue-Aware Encoder, and two Mamba-based fusion mechanisms: Modality-Group Fusion and Partner-Group Fusion, these modules achieve expressive dialogue understanding. Extensive experiments on three standard benchmarks (NoXi, NoXi-Add, and MPIIGI) show that DA-Mamba surpasses prior state-of-the-art (SOTA) methods in concordance correlation coefficient (CCC), while reducing training time and peak memory; these gains enable processing much longer sequences and facilitate real-time deployment in resource-constrained, multiparty conversational settings. The source code will be available at: https://github.com/kksssssss-ssda/MMEA.

### 6. Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling

**主要机构**: University of Electronic, Science and Technology
**作者数量**: 9人

**摘要**:
The explosive growth in sequence length has intensified the demand for effective and efficient long sequence modeling. Benefiting from intrinsic oscillatory membrane dynamics, Resonate-and-Fire (RF) neurons can efficiently extract frequency components from input signals and encode them into spatiotemporal spike trains, making them well-suited for long sequence modeling. However, RF neurons exhibit limited effective memory capacity and a trade-off between energy efficiency and training speed on complex temporal tasks. Inspired by the dendritic structure of biological neurons, we propose a Dendritic Resonate-and-Fire (D-RF) model, which explicitly incorporates a multi-dendritic and soma architecture. Each dendritic branch encodes specific frequency bands by utilizing the intrinsic oscillatory dynamics of RF neurons, thereby collectively achieving comprehensive frequency representation. Furthermore, we introduce an adaptive threshold mechanism into the soma structure that adjusts the threshold based on historical spiking activity, reducing redundant spikes while maintaining training efficiency in long sequence tasks. Extensive experiments demonstrate that our method maintains competitive accuracy while substantially ensuring sparse spikes without compromising computational efficiency during training. These results underscore its potential as an effective and efficient solution for long sequence modeling on edge platforms.

### 7. Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs

**主要机构**: ZTE Corporation
**作者数量**: 7人

**摘要**:
LLM-based applications have been widely used in various industries, but with the increasing of models size, an efficient large language model (LLM) inference system is an urgent problem to be solved for service providers. Since the inference system is divided into two stage with different characteristics: Prefill and Decode, the two stage will interfere with each other during the inference process. Toward this end, a P-D disaggregated inference framework is proposed by some researchers. Current research is done on homogeneous GPUs, and lacks deployment solutions based on business scenarios. Compared with homogeneous GPUs, using heterogeneous GPUs to construct inference systems can better improve resource utilization and reduce costs. Even if GPUs from different vendors are used to build inference systems, on the basis of reducing costs, the resource utilization rate can be improved and the dependence on a single vendor can be reduced. Therefore, a P-D disaggreagetd inference system based on heterogeneous GPUs is designed, and the heterogeneous compatible transmission module in the system is designed to address heterogeneous GPU data compatibility issues. Then, a joint optimization algorithm of parallel strategy and instance number allocation is proposed to obtain the deployment solutions. Finally, the experimental results show that the P-D disaggregated inference system can well solve the hybrid inference problem of heterogeneous GPUs from different vendors, and the joint optimization algorithm can obtain the optimal deployment solution.

### 8. DOMAIN ADAPTIVE OBJECT DETECTION FOR SPACE APPLICATIONS WITH REAL-TIME CONSTRAINTS

**主要机构**: University of Luxembourg
**作者数量**: 5人

**摘要**:
Object detection is essential in space applications targeting Space Domain Awareness and also applications involving relative navigation scenarios. Current deep learning models for Object Detection in space applications are often trained on synthetic data from simulators, however, the model performance drops significantly on real-world data due to the domain gap. However, domain adaptive object detection is an overlooked problem in the community. In this work, we first show the importance of domain adaptation and then explore Supervised Domain Adaptation (SDA) to reduce this gap using minimal labeled real data. We build on a recent semi-supervised adaptation method and tailor it for object detection. Our approach combines domain-invariant feature learning with a CNNbased domain discriminator and invariant risk minimization using a domain-independent regression head. To meet real-time deployment needs, we test our method on a lightweight Single Shot Multibox Detector (SSD) with MobileNet backbone and on the more advanced Fully Convolutional One-Stage object detector (FCOS) with ResNet-50 backbone. We evaluated on two space datasets, SPEED+ and SPARK. The results show up to 20-point improvements in average precision (AP) with just 250 labeled real images.

### 9. Dynamic Expert Specialization: Towards Catastrophic Forgetting-Free Multi-Domain MoE Adaptation

**主要机构**: The Hong Kong University of Science and Technology (Guangzhou), The Hong Kong University of Science and Technology
**作者数量**: 4人

**摘要**:
Mixture-of-Experts (MoE) models offer immense capacity via sparsely gated expert subnetworks, yet adapting them to multiple domains without catastrophic forgetting remains an open challenge. Existing approaches either incur prohibitive computation, suffer crossdomain interference, or require separate runs per domain. We propose DES-MoE, a dynamic expert specialization framework for multi-domain adaptation of Mixture-of-Experts models. DES-MoE addresses catastrophic forgetting through three innovations: (1) an adaptive router balancing pre-trained knowledge retention and task-specific updates via distillation, (2) real-time expert-domain correlation mapping to isolate domain-specific gradients, and (3) a three-phase adaptive fine-tuning schedule that progressively freezes non-specialized parameters. Evaluated on six domains (math, code, law, etc.), DES-MoE matches singledomain ESFT performance while training one unified model, reduces forgetting by 89% compared to full fine-tuning as domains scale from 2 to 6, and achieves 68% faster convergence than conventional methods. Our work establishes dynamic expert isolation as a scalable paradigm for multi-task MoE adaptation.

### 10. EG-MLA: Embedding-Gated Multi-head Latent Attention for Scalable and Efficient LLMs

**主要机构**: Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)
**作者数量**: 2人

**摘要**:
Reducing the key-value (KV) cache size is a crucial step toward enabling efficient inference in large language models (LLMs), especially under latency and memory constraints. While Multi-Head Attention (MHA) offers strong representational power, it incurs significant memory overhead. Recent work on Multi-head Latent Attention (MLA) mitigates this by compressing KV representations into a shared latent space, achieving a better trade-off between performance and cache efficiency. While MLA already achieves significant KV cache reduction, the scope for further compression remains limited without performance loss. In this paper, we propose Embedding-Gated Multi-head Latent Attention (EG-MLA), a novel extension of MLA that further reduces KV cache size while enhancing representational expressiveness. EG-MLA introduces a token-specific embedding gating mechanism applied in the latent space, enabling finegrained modulation of compressed KV vectors with minimal additional computation. Compared to MHA, EG-MLA achieves over 91.6% reduction in KV cache size with negligible performance degradation. Relative to MLA, EG-MLA consistently improves task accuracy across diverse reasoning benchmarks while achieving up to 59.9% additional memory savings. Our theoretical analysis highlights how embedding gating induces implicit high-order interactions, and empirical evaluations demonstrate robust generalization across model scales and compression regimes. Notably, we successfully scale EG-MLA to over 1 billion parameters, demonstrating its practical viability for large-scale LLM deployment. These results establish EG-MLA as a memory-and compute-efficient attention mechanism that enables scalable, high-performance inference in modern LLMs.

### 11. EPICACHE: EPISODIC KV CACHE MANAGEMENT FOR LONG CONVERSATIONAL QUESTION ANSWERING

**主要机构**: Hanyang University
**作者数量**: 6人

**摘要**:
Recent advances in large language models (LLMs) have extended context lengths, enabling assistants to sustain long histories for coherent, personalized responses. This ability, however, hinges on Key-Value (KV) caching, whose memory grows linearly with dialogue length and quickly dominates under strict resource constraints. An active line of research for reducing this overhead is KV cache compression, which seeks to limit cache size while preserving accuracy. Yet existing methods face two major limitations: (i) evicting entries after full-context prefill causes unbounded peak memory, and (ii) query-dependent eviction narrows the cache to a single query, leading to degraded accuracy in multi-turn conversations. We introduce EPICACHE, a training-free KV cache management framework for long conversational question answering (LongConvQA) under fixed memory budgets. EPICACHE bounds cache growth through block-wise prefill and preserves topic-relevant context via episodic KV compression, which clusters conversation history into coherent episodes and applies episode-specific KV cache eviction. We further design an adaptive layer-wise budget allocation strategy that measures each layer's sensitivity to eviction and distributes the memory budget across layers accordingly. Across three LongConvQA benchmarks, EPICACHE improves accuracy by up to 40% over recent baselines, sustains near-full KV accuracy under 4-6× compression, and reduces latency and memory by up to 2.4× and 3.5×, thereby enabling efficient multi-turn interaction under strict resource constraints.

### 12. Equip Pre-ranking with Target Attention by Residual Quantization

**主要机构**: Tmall Group of Alibaba Hangzhou, Xidian University Xi'an, Shanghai Jiao Tong University Shanghai
**作者数量**: 13人

**摘要**:
The pre-ranking stage in industrial recommendation systems faces a fundamental conflict between efficiency and effectiveness. While powerful models like Target Attention (TA) excel at capturing complex feature interactions in the ranking stage, their high computational cost makes them infeasible for pre-ranking, which often relies on simplistic vector-product models. This disparity creates a significant performance bottleneck for the entire system. To bridge this gap, we propose TARQ, a novel pre-ranking framework. Inspired by generative models, TARQ's key innovation is to equip pre-ranking with an architecture approximate to TA by Residual Quantization. This allows us to bring the modeling power of TA into the latency-critical pre-ranking stage for the first time, establishing a new state-of-the-art trade-off between accuracy and efficiency. Extensive offline experiments and large-scale online A/B tests at Taobao demonstrate TARQ's significant improvements in ranking performance. Consequently, our model has been fully deployed in production, serving tens of millions of daily active users and yielding substantial business improvements. CCS Concepts • Information systems → Learning to rank.

### 13. Evaluating the Energy Efficiency of NPU-Accelerated Machine Learning Inference on Embedded Microcontrollers

**主要机构**: 
**作者数量**: 3人

**摘要**:
The deployment of machine learning (ML) models on microcontrollers (MCUs) is constrained by strict energy, latency, and memory requirements, particularly in batteryoperated and real-time edge devices. While software-level optimizations such as quantization and pruning reduce model size and computation, hardware acceleration has emerged as a decisive enabler for efficient embedded inference. This paper evaluates the impact of Neural Processing Units (NPUs) on MCU-based ML execution, using the ARM Cortex-M55 core combined with the Ethos-U55 NPU on the Alif Semiconductor Ensemble E7 development board as a representative platform. A rigorous measurement methodology was employed, incorporating perinference net energy accounting via GPIO-triggered highresolution digital multimeter synchronization and idle-state subtraction, ensuring accurate attribution of energy costs. Experimental results across six representative ML modelsincluding MiniResNet, MobileNetV2, FD-MobileNet, MNIST, TinyYolo, and SSD-MobileNet-demonstrate substantial efficiency gains when inference is offloaded to the NPU. For moderate to large networks, latency improvements ranged from 7× to over 125×, with per-inference net energy reductions up to 143×. Notably, the NPU enabled execution of models unsupported on CPU-only paths, such as SSD-MobileNet, highlighting its functional as well as efficiency advantages. These findings establish NPUs as a cornerstone of energy-aware embedded AI, enabling real-time, power-constrained ML inference at the MCU level.

### 14. Expert-as-a-Service: Towards Efficient, Scalable, and Robust Large-scale MoE Serving

**主要机构**: Shanghai Qiji Zhifeng Co., Ltd, National University of Singapore, Nanyang Technology University, National University of Singapore Shanghai Qiji Zhifeng Co., Ltd, Shanghai Innovation Institute Shanghai Qiji Zhifeng Co., Ltd, Tsinghua University
**作者数量**: 15人

**摘要**:
Mixture-of-Experts (MoE) models challenge serving infrastructures with dynamic, sparse expert utilization, causing instability on conventional systems designed for dense architectures. We propose EaaS, a novel serving system to enable efficient, scalable, and robust MoE deployment. Our system disaggregates MoE modules into independent, stateless services. This design enables fine-grained resource scaling and provides inherent fault tolerance by decoupling compute units. The architecture is powered by a high-performance, CPU-free peer-to-peer communication library that ensures minimal overhead and high throughput. Experiments confirm EaaS's scalability and efficiency, achieving performance comparable to monolithic systems while providing robust fault tolerance and strong scalability. EaaS incurs less than a 2% throughput reduction under simulated hardware failures that would otherwise halt monolithic architectures. It further saves up to 37.5% of computing resources through * Work done during Ziming Liu's internship at Shanghai Qiji Zhifeng Co., Ltd. † Corresponding authors. dynamic fine-grained adaptation to serving traffic, demonstrating strong resilience for large-scale MoE deployment in production.

### 15. EYE GAZE TELLS YOU WHERE TO COMPUTE: GAZE-DRIVEN EFFICIENT VLMS

**主要机构**: Leiden University Leiden, Leiden Institute of Advanced Computer Science (LIACS)
**作者数量**: 2人

**摘要**:
Vision-Language Models (VLMs) deliver impressive performance in understanding visual content with language instructions. However, redundancy in vision tokens results in the degenerated inference efficiency of VLMs, which hinders real-time use on edge consumer devices such as Virtual Reality (VR) headsets and Augmented Reality (AR) glasses. Existing efficiency methods commonly prune visual tokens using learned saliency, sparse attention schedules, or controller policies, but they often require architectural modification or access to intermediate activations. These pipelines add inference-time modules that increase compute and memory and often lead to an accuracy trade-off. Moreover, they also suffer from misalignment between the prompts and the region of interest in the images. Without human guidance, the model may focus on the wrong regions and miss small, high-frequency details when prompts or scenes change. In this paper, we propose GazeVLM, a training-free framework that uses the human eye gaze as a natural supervisory signal to allocate computation where it matters. By extracting gaze-driven regions of interest (ROIs) and optionally combining them with a low-resolution global view, GazeVLM mimics fovea-periphery perception to cut redundant visual tokens while preserving task-relevant details. We evaluate the visual question answering tasks on Qwen2.5-VL-3B/7B on the VOILA-COCO benchmark with human gaze. Quality of the answer is assessed by GPT-4o pairwise judging and a weighted score over coverage, accuracy, details, and fluency. Efficiency is measured by token counts and FLOPs. GazeVLM reduces visual tokens by up to 93.1%, total tokens by up to 59.6%, and FLOPs by 50%, while keeping better answer quality relative to full-resolution baselines. Our results show that aligning model computation with human gaze offers a simple, plugand-play path toward efficient VLM inference on consumer devices. The code is available at https://github.com/qinche106/GazeVLM.

### 16. FG-ATTN: LEVERAGING FINE-GRAINED SPARSITY IN DIFFUSION TRANSFORMERS

**主要机构**: 
**作者数量**: 8人

**摘要**:
Generating realistic videos with diffusion transformers demands significant computation, with attention layers becoming the central bottleneck. Even producing a short clip requires running a transformer over a very long sequence of embeddings, e.g., more than 30K embeddings for a 5-second video. These long sequence lengths thus incur significant compute latencies. Prior work aims to mitigate this bottleneck by exploiting sparsity in the attention layers to reduce the computation required. However, these works typically rely on block-sparse attention, which skips score computation only when all entries in a block of attention scores (corresponding to M queries and M keys, with M = 64 typically) are zero. This coarse-granular skipping of attention scores does not fully exploit sparsity in the attention map and leaves significant room for improvement. In this work, we propose FG-Attn, a sparse attention mechanism for long-context diffusion transformers that leverages sparsity at a fine granularity. Unlike block-sparse attention, which skips entire M × M blocks, our approach skips computations at the granularity of M × 1 slices of the attention map. Each slice is produced as a result of query-key dot products between a block of query vectors and a single key. To implement our proposed sparse attention mechanism, we construct a new highly efficient bulk-load operation called asynchronous-gather load. This load operation gathers a sparse set of relevant key-value vectors from memory and arranges them into packed tiles in the GPU's shared memory. In this manner, only a sparse set of keys relevant to those queries are loaded into shared memory when computing attention for a block of queries, in contrast to loading full blocks of key tokens in block-sparse attention. Our fine-grained sparse attention, applied to video diffusion models, achieves an average 1.55x (up to 1.65x) speedup for 5 second, 480p videos, and an average 1.41x (up to 1.49x) for 5 second, 720p videos on a single H100 GPU.

### 17. Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics

**主要机构**: School of Computer Science, Sichuan University Chengdu
**作者数量**: 12人

**摘要**:
As the third generation of neural networks, spiking neural networks (SNNs) have recently gained widespread attention for their biological plausibility, energy efficiency, and effectiveness in processing neuromorphic datasets. To better emulate biological neurons, various models such as Integrate-and-Fire (IF) and Leaky Integrate-and-Fire (LIF) have been widely adopted in SNNs. However, these neuron models overlook the refractory period, a fundamental characteristic of biological neurons. Research on excitable neurons reveal that after firing, neurons enter a refractory period during which they are temporarily unresponsive to subsequent stimuli. This mechanism is critical for preventing over-excitation and mitigating interference from aberrant signals. Therefore, we propose a simple yet effective method to incorporate the refractory period into spiking LIF neurons through spike-triggered threshold dynamics, termed RPLIF. Our method ensures that each spike accurately encodes neural information, effectively preventing neuron over-excitation under continuous inputs and interference from anomalous inputs. Incorporating the refractory period into LIF neurons is seamless and computationally efficient, enhancing robustness and efficiency while yielding better performance with negligible overhead. To the best of our knowledge, RPLIF achieves state-of-the-art performance on Cifar10-DVS(82.40%) and N-Caltech101(83.35%) with fewer timesteps and demonstrates superior performance on DVS128 Gesture(97.22%) at low latency.

### 18. ISCS: Parameter-Guided Channel Ordering and Grouping for Learned Image Compression

**主要机构**: Santa Clara University Santa Clara, Futurewei Technologies, Inc. San Jose
**作者数量**: 5人

**摘要**:
Prior studies in learned image compression (LIC) consistently show that only a small subset of latent channels is critical for reconstruction, while many others carry limited information. Exploiting this imbalance could improve both coding and computational efficiency, yet existing approaches often rely on costly, dataset-specific ablation tests and typically analyze channels in isolation, ignoring their interdependencies. We propose a generalizable, dataset-agnostic method to identify and organize important channels in pretrained VAE-based LIC models. Instead of brute-force empirical evaluations, our approach leverages intrinsic parameter statistics-weight variances, bias magnitudes, and pairwise correlations-to estimate channel importance. This analysis reveals a consistent organizational structure, termed the Invariant Salient Channel Space (ISCS), where Salient-Core channels capture dominant structures and Salient-Auxiliary channels provide complementary details. Building on ISCS, we introduce a deterministic channel ordering and grouping strategy that enables slice-parallel decoding, reduces redundancy, and improves bitrate efficiency. Experiments across multiple LIC architectures demonstrate that our method effectively reduces bitrate and computation while maintaining reconstruction quality, providing a practical and modular enhancement to existing learned compression frameworks.

### 19. MCP: A Control-Theoretic Orchestration Framework for Synergistic Efficiency and Interpretability in Multimodal Large Language Models

**主要机构**: Northeastern University
**作者数量**: 1人

**摘要**:
Aiming at the problems of computational inefficiency and insufficient interpretability faced by large models in complex tasks such as multi-round reasoning and multi-modal collaboration, this study proposes a three-layer collaboration framework based on model-controller-task adaptation (MCP). By decoupling large model functions into reasoning, generation and retrieval modules, and combining reinforcement learning-driven dynamic routing algorithms and task adaptation mechanisms, the systematic integration of control theory and large model dynamic reasoning is achieved for the first time. Experiments show that the MCP framework improves the performance of cross-modal benchmarking tasks, such as GLUE, COCO, ScienceQA, etc., by 15-30% compared with the baseline model, improves the reasoning efficiency by 40%, and generates the interpretable intermediate results through the Presenter layer, obtaining 90% of the manual interpretability scores, which provides a brand-new technological path to solve the bottleneck of the practical application of the large model.

### 20. MOA-OFF: ADAPTIVE HETEROGENEOUS MODALITY-AWARE OFFLOADING WITH EDGE-CLOUD COLLABORATION FOR EFFICIENT MULTIMODAL LLM INFERENCE

**主要机构**: Institute of Computing Technology, Chinese Academy of Sciences
**作者数量**: 7人

**摘要**:
Multimodal large language models (MLLMs) enable powerful cross-modal inference but impose significant computational and latency burdens, posing severe challenges for deployment in resource-constrained environments. In this paper, we propose MoA-Off, an adaptive heterogeneous modality-aware offloading framework with edge-cloud collaboration for efficient MLLM inference. MoA-Off introduces a lightweight heterogeneous modality-aware module that estimates the complexity of heterogeneous inputs through multi-dimensional feature analysis. Then, an adaptive edge-cloud collaborative offloading strategy is proposed that dynamically schedules workloads between edge and cloud based on modality-aware complexity scores and realtime system states. The experimental results demonstrate that MoA-Off can achieve over 30% reduction in latency and 30%-65% decrease in resource overhead while maintaining competitive accuracy compared to traditional approaches.

### 21. MOES ARE STRONGER THAN YOU THINK: HYPER-PARALLEL INFERENCE SCALING WITH ROE

**主要机构**: University of California San Diego
**作者数量**: 7人

**摘要**:
The generation quality of large language models (LLMs) is often improved by utilizing inference-time sequence-level scaling methods (e.g., Chain-of-Thought). We introduce hyper-parallel scaling, a complementary framework that improves prediction quality at the token level. Hyper-parallel scaling computes and aggregates multiple output proposals for a single token from the model. We implement this concept in Mixture-of-Experts (MoE) models, which we refer to as Roster of Experts (RoE). RoE is a training-free inference algorithm that turns a single MoE into a dynamic ensemble of MoEs. RoE injects controlled stochasticity into the expert routing mechanism, enabling it to sample multiple diverse experts for each token and aggregate their outputs for a more accurate final prediction. To overcome the computational cost, we introduce an efficient batching strategy and a specialized KV-caching mechanism that minimizes compute and memory overhead. For example, RoE enables a 7B MoE model to match the performance of a 10.5B MoE model while using 30% less compute for inference. These gains are achieved without any fine-tuning of model parameters.

### 22. MRADNET: A COMPACT RADAR OBJECT DETECTOR WITH METAFORMER

**主要机构**: School of Electrical Engineering and Computer Science, University of Ottawa, Sensor Cortek Inc
**作者数量**: 5人

**摘要**:
Frequency-modulated continuous wave radars have gained increasing popularity in the automotive industry. Its robustness against adverse weather conditions makes it a suitable choice for radar object detection in advanced driver assistance systems. These real-time embedded systems have requirements for the compactness and efficiency of the model, which have been largely overlooked in previous work. In this work, we propose mRadNet, a novel radar object detection model with compactness in mind. mRadNet employs a U-net style architecture with MetaFormer blocks, in which separable convolution and attention token mixers are used to capture both local and global features effectively. More efficient token embedding and merging strategies are introduced to further facilitate the lightweight design. The performance of mRadNet is validated on the CRUW dataset, improving state-of-the-art performance with the least number of parameters and FLOPs. Source code: huaiyu-chen/mRadNet.

### 23. OS-DiffVSR: Towards One-step Latent Diffusion Model for High-detailed Real-world Video Super-Resolution

**主要机构**: Huawei Noah's Ark Lab
**作者数量**: 8人

**摘要**:
Recently, latent diffusion models has demonstrated promising performance in real-world video super-resolution (VSR) task, which can reconstruct high-quality videos from distorted low-resolution input through multiple diffusion steps. Compared to image super-resolution (ISR), VSR methods needs to process each frame in a video, which poses challenges to its inference efficiency. However, video quality and inference efficiency have always been a trade-off for the diffusion-based VSR methods. In this work, we propose One-Step Diffusion model for real-world Video Super-Resolution, namely OS-DiffVSR. Specifically, we devise a novel adjacent frame adversarial training paradigm, which can significantly improve the quality of synthetic videos. Besides, we devise a multi-frame fusion mechanism to maintain inter-frame temporal consistency and reduce the flicker in video. Extensive experiments on several popular VSR benchmarks demonstrate that OS-DiffVSR can even achieve better quality than existing diffusion-based VSR methods that require dozens of sampling steps.

### 24. PGSTalker: Real-Time Audio-Driven Talking Head Generation via 3D Gaussian Splatting with Pixel-Aware Density Control

**主要机构**: Xinjiang University, Department of Computer Science and Technology, School of Computer Science and Technology, Tianjin University of Technology, Xinjiang Multimodal Intelligent Processing and Information Security Engineering Technology Research Center, Tsinghua University, School of Electrical Engineering and Automation
**作者数量**: 9人

**摘要**:
fuse audio and spatial features, thereby improving the accuracy of Gaussian deformation prediction. Extensive experiments on public datasets demonstrate that PGSTalker outperforms existing NeRF-and 3DGSbased approaches in rendering quality, lip-sync precision, and inference speed. Our method exhibits strong generalization capabilities and practical potential for real-world deployment.

### 25. Preference Distillation via Value based Reinforcement Learning

**主要机构**: Gwangju Institute of Science and Technology (GIST), Korea Advanced Institute of Science and Technology (KAIST)
**作者数量**: 4人

**摘要**:
Direct Preference Optimization (DPO) is a powerful paradigm to align language models with human preferences using pairwise comparisons. However, its binary win-or-loss supervision often proves insufficient for training small models with limited capacity. Prior works attempt to distill information from large teacher models using behavior cloning or KL divergence. These methods often focus on mimicking current behavior and overlook distilling reward modeling. To address this issue, we propose Teacher Value-based Knowledge Distillation (TVKD), which introduces an auxiliary reward from the value function of the teacher model to provide a soft guide. This auxiliary reward is formulated to satisfy potential-based reward shaping, ensuring that the global reward structure and optimal policy of DPO are preserved. TVKD can be integrated into the standard DPO training framework and does not require additional rollouts. Our experimental results show that TVKD consistently improves performance across various benchmarks and model sizes.

### 26. PRISM: PRECISION-RECALL INFORMED DATA-FREE KNOWLEDGE DISTILLATION VIA GENERATIVE DIFFUSION

**主要机构**: University of Electronic Science and Technology of China
**作者数量**: 6人

**摘要**:
Data-free knowledge distillation (DFKD) transfers knowledge from a teacher to a student without access to the real in-distribution (ID) data. While existing methods perform well on small-scale images, they suffer from mode collapse when synthesizing large-scale images, resulting in limited knowledge transfer. Recently, leveraging advanced generative models to synthesize photorealistic images has emerged as a promising alternative. Nevertheless, directly using off-the-shelf diffusion to generate datasets faces the precision-recall challenges: 1) ensuring synthetic data aligns with the real distribution, and 2) ensuring coverage of the real ID manifold. In response, we propose PRISM, a precision-recall informed synthesis method. Specifically, we introduce Energy-guided Distribution Alignment to avoid the generation of out-of-distribution samples, and design the Diversified Prompt Engineering to enhance coverage of the real ID manifold. Extensive experiments on various large-scale image datasets demonstrate the superiority of PRISM. Moreover, we demonstrate that models trained with PRISM exhibit strong domain generalization.

### 27. PruneCD: Contrasting Pruned Self Model to Improve Decoding Factuality

**主要机构**: Pohang University of Science and Technology (POSTECH), Graduate School of Artificial Intelligence, Department of Computer Science and Engineering, Department of Convergence IT Engineering
**作者数量**: 4人

**摘要**:
To mitigate the hallucination problem in large language models, DoLa exploits early exit logits from the same model as a contrastive prior. However, we found that these early exit logits tend to be flat, low in magnitude, and fail to reflect meaningful contrasts. To address this, we propose PruneCD, a novel contrastive decoding method that constructs the amateur model via layer pruning rather than early exit. This design leads to more informative and well-aligned logits, enabling more effective contrastive decoding. Through qualitative and quantitative analyses, we demonstrate that PruneCD consistently improves factuality with minimal inference overhead, offering a robust and practical approach to mitigating hallucinations in LLMs.

### 28. PTQTP: POST-TRAINING QUANTIZATION TO TRIT-PLANES FOR LARGE LANGUAGE MODELS

**主要机构**: The Hong Kong Polytechnic University, The University of Hong, University of California
**作者数量**: 9人

**摘要**:
Post-training quantization (PTQ) of large language models (LLMs) to extremely low bit-widths remains challenging due to the fundamental trade-off between computational efficiency and model expressiveness. While existing ultra-low-bit PTQ methods rely on binary approximations or complex compensation mechanisms, they suffer from either limited representational capacity or computational overhead that undermines their efficiency gains. We introduce PTQ to Trit-Planes (PTQTP), the first ternary-weight PTQ framework that decomposes weight matrices into structured ternary {-1, 0, 1} trit-planes using 2×1.58-bit representation. PTQTP achieves multiplication-free inference, identical to 1-bit quantization, while maintaining superior expressiveness through its novel structured decomposition. Our approach provides: (1) a theoretically grounded progressive approximation algorithm ensuring global weight consistency; (2) model-agnostic deployment across diverse modern LLMs without architectural modifications; and (3) uniform ternary operations that eliminate the need for mixed-precision or compensation schemes. Comprehensive experiments across LLaMA3.x and Qwen3 model families (0.6B-70B parameters) demonstrate that PTQTP significantly outperforms existing low-bit PTQ methods, achieving 82.4% mathematical reasoning retention versus 0% for competing approaches. PTQTP approaches and sometimes surpasses 1.58-bit quantization-aware training performance while requiring only single-hour quantization compared to 10-14 GPU days for training-based methods. These results establish PTQTP as a practical solution for efficient LLM deployment in resource-constrained environments.

### 29. 

**主要机构**: 
**作者数量**: 0人

**摘要**:
We present Qwen3-Omni, a single multimodal model that for the first time maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to single-modal counterparts. Qwen3-Omni matches the performance of same-sized single-modal models within the Qwen series and excels particularly on audio tasks. Across 36 audio and audiovisual benchmarks, Qwen3-Omni achieves opensource state-of-the-art (SOTA) on 32 benchmarks and overall SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker Mixture-of-Experts (MoE) architecture that unifies perception and generation across text, images, audio, and video, yielding fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages and speech generation in 10 languages. The system can process audio recordings up to 40 minutes per instance for ASR and spoken-language understanding, enabling high-quality audio and audiovisual experiences across locales. It demonstrates strong instruction following and allows fine-grained customization of conversational tone and persona via user-defined system prompts. To reduce first-packet latency in streaming synthesis, the Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme. Leveraging the representational capacity of these codebooks, we replace computationally intensive block-wise diffusion with a lightweight causal ConvNet, enabling streaming from the first codec frame. In cold-start settings (no prior context), Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms. To further strengthen multimodal reasoning, we introduce a Thinking model that explicitly reasons over inputs from any modality. Since the research community currently lacks a general-purpose audio captioning model, we fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which produces detailed, low-hallucination captions for arbitrary audio inputs. Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking and Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0 license.

### 30. QWHA: QUANTIZATION-AWARE WALSH-HADAMARD ADAPTATION FOR PARAMETER-EFFICIENT FINE-TUNING ON LARGE LANGUAGE MODELS

**主要机构**: Sungkyunkwan University, Seoul National University
**作者数量**: 5人

**摘要**:
The demand for efficient deployment of large language models (LLMs) has driven interest in quantization, which reduces inference cost, and parameter-efficient fine-tuning (PEFT), which lowers training overhead. This motivated the development of quantization-aware PEFT to produce accurate yet efficient quantized models. In this setting, reducing quantization error prior to fine-tuning is crucial for achieving high model accuracy. However, existing methods that rely on low-rank adaptation suffer from limited representational capacity. Recent Fourier-related transform (FT)-based adapters offer greater representational power than low-rank adapters, but their direct integration into quantized models often results in ineffective error reduction and increased computational overhead. To overcome these limitations, we propose QWHA, a method that integrates FT-based adapters into quantized models by employing the Walsh-Hadamard Transform (WHT) as the transform kernel, together with a novel adapter initialization scheme incorporating adaptive parameter selection and value refinement. We demonstrate that QWHA effectively mitigates quantization errors while facilitating fine-tuning, and that its design substantially reduces computational cost. Experimental results show that QWHA consistently outperforms baselines in low-bit quantization accuracy and achieves significant training speedups over existing FT-based adapters. The code is available at https://github.com/vantaa89/qwha.

### 31. R-Net: A Reliable and Resource-Efficient CNN for Colorectal Cancer Detection with XAI Integration

**主要机构**: Department of Computer Science and Engineering Daffodil, Lecturer (Industrial Automation, University of Southern Queensland, School of Engineering, International University, 4IR Research Cell, School of Mathematics, Physics and Computing Toowoomba Campus
**作者数量**: 6人

**摘要**:


### 32. RCTDistill: Cross-Modal Knowledge Distillation Framework for Radar-Camera 3D Object Detection with Temporal Fusion

**主要机构**: Hanyang University, Hyundai Motor Company, Seoul National University
**作者数量**: 8人

**摘要**:
Radar-camera fusion methods have emerged as a costeffective approach for 3D object detection but still lag behind LiDAR-based methods in performance. Recent works have focused on employing temporal fusion and Knowledge Distillation (KD) strategies to overcome these limitations. However, existing approaches have not sufficiently accounted for uncertainties arising from object motion or sensor-specific errors inherent in radar and camera modalities. In this work, we propose RCTDistill, a novel cross-modal KD method based on temporal fusion, comprising three key modules: Range-Azimuth Knowledge Distillation (RAKD), Temporal Knowledge Distillation (TKD), and Region-Decoupled Knowledge Distillation (RDKD). RAKD is designed to consider the inherent errors in the range and azimuth directions, enabling effective knowledge transfer from LiDAR features to refine inaccurate BEV representations. TKD mitigates temporal misalignment caused by dynamic objects by aligning historical radar-camera BEV features with current LiDAR representations. RDKD enhances feature discrimination by distilling relational knowledge from the teacher model, allowing the student to differentiate foreground and background features. RCTDistill achieves state-of-the-art radar-camera fusion performance on both the nuScenes and View-of-Delft (VoD) datasets, with the fastest inference speed of 26.2 FPS.

### 33. SAEC: SCENE-AWARE ENHANCED EDGE-CLOUD COLLABORATIVE INDUSTRIAL VISION INSPECTION WITH MULTIMODAL LLM

**主要机构**: Institute of Computing Technology, Chinese Academy of Sciences
**作者数量**: 2人

**摘要**:
Industrial vision inspection requires high accuracy under stringent resource constraints, yet existing approaches face a fundamental trade-off. Multimodal LLMs (MLLMs) deliver strong reasoning capabilities but incur prohibitive computational costs, while lightweight edge models often fail on complex cases. In this paper, we present SAEC, a sceneaware enhanced edge-cloud collaborative industrial vision inspection framework with MLLM. The framework is composed of three synergistic components: (1) Efficient MLLM Fine-Tuning for Complex Defect Inspection, (2) Lightweight Multiscale Scene-Complexity Estimation, and (3) Adaptive Edge-Cloud Scheduler. Together, these modules enable robust defect detection by tailoring multimodal reasoning to scene complexity and dynamically balancing computation between edge and cloud resources. Experimental results on MVTec AD and KSDD2 datasets demonstrate that SAEC attains 85.11% and 82.72% accuracy, surpassing Qwen by 22.1% and 20.8%, and LLaVA by 33.3% and 31.6%. It also reduces runtime by up to 22.4% and cuts energy per correct decision by 40%-74%. The code is available at https://github.com/YuHao-Tian/SAEC.

### 34. SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning

**主要机构**: Soochow University
**作者数量**: 6人

**摘要**:
Process reward models (PRMs) offer fine-grained, step-level evaluations that facilitate deeper reasoning processes in large language models (LLMs), proving effective in complex tasks like mathematical reasoning. However, developing PRMs is challenging due to the high cost and limited scalability of human-annotated data. Synthetic data from Monte Carlo (MC) estimation is a promising alternative but suffers from a high noise ratio, which can cause overfitting and hinder large-scale training. In this work, we conduct a preliminary study on the noise distribution in synthetic data from MC estimation, identifying that annotation models tend to both underestimate and overestimate step correctness due to limitations in their annotation capabilities. Building on these insights, we propose Self-Denoising Monte Carlo Annotation (SCAN), an efficient data synthesis and noise-tolerant learning framework. Our key findings indicate that: (1) Even lightweight models (e.g., 1.5B parameters) can produce high-quality annotations through a self-denoising strategy, enabling PRMs to achieve superior performance with only 6% the inference cost required by vanilla MC estimation. (2) With our robust learning strategy, PRMs can effectively learn from this weak supervision, achieving a 39.2 F1 score improvement (from 19.9 to 59.1) in ProcessBench. Despite using only a compact synthetic dataset, our models surpass strong baselines, including those trained on large-scale human-annotated datasets such as PRM800K. Furthermore, performance continues to improve as we scale up the synthetic data, highlighting the potential of SCAN for scalable, cost-efficient, and robust PRM training.

### 35. SecureFixAgent: A Hybrid LLM Agent for Automated Python Static Vulnerability Repair

**主要机构**: Computer Science Department, The George
**作者数量**: 1人

**摘要**:
Modern software development pipelines face growing challenges in securing large codebases with extensive dependencies. Static analysis tools like Bandit are effective at vulnerability detection but suffer from high false positives and lack repair capabilities. Large Language Models (LLMs), in contrast, can suggest fixes but often hallucinate changes and lack self-validation. We present SecureFixAgent, a hybrid repair framework integrating Bandit with lightweight local LLMs (<8B parameters) in an iterative detect-repair-validate loop. To improve precision, we apply parameter-efficient LoRA-based fine-tuning on a diverse, curated dataset spanning multiple Python project domains, mitigating dataset bias and reducing unnecessary edits. SecureFixAgent uses Bandit for detection, the LLM for candidate fixes with explanations, and Bandit revalidation for verification, all executed locally to preserve privacy and reduce cloud reliance. Experiments show SecureFixAgent reduces false positives by 10.8% over static analysis, improves fix accuracy by 13.51%, and lowers false positives by 5.46% compared to pre-trained LLMs, typically converging within three iterations. Beyond metrics, developer studies rate explanation quality 4.5/5, highlighting its value for human trust and adoption. By combining verifiable security improvements with transparent rationale in a resource-efficient local framework, SecureFixAgent advances trustworthy, automated vulnerability remediation for modern pipelines.

### 36. ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching

**主要机构**: University of Washington ¶ UC Davis, Harvard University, University of Chicago
**作者数量**: 9人

**摘要**:
Distributed prefix caching accelerates long-context LLM serving by reusing KV cache entries for common context prefixes. However, KV cache fetches can become a bottleneck when network bandwidth is limited. Compression mitigates the bandwidth issue, but can degrade overall performance when decompression interferes with model computation. We present ShadowServe, the first SmartNIC-accelerated, interference-free prefix caching system for LLM serving. ShadowServe separates a control plane on the host and a data plane fully offloaded to the SmartNIC, which eliminates interference to both host GPU and CPU. To overcome the Smart-NIC's limited compute and memory resources, we design a chunked pipeline that parallelizes data plane operations across the SmartNIC's compute resources, and a minimal-copy memory management scheme that reduces memory pressure on the SmartNIC. Compared to state-of-the-art solutions, Shad-owServe achieves up to 2.2× lower loaded time-per-outputtoken (TPOT), and reduces time-to-first-token (TTFT) by up to 1.38× in low-bandwidth scenarios (≤ 20 Gbps), translating to up to 1.35× higher throughput.

### 37. Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

**主要机构**: Snowflake AI Research
**作者数量**: 6人

**摘要**:
Efficient parallelism is necessary for achieving low-latency, high-throughput inference with large language models (LLMs). Tensor parallelism (TP) is the state-of-the-art method for reducing LLM response latency, however GPU communications reduces combined token throughput. On the other hand, data parallelism (DP) obtains a higher throughput yet is slow in response latency. Best of both worlds does not exist, and it is not possible to combine TP and DP because of the KV cache variance across the parallelisms. We notice Sequence Parallelism (SP-Ulysses in training) has similar properties as DP but with KV cache invariance. We adapt SP to inference, and combine it with TP to get the best of both worlds. Our solution: Shift Parallelism. Shift Parallelism dynamically switches across TP and SP, and minimizes latency in low traffic without losing throughput in high traffic. The efficient GPU communications of Shift Parallelism yields up to i) 1.51× faster response in interactive workloads and ii) 50% higher throughput in batch workloads, compared to a TP-only solution. We evaluate Shift Parallelism with real-world production traces with dynamic traffic patterns as well as synthetic benchmarking patterns across models, context sizes, and arrival rates. All results affirm the same: Shift Parallelism has a better the latency vs. throughput tradeoff than TP or DP, and hence obtains low latency without degrading throughput in dynamic workloads.

### 38. SISMA: Semantic Face Image Synthesis with Mamba

**主要机构**: Department of Information Engineering and Mathematical Sciences, University of Parma, University of Siena, Department of Engineering and Architecture
**作者数量**: 6人

**摘要**:
Diffusion Models have become very popular for Semantic Image Synthesis (SIS) of human faces. Nevertheless, their training and inference is computationally expensive and their computational requirements are high due to the quadratic complexity of attention layers. In this paper, we propose a novel architecture called SISMA, based on the recently proposed Mamba. SISMA generates high quality samples by controlling their shape using a semantic mask at a reduced computational demand. We validated our approach through comprehensive experiments with CelebAMask-HQ, revealing that our architecture not only achieves a better FID score yet also operates at three times the speed of state-ofthe-art architectures. This indicates that the proposed design is a viable, lightweight substitute to transformer-based models.

### 39. SlowFast-SCI: Slow-Fast Deep Unfolding Learning for Spectral Compressive Imaging

**主要机构**: Shanghai Jiaotong University, Harbin Institute of Technology, Harvard University
**作者数量**: 6人

**摘要**:
Humans learn in two complementary ways: a slow, cumulative process that builds broad, general knowledge, and a fast, on-the-fly process that captures specific experiences. Existing deep-unfolding methods for spectral compressive imaging (SCI) mirror only the slow component-relying on heavy pre-training with many unfolding stages-yet they lack the rapid adaptation needed to handle new optical configurations. As a result, they falter on out-of-distribution cameras, especially in bespoke spectral setups unseen during training. This depth also incurs heavy computation and slow inference. To bridge this gap, we introduce SlowFast-SCI, a dual-speed framework seamlessly integrated into any deep unfolding network beyond SCI systems. During slow learning, we pre-train or reuse a priors-based backbone and distill it via imaging guidance into a compact fast-unfolding model. In the fast learning stage, lightweight adaptation modules are embedded within each block and trained self-supervised at test time via a dual-domain loss-without retraining the backbone. To the best of our knowledge, SlowFast-SCI is the first test-time adaptation-driven deep unfolding framework for efficient, self-adaptive spectral reconstruction. Its dual-stage design unites offline robustness with on-thefly per-sample calibration-yielding over 70% reduction in parameters and FLOPs, up to 5.79 dB PSNR improvement on out-of-distribution data, preserved crossdomain adaptability, and a 4× faster adaptation speed. In addition, its modularity integrates with any deep-unfolding network, paving the way for self-adaptive, field-deployable imaging and expanded computational imaging modalities. Code and models are available at https://github.com/XuanLu11/SlowFast-SCI

### 40. Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding

**主要机构**: Qualcomm AI Research
**作者数量**: 6人

**摘要**:
Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by 2.8-3.1× while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to 7.9×.

### 41. STENCIL: SUBJECT-DRIVEN GENERATION WITH CONTEXT GUIDANCE

**主要机构**: S-Lab, Nanyang Technological University, IHPC, CFAR
**作者数量**: 4人

**摘要**:


### 42. The Role of Vocabularies in Learning Sparse Representations for Ranking

**主要机构**: 
**作者数量**: 3人

**摘要**:
Learned Sparse Retrieval (LSR) such as SPLADE has growing interest for effective semantic 1st stage matching while enjoying the efficiency of inverted indices. A recent work on learning SPLADE models with expanded vocabularies (ESPLADE) was proposed to represent queries and documents into a sparse space of custom vocabulary which have different levels of vocabularic granularity. Within this effort, however, there have not been many studies on the role of vocabulary in SPLADE models and their relationship to retrieval efficiency and effectiveness. To study this, we construct BERT models with 100K-sized output vocabularies, one initialized with the ESPLADE pretraining method and one initialized randomly. After finetune on real-world search click logs, we applied logit score-based queries and documents pruning to max size for further balancing efficiency. The experimental result in our evaluation set shows that, when pruning is applied, the two models are effective compared to the 32K-sized normal SPLADE model in the computational budget under the BM25. And the ESPLADE models are more effective than the random vocab model, while having a similar retrieval cost. The result indicates that the size and pretrained weight of output vocabularies play the role of configuring the representational specification for queries, documents, and their interactions in the retrieval engine, beyond their original meaning and purposes in NLP. These findings can provide a new room for improvement for LSR by identifying the importance of representational specification from vocabulary configuration for efficient and effective retrieval.

### 43. Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few

**主要机构**: Beijing University of Posts and Telecommunications
**作者数量**: 3人

**摘要**:
Attention mechanisms in Transformers have gained significant empirical success. Nonetheless, the optimization objectives underlying their forward pass are still unclear. Additionally, the quadratic complexity of self-attention is increasingly prohibitive. Unlike the prior work on addressing the interpretability or efficiency issue separately, we propose a unified optimization objective to alleviate both issues simultaneously. By unrolling the optimization over the objective, we derive an inherently interpretable and efficient attention mechanism, which compresses all tokens into low-dimensional structures by contracting a few representative tokens and then broadcasting the contractions back. This Contract-and-Broadcast Self-Attention (CBSA) mechanism can not only scale linearly but also generalize existing attention mechanisms as its special cases. Experiments further demonstrate comparable performance and even superior advantages of CBSA on several visual tasks. Code is available at this https URL.

### 44. Visual Detector Compression via Location-Aware Discriminant Analysis

**主要机构**: Bowling Green State University, University of Alabama at Birmingham, Dept. of Computer Science
**作者数量**: 3人

**摘要**:
Deep neural networks are powerful, yet their high complexity greatly limits their potential to be deployed on billions of resource-constrained edge devices. Pruning is a crucial network compression technique, yet most existing methods focus on classification models, with limited attention to detection. Even among those addressing detection, there is a lack of utilization of essential localization information. Also, many pruning methods passively rely on pre-trained models, in which useful and useless components are intertwined, making it difficult to remove the latter without harming the former at the neuron/filter level. To address the above issues, in this paper, we propose a proactive detection-discriminants-based network compression approach for deep visual detectors, which alternates between two steps: (1) maximizing and compressing detection-related discriminants and aligning them with a subset of neurons/filters immediately before the detection head, and (2) tracing the detection-related discriminating power across the layers and discarding features of lower importance. Object location information is exploited in both steps. Extensive experiments, employing four advanced detection models and four state-of-the-art competing methods on the KITTI and COCO datasets, highlight the superiority of our approach. Remarkably, our compressed models can even beat the original base models with a substantial reduction in complexity.

### 45. ViTCAE: ViT-based Class-conditioned Autoencoder

**主要机构**: ECE Department ECE Department ECE Department, NC State University Raleigh, NC State University
**作者数量**: 3人

**摘要**:
Vision Transformer (ViT) based autoencoders often underutilize the global Class token and employ static attention mechanisms, limiting both generative control and optimization efficiency. This paper introduces ViTCAE, a framework that addresses these issues by re-purposing the Class token into a generative linchpin. In our architecture, the encoder maps the Class token to a global latent variable that dictates the prior distribution for local, patch-level latent variables, establishing a robust dependency where global semantics directly inform the synthesis of local details. Drawing inspiration from opinion dynamics, we treat each attention head as a dynamical system of interacting tokens seeking consensus. This perspective motivates a convergence-aware temperature scheduler that adaptively anneals each head's influence function based on its distributional stability. This process enables a principled head-freezing mechanism, guided by theoretically-grounded diagnostics like an attention evolution distance and a consensus/cluster functional. This technique prunes converged heads during training to significantly improve computational efficiency without sacrificing fidelity. By unifying a generative Class token with an adaptive attention mechanism rooted in multi-agent consensus theory, ViTCAE offers a more efficient and controllable approach to transformer-based generation. 1

### 46. When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using Small VLMs

**主要机构**: Indian Institute of Technology Jodhpur
**作者数量**: 4人

**摘要**:
Large Vision-Language Models (L-VLMs) have demonstrated remarkable performance in various vision and language tasks, including visual question answering (VQA). However, their high computational cost makes them impractical for resource-constrained settings and inference-heavy applications. In contrast, Small Vision-Language Models (S-VLMs) offer efficiency but suffer from a significant performance gap compared to their larger counterparts. In this work, we introduce the Model Parity Aligner (MPA), a novel framework designed to systematically improve S-VLMs by leveraging unlabeled images and effective knowledge transfer from L-VLMs. Instead of traditional knowledge distillation methods that rely on labeled training data, MPA employs a strategic parity-based approach that precisely identifies the knowledge disparities between S-VLMs and L-VLMs, and optimizes training by targeting only these disparities. We conduct extensive experiments on four diverse VQA benchmarks, namely TextVQA, ST-VQA, ChartQA, and OKVQA, each of which required specialized reasoning capabilities such as text recognition, chart interpretation, and commonsense and factual understanding. Our results demonstrate that MPA consistently enhances the performance of S-VLM on all benchmarks, reducing the performance gap while maintaining computational efficiency. We make our code publicly available.
