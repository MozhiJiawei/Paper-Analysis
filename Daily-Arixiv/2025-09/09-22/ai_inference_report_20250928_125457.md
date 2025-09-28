# AI推理加速技术论文分析报告
生成时间: 2025-09-28 12:54:57
分析论文数量: 29篇

## 论文技术简报

### 1. Adversarially-Refined VQ-GAN with Dense Motion Tokenization for Spatio-Temporal Heatmaps

University of North Carolina at Charlotte和University of Wyoming发布了相关论文，使用对抗性优化的VQ-GAN框架结合密集运动标记化技术，解决了连续人体运动理解中的高维度和冗余性问题，实现时空热图高效压缩并保留细粒度运动轨迹，在CMU Panoptic数据集上较dVAE基线SSIM提升9.31%、时间不稳定性降低37.1%。

### 2. Bi-VLM: Pushing Ultra-Low Precision Post-Training Quantization Boundaries in Vision-Language Models

University of Maryland发布了Bi-VLM论文，使用基于高斯分位数非均匀分离权重的显著性感知混合量化技术，解决了视觉语言模型（VLMs）的计算需求与超低比特（≤2 bits）权重精度之间的差距以提升效率，达成在视觉问答任务上语言模型部分较SOTA提升3%-47%、整体VLM提升4%-45%，并发现量化模型中图像token 90%-99%冗余可进一步剪枝的效果

### 3. Chiplet-Based RISC-V SoC with Modular AI Acceleration

美国沙迦大学发布了Chiplet-Based RISC-V SoC with Modular AI Acceleration论文，使用模块化AI加速与智能系统级优化（含自适应跨chiplet DVFS、AI感知UCIe协议扩展等）技术，解决了边缘AI设备高性能、能效、成本效益与架构灵活性的平衡及单片SoC低良率问题，达成效率提升40.1%、延迟降低14.7%、吞吐量提升17.3%、功耗降低16.2%并保持亚5ms实时能力的效果

### 4. CODEBOOK-BASED ADAPTIVE FEATURE COMPRESSION WITH SEMANTIC ENHANCEMENT FOR EDGE-CLOUD SYSTEMS

哈尔滨工业大学、鹏城实验室发布了基于码本的自适应特征压缩与语义增强（CAFC-SE）论文，使用码本结合矢量量化（VQ）技术，解决了现有方法在低比特率下因冗余细节或符号分布集中导致的性能不佳问题，达成了在速率和准确性方面的优越性。

### 5. COMPLLM: COMPRESSION FOR LONG CONTEXT Q&A

University of Central Florida发布了COMPLLM: COMPRESSION FOR LONG CONTEXT Q&A论文，使用将长上下文分段独立压缩的软压缩技术，解决了大型语言模型处理长上下文时自注意力二次复杂度及现有压缩方法二次压缩复杂度和不可重用的问题，达成2倍压缩率下高上下文长度时TTFT加速4倍、KV缓存减少50%且性能与未压缩相当甚至在超长序列上更优的效果。

### 6. Evaluating the Safety and Skill Reasoning of Large Reasoning Models Under Compute Constraints

阿贡国家实验室发布了关于计算约束下大型推理模型安全与技能推理评估的论文，使用长度控制策略优化（LCPO）强化学习微调与模型量化的计算约束策略，解决了大型推理模型生成更长CoT序列提升性能时计算成本显著增加的问题，达成了在减少计算需求的同时研究计算效率与安全性权衡的效果。

### 7. FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction

相关机构发布了FastMTP论文，使用增强型多token预测技术（通过对齐MTP训练与推理模式、自蒸馏数据微调位置共享权重的MTP头及语言感知动态词汇压缩），解决了LLM推理的吞吐量瓶颈，达成了平均2.03倍加速且输出质量无损、性能超越普通MTP 82%的效果。

### 8. FC3DNET: A FULLY CONNECTED ENCODER-DECODER FOR EFFICIENT DEMOIR ÉING

中国科学技术大学发布了FC3DNet论文，使用全连接编解码器结合多特征多注意力融合（MFMAF）模块及解码器多尺度特征，解决了高效去除莫尔条纹中平衡速度与质量的问题，达成了性能媲美SOTA且参数、FLOPs和运行时间大幅减少的效果

### 9. FLOW MARCHING FOR A GENERATIVE PDE FOUNDATION MODEL A PREPRINT

麻省理工学院发布了FLOW MARCHING FOR A GENERATIVE PDE FOUNDATION MODEL论文，使用Flow Marching算法（结合神经算子学习与流匹配，联合采样噪声水平和物理时间步）及P2VAE、FMT，解决了现有PDE基础模型生成灵活性不足及长期滚动漂移问题，达成计算效率较全长视频扩散模型提升15倍、长期滚动稳定性增强及少样本适应柯尔莫哥洛夫湍流的效果。

### 10. Hyper-Bagel: A Unified Acceleration Framework for Multimodal Understanding and Generation

发布了Hyper-Bagel论文，使用统一加速框架（采用分治策略，结合推测解码与多阶段蒸馏），解决了多模态模型因扩散去噪和自回归解码迭代过程导致的计算开销问题，达成了理解任务2倍以上加速、文本到图像/图像编辑16.67倍/22倍无损加速及1-NFE模型近实时生成的效果。

### 11. LAWCAT: Efficient Distillation from Quadratic to Linear Attention with Convolution across Tokens for Long Context Modeling

亚马逊AGI与南加州大学发布了LAWCAT论文，使用整合因果Conv1D层与归一化门控线性注意力的线性化框架，解决了Transformer二次复杂度瓶颈及线性注意力模型训练资源密集问题，实现了长上下文建模中高效知识蒸馏（如Mistral-7B蒸馏后22K tokens passkey检索准确率超90%）且prefill速度优于FlashAttention-2（8K+ tokens）。

### 12. LCMF: Lightweight Cross-Modality Mambaformer for Embodied Robotics VQA

University Sorbonne等机构发布了LCMF: Lightweight Cross-Modality Mambaformer论文，使用轻量级跨模态Mambaformer（LCMF）级联注意力框架（引入多级别跨模态参数共享机制，融合Cross-Attention与选择性参数共享状态空间模型），解决了异构数据有效融合及资源受限环境下的计算效率问题，达成了VQA任务准确率74.29%、FLOPs较同类基线平均降低4.35倍且参数轻量化（图文166.51M、视频文本219M）的效果。

### 13. LEAF-Mamba: Local Emphatic and Adaptive Fusion State Space Model for RGB-D Salient Object Detection

大连理工大学与新加坡国立大学发布了LEAF-Mamba论文，使用含局部强调状态空间模块(LE-SSM)与基于SSM的自适应融合模块(AFM)的LEAF-Mamba模型，解决了RGB-D显著目标检测中局部语义不足、跨模态融合不充分及性能与效率难以平衡的问题，达成了在RGB-D SOD任务上超越16种现有方法且兼顾效能与效率，并在RGB-T SOD任务上展现优异泛化能力的效果

### 14. LIGHTWEIGHT VISION TRANSFORMER WITH WINDOW AND SPATIAL ATTENTION FOR FOOD IMAGE CLASSIFICATION

江南大学发布了轻量级食物图像分类论文，使用集成窗口多头注意力机制（WMHAM）和空间注意力机制（SAM）的技术，解决了Vision Transformer模型参数多、计算复杂度高的问题，达成了在Food-101和Vireo Food-172数据集上准确率分别达95.24%和94.33%且显著减少参数和FLOPs的效果。

### 15. LOCALIZED PCA-NET NEURAL OPERATORS FOR SCALABLE SOLUTION RECONSTRUCTION OF ELLIPTIC PDES A PREPRINT

宾夕法尼亚州立大学发布了关于椭圆型偏微分方程可扩展解重构的论文，使用基于补丁的PCA-Net框架（含局部到全局和局部到局部方法及重叠补丁平滑滤波、CNN两步优化），解决了传统PCA在高维解场中计算开销大的问题，达成计算复杂度显著降低、端到端处理时间比全局PCA减少3.7到4倍且保持高精度的效果。

### 16. LYRA: GENERATIVE 3D SCENE RECONSTRUCTION VIA VIDEO DIFFUSION MODEL SELF-DISTILLATION

University of Toronto发布了LYRA论文，使用视频扩散模型自蒸馏技术，解决了从视频中生成3D场景的挑战，实现了高效的生成式3D场景重建。

### 17. MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipes

MiniCPM-V Team发布了MiniCPM-V 4.5论文，使用统一3D-Resampler架构、文档知识与文本识别的统一学习范式及混合强化学习策略，解决了多模态大语言模型（MLLMs）的训练与推理效率瓶颈，达成了在OpenCompass评估中超越GPT-4o-latest等专有模型及Qwen2.5-VL 72B等更大开源模型的性能，且效率显著提升，如在VideoMME基准上以30B以下模型尺寸实现SOTA，GPU内存成本仅为Qwen2.5-VL 7B的46.7%、推理时间仅8.7%。

### 18. MMCD: Multi-Modal Collaborative Decision-Making for Connected Autonomy with Knowledge Distillation

研究团队发布了MMCD论文，使用多模态协作决策框架与跨模态知识蒸馏师生模型，解决了自动驾驶中传感器故障或联网车辆缺失时决策不可靠的问题，将驾驶安全性提升20.7%，超越现有最佳基线。

### 19. MOCROP: TRAINING FREE MOTION GUIDED CROPPING FOR EFFICIENT VIDEO ACTION RECOGNITION

深圳大学、都柏林大学学院发布了MoCrop论文，提出运动感知自适应裁剪模块，利用H.264运动向量定位运动密集区域实现训练free裁剪，解决压缩域视频动作识别效率问题，在ResNet-50上同等FLOPs提升3.5% Top-1准确率或减少26.5% FLOPs并提升2.4%准确率，CoViAR原成本达89.2%准确率。

### 20. NeuCODEX: Edge-Cloud Co-Inference with Spike-Driven Compression and Dynamic Early-Exit

Huawei Ireland Research Center Dublin发布了NeuCODEX论文，使用spike-driven compression和dynamic early-exit的边缘云协同推理技术，解决了边缘部署脉冲神经网络（SNN）时的高延迟、高传输成本及能耗问题，达成数据传输减少2048x、边缘能耗降低90%以上、端到端延迟减少3倍，精度损失小于2%的效果。

### 21. READING IMAGES LIKE TEXTS: SEQUENTIAL IMAGE UNDERSTANDING IN VISION-LANGUAGE MODELS

北京邮电大学发布了《像阅读文本一样阅读图像：视觉语言模型中的序列图像理解》论文，使用分解视觉处理为物体识别与空间感知、基于即插即用视觉解码器的指令无关token压缩算法及RoPE缩放技术，解决了现有视觉语言模型处理视觉信息时序列化图像与人类视觉并行性不符、内部机制不透明的问题，达成了提高解码效率、增强空间推理，加深对VLM内部理解并为设计更强大架构提供明确原则的效果。

### 22. Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding

康奈尔大学发布了SUBSPEC论文，使用无需训练、通过生成低比特量化替代层构建高度对齐草稿模型并共享GPU驻留层和KV-Cache的参数卸载加速技术，解决了大语言模型在内存有限消费级GPU上参数卸载推理速度慢的问题，达成了Qwen2.5 7B达9.1倍加速、32B平均12.5倍加速的效果。

### 23. Symphony-MoE: Harmonizing Disparate Pre-trained Models into a Coherent Mixture-of-Experts

鹏程实验室发布了Symphony-MoE论文，使用两阶段框架（层感知融合构建共享骨干、激活基功能对齐缓解专家参数错位及轻量级路由训练），解决了多异源预训练模型作为专家时参数空间差异导致的性能下降问题，达成整合异源专家并在多域任务和分布外泛化上显著超越基线的效果。

### 24. TEACHING AUDIO MODELS TO REASON: A UNIFIED FRAMEWORK FOR SOURCE-AND LAYER-WISE DISTILLATION

中国移动发布了TEACHING AUDIO MODELS TO REASON: A UNIFIED FRAMEWORK FOR SOURCE-AND LAYER-WISE DISTILLATION论文，使用源蒸馏（利用文本和声学教师提供互补模态监督）和层蒸馏（教师信号与学生层对齐）的统一知识蒸馏框架，解决了音频模型因音频与文本模态差距及缺乏结构化中间监督导致的复杂推理能力弱的问题，达成了音频推理性能显著提升的效果。

### 25. TinyBEV: Cross-Modal Knowledge Distillation for Efficient Multi-Task Bird's-Eye-View Perception and Planning

University of Arkansas Fayetteville发布了TinyBEV论文，使用跨模态多阶段知识蒸馏策略（结合特征级、输出级及自适应区域感知监督），解决大规模多模态感知规划模型在资源受限环境下难以实时部署完整自动驾驶栈的问题，达成参数减少78%、速度提升5倍（11 FPS）并保持全栈驾驶智能（39.0 mAP检测、1.08 minADE运动预测、0.32碰撞率）的效果。

### 26. TinyEcoWeedNet: Edge Efficient Real-Time Aerial Agricultural Weed Detection

College of Computer and Information Sciences发布了TinyEcoWeedNet论文，使用结构化通道剪枝、量化感知训练结合TensorRT加速技术，解决了边缘设备资源有限下农业杂草检测模型部署难题，达成参数减少68.5%、推理速度达184 FPS且mAP50 85.9%优于YOLO11n/12n的效果。

### 27. Towards General Computer Control with Hierarchical Agents and Multi-Level Action Spaces

东京大学与联想美国发布了关于通用电脑控制的论文，使用轻量级分层强化学习框架ComputerAgent（含两级选项过程、三重模态状态编码器及元动作早停机制），解决了现有多模态大语言模型在电脑控制中推理延迟、样本效率低及无法在设备上部署的问题，达成在135个真实桌面任务上简单任务成功率92.1%、硬任务58.8%，匹配200B+ MLLM基线且模型大小减少四个数量级、推理时间减半的效果。

### 28. VGGT-DP: Generalizable Robot Control via Vision Foundation Models

哈尔滨工业大学、清华大学发布了VGGT-DP论文，使用整合预训练3D感知模型几何先验与本体感受反馈的视觉-运动策略框架（含VGGT视觉编码器、本体感受引导视觉学习、帧-wise令牌重用机制及随机令牌剪枝），解决了现有视觉模仿学习中视觉编码器结构和能力不足导致的空间理解与泛化限制问题，在MetaWorld任务上显著优于DP、DP3等基线，尤其在精度关键和长时程场景表现突出。

### 29. Weight Mapping Properties of a Dual Tree Single Clock Adiabatic Capacitive Neuron

研究团队发布了《Weight Mapping Properties of a Dual Tree Single Clock Adiabatic Capacitive Neuron》论文，使用Dual Tree Single Clock (DTSC) Adiabatic Capacitive Neuron (ACN)电路及最优AN到ACN方法论，解决了人工神经元抽象权重到物理ACN电容值的映射复杂性与挑战问题，达成100%功能等效、促进更小芯片尺寸及提高分类精度的效果。

## 论文详细信息

### 1. Adversarially-Refined VQ-GAN with Dense Motion Tokenization for Spatio-Temporal Heatmaps

**主要机构**: University of Wyoming, University of North Carolina at Charlotte
**作者数量**: 6人

**摘要**:
Continuous human motion understanding remains a core challenge in computer vision due to its high dimensionality and inherent redundancy. Efficient compression and representation are crucial for analyzing complex motion dynamics. In this work, we introduce an adversarially-refined VQ-GAN framework with dense motion tokenization for compressing spatio-temporal heatmaps while preserving the fine-grained traces of human motion. Our approach combines dense motion tokenization with adversarial refinement, which eliminates reconstruction artifacts like motion smearing and temporal misalignment observed in non-adversarial baselines. Our experiments on the CMU Panoptic dataset [7] provide conclusive evidence of our method's superiority, outperforming the dVAE baseline by 9.31% SSIM and reducing temporal instability by 37.1%. Furthermore, our dense tokenization strategy enables a novel analysis of motion complexity, revealing that 2D motion can be optimally represented with a compact 128-token vocabulary, while 3D motion's complexity demands a much larger 1024-token codebook for faithful reconstruction. These results establish practical deployment feasibility across diverse motion analysis applications.

### 2. Bi-VLM: Pushing Ultra-Low Precision Post-Training Quantization Boundaries in Vision-Language Models

**主要机构**: University of Maryland
**作者数量**: 6人

**摘要**:
We address the critical gap between the computational demands of vision-language models and the possible ultralow-bit weight precision (bitwidth ≤ 2 bits) we can use for higher efficiency. Our work is motivated by the substantial computational cost and memory requirements of VLMs, which restrict their applicability in hardware-constrained environments. We propose Bi-VLM, which separates model weights non-uniformly based on the Gaussian quantiles. Our formulation groups the model weights into outlier (salient) and multiple inlier (unsalient) subsets, ensuring that each subset contains a proportion of weights corresponding to its quantile in the distribution. We propose a saliency-aware hybrid quantization algorithm and use it to quantize weights by imposing different constraints on the scaler and binary matrices based on the saliency metric and compression objective. We have evaluated our approach on different VLMs. For the language model part of the VLM, our Bi-VLM outperforms the SOTA by 3%-47% on the visual question answering task in terms of four different benchmarks and three different models. For the overall VLM, our Bi-VLM outperforms the SOTA by 4%-45%. We also perform token pruning on the quantized models and observe that there is redundancy of image tokens 90%-99% in the quantized models. This helps us to further prune the visual tokens to improve efficiency.

### 3. Chiplet-Based RISC-V SoC with Modular AI Acceleration

**主要机构**: Department of Electrical and Electronics Engineering, American University of Sharjah, College of Engineering, Birla Institute of Technology and Science
**作者数量**: 4人

**摘要**:
Achieving high performance, energy efficiency, and cost-effectiveness while maintaining architectural flexibility has remained a critical challenge in the development and deployment of Edge AI devices. Current monolithic SoC designs struggle with this complex balance which results in low manufacturing yields (below 16%) at advanced 360 mm 2 process nodes. This paper presents a novel chiplet-based RISC-V SoC architecture that addresses these limitations through modular AI acceleration and intelligent system level optimization. Our proposed design integrates four different key innovations in a 30 mm x 30 mm silicon interposer: adaptive cross-chiplet Dynamic Voltage and Frequency Scaling (DVFS); AI-aware Universal Chiplet Interconnect Express (UCIe) protocol extensions featuring streaming flow control units and compression-aware transfers; distributed cryptographic security across heterogeneous chiplets; and intelligent sensordriven load migration. The proposed architecture integrates a 7 nm RISC-V CPU chiplet with dual 5 nm AI accelerators (15 TOPS INT8 each), 16GB HBM3 memory stacks, and dedicated power management controllers. Experimental results across industry standard benchmarks like MobileNetV2, ResNet-50 and real-time video processing demonstrate significant performance improvements. The AI-optimized configuration achieves ~14.7% latency reduction, 17.3% throughput improvement, and 16.2% power reduction compared to previous basic chiplet implementations. These improvements collectively translate to a 40.1% efficiency gain corresponding to ≈ 3.5 mJ per MobileNetV2 inference (860 mW / 244 images/s), while maintaining sub-5ms real-time capability across all experimented workloads. These performance upgrades demonstrate that modular chiplet designs can achieve near-monolithic computational density while enabling cost efficiency, scalability and upgradeability, crucial for nextgeneration edge AI device applications.

### 4. CODEBOOK-BASED ADAPTIVE FEATURE COMPRESSION WITH SEMANTIC ENHANCEMENT FOR EDGE-CLOUD SYSTEMS

**主要机构**: Harbin Institute of Technology, Pengcheng Laboratory
**作者数量**: 5人

**摘要**:
Coding images for machines with minimal bitrate and strong analysis performance is key to effective edge-cloud systems. Several approaches deploy an image codec and perform analysis on the reconstructed image. Other methods compress intermediate features using entropy models and subsequently perform analysis on the decoded features. Nevertheless, these methods both perform poorly under low-bitrate conditions, as they retain many redundant details or learn over-concentrated symbol distributions. In this paper, we propose a Codebookbased Adaptive Feature Compression framework with Semantic Enhancement, named CAFC-SE. It maps continuous visual features to discrete indices with a codebook at the edge via Vector Quantization (VQ) and selectively transmits them to the cloud. The VQ operation that projects feature vectors onto the nearest visual primitives enables us to preserve more informative visual patterns under low-bitrate conditions. Hence, CAFC-SE is less vulnerable to low-bitrate conditions. Extensive experiments demonstrate the superiority of our method in terms of rate and accuracy.

### 5. COMPLLM: COMPRESSION FOR LONG CONTEXT Q&A

**主要机构**: Center For Research in Computer Vision, University of Central Florida
**作者数量**: 4人

**摘要**:
Large Language Models (LLMs) face significant computational challenges when processing long contexts due to the quadratic complexity of self-attention. While soft context compression methods, which map input text to smaller latent representations, have shown promise, their real-world adoption is limited. Existing techniques typically compress the context as a single unit, which leads to quadratic compression complexity and an inability to reuse computations across queries with overlapping contexts. In this work, we introduce CompLLM, a soft compression technique designed for practical deployment. Instead of processing the context holistically, CompLLM divides it into segments and compresses each one independently. This simple design choice yields three critical properties: efficiency, as the compression step scales linearly with the context length; scalability, enabling models trained on short sequences (e.g., 1k tokens) to generalize to contexts of 100k tokens; and reusability, allowing compressed segments to be cached and reused across different queries. Our experiments show that with a 2x compression rate, at high context lengths CompLLM speeds up Time To First Token (TTFT) by up to 4x and reduces the KV cache size by 50%. Furthermore, Com-pLLM achieves performance comparable to that obtained with the uncompressed context, and even surpasses it on very long sequences, demonstrating its effectiveness and practical utility.

### 6. Evaluating the Safety and Skill Reasoning of Large Reasoning Models Under Compute Constraints

**主要机构**: Mathematics and Computer Science Division, Data Science and Learning Division, Argonne National Laboratory Lemont
**作者数量**: 5人

**摘要**:
Test-time compute scaling has demonstrated the ability to improve the performance of reasoning language models by generating longer chain-of-thought (CoT) sequences. However, this increase in performance comes with a significant increase in computational cost. In this work, we investigate two compute constraint strategies: (1) reasoning length constraint and (2) model quantization, as methods to reduce the compute demand of reasoning models and study their impact on their safety performance. Specifically, we explore two approaches to apply compute constraints to reasoning models: (1) fine-tuning reasoning models using a lengthcontrolled policy optimization (LCPO) based reinforcement learning method to satisfy a user-defined CoT reasoning length, and (2) applying quantization to maximize the generation of CoT sequences within a user-defined compute constraint. Furthermore, we study the trade-off between the computational efficiency and the safety of the model.

### 7. FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction

**主要机构**: 
**作者数量**: 11人

**摘要**:
As large language models (LLMs) become increasingly powerful, the sequential nature of autoregressive generation creates a fundamental throughput bottleneck that limits the practical deployment. While Multi-Token Prediction (MTP) has demonstrated remarkable benefits for model training efficiency and performance, its inherent potential for inference acceleration remains largely unexplored. This paper introduces FastMTP, a simple yet effective method that improves multi-step draft quality by aligning MTP training with its inference pattern, significantly enhancing speculative decoding performance. Our approach fine-tunes a single MTP head with position-shared weights on self-distilled data, enabling it to capture dependencies among consecutive future tokens and maintain high acceptance rates across multiple recursive draft steps. By integrating language-aware dynamic vocabulary compression into the MTP head, we further reduce computational overhead in the drafting process. Experimental results across seven diverse benchmarks demonstrate that FastMTP achieves an average of 2.03× speedup compared to standard next token prediction with lossless output quality, outperforming vanilla MTP by 82%. FastMTP requires only lightweight training and seamlessly integrates with existing inference frameworks, offering a practical and rapidly deployable solution for accelerating LLM inference.

### 8. FC3DNET: A FULLY CONNECTED ENCODER-DECODER FOR EFFICIENT DEMOIR ÉING

**主要机构**: Department of Automation Hefei, University of Science and Technology of China
**作者数量**: 5人

**摘要**:
Moiré patterns are commonly seen when taking photos of screens. Camera devices usually have limited hardware performance but take high-resolution photos. However, users are sensitive to the photo processing time, which presents a hardly considered challenge of efficiency for demoiréing methods. To balance the network speed and quality of results, we propose a Fully Connected enCoder-deCoder based Demoiréing Network (FC3DNet). FC3DNet utilizes features with multiple scales in each stage of the decoder for comprehensive information, which contains long-range patterns as well as various local moiré styles that both are crucial aspects in demoiréing. Besides, to make full use of multiple features, we design a Multi-Feature Multi-Attention Fusion (MFMAF) module to weigh the importance of each feature and compress them for efficiency. These designs enable our network to achieve performance comparable to state-of-the-art (SOTA) methods in real-world datasets while utilizing only a fraction of parameters, FLOPs, and runtime.

### 9. FLOW MARCHING FOR A GENERATIVE PDE FOUNDATION MODEL A PREPRINT

**主要机构**: Department of Mechanical Engineering, Massachusetts Institute of Technology Cambridge
**作者数量**: 4人

**摘要**:
Pretraining on large-scale collections of PDE-governed spatiotemporal trajectories has recently shown promise for building generalizable models of dynamical systems. Yet most existing PDE foundation models rely on deterministic Transformer architectures, which lack generative flexibility for many science and engineering applications. We propose Flow Marching, an algorithm that bridges neural operator learning with flow matching motivated by an analysis of error accumulation in physical dynamical systems, and we build a generative PDE foundation model on top of it. By jointly sampling the noise level and the physical time step between adjacent states, the model learns a unified velocity field that transports a noisy current state toward its clean successor, reducing long-term rollout drift while enabling uncertainty-aware ensemble generations. Alongside this core algorithm, we introduce a Physics-Pretrained Variational Autoencoder (P2VAE) to embed physical states into a compact latent space, and an efficient Flow Marching Transformer (FMT) that combines a diffusion-forcing scheme with latent temporal pyramids, achieving up to 15× greater computational efficiency than full-length video diffusion models and thereby enabling large-scale pretraining at substantially reduced cost. We curate a corpus of ∼2.5M trajectories across 12 distinct PDE families and train suites of P2VAEs and FMTs at multiple scales. On downstream evaluation, we benchmark on unseen Kolmogorov turbulence with few-shot adaptation, demonstrate long-term rollout stability over deterministic counterparts, and present uncertainty-stratified ensemble results, highlighting the importance of generative PDE foundation models for real-world applications.

### 10. Hyper-Bagel: A Unified Acceleration Framework for Multimodal Understanding and Generation

**主要机构**: 
**作者数量**: 7人

**摘要**:
Unified multimodal models have recently attracted considerable attention for their remarkable abilities in jointly understanding and generating diverse content. However, as contexts integrate increasingly numerous interleaved multimodal tokens, the iterative processes of diffusion denoising and autoregressive decoding impose significant computational overhead. To address this, we propose Hyper-Bagel, a unified acceleration framework designed to simultaneously speed up both multimodal understanding and generation tasks. Our approach uses a divide-and-conquer strategy, employing speculative decoding for next-token prediction and a multi-stage distillation process for diffusion denoising. The framework delivers substantial performance gains, achieving over a 2x speedup in multimodal understanding. For generative tasks, our resulting lossless 6-NFE model yields a 16.67x speedup in text-to-image generation and a 22x speedup in image editing, all while preserving the high-quality output of the original model. We further develop a highly efficient 1-NFE model that enables near real-time interactive editing and generation. By combining advanced adversarial distillation with human feedback learning, this model achieves ultimate cost-effectiveness and responsiveness, making complex multimodal interactions seamless and instantaneous.

### 11. LAWCAT: Efficient Distillation from Quadratic to Linear Attention with Convolution across Tokens for Long Context Modeling

**主要机构**: University of Southern California, Case Western Reserve University, Amazon AGI, Intel Labs
**作者数量**: 8人

**摘要**:
Although transformer architectures have achieved state-of-the-art performance across diverse domains, their quadratic computational complexity with respect to sequence length remains a significant bottleneck, particularly for latency-sensitive long-context applications. While recent linear-complexity alternatives are increasingly powerful, effectively training them from scratch is still resource-intensive. To overcome these limitations, we propose LAWCAT (Linear Attention with Convolution Across Time), a novel linearization framework designed to efficiently transfer the capabilities of pre-trained transformers into a performant linear attention architecture. LAWCAT integrates causal Conv1D layers to enhance local dependency modeling and employs normalized gated linear attention to improve generalization across varying context lengths. Our comprehensive evaluations demonstrate that, distilling Mistral-7B with only 1K-length sequences yields over 90% passkey retrieval accuracy up to 22K tokens, significantly extending its effective context window. Similarly, Llama3.2-1B LAWCAT variant achieves competitive performance on S-NIAH 1&2&3 tasks (1K-8K context length) and BABILong benchmark (QA2&QA3, 0K-16K context length), requiring less than 0.1% pre-training tokens compared with pre-training models. Furthermore, LAWCAT exhibits faster prefill speeds than FlashAttention-2 for sequences exceeding 8K tokens. LAWCAT thus provides an efficient pathway to high-performance, long-context linear models suitable for edge deployment, reducing reliance on extensive long-sequence training data and computational resources.

### 12. LCMF: Lightweight Cross-Modality Mambaformer for Embodied Robotics VQA

**主要机构**: School of Software, Yangtze River Delta Research Institute (Taicang), Laboratoire L2Tl University Sorbonne, Yanxin Zhang School of Software, Liang He School of Software
**作者数量**: 7人

**摘要**:
Multimodal semantic learning plays a critical role in embodied intelligence, especially when robots perceive their surroundings, understand human instructions, and make intelligent decisions. However, the field faces technical challenges such as effective fusion of heterogeneous data and computational efficiency in resource-constrained environments. To address these challenges, this study proposes the lightweight LCMF cascaded attention framework, introducing a multi-level crossmodal parameter sharing mechanism into the Mamba module. By integrating the advantages of Cross-Attention and Selective parameter-sharing State Space Models (SSMs), the framework achieves efficient fusion of heterogeneous modalities and semantic complementary alignment. Experimental results show that LCMF surpasses existing multimodal baselines with an accuracy of 74.29% in VQA tasks and achieves competitive mid-tier performance within the distribution cluster of Large Language Model Agents (LLM Agents) in EQA video tasks. Its lightweight design achieves a 4.35-fold reduction in FLOPs relative to the average of comparable baselines while using only 166.51M parameters (image-text) and 219M parameters (videotext), providing an efficient solution for Human-Robot Interaction (HRI) applications in resource-constrained scenarios with strong multimodal decision generalization capabilities.

### 13. LEAF-Mamba: Local Emphatic and Adaptive Fusion State Space Model for RGB-D Salient Object Detection

**主要机构**: Dalian University of Technology Dalian, National University of Singapore Singapore
**作者数量**: 10人

**摘要**:
RGB-D salient object detection (SOD) aims to identify the most conspicuous objects in a scene with the incorporation of depth cues. Existing methods mainly rely on CNNs, limited by the local receptive fields, or Vision Transformers that suffer from the cost of quadratic complexity, posing a challenge in balancing performance and computational efficiency. Recently, state space models (SSM), Mamba, have shown great potential for modeling long-range dependency with linear complexity. However, directly applying SSM to RGB-D SOD may lead to deficient local semantics as well as the inadequate cross-modality fusion. To address these issues, we propose a Local Emphatic and Adaptive Fusion state space model (LEAF-Mamba) that contains two novel components: 1) a local emphatic state space module (LE-SSM) to capture multi-scale local dependencies for both modalities. 2) an SSM-based adaptive fusion module (AFM) for complementary cross-modality interaction and reliable cross-modality integration. Extensive experiments demonstrate that the LEAF-Mamba consistently outperforms 16 state-of-the-art RGB-D SOD methods in both efficacy and efficiency. Moreover, our method can achieve excellent performance on the RGB-T SOD task, proving a powerful generalization ability. CCS Concepts • Computing methodologies → Interest point and salient region detections.

### 14. LIGHTWEIGHT VISION TRANSFORMER WITH WINDOW AND SPATIAL ATTENTION FOR FOOD IMAGE CLASSIFICATION

**主要机构**: Jiangnan University Wuxi, School of Artificial Intelligence and Computer Science
**作者数量**: 5人

**摘要**:
With the rapid development of society and continuous advances in science and technology, the food industry increasingly demands higher production quality and efficiency. Food image classification plays a vital role in enabling automated quality control on production lines, supporting food safety supervision, and promoting intelligent agricultural production. However, this task faces challenges due to the large number of parameters and high computational complexity of Vision Transformer models. To address these issues, we propose a lightweight food image classification algorithm that integrates a Window Multi-Head Attention Mechanism (WMHAM) and a Spatial Attention Mechanism (SAM). The WMHAM reduces computational cost by capturing local and global contextual features through efficient window partitioning, while the SAM adaptively emphasizes key spatial regions to improve discriminative feature representation. Experiments conducted on the Food-101 and Vireo Food-172 datasets demonstrate that our model achieves accuracies of 95.24% and 94.33%, respectively, while significantly reducing parameters and FLOPs compared with baseline methods. These results confirm that the proposed approach achieves an effective balance between computational efficiency and classification performance, making it well-suited for deployment in resource-constrained environments.

### 15. LOCALIZED PCA-NET NEURAL OPERATORS FOR SCALABLE SOLUTION RECONSTRUCTION OF ELLIPTIC PDES A PREPRINT

**主要机构**: The Pennsylvania State University University Park, Informatics and Intelligent Systems, Department of Mechanical and Aerospace Engineering, University of Tennessee, Norwegian University of Science and Technology Trondheim, Department of Engineering Cybernetics
**作者数量**: 4人

**摘要**:
Neural operator learning has emerged as a powerful approach for solving partial differential equations (PDEs) in a data-driven manner. However, applying principal component analysis (PCA) to highdimensional solution fields incurs significant computational overhead. To address this, we propose a patch-based PCA-Net framework that decomposes the solution fields into smaller patches, applies PCA within each patch, and trains a neural operator in the reduced PCA space. We investigate two different patch-based approaches that balance computational efficiency and reconstruction accuracy: (1) local-to-global patch PCA, and (2) local-to-local patch PCA. The trade-off between computational cost and accuracy is analyzed, highlighting the advantages and limitations of each approach. Furthermore, within each approach, we explore two refinements for the most computationally efficient method: (i) introducing overlapping patches with a smoothing filter and (ii) employing a two-step process with a convolutional neural network (CNN) for refinement. Our results demonstrate that patch-based PCA significantly reduces computational complexity while maintaining high accuracy, reducing end-to-end pipeline processing time by a factor of 3.7 to 4× compared to global PCA, thefore making it a promising technique for efficient operator learning in PDE-based systems.

### 16. LYRA: GENERATIVE 3D SCENE RECONSTRUCTION VIA VIDEO DIFFUSION MODEL SELF-DISTILLATION

**主要机构**: University of Toronto
**作者数量**: 14人

**摘要**:


### 17. MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipes

**主要机构**: MiniCPM-V Team
**作者数量**: 34人

**摘要**:
Multimodal Large Language Models (MLLMs) are undergoing rapid progress and represent the frontier of AI development. However, their training and inference efficiency have emerged as a core bottleneck in making MLLMs more accessible and scalable. To address the challenges, we present MiniCPM-V 4.5, an 8B parameter model designed for high efficiency and strong performance. We introduce three core improvements in model architecture, data strategy and training method: a unified 3D-Resampler model architecture for highly compact encoding over images and videos, a unified learning paradigm for document knowledge and text recognition without heavy data engineering, and a hybrid reinforcement learning strategy for proficiency in both short and long reasoning modes. Comprehensive experimental results in OpenCompass evaluation show that MiniCPM-V 4.5 surpasses widely used proprietary models such as GPT-4o-latest, and significantly larger open-source models such as Qwen2.5-VL 72B. Notably, the strong performance is achieved with remarkable efficiency. For example, on the widely adopted VideoMME benchmark, MiniCPM-V 4.5 achieves state-of-the-art performance among models under 30B size, using just 46.7% GPU memory cost and 8.7% inference time of Qwen2.5-VL 7B.

### 18. MMCD: Multi-Modal Collaborative Decision-Making for Connected Autonomy with Knowledge Distillation

**主要机构**: 
**作者数量**: 6人

**摘要**:
Autonomous systems have advanced significantly, but challenges persist in accident-prone environments where robust decision-making is crucial. A single vehicle's limited sensor range and obstructed views increase the likelihood of accidents. Multi-vehicle connected systems and multi-modal approaches, leveraging RGB images and LiDAR point clouds, have emerged as promising solutions. However, existing methods often assume the availability of all data modalities and connected vehicles during both training and testing, which is impractical due to potential sensor failures or missing connected vehicles. To address these challenges, we introduce a novel framework MMCD (Multi-Modal Collaborative Decision-making) for connected autonomy. Our framework fuses multi-modal observations from ego and collaborative vehicles to enhance decision-making under challenging conditions. To ensure robust performance when certain data modalities are unavailable during testing, we propose an approach based on cross-modal knowledge distillation with a teacher-student model structure. The teacher model is trained with multiple data modalities, while the student model is designed to operate effectively with reduced modalities. In experiments on connected autonomous driving with ground vehicles and aerial-ground vehicles collaboration, our method improves driving safety by up to 20 .7 %, surpassing the bestexisting baseline in detecting potential accidents and making safe driving decisions. More information can be found on our website https://ruiiu.github.io/mmcd.

### 19. MOCROP: TRAINING FREE MOTION GUIDED CROPPING FOR EFFICIENT VIDEO ACTION RECOGNITION

**主要机构**: School of Computer Science, Shenzhen University, University College Dublin ‡ College of Electronics and Information Engineering, University College Dublin, School of Electrical and Electronic Engineering
**作者数量**: 6人

**摘要**:
We introduce MoCrop, a motion-aware adaptive cropping module for efficient video action recognition in the compressed domain. MoCrop uses motion vectors that are available in H.264 video to locate motion-dense regions and produces a single clip-level crop that is applied to all I-frames at inference. The module is training free, adds no parameters, and can be plugged into diverse backbones. A lightweight pipeline that includes denoising & merge (DM), Monte Carlo sampling (MCS), and adaptive cropping (AC) via a motiondensity submatrix search yields robust crops with negligible overhead. On UCF101, MoCrop improves accuracy or reduces compute. With ResNet-50, it delivers +3.5% Top-1 accuracy at equal FLOPs (attention setting), or +2.4% Top-1 accuracy with 26.5% fewer FLOPs (efficiency setting). Applied to CoViAR, it reaches 89.2% Top-1 accuracy at the original cost and 88.5% Top-1 accuracy while reducing compute from 11.6 to 8.5 GFLOPs. Consistent gains on MobileNet-V3, EfficientNet-B1, and Swin-B indicate strong generality and make MoCrop practical for real-time deployment in the compressed domain. Our code and models are available at https://github.com/microa/MoCrop.

### 20. NeuCODEX: Edge-Cloud Co-Inference with Spike-Driven Compression and Dynamic Early-Exit

**主要机构**: Centre for Sustainable Digital, Technologies Technological University, Smart Network Innovation Lab Huawei Ireland Research Center Dublin
**作者数量**: 9人

**摘要**:
Spiking Neural Networks (SNNs) offer significant potential for enabling energy-efficient intelligence at the edge. However, performing full SNN inference at the edge can be challenging due to the latency and energy constraints arising from fixed and high timestep overheads. Edge-cloud co-inference systems present a promising solution, but their deployment is often hindered by high latency and feature transmission costs. To address these issues, we introduce NeuCODEX, a neuromorphic co-inference architecture that jointly optimizes both spatial and temporal redundancy. NeuCODEX incorporates a learned spikedriven compression module to reduce data transmission and employs a dynamic early-exit mechanism to adaptively terminate inference based on output confidence. We evaluated NeuCODEX on both static images (CIFAR10 and Caltech) and neuromorphic event streams (CIFAR10-DVS and N-Caltech). To demonstrate practicality, we prototyped NeuCODEX on ResNet-18 and VGG-16 backbones in a real edge-to-cloud testbed. Our proposed system reduces data transfer by up to 2048x and edge energy consumption by over 90%, while reducing end-to-end latency by up to 3× compared to edge-only inference, all with a negligible accuracy drop of less than 2%. In doing so, NeuCODEX enables practical, high-performance SNN deployment in resourceconstrained environments.

### 21. READING IMAGES LIKE TEXTS: SEQUENTIAL IMAGE UNDERSTANDING IN VISION-LANGUAGE MODELS

**主要机构**: Beijing University of Posts and Telecommunications
**作者数量**: 5人

**摘要**:
Vision-Language Models (VLMs) have demonstrated remarkable performance across a variety of real-world tasks. However, existing VLMs typically process visual information by serializing images, a method that diverges significantly from the parallel nature of human vision. Moreover, their opaque internal mechanisms hinder both deeper understanding and architectural innovation. Inspired by the dual-stream hypothesis of human vision, which distinguishes the "what" and "where" pathways, we deconstruct the visual processing in VLMs into object recognition and spatial perception for separate study. For object recognition, we convert images into text token maps and find that the model's perception of image content unfolds as a two-stage process from shallow to deep layers, beginning with attribute recognition and culminating in semantic disambiguation. For spatial perception, we theoretically derive and empirically verify the geometric structure underlying the positional representation in VLMs. Based on these findings, we introduce an instruction-agnostic token compression algorithm based on a plug-and-play visual decoder to improve decoding efficiency, and a RoPE scaling technique to enhance spatial reasoning. Through rigorous experiments, our work validates these analyses, offering a deeper understanding of VLM internals and providing clear principles for designing more capable future architectures. Code is available at https://github.com/Siriuslala/vlm_interp.

### 22. Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding

**主要机构**: National Yang Ming Chiao Tung University, Cornell University
**作者数量**: 7人

**摘要**:
The immense model sizes of large language models (LLMs) challenge deployment on memory-limited consumer GPUs. Although model compression and parameter offloading are common strategies to address memory limitations, compression can degrade quality, and offloading maintains quality but suffers from slow inference. Speculative decoding presents a promising avenue to accelerate parameter offloading, utilizing a fast draft model to propose multiple draft tokens, which are then verified by the target LLM in parallel with a single forward pass. This method reduces the time-consuming data transfers in forward passes that involve offloaded weight transfers. Existing methods often rely on pretrained weights of the same family, but require additional training to align with custom-trained models. Moreover, approaches that involve draft model training usually yield only modest speedups. This limitation arises from insufficient alignment with the target model, preventing higher token acceptance lengths. To address these challenges and achieve greater speedups, we propose SUBSPEC, a plug-and-play method to accelerate parameter offloading that is lossless and training-free. SubSpec constructs a highly aligned draft model by generating low-bit quantized substitute layers from offloaded target LLM portions. Additionally, our method shares the remaining GPU-resident layers and the KV-Cache, further reducing memory overhead and enhance alignment. SubSpec achieves a high average acceptance length, delivering 9.1× speedup for Qwen2.5 7B on MT-Bench (8GB VRAM limit) and an average of 12.5× speedup for Qwen2.5 32B on popular generation benchmarks (24GB VRAM limit). The code is available at https://github.com/NYCU-EDgeAi/subspec.

### 23. Symphony-MoE: Harmonizing Disparate Pre-trained Models into a Coherent Mixture-of-Experts

**主要机构**: Institute of Computing Technology, Peng Cheng Laboratory, State Key Laboratory of AI Safety, University of Chinese Academy of Sciences, Chinese Academy of Sciences
**作者数量**: 3人

**摘要**:
Mixture-of-Experts (MoE) models enable scalable performance by activating large parameter sets sparsely, minimizing computational overhead. To circumvent the prohibitive cost of training MoEs from scratch, recent work employs upcycling, reusing a single pre-trained dense model by replicating its feed-forward network (FFN) layers into experts. However, this limits expert diversity, as all experts originate from a single pre-trained dense model. This paper addresses this limitation by constructing powerful MoE models using experts sourced from multiple identically-architected but disparate pre-trained models (e.g., Llama2-Chat and Code Llama). A key challenge lies in the fact that these source models occupy disparate, dissonant regions of the parameter space, making direct upcycling prone to severe performance degradation. To overcome this, we propose Symphony-MoE, a novel two-stage framework designed to harmonize these models into a single, coherent expert mixture. First, we establish this harmony in a training-free manner: we construct a shared backbone via a layer-aware fusion strategy and, crucially, alleviate parameter misalignment among experts using activation-based functional alignment. Subsequently, a single lightweight stage of router training coordinates the entire architecture. Experiments demonstrate that our method successfully integrates experts from heterogeneous sources, achieving an MoE model that significantly surpasses baselines in multi-domain tasks and out-of-distribution generalization.

### 24. TEACHING AUDIO MODELS TO REASON: A UNIFIED FRAMEWORK FOR SOURCE-AND LAYER-WISE DISTILLATION

**主要机构**: China Mobile, Jiutian Artificial Intelligence Research Institute
**作者数量**: 6人

**摘要**:
While large audio language models excel at tasks like ASR and emotion recognition, they still struggle with complex reasoning due to the modality gap between audio and text as well as the lack of structured intermediate supervision. To address this, we propose a unified knowledge distillation framework to transfer reasoning capabilities from a high-capacity textual teacher model to a student audio models while preserving its acoustic competence. Our method introduces two key dimensions: source-wise distillation, which leverages both textual and acoustic teachers to provide complementary modalityspecific supervision; and layer-wise distillation, which aligns teacher signals with appropriate student layers to improve transfer efficiency. This dual-dimensional strategy enables fine-grained control over the distillation process, effectively bridging the gap between symbolic reasoning and speech representations. Experimental results show significant improvements in audio reasoning performance, demonstrating the effectiveness of our framework as a reasoning transfer solution for audio modeling.

### 25. TinyBEV: Cross-Modal Knowledge Distillation for Efficient Multi-Task Bird's-Eye-View Perception and Planning

**主要机构**: University of Arkansas Fayetteville
**作者数量**: 2人

**摘要**:
We present TinyBEV, a unified, camera-only Bird's-Eye-View (BEV) framework that distills the full-stack capabilities of a large planning-oriented teacher (UniAD [19]) into a compact, real-time student model. Unlike prior efficient camera-only baselines such as VAD[23] and VADv2[7], TinyBEV supports the complete autonomy stack-3D detection, HD-map segmentation, motion forecasting, occupancy prediction, and goal-directed planning-within a streamlined 28M-parameter backbone, achieving a 78% reduction in parameters over UniAD [19]. Our modelagnostic, multi-stage distillation strategy combines featurelevel, output-level, and adaptive region-aware supervision to effectively transfer high-capacity multi-modal knowledge to a lightweight BEV representation. On nuScenes[4], Tiny-BEV achieves 39.0 mAP for detection, 1.08 minADE for motion forecasting, and a 0.32 collision rate, while running 5× faster (11 FPS) and requiring only camera input. These results demonstrate that full-stack driving intelligence can be retained in resource-constrained settings, bridging the gap between large-scale, multi-modal perception-planning models and deployment-ready real-time autonomy.

### 26. TinyEcoWeedNet: Edge Efficient Real-Time Aerial Agricultural Weed Detection

**主要机构**: College of Computer and Information Sciences, Research Chair of Pervasive and Mobile Computing, Department of Software Engineering, Department of Computer Engineering, SDAIA-KFUPM Joint Research Center on Artificial Intelligence, Center for Intelligent Secure Systems, Computer Engineering Department, King Saud University, King Fahd University of Petroleum & Minerals, King Fahd University of Petroleum and Minerals (KFUPM)
**作者数量**: 8人

**摘要**:
Deploying efficient deep models for real-world applications in agriculture is challenging because the computations and available resources are limited on edge devices. This work proposes a strategy for deploying a compressed version of EcoWeedNet. This efficient deep learning model employs structured channel pruning and quantization-aware training (QAT), while accelerating computations using NVIDIA's Ten-sorRT framework on the Jetson Orin Nano. The proposed approach's critical modules are pruned while preserving their consistency in the pruned architecture, compared with the pruned state-of-the-art YOLO models (YOLO11n and YOLO12n), with performance tested over two benchmarking datasets: an aerial soybean dataset and the benchmarking dataset of Cotton-WeedDet12. The inherent complexity of structured pruning for EcoWeedNet and YOLO models arises due to their sophisticated architectural features, that is, residual shortcuts, complex attention mechanisms, concatenations, and CSP blocks, which make it highly challenging to maintain model consistency with structured pruning. Despite all these, structured pruning effectively reduces model parameters by up to 68.5%, and decreases computational load up to 3.2 GFLOPs. Accelerated inference speed up to 184 FPS with an 11.6% pruning ratio at lower precision (FP16), improving over 28.7% compared to Baseline EcoWeedNet at FP16. Our pruned EcoWeedNet model with a pruning ratio of 39.5% even outperformed YOLO11n and YOLO12n models with significantly lesser pruning ratios of about 20%, with the precision of 83.7%, a recall of 77.5%, and a mAP50 of 85.9% on the CottonWeedDet12 dataset. These observations validate that the compressed EcoWeedNet is optimal and efficient for precision agriculture.

### 27. Towards General Computer Control with Hierarchical Agents and Multi-Level Action Spaces

**主要机构**: The University of Tokyo, Information Science and Technology, Advanced AI Technology Center Lenovo US Morrisville, Workstation Solution Lenovo US Morrisville
**作者数量**: 7人

**摘要**:
Controlling desktop applications via software remains a fundamental yet under-served problem: existing multi-modal large language models (MLLMs) ingest screenshots and task instructions to generate keystrokes and mouse events, but suffer from prohibitive inference latency, poor sample efficiency on long-horizon sparse-reward tasks, and infeasible on-device deployment. We introduced a lightweight hierarchical reinforcement learning framework, ComputerAgent, that formulates OS control as a two-level option process (manager/subpolicy), employs a triple-modal state encoder (screenshot, task ID, numeric state) to handle visual and contextual diversity, integrates meta-actions with an early-stop mechanism to curb wasted interactions, and uses a compact vision backbone plus small policy networks for on-device inference (0.015 B parameters). On a suite of 135 real-world desktop tasks, ComputerAgent attains 92.1% success on simple tasks (<8 steps) and 58.8% on hard tasks (≥ 8 steps), matching or exceeding 200 B+ MLLM baselines on simple scenarios while reducing model size by over four orders of magnitude and halving inference time. Our results demonstrate that hierarchical RL offers a practical, scalable alternative to monolithic MLLM-based automation for computer control.

### 28. VGGT-DP: Generalizable Robot Control via Vision Foundation Models

**主要机构**: Harbin Institute of Technology, CASBOT, Tsinghua University, Shenzhen International Graduate School
**作者数量**: 6人

**摘要**:
Visual imitation learning frameworks allow robots to learn manipulation skills from expert demonstrations. While existing approaches mainly focus on policy design, they often neglect the structure and capacity of visual encoders-limiting spatial understanding and generalization. Inspired by biological vision systems, which rely on both visual and proprioceptive cues for robust control, we propose VGGT-DP, a visuomotor policy framework that integrates geometric priors from a pretrained 3D perception model with proprioceptive feedback. We adopt the Visual Geometry Grounded Transformer (VGGT) as the visual encoder and introduce a proprioception-guided visual learning strategy to align perception with internal robot states, improving spatial grounding and closed-loop control. To reduce inference latency, we design a frame-wise token reuse mechanism that compacts multi-view tokens into an efficient spatial representation. We further apply random token pruning to enhance policy robustness and reduce overfitting. Experiments on challenging MetaWorld tasks show that VGGT-DP significantly outperforms strong baselines such as DP and DP3, particularly in precision-critical and long-horizon scenarios.

### 29. Weight Mapping Properties of a Dual Tree Single Clock Adiabatic Capacitive Neuron

**主要机构**: 
**作者数量**: 3人

**摘要**:
Dual Tree Single Clock (DTSC) Adiabatic Capacitive Neuron (ACN) circuits offer the potential for highly energyefficient Artificial Neural Network (ANN) computation in full custom analog IC designs. The efficient mapping of Artificial Neuron (AN) abstract weights, extracted from the software-trained ANNs, onto physical ACN capacitance values has, however, yet to be fully researched. In this paper, we explore the unexpected hidden complexities, challenges and properties of the mapping, as well as, the ramifications for IC designers in terms accuracy, design and implementation. We propose an optimal, AN to ACN methodology, that promotes smaller chip sizes and improved overall classification accuracy, necessary for successful practical deployment. Using TensorFlow and Larq software frameworks, we train three different ANN networks and map their weights into the energy-efficient DTSC ACN capacitance value domain to demonstrate 100% functional equivalency. Finally, we delve into the impact of weight quantization on ACN performance using novel metrics related to practical IC considerations, such as IC floor space and comparator decision-making efficacy.
