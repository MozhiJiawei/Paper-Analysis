# AI推理加速技术论文分析报告
生成时间: 2025-10-15 20:30:36
分析论文数量: 25篇

## 论文技术简报

### 1. Adaptive and Resource-efficient Agentic AI Systems for Mobile and Embedded Devices: A Survey

相关研究机构发布了《Adaptive and Resource-efficient Agentic AI Systems for Mobile and Embedded Devices: A Survey》论文，使用提出涵盖弹性基础模型推理、测试时适应等的新分类法，解决了现有综述忽略移动设备资源限制、运行时适应性等特有挑战的问题，达成了填补空白并启发未来研究的效果。

### 2. Adaptive Event Stream Slicing for Open-Vocabulary Event-Based Object Detection via Vision-Language Knowledge Distillation

发布了《Adaptive Event Stream Slicing for Open-Vocabulary Event-Based Object Detection via Vision-Language Knowledge Distillation》论文，使用事件-图像知识蒸馏及混合SNN-CNN框架的自适应事件流切片技术，解决了事件相机因缺乏纹理颜色导致开放词汇目标检测困难及图像与事件流模态差距的问题，达成了开放词汇事件基目标检测并继承CLIP视觉知识、提取关键时间特征的效果。

### 3. A Deep Learning Pipeline for Epilepsy Genomic Analysis Using GPT-2 XL and NVIDIA H100

发布了癫痫基因组分析的深度学习 pipeline论文，使用整合GPT-2 XL与NVIDIA H100的深度学习 pipeline，解决了癫痫复杂转录组数据解读难题，揭示了显著转录组修饰（如ketogenic diet治疗后海马星形胶质细胞增生减少、斑马鱼癫痫模型兴奋-抑制信号平衡恢复）并证明LLM结合硬件加速在神经疾病转录组表征中的有效性。

### 4. Collaborative-Distilled Diffusion Models (CDDM) for Accelerated and Lightweight Trajectory Prediction

发布了Collaborative-Distilled Diffusion Models (CDDM)论文，使用基于协作渐进蒸馏(CPD)和双信号正则化蒸馏损失的CDDM技术，解决了扩散模型在轨迹预测中模型尺寸大、采样速度慢的部署难题，达成了保留96.2% ADE和95.5% FDE精度、161×压缩、31×加速及9ms延迟的效果

### 5. Efficient Multi-modal Large Language Models via Progressive Consistency Distillation

北京大学、上海交通大学发布了Efficient Multi-modal Large Language Models via Progressive Consistency Distillation论文，使用EPIC渐进式一致性蒸馏框架（通过令牌和层一致性蒸馏分解特征空间扰动），解决了多模态大模型中视觉令牌压缩导致的学习难度增加及效率低的问题，达成了优越的有效性、鲁棒性和泛化能力。

### 6. Enhancing Certifiable Semantic Robustness via Robust Pruning of Deep Neural Networks

研究团队发布了Enhancing Certifiable Semantic Robustness via Robust Pruning of Deep Neural Networks论文，使用基于Unbiased and Smooth Neuron (USN)指标的剪枝方法及Wasserstein距离损失，解决了深度神经网络过参数化导致的语义鲁棒性验证紧密度和可扩展性不足问题，达成了在鲁棒关键点检测任务上较基线更优的鲁棒性认证性能和效率。

### 7. Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution

NVIDIA发布了Expected Attention论文，使用通过预测未来查询的注意力分布来估计KV对重要性的无训练压缩技术，解决了LLM推理中KV cache内存消耗瓶颈及现有注意力分数剪枝方法的局限，达成了在预填充和解码阶段均优于最先进基线、实现有效压缩且无性能下降的效果，并发布KVPress库。

### 8. FIN: Fast Inference Network for Map Segmentation

利默里克大学发布了FIN: Fast Inference Network for Map Segmentation论文，使用摄像头和雷达在BEV空间结合先进损失集与轻量级头部的新型高效地图分割架构，解决了地图分割在高精度和实时性能方面的挑战，达成53.5 mIoU且推理时间较最强基线提升260%的效果。

### 9. Free Draft-and-Verification: Toward Lossless Parallel Decoding for Diffusion Large Language Models

威斯康星大学麦迪逊分校发布了Free Draft-and-Verification论文，使用FreeDave无损并行解码算法，解决了扩散大型语言模型（DLLMs）并行解码性能下降与推理效率不足的问题，达成了数学推理任务吞吐量提升2.8倍且无性能损失的效果。

### 10. Gather-Scatter Mamba: Accelerating Propagation with Efficient State Space Model

Yonsei University与Sungkyunkwan University发布了Gather-Scatter Mamba论文，使用Gather-Scatter Mamba (GSM)机制（结合shifted window self-attention进行空间上下文聚合与Mamba-based selective scanning实现高效时序传播），解决了传统视频超分辨率方法中RNN的消失梯度、缺乏并行性、推理慢及Mamba单独使用时细粒度空间依赖捕捉不足的问题，达成了减少遮挡伪影并有效重新分配聚合信息的效果

### 11. GUI-KV: EFFICIENT GUI AGENTS VIA KV CACHE WITH SPATIO-TEMPORAL AWARENESS

加州大学与Salesforce AI Research发布了GUI-KV论文，使用结合空间显著性引导和时间冗余评分的KV缓存压缩技术（含均匀预算分配策略），解决了GUI代理处理长序列高分辨率截图和长任务时的推理效率低、成本高问题，在AgentNetBench 5截图设置中减少解码FLOPs 38.9%并提高步准确率4.1%

### 12. INFVSR: BREAKING LENGTH LIMITS OF GENERIC VIDEO SUPER-RESOLUTION

美团与上海交通大学发布了InfVSR论文，使用自回归单步扩散范式（含因果结构DiT调整与单步扩散蒸馏）技术，解决了长视频超分辨率的效率低和可扩展性差（长度限制）问题，达成了长视频超分辨率的高效可扩展处理，实现SOTA质量与语义一致性，速度提升达58倍。

### 13. Instant4D: 4D Gaussian Splatting in Minutes

Carnegie Mellon University发布了Instant4D论文，使用4D Gaussian Splatting技术，解决了传统4D重建耗时的问题，达成了几分钟内完成4D重建的效果。

### 14. LongCodeZip: Compress Long Context for Code Language Models

斯坦福大学与上海交通大学发布了LongCodeZip论文，使用双阶段代码上下文压缩策略（粗粒度函数级筛选与细粒度块级选择，结合条件困惑度与自适应token预算），解决了现有技术忽略代码结构依赖导致的编程任务性能不佳及长上下文高成本与延迟问题，达成了5.6倍压缩比且不降低任务性能的效果。

### 15. MILCO: LEARNED SPARSE RETRIEVAL ACROSS LANGUAGES VIA A MULTILINGUAL CONNECTOR

约翰斯·霍普金斯大学和阿姆斯特丹大学发布了MILCO论文，使用多语言连接器、两阶段训练机制及LexEcho头的MILCO架构，解决了现有Learned Sparse Retrieval难以扩展到多语言/跨语言且不常见实体丢失的问题，达成多语言跨语言LSR SOTA，超过BGE-M3等基线并支持动态效率（如30维文档表示优于1024维同规模模型）。

### 16. NSARM: Next-Scale Autoregressive Modeling for Robust Real-World Image Super-Resolution

香港理工大学发布了NSARM论文，使用基于next-scale预测的两阶段训练（先转换网络映射低质图到初步尺度，再端到端全模型微调）的Next-Scale Autoregressive Modeling技术，解决了现有Real-ISR方法在效率与质量间的权衡及对不同退化输入鲁棒性差的问题，达成了视觉效果优于现有方法、推理速度快且对输入图像质量鲁棒性更高的效果。

### 17. PRISM-Consult: A Panel-of-Experts Architecture for Clinician-Aligned Diagnosis

加州大学伯克利分校发布了PRISM-Consult论文，使用临床医生对齐的专家小组架构（轻量级路由器分派临床事件至领域专家模型并继承PRISM小Transformer骨干实现参数高效与可解释），解决了临床诊断中专家诊断的需求，达成了高路由质量、大幅计算节省及跨领域低困惑度收敛的效果。

### 18. PrunedLoRA: Robust Gradient-Based structured pruning for Low-rank Adaptation in Fine-tuning

宾夕法尼亚州立大学发布了PrunedLoRA论文，使用基于梯度的结构化剪枝技术（动态剪枝不重要组件并防止重新激活，实现灵活自适应秩分配），解决了LoRA在参数高效微调中表征能力落后于全微调的问题，达成了在数学推理、代码生成、自然语言理解等监督微调任务上持续优于LoRA及其变体和现有结构化剪枝方法的效果。

### 19. RETHINKING ROPE SCALING IN QUANTIZED LLM: THEORY, OUTLIER, AND CHANNEL-BAND ANALYSIS WITH WEIGHT RESCALING

加州大学发布了关于量化LLM中RoPE缩放的论文，使用Q-ROAR（通过分组RoPE维度为频率带并对Key和Query权重进行每频段缩放搜索的权重仅插值感知稳定化）技术，解决了RoPE位置插值与后训练量化结合导致的精度下降问题，达成了长上下文工作负载困惑度降低超14%且保持短上下文性能、推理吞吐量及系统兼容性的效果。

### 20. SAGE-MUSIC: LOW-LATENCY SYMBOLIC MUSIC GENERATION VIA ATTRIBUTE-SPECIALIZED KEY-VALUE HEAD SHARING

斯坦福大学发布了SAGE-MUSIC论文，使用Attribute-Specialized Key-Value Head Sharing (AS-KVHS)技术，解决了transformer模型在低延迟符号音乐生成中推理速度与音乐质量的权衡问题，达成了约30%推理加速且音乐质量几乎无损失（仅≈0.4%客观质量下降，主观测试略有提升）的效果。

### 21. Semantic-Driven AI Agent Communications: Challenges and Solutions

Peng Cheng Laboratory and State Key Laboratory of Networking and Switching Technology发布了Semantic-Driven AI Agent Communications: Challenges and Solutions论文，使用语义驱动AI智能体通信框架及语义自适应传输、语义轻量化传输、语义自演化控制三项使能技术，解决了动态环境和资源受限导致语义通信实际部署受限的问题，达成了更快收敛性、更强鲁棒性且分布式分层优化方法显著优于传统决策方案的效果。

### 22. Spiralformer: Low Latency Encoder for Streaming Speech Recognition with Circular Layer Skipping and Early Exiting

卡内基梅隆大学与索尼集团发布了Spiralformer论文，使用循环层跳过与提前退出的螺旋层计算技术，解决了块处理流式语音识别的编码延迟问题，在Librispeech上平均令牌发射延迟减少21.6%、CSJ上减少7.0%，且计算成本和词错误率与基线相近。

### 23. THE PITFALLS OF KV CACHE COMPRESSION

加州大学发布了THE PITFALLS OF KV CACHE COMPRESSION论文，使用改进的KV缓存驱逐策略，解决了KV缓存压缩在多指令提示等现实场景中的隐患（如指令退化、系统提示泄露），达成了减少相关因素影响并提升多指令任务整体性能的效果。

### 24. THOUGHTBUBBLES: AN UNSUPERVISED METHOD FOR PARALLEL THINKING IN LATENT SPACE

斯坦福大学发布了THOUGHTBUBBLES论文，使用通过学习在潜在空间中分叉或删除残差流来原生执行并行自适应计算的transformer变体技术，解决了现有transformers推理时依赖显式思维链tokens、无法应用于预训练且仅能串行生成自然语言以扩展计算的问题，达成了在预训练后于OpenWebText、peS2o困惑度及HellaSwag、LAMBADA等零样本评估中性能超过标准解码器语言模型及非自适应并行计算方法的效果。

### 25. ToolBrain: A Flexible Reinforcement Learning Framework for Agentic Tools

IBM Research Lab发布了ToolBrain论文，使用灵活的强化学习框架（支持RL算法、自定义奖励/LLM-as-judge奖励生成、知识蒸馏等），解决智能体工具使用训练中手动设计奖励、数据有限、多工具选择差导致的适应慢、资源浪费、性能不佳问题，达成工具使用技能提升高达30.0%的效果。

## 论文详细信息

### 1. Adaptive and Resource-efficient Agentic AI Systems for Mobile and Embedded Devices: A Survey

**主要机构**: 
**作者数量**: 5人

**摘要**:
Large foundation models (FMs) such as LLMs, VLMs, diffusion models, and MLLMs have shifted AI from fragmented, task-specific models toward versatile cognitive systems. In parallel, the AI agents has been refreshed by FMs as their cognitive core, enabling autonomy, perception, planning, and selfreflection in dynamic environments. Together, these shifts open opportunities for agentic AI on mobile and edge platforms, where real-world applications demand low-latency, energy-efficient, and adaptive intelligence. However, current surveys mainly focus on static FM optimization or generic agents, overlooking mobilespecific challenges of resource constraints, runtime adaptability, and diverse conditions. This article fills that gap by providing the first systematic survey on adaptive and resource-efficient agentic AI systems on mobile/edge devices. We propose a novel taxonomy covering elastic FM inference, test-time adaptation, dynamic multimodal integration, and application-driven optimization, and we outline open issues and evaluation methodologies to inspire future research at the intersection of FMs, agents, and mobile/edge intelligence. We believe this survey can help readers to understand the connections between enabling technologies while promoting further discussions.

### 2. Adaptive Event Stream Slicing for Open-Vocabulary Event-Based Object Detection via Vision-Language Knowledge Distillation

**主要机构**: 
**作者数量**: 4人

**摘要**:
Event camera offers advantages in object detection tasks for its properties such as high-speed response, low latency, and robustness to motion blur. However, event cameras inherently lack texture and color information, making openvocabulary detection particularly challenging. Current eventbased detection methods are typically trained on predefined target categories, limiting their ability to generalize to novel objects, where encountering previously unseen objects is common. Vision-language models (VLMs) have enabled open-vocabulary object detection in RGB images. However, the modality gap between images and event streams makes it ineffective to directly transfer CLIP to event data, as CLIP was not designed for event streams. To bridge this gap, we propose an event-image knowledge distillation framework, leveraging CLIP's semantic understanding to achieve open-vocabulary object detection on event data. Instead of training CLIP directly on event streams, we use image frames as teacher model inputs, guiding the eventbased student model to learn CLIP's rich visual representations. Through spatial attention-based distillation, the student network learns meaningful visual features directly from raw event inputs, while inheriting CLIP's broad visual knowledge. Furthermore, to prevent information loss due to event data segmentation, we design a hybrid Spiking Neural Network (SNN) and Convolutional Neural Network (CNN) framework. Unlike fixed-group event segmentation methods, which often discard crucial temporal information, our SNN adaptively determines the optimal event segmentation moments, ensuring that key temporal features are extracted. The extracted event features are then processed by CNNs for object detection.

### 3. A Deep Learning Pipeline for Epilepsy Genomic Analysis Using GPT-2 XL and NVIDIA H100

**主要机构**: 
**作者数量**: 4人

**摘要**:
Epilepsy is a chronic neurological condition characterized by recurrent seizures, with global prevalence estimated at 50 million people worldwide. While progress in high-throughput sequencing has allowed for broad-based transcriptomic profiling of brain tissues, the deciphering of these highly complex datasets remains one of the challenges. To address this issue, in this paper we propose a new analysis pipeline that integrates the power of deep learning strategies with GPU-acceleration computation for investigating Gene expression patterns in epilepsy. Specifically, our proposed approach employs GPT-2 XL, a transformerbased Large Language Model (LLM) with 1.5 billion parameters for genomic sequence analysis over the latest NVIDIA H100 Tensor Core GPUs based on Hopper architecture. Our proposed method enables efficient preprocessing of RNA sequence data, gene sequence encoding, and subsequent pattern identification. We conducted experiments on two epilepsy datasets including GEO accession GSE264537 and GSE275235. The obtained results reveal several significant transcriptomic modifications, including reduced hippocampal astrogliosis after ketogenic diet treatment as well as restored excitatory-inhibitory signaling equilibrium in zebrafish epilepsy model. Moreover, our results highlight the effectiveness of leveraging LLMs in combination with advanced hardware acceleration for transcriptomic characterization in neurological diseases.

### 4. Collaborative-Distilled Diffusion Models (CDDM) for Accelerated and Lightweight Trajectory Prediction

**主要机构**: 
**作者数量**: 3人

**摘要**:
Trajectory prediction is a fundamental task in Autonomous Vehicles (AVs) and Intelligent Transportation Systems (ITS), supporting efficient motion planning and real-time traffic safety management. Diffusion models have recently demonstrated strong performance in probabilistic trajectory prediction, but their large model size and slow sampling process hinder real-world deployment. This paper proposes Collaborative-Distilled Diffusion Models (CDDM), a novel method for real-time and lightweight trajectory prediction. Built upon Collaborative Progressive Distillation (CPD), CDDM progressively transfers knowledge from a high-capacity teacher diffusion model to a lightweight student model, jointly reducing both the number of sampling steps and the model size across distillation iterations. A dual-signal regularized distillation loss is further introduced to incorporate guidance from both the teacher and groundtruth data, mitigating potential overfitting and ensuring robust performance. Extensive experiments on the ETH-UCY pedestrian benchmark and the nuScenes vehicle benchmark demonstrate that CDDM achieves state-of-the-art prediction accuracy. The well-distilled CDDM retains 96.2% and 95.5% of the baseline model's ADE and FDE performance on pedestrian trajectories, while requiring only 231K parameters and 4 or 2 sampling steps, corresponding to 161× compression, 31× acceleration, and 9 ms latency. Qualitative results further show that CDDM generates diverse and accurate trajectories under dynamic agent behaviors and complex social interactions. By bridging high-performing generative models with practical deployment constraints, CDDM enables resource-efficient probabilistic prediction for AVs and ITS. Code is available at https://github.com/bingzhangw/CDDM.

### 5. Efficient Multi-modal Large Language Models via Progressive Consistency Distillation

**主要机构**: Peking University, Shanghai Jiao Tong University, Shanghai AI Laboratory, Duke University, EPIC Lab, University of Chicago, The University of Hong
**作者数量**: 11人

**摘要**:
Visual tokens consume substantial computational resources in multi-modal large models (MLLMs), significantly compromising their efficiency. Recent works have attempted to improve efficiency by compressing visual tokens during training, either through modifications to model components or by introducing additional parameters. However, they often overlook the increased learning difficulty caused by such compression, as the model's parameter space struggles to quickly adapt to the substantial perturbations in the feature space induced by token compression. In this work, we propose to develop Efficient MLLMs via ProgressIve Consistency Distillation (EPIC), a progressive learning framework. Specifically, by decomposing the feature space perturbations introduced by token compression along the token-wise and layer-wise dimensions, we introduce token consistency distillation and layer consistency distillation, respectively, aiming to reduce the training difficulty by leveraging guidance from a teacher model and following a progressive learning trajectory. Extensive experiments demonstrate the superior effectiveness, robustness, and generalization capabilities of our proposed framework.

### 6. Enhancing Certifiable Semantic Robustness via Robust Pruning of Deep Neural Networks

**主要机构**: 
**作者数量**: 7人

**摘要**:
Deep neural networks have been widely adopted in many vision and robotics applications with visual inputs. It is essential to verify its robustness against semantic transformation perturbations, such as brightness and contrast. However, current certified training and robustness certification methods face the challenge of over-parameterization, which hinders the tightness and scalability due to the over-complicated neural networks. To this end, we first analyze stability and variance of layers and neurons against input perturbation, showing that certifiable robustness can be indicated by a fundamental Unbiased and Smooth Neuron metric (USN). Based on USN, we introduce a novel neural network pruning method that removes neurons with low USN and retains those with high USN, thereby preserving model expressiveness without over-parameterization. To further enhance this pruning process, we propose a new Wasserstein distance loss to ensure that pruned neurons are more concentrated across layers. We validate our approach through extensive experiments on the challenging robust keypoint detection task, which involves realistic brightness and contrast perturbations, demonstrating that our method achieves superior robustness certification performance and efficiency compared to baselines.

### 7. Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution

**主要机构**: NVIDIA NVIDIA, Sapienza University of Rome
**作者数量**: 4人

**摘要**:
Memory consumption of the Key-Value (KV) cache represents a major bottleneck for efficient large language model (LLM) inference. While attention-score-based KV cache pruning shows promise, it faces critical practical limitations: attention scores from future tokens are unavailable during compression, and modern implementations like Flash Attention do not materialize the full attention matrix, making past scores inaccessible. To overcome these challenges, we introduce Expected Attention, a training-free compression method that estimates KV pairs importance by predicting how future queries will attend to them. Our approach leverages the distributional properties of LLM activations to compute expected attention scores in closed form for each KV pair. These scores enable principled ranking and pruning of KV pairs with minimal impact on the residual stream, achieving effective compression without performance degradation. Importantly, our method operates seamlessly across both prefilling and decoding phases, consistently outperforming state-of-the-art baselines in both scenarios. Finally, we release KVPress, a comprehensive library to enable researchers to implement and benchmark KV cache compression methods, already including more than 20 techniques.

### 8. FIN: Fast Inference Network for Map Segmentation

**主要机构**: Department of Electronic and Computer Engineering, University of Limerick
**作者数量**: 4人

**摘要**:
Multi-sensor fusion in autonomous vehicles is becoming more common to offer a more robust alternative for several perception tasks. This need arises from the unique contribution of each sensor in collecting data: camera-radar fusion offers a cost-effective solution by combining rich semantic information from cameras with accurate distance measurements from radar, without incurring excessive financial costs or overwhelming data processing requirements. Map segmentation is a critical task for enabling effective vehicle behaviour in its environment, yet it continues to face significant challenges in achieving high accuracy and meeting real-time performance requirements. Therefore, this work presents a novel and efficient map segmentation architecture, using cameras and radars, in the Bird's-Eye View (BEV) space. Our model introduces a real-time map segmentation architecture considering aspects such as high accuracy, per-class balancing, and inference time. To accomplish this, we use an advanced loss set together with a new lightweight head to improve the perception results. Our results show that, with these modifications, our approach achieves results comparable to large models, reaching 53.5 mIoU, while also setting a new benchmark for inference time, improving it by 260% over the strongest baseline models.

### 9. Free Draft-and-Verification: Toward Lossless Parallel Decoding for Diffusion Large Language Models

**主要机构**: Department of Computer Sciences, University of Wisconsin -Madison
**作者数量**: 2人

**摘要**:
Diffusion Large Language Models (DLLMs) have emerged as a new paradigm of language modeling beyond autoregressive next-token prediction. Thanks to their bidirectional attention mechanism, DLLMs are more capable of capturing the connection of context, and thus show unique advantages in challenges like the famous "reversal curse" or learning under data-constrained scenarios. However, this bidirectional nature also brings an obstacle that DLLMs are not inherently compatible with KV Cache, and consequently, the inference efficiency is not competitive compared with autoregressive models. Taking advantage of their inherent capability of multi-token prediction, existing parallel decoding algorithms can speed up the DLLM inference, but at the cost of non-negligible performance degradation. To overcome this challenge, we introduce Free Draft-and-Verification (FreeDave), a novel fast sampling algorithm tailored for DLLMs that achieves lossless parallel decoding. Specifically, we propose a pipeline of parallel-decoded candidate generation and verification, which is guaranteed to reproduce the same sequence generated by static sampling, without introducing extra model forward calls. By applying FreeDave, the throughput of DLLMs can be boosted up to 2.8× without performance degradation on math reasoning tasks.

### 10. Gather-Scatter Mamba: Accelerating Propagation with Efficient State Space Model

**主要机构**: Yonsei University, Department of Electrical and Computer Engineering, Sungkyunkwan University, Department of Artificial Intelligence, Hanwha Systems
**作者数量**: 8人

**摘要**:
State Space Models (SSMs)-most notably RNNs-have historically played a central role in sequential modeling. Although attention mechanisms such as Transformers have since dominated due to their ability to model global context, their quadratic complexity and limited scalability make them less suited for long sequences. Video super-resolution (VSR) methods have traditionally relied on recurrent architectures to propagate features across frames. However, such approaches suffer from well-known issues including vanishing gradients, lack of parallelism, and slow inference speed. Recent advances in selective SSMs like Mamba [11] offer a compelling alternative: by enabling input-dependent state transitions with linear-time complexity, Mamba mitigates these issues while maintaining strong long-range modeling capabilities. Despite this potential, Mamba alone struggles to capture fine-grained spatial dependencies due to its causal nature and lack of explicit context aggregation. To address this, we propose a hybrid architecture that combines shifted window self-attention for spatial context aggregation with Mamba-based selective scanning for efficient temporal propagation. Furthermore, we introduce Gather-Scatter Mamba (GSM), an alignment-aware mechanism that warps features toward a center anchor frame within the temporal window before Mamba propagation and scatters them back afterward, effectively reducing occlusion artifacts and ensuring effective redistribution of aggregated information across all frames. The official implementation is provided at: https:// github.com/Ko-Lani/GSMamba.

### 11. GUI-KV: EFFICIENT GUI AGENTS VIA KV CACHE WITH SPATIO-TEMPORAL AWARENESS

**主要机构**: University of California, Salesforce AI Research
**作者数量**: 5人

**摘要**:
Graphical user interface (GUI) agents built on vision-language models have emerged as a promising approach to automate human-computer workflows. However, they also face the inefficiency challenge as they process long sequences of high-resolution screenshots and solving long-horizon tasks, making inference slow, costly and memory-bound. While key-value (KV) caching can mitigate this, storing the full cache is prohibitive for image-heavy contexts. Existing cache-compression methods are sub-optimal as they do not account for the spatial and temporal redundancy of GUIs. In this work, we first analyze attention patterns in GUI agent workloads and find that, unlike in natural images, attention sparsity is uniformly high across all transformer layers.This insight motivates a simple uniform budget allocation strategy, which we show empirically outperforms more complex layer-varying schemes. Building on this, we introduce GUI-KV, a plug-and-play KV cache compression method for GUI agents that requires no retraining. GUI-KV combines two novel techniques: (i) spatial saliency guidance, which augments attention scores with the L2 norm of hidden states to better preserve semantically important visual tokens, and (ii) temporal redundancy scoring, which projects previous frames' keys onto the current frame's key subspace to preferentially prune redundant history. Across standard GUI agent benchmarks and models, GUI-KV outperforms competitive KV compression baselines, closely matching full-cache accuracy at modest budgets. Notably, in a 5-screenshot setting on the AgentNetBench benchmark, GUI-KV reduces decoding FLOPs by 38.9% while increasing step accuracy by 4.1% over the full-cache baseline. These results demonstrate that exploiting GUI-specific redundancies enables efficient and reliable agent performance.

### 12. INFVSR: BREAKING LENGTH LIMITS OF GENERIC VIDEO SUPER-RESOLUTION

**主要机构**: Meituan Inc, Shanghai Jiao Tong University
**作者数量**: 8人

**摘要**:
Real-world videos often extend over thousands of frames. Existing video superresolution (VSR) approaches, however, face two persistent challenges when processing long sequences: (1) Inefficiency due to the heavy cost of multi-step denoising for full-length sequences; and (2) poor scalability hindered by temporal decomposition that causes artifacts and discontinuities. To break these limits, we propose InfVSR, which novelly reformulates VSR as an autoregressive-onestep-diffusion paradigm. This enables streaming inference while fully leveraging pre-trained video diffusion priors. First, we adapt the pre-trained DiT into a causal structure, maintaining both local and global coherence via rolling KV-cache and joint visual guidance. Second, we distill the diffusion process into a single step efficiently, with patch-wise pixel supervision and cross-chunk distribution matching. Together, these designs enable efficient and scalable VSR for unbounded-length videos. To fill the gap in long-form video evaluation, we build a new benchmark tailored for extended sequences and further introduce semantic-level metrics to comprehensively assess temporal consistency. Our method pushes the frontier of long-form VSR, achieves state-of-the-art quality with enhanced semantic consistency, and delivers up to 58× speed-up over existing methods such as MGLD-VSR. Code will be available at https://github.com/Kai-Liu001/InfVSR.

### 13. Instant4D: 4D Gaussian Splatting in Minutes

**主要机构**: University of Pittsburgh, Carnegie Mellon University, Sichuan Univeristy
**作者数量**: 3人

**摘要**:


### 14. LongCodeZip: Compress Long Context for Code Language Models

**主要机构**: Stanford University, Shanghai Jiao Tong University, Chongqing University
**作者数量**: 5人

**摘要**:
Code generation under long contexts is becoming increasingly critical as Large Language Models (LLMs) are required to reason over extensive information in the codebase. While recent advances enable code LLMs to process long inputs, high API costs and generation latency remain substantial bottlenecks. Existing context pruning techniques, such as LLMLingua, achieve promising results for general text but overlook code-specific structures and dependencies, leading to suboptimal performance in programming tasks. In this paper, we propose LongCodeZip, a novel plug-and-play code compression framework designed specifically for code LLMs. LongCodeZip employs a dual-stage strategy: (1) coarse-grained compression, which identifies and ranks function-level chunks using conditional perplexity with respect to the instruction, retaining only the most relevant functions; and (2) fine-grained compression, which segments retained functions into blocks based on perplexity and selects an optimal subset under an adaptive token budget to maximize relevance. Evaluations across multiple tasks, including code completion, summarization, and question answering, show that LongCodeZip consistently outperforms baseline methods, achieving up to a 5.6× compression ratio without degrading task performance. By effectively reducing context size while preserving essential information, LongCodeZip enables LLMs to better scale to real-world, large-scale code scenarios, advancing the efficiency and capability of code intelligence applications 1 .

### 15. MILCO: LEARNED SPARSE RETRIEVAL ACROSS LANGUAGES VIA A MULTILINGUAL CONNECTOR

**主要机构**: Johns Hopkins University, University of Amsterdam
**作者数量**: 5人

**摘要**:
Learned Sparse Retrieval (LSR) combines the efficiency of bi-encoders with the transparency of lexical matching, but existing approaches struggle to scale beyond English. We introduce MILCO, an LSR architecture that maps queries and documents from different languages into a shared English lexical space via a multilingual connector. MILCO is trained with a specialized two-stage regime that combines Sparse Alignment Pretraining with contrastive training to provide representation transparency and effectiveness while mitigating semantic collapse. Motivated by the observation that uncommon entities are often lost when projected into English, we propose a new LexEcho head, which enhances robustness by augmenting the English lexical representation with a source-language view obtained through a special [ECHO] token. MILCO achieves state-of-the-art multilingual and cross-lingual LSR performance, outperforming leading dense, sparse, and multi-vector baselines such as BGE-M3 and Qwen3-Embed on standard multilingual benchmarks, while supporting dynamic efficiency through post-hoc pruning. Notably, when using mass-based pruning to reduce document representations to only 30 active dimensions on average, MILCO 560M outperforms the similarlysized Qwen3-Embed 0.6B with 1024 dimensions. 1

### 16. NSARM: Next-Scale Autoregressive Modeling for Robust Real-World Image Super-Resolution

**主要机构**: The Hong Kong Polytechnic University
**作者数量**: 5人

**摘要**:
Most recent real-world image super-resolution (Real-ISR) methods employ pre-trained text-to-image (T2I) diffusion models to synthesize the high-quality image either from random Gaussian noise, which yields realistic results but is slow due to iterative denoising, or directly from the input low-quality image, which is efficient but at the price of lower output quality. These approaches train Control-Net or LoRA modules while keeping the pre-trained model fixed, which often introduces over-enhanced artifacts and hallucinations, suffering from the robustness to inputs of varying degradations. Recent visual autoregressive (AR) models, such as pre-trained Infinity, can provide strong T2I generation capabilities while offering superior efficiency by using the bitwise next-scale prediction strategy. Building upon next-scale prediction, we introduce a robust Real-ISR framework, namely Next-Scale Autoregressive Modeling (NSARM). Specifically, we train NSARM in two stages: a transformation network is first trained to map the input low-quality image to preliminary scales, followed by an end-to-end full-model fine-tuning. Such a comprehensive fine-tuning enhances the robustness of NSARM in Real-ISR tasks without compromising its generative capability. Extensive quantitative and qualitative evaluations demonstrate that as a pure AR model, NSARM achieves superior visual results over existing Real-ISR methods while maintaining a fast inference speed. Most importantly, it demonstrates much higher robustness to the quality of input images, showing stronger generalization performance.

### 17. PRISM-Consult: A Panel-of-Experts Architecture for Clinician-Aligned Diagnosis

**主要机构**: MITRE Corporation Bedford, School of Information Berkeley, University of California, ABLE Medical Consulting Savannah, UCLA David Geffen School of Medicine, UC Berkeley
**作者数量**: 6人

**摘要**:
We present PRISM-Consult, a clinician-aligned panel-of-experts architecture that extends the compact PRISM sequence model into a routed family of domain specialists. Episodes are tokenized as structured clinical events; a lightweight router reads the first few tokens and dispatches to specialist models (Cardiac-Vascular, Pulmonary, Gastro-Oesophageal, Musculoskeletal, Psychogenic). Each specialist inherits PRISM's small transformer backbone and token template, enabling parameter efficiency and interpretability. On real-world Emergency Department cohorts, specialists exhibit smooth convergence with low development perplexities across domains, while the router achieves high routing quality and large compute savings versus consult-all under a safety-first policy. We detail the data methodology (initial vs. conclusive ICD-9 families), routing thresholds and calibration, and report per-domain results to avoid dominance by common events. The framework provides a practical path to safe, auditable, and low-latency consult at scale, and we outline validation steps-external/temporal replication, asymmetric life-threat thresholds, and multi-label arbitration-to meet prospective clinical deployment standards.

### 18. PrunedLoRA: Robust Gradient-Based structured pruning for Low-rank Adaptation in Fine-tuning

**主要机构**: The Pennsylvania State University
**作者数量**: 6人

**摘要**:
Low-rank adaptation (LoRA) has become a widely used paradigm for parameter-efficient fine-tuning of large language models, yet its representational capacity often lags behind full fine-tuning. Within the context of LoRA, a key open question is how to obtain expressive low-rank adapters from over-parameterized spaces. We propose PrunedLoRA, a new framework that leverages structured pruning to obtain highly representative low-rank adapters from an over-parameterized initialization. Unlike prior approaches that impose a fixed low-rank budget, PrunedLoRA dynamically prunes less important components during fine-tuning and prevents their reactivation, enabling flexible and adaptive rank allocation. For structured pruning, by minimizing the pruning error for overall loss, we provide fine-grained pruning and recovery updates in a gradient-based pruning strategy with grounded interpretation. We provide the first theoretical analysis of the robustness of structured pruning and provably show that under the impact of weight perturbation, gradient-based pruning is more robust than activation-based pruning with respect to overall loss. Empirically, PrunedLoRA consistently outperforms LoRA and its variants across supervised fine-tuning tasks in mathematical reasoning, code generation, and natural language understanding, and it also demonstrates advantages over existing structured pruning methods across diverse sparsity levels.

### 19. RETHINKING ROPE SCALING IN QUANTIZED LLM: THEORY, OUTLIER, AND CHANNEL-BAND ANALYSIS WITH WEIGHT RESCALING

**主要机构**: University of California, Department of EECS
**作者数量**: 4人

**摘要**:
Extending the context window support of large language models (LLMs) is crucial for tasks with long-distance dependencies. RoPE-based interpolation and extrapolation methods, such as linear scaling and frequency-aware schemes, enable longer input length support without retraining, while post-training quantization (PTQ) makes deployment practical. However, we show that combining RoPE position interpolation (PI) with PTQ degrades accuracy due to coupled effects including long-context aliasing, dynamic-range dilation, anisotropy from axis-aligned quantizers vs. rotated RoPE pairs, and outlier shifting that produces positiondependent logit noise. We provide, to the best of our knowledge, the first systematic analysis of the PI+PTQ approach and introduce two practical diagnostics: interpolation pressure (per-band sensitivity to phase scaling) and tail-inflation ratios (outlier shift from short to long contexts). Following the analysis results, we propose Q-ROAR (Quantization, RoPE-interpolation, and Outlier Aware Rescaling), a weight-only, interpolation-aware stabilization of PI for quantized LLMs. Q-ROAR groups RoPE dimensions into a small number of frequency bands and performs a lightweight search over per-band scales for Key and Query weights (with an optional symmetric variant to preserve logit scale). The search is guided by our diagnostics and uses a tiny long-context development dataset, requiring no fine-tuning to the model, no architecture or kernel changes, and no additional deployment overhead. Empirically, Q-ROAR reduces the model's perplexity on long-context workloads by more than 14%, while preserving short-context performance, inference throughput, and compatibility with existing LLM system stacks.

### 20. SAGE-MUSIC: LOW-LATENCY SYMBOLIC MUSIC GENERATION VIA ATTRIBUTE-SPECIALIZED KEY-VALUE HEAD SHARING

**主要机构**: University of Waterloo, Stanford University, University of Michigan, University of Pennsylvania, University of Chinese Academy of Social Sciences
**作者数量**: 13人

**摘要**:
Low-latency symbolic music generation is essential for real-time improvisation and human-AI co-creation. Existing transformer-based models, however, face a trade-off between inference speed and musical quality. Traditional acceleration techniques such as embedding pooling significantly degrade quality, while recently proposed Byte Pair Encoding (BPE) methods-though effective on singletrack piano data-suffer large performance drops in multi-track settings, as revealed by our analysis. We propose Attribute-Specialized Key-Value Head Sharing (AS-KVHS) adapted to music's structured symbolic representation, achieving ≈30% inference speedup with only a negligible (≈0.4%) quality drop in objective evaluations and slight improvements in subjective listening tests. Our main contributions are (1) the first systematic study of BPE's generalizability in multi-track symbolic music, and (2) the introduction of AS-KVHS for low-latency symbolic music generation. Beyond these, we also release SAGE-Music, an open-source benchmark that matches or surpasses state-of-the-art models in generation quality.

### 21. Semantic-Driven AI Agent Communications: Challenges and Solutions

**主要机构**: State Key Laboratory of Net- working and Switching Technology, Peng Cheng Laboratory, Research Center for Information Science and Technology, University of Electronic Sci- ence and Technology of China, Key Laboratory of Wireless Communications, Tsinghua University, Beijing University of Posts and Telecom- munications, Department of Electronic Engineering, Department of Broad band Communication
**作者数量**: 11人

**摘要**:
With the rapid growth of intelligent services, communication targets are shifting from humans to artificial intelligent (AI) agents, which require new paradigms to enable realtime perception, decision-making, and collaboration. Semantic communication, which conveys task-relevant meaning rather than raw data, offers a promising solution. However, its practical deployment remains constrained by dynamic environments and limited resources. To address these issues, this article proposes a semantic-driven AI agent communication framework and develops three enabling techniques. First, semantic adaptation transmission applies fine-tuning with real or generative samples to efficiently adapt models to varying environments. Second, semantic lightweight transmission incorporates pruning, quantization, and perception-aware sampling to reduce model complexity and alleviate computational burden on edge agents. Third, semantic self-evolution control employs distributed hierarchical decisionmaking to optimize multi-dimensional resources, enabling robust multi-agent collaboration in dynamic environments. Simulation results show that the proposed solutions achieve faster convergence and stronger robustness, while the proposed distributed hierarchical optimization method significantly outperforms conventional decision-making schemes, highlighting its potential for AI agent communication networks.

### 22. Spiralformer: Low Latency Encoder for Streaming Speech Recognition with Circular Layer Skipping and Early Exiting

**主要机构**: Carnegie Mellon University, Sony Group Corporation
**作者数量**: 5人

**摘要**:
For streaming speech recognition, a Transformerbased encoder has been widely used with block processing. Although many studies addressed improving emission latency of transducers, little work has been explored for improving encoding latency of the block processing. We seek to reduce latency by frequently emitting a chunk with a small shift rather than scarce large-chunk emissions, resulting in higher computational costs. To efficiently compute with the small chunk shift, we propose a new encoder, Spiralformer, tailored for block processing by combining layer dropping and early exiting. We skip layer computation in a cyclic manner and shift the computed layer in each block spirally, which completes computation for all the layers over the block processing. Experimentally, we observed that our method achieved 21.6% reduction in the averaged token emission delay in Librispeech, and 7.0% in CSJ, compared with the baseline with similar computational cost and word error rates.

### 23. THE PITFALLS OF KV CACHE COMPRESSION

**主要机构**: University of California
**作者数量**: 5人

**摘要**:
KV cache compression promises increased throughput and efficiency with negligible loss in performance. While the gains in throughput are indisputable and recent literature has indeed shown minimal degradation on particular benchmarks, in general the consequences of compression in realistic scenarios such as multiinstruction prompting have been insufficiently studied. In this paper, we identify several pitfalls practitioners should be aware of when deploying KV cache compressed LLMs. Importantly, we show that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM. As a practical example of that, we highlight system prompt leakage as a case study, empirically showing the impact of compression on leakage and general instruction following. We show several factors that play a role in prompt leakage: compression method, instruction order, and KV eviction bias. We then propose simple changes to KV cache eviction policies that can reduce the impact of these factors and improve the overall performance in multi-instruction tasks.

### 24. THOUGHTBUBBLES: AN UNSUPERVISED METHOD FOR PARALLEL THINKING IN LATENT SPACE

**主要机构**: Stanford University Stanford, Department of Computer Science
**作者数量**: 4人

**摘要**:
Current approaches for scaling inference-time compute in transformers rely on training them to emit explicit chain-of-thought tokens before producing an answer. While these methods are powerful, they are limited because they cannot be applied during pretraining and are limited to only serially-generated, naturallanguage verbalization to scale inference-time compute. In this work, we propose Thoughtbubbles, a transformer variant that natively performs parallel adaptive computation in latent space by learning to fork or delete residual streams. Thus, tokens that require a large amount of computation can form a "bubble" of cloned residuals in the middle of the network for additional thinking. Crucially, this behavior is learned during pretraining with only language modeling loss. Thoughtbubbles outperforms both standard decoder LMs as well as non-adaptive parallel computation approaches on OpenWebText and peS2o perplexity and in zero-shot evaluations such as HellaSwag and LAMBADA after pretraining across 150M to 772M parameter scales. The implicit nature of our method enables adaptive computation to be learned starting at pretraining time, paving the way to unify train and test-time behavior for reasoning models.

### 25. ToolBrain: A Flexible Reinforcement Learning Framework for Agentic Tools

**主要机构**: CeADAR University College Dublin, ToolBrain Research, IBM Research Lab, University College Cork
**作者数量**: 11人

**摘要**:
Effective tool use is essential for agentic AI, yet training agents to utilize tools remains challenging due to manually designed rewards, limited training data, and poor multi-tool selection, resulting in slow adaptation, wasted computational resources, and suboptimal performance. We introduce Tool-Brain, a lightweight and user-friendly framework for coaching tool use in agentic models with flexible reinforcement learning (RL), easing the barriers for researchers and practitioners to adapt LLM-based agents to specific domains. It supports a wide range of training strategies, including RL algorithms such as GRPO and DPO, as well as supervised learning. Tool-Brain enables custom reward callables directly on an agent's execution traces or simply utilizes an automated LLM-as-ajudge system for reward generation. It is packed with useful capabilities, including knowledge distillation from large to small models for efficient development, automatic task generation from tool descriptions, seamless tool retrieval, efficient fine-tuning pipelines with QLoRA through Unsloth, and quantized inference via bitsandbytes. We demonstrate ToolBrain through diverse use cases, such as training a CodeAct agent to autonomously execute email search tasks, showing fast, targeted improvements (up to 30.0%) in tool-use skills while keeping the codebase simple and extensible in Agentic AI. Our framework is publicly available 1 .
