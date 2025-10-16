# AI推理加速技术论文分析报告
生成时间: 2025-10-16 12:27:34
分析论文数量: 47篇

## 论文技术简报

### 1. A.I.R.: ADAPTIVE, ITERATIVE, AND REASONING-BASED FRAME SELECTION FOR VIDEO QUESTION ANSWERING

University of Central Florida与Weill Cornell Medicine发布了A.I.R.框架相关论文，使用自适应、迭代、基于推理的帧选择方法（A.I.R.），解决了VideoQA中帧选择的准确性与计算成本权衡问题，达成了优于现有方法、显著提升基础VLM性能并大幅提高计算效率的效果。

### 2. Best of mini-N in-loop Sampling: A Contextual Quality Reward Model for Reliable and Efficient Best-of-N Sampling

University of Mississippi发布了Best of mini-N in-loop Sampling论文，使用添加外部选项的上下文质量奖励模型及mini-N循环内最佳自适应推理策略，解决了Best-of-N采样中无法捕捉响应可接受性信号导致的可靠性问题，达成减少70%可靠性失败并提升超22%平均推理速度的效果。

### 3. CoDA: Coding LM via Diffusion Adaptation

AI Research发布了CoDA: Coding LM via Diffusion Adaptation论文，使用大规模扩散预训练结合代码中心中期训练与指令调优及置信引导采样技术，解决了扩散语言模型实用系统过重的问题，达成了在Humaneval、MBPP、EvalPlus上1.7B参数模型匹配或超过7B参数扩散模型的效果

### 4. Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space

研究团队发布了Compressed Convolutional Attention论文，提出在共享压缩潜在空间执行注意力操作并结合头共享的CCGQA技术，解决了长上下文Transformer中MHA计算量大、KV缓存增长快的问题，实现8倍KV缓存压缩且性能无下降，预填充延迟降低约1.7倍、反向传播加速约1.3倍。

### 5. Conditional pseudo-supervised contrast for data-Free knowledge distillation

华东师范大学发布了Conditional pseudo-supervised contrast for data-Free knowledge distillation论文，使用条件生成对抗网络、改进生成器区分类别分布及伪监督对比学习技术，解决了现有数据无关知识蒸馏中生成样本类别分布混淆、模糊及多样性不足的问题，在三个常用数据集上提升了学生模型和生成器的性能。

### 6. CONTEXTVLA: VISION-LANGUAGE-ACTION MODEL WITH AMORTIZED MULTI-FRAME CONTEXT

研究团队发布了CONTEXTVLA论文，使用将过去观察压缩为单个上下文token的Vision-Language-Action模型技术，解决了部分可观测机器人任务中多帧观察利用性能提升不一致及视频输入高维计算开销大的问题，达成了比单帧VLA持续改进、实现全多帧训练性能且减少训练和推理时间的效果。

### 7. Efficient Test-Time Scaling for Small Vision-Language Models EFFICIENT TEST-TIME SCALING FOR SMALL VISION-LANGUAGE MODELS

丹麦技术大学发布了Efficient Test-Time Scaling for Small Vision-Language Models论文，使用Test-Time Augmentation (TTAug)和Test-Time Adaptation (TTAdapt)两种利用模型内部特征的高效测试时缩放策略，解决了小视觉语言模型泛化能力与下游任务性能弱且现有测试时缩放技术计算量大的矛盾，达成了在九个基准上持续提升性能、保持计算效率并适用于资源受限环境的效果。

### 8. ERDE: Entropy-Regularized Distillation for Early-exit

LIRIS发布了ERDE论文，使用融合早期退出与知识蒸馏并对教师分类错误图像引入熵基损失的技术，解决了深度神经网络计算成本高不适合实时和边缘应用的问题，在CIFAR10等数据集上显著降低计算复杂度且不影响分类性能。

### 9. EVOENGINEER: MASTERING AUTOMATED CUDA KERNEL CODE EVOLUTION WITH LARGE LANGUAGE MODELS

香港城市大学发布了EVOENGINEER论文，使用系统化基于大语言模型的代码进化框架EvoEngineer，解决了CUDA内核优化的性能瓶颈及现有LLM方法生态碎片化、正确性不足的问题，达成了平均中值加速比2.72倍、代码有效率69.8%且优于现有方法的效果。

### 10. FINISH FIRST, PERFECT LATER: TEST-TIME TOKEN-LEVEL CROSS-VALIDATION FOR DIFFUSION LARGE LANGUAGE MODELS

University of Illinois Urbana-Champaign, University of California San Diego发布了《FINISH FIRST, PERFECT LATER: TEST-TIME TOKEN-LEVEL CROSS-VALIDATION FOR DIFFUSION LARGE LANGUAGE MODELS》论文，使用TOLERATOR（Token-Level Cross-Validation Refinement）训练无关解码策略，解决了离散扩散大型语言模型（dLLMs）vanilla解码中已接受token无法修正导致早期错误持续、输出质量受损的问题，在语言理解、代码生成和数学等五个标准基准上相比基线在相同计算预算下实现一致改进。

### 11. Flexible and Efficient Spatio-Temporal Transformer for Sequential Visual Place Recognition

研究团队发布了《Flexible and Efficient Spatio-Temporal Transformer for Sequential Visual Place Recognition》论文，使用基于新型Recurrent Deformable Transformer Encoder (Recurrent-DTE)的Adapt-STformer技术，解决了现有Transformer-based Seq-VPR方法无法同时实现灵活性（支持可变序列长度）和效率（快速推理、低内存使用）的问题，达成了召回率提升高达17%、序列提取时间减少36%、内存使用降低35%的效果。

### 12. From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing

加州大学、哈佛大学发布了相关论文，使用LASER即插即用推理时路由算法，解决了MoE模型推理时专家负载不平衡导致的系统性能下降问题，达成了改善负载平衡、降低延迟、提高吞吐量且精度变化可忽略的效果

### 13. FT-MDT: Extracting Decision Trees from Medical Texts via a Novel Low-rank Adaptation Method

斯坦福大学发布了FT-MDT论文，使用新型低秩适应方法PI-LoRA（Path-Integrated LoRA），解决了医学决策树构建依赖手动标注的耗时问题，达成在Text2MDT任务上准确率更高、模型复杂度大幅降低的SOTA效果。

### 14. Hybrid Architectures for Language Models: Systematic Analysis and Design Insights

研究团队发布了《Hybrid Architectures for Language Models: Systematic Analysis and Design Insights》论文，通过对结合自注意力与结构化状态空间模型（如Mamba）的混合架构进行层间/层内融合的系统性分析并提出最优设计方案，解决了混合模型杂交策略缺乏系统比较及效果关键因素分析的问题，为混合语言模型开发提供实用指导，优化架构配置，尤其在长上下文任务中平衡建模质量与计算效率。

### 15. HYPERVLA: EFFICIENT INFERENCE IN VISION-LANGUAGE-ACTION MODELS VIA HYPERNETWORKS

牛津大学发布了HyperVLA论文，使用基于超网络（HN）的架构，解决了现有视觉-语言-动作（VLA）模型推理成本极高的问题，达成了与OpenVLA相比测试时激活参数减少90×、推理速度提升120×且成功率相似或更高的效果。

### 16. Learning Efficient Meshflow and Optical Flow from Event Cameras

研究团队发布了《Learning Efficient Meshflow and Optical Flow from Event Cameras》论文，使用HREM数据集、EEMFlow轻量级网络及ADM模块等技术，解决了事件相机网格流估计缺乏专用数据集与方法及事件数据密度挑战的问题，达成EEMFlow比最新SOTA快30倍、ADM提升性能8%和10%的效果

### 17. Learning from All: Concept Alignment for Autonomous Distillation from Multiple Drifting MLLMs

Australian Artificial Intelligence Institute (AAII)与University of Technology发布了《Learning from All: Concept Alignment for Autonomous Distillation from Multiple Drifting MLLMs》论文，使用learn-compare-critique范式及autonomous preference optimization (APO)进行概念对齐，解决了多漂移MLLM教师蒸馏中概念漂移导致学生模型性能受损的问题，达成了知识蒸馏中一致性、鲁棒性和泛化能力的提升，并贡献了CXR-MAX数据集的效果。

### 18. LET FEATURES DECIDE THEIR OWN SOLVERS: HYBRID FEATURE CACHING FOR DIFFUSION TRANSFORMERS

上海交通大学发布了相关论文，使用HyCa混合ODE求解器启发的维度感知特征缓存框架，解决了扩散Transformer迭代采样中Transformer前向传播成本高的瓶颈问题，达成了在FLUX、HunyuanVideo、Qwen-Image等模型上5.55×-6.24×近无损加速的效果。

### 19. MACE: A HYBRID LLM SERVING SYSTEM WITH COLOCATED SLO-AWARE CONTINUOUS RETRAINING ALIGNMENT

NVIDIA发布了MACE混合LLM服务系统论文，使用混合LLM服务系统通过共存并发推理与微调、迭代级SLO感知持续再训练对齐及智能内存管理技术，解决了边缘服务器部署的LLM在GPU资源受限下推理延迟与模型准确性的矛盾，达成了减少推理延迟达63%、维持吞吐量、改善多阶段延迟分布并在NVIDIA AGX Orin上保持GPU利用率超85%的效果。

### 20. MambaCAFU: Hybrid Multi-Scale and Multi-Attention Model with Mamba-Based Fusion for Medical Image Segmentation

University of the Basque Country UPV/EHU发布了MambaCAFU论文，使用三分支编码器（集成CNN、Transformer与Mamba-based注意力融合机制）、多尺度注意力解码器及共注意力门的混合多尺度多注意力模型，解决了现有模型任务特定、跨模态与解剖区域性能差异大及复杂度与性能难以权衡的问题，达成在多个基准数据集上优于SOTA的准确性和泛化性且计算复杂度与平均模型相当的效果。

### 21. MECKD: Deep Learning-Based Fall Detection in Multilayer Mobile Edge Computing With Knowledge Distillation

研究团队发布了MECKD论文，使用多层移动边缘计算（MLMEC）与知识蒸馏（KD）的深度学习技术，解决了跌倒检测系统中边缘设备模型尺寸有限及数据传输延迟问题，达成了在SisFall数据集准确率提升11.65%、延迟率降低46.67%，FallAllD数据集准确率提升2.78%、延迟率降低54.15%的效果

### 22. Under review as a conference paper at ICLR 2026 MEMMAMBA: RETHINKING MEMORY PATTERNS IN STATE SPACE MODEL

中国人民大学发布了MemMamba论文，使用状态总结机制与跨层跨token注意力技术，解决了Mamba长程记忆指数衰减问题，达成了长序列建模性能超越现有Mamba变体和Transformer、推理效率提升48%的效果。

### 23. MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition

帝国理工学院发布了MoME论文，使用Mixture of Matryoshka Experts (MoME)框架（整合稀疏Mixture-of-Experts与Matryoshka表示学习），解决了大语言模型在音视频语音识别(AVSR)中计算需求高、对token粒度敏感及现有MRL方法跨尺度泛化不足的问题，在LRS2和LRS3数据集上达成音视频语音识别等任务的最先进性能，且参数显著减少、噪声下稳健。

### 24. MonitorVLM: A Vision-Language Framework for Safety Violation Detection in Mining Operations

北京科技大学发布了MonitorVLM论文，使用结合领域特定违规数据集、条款过滤器(CF)模块及行为放大器(BM)模块的视觉语言框架，解决了采矿作业中传统人工检查劳动密集、易出错的安全监控问题，达成相比72B未微调基线精度提升22.01%、召回34.22%、F1 28.37%的效果。

### 25. On Structured State-Space Duality

西北大学和加州大学发布了关于结构化状态空间对偶性的论文，通过形式化并推广结构化状态空间对偶性（SSD）至一般对角结构化状态空间模型（SSMs）、建立其与1-半可分掩码注意力的等价条件，解决了循环SSMs与Transformers之间的连接问题，达成了加强两者桥梁、拓宽高效且有表达力的序列模型设计空间的效果。

### 26. OPTIMIZED MINIMAL 4D GAUSSIAN SPLATTING

首尔国立大学发布了OPTIMIZED MINIMAL 4D GAUSSIAN SPLATTING论文，使用优化的最小化4D高斯溅射技术，解决了4D高斯溅射模型复杂度与计算效率问题，达成了在减少资源消耗的同时保持动态场景重建性能的效果。

### 27. POLYKAN: A POLYHEDRAL ANALYSIS FRAMEWORK FOR PROVABLE AND APPROXIMATELY OPTIMAL KAN COMPRESSION

西安交通大学发布了PolyKAN论文，使用多面体分析框架与动态规划算法，解决了KAN模型的压缩难题，达成了可证明近最优压缩并保持严格误差控制，为KAN压缩提供首个带数学保证的理论基础。

### 28. POST-TRAINING QUANTIZATION OF VISION ENCODERS NEEDS PREFIXING REGISTERS

Dankook University和Google发布了POST-TRAINING QUANTIZATION OF VISION ENCODERS NEEDS PREFIXING REGISTERS论文，使用RegCache（含中间层前缀和令牌删除）技术，解决了视觉编码器后训练量化中因大规模激活异常值导致的精度下降问题，达成了在文本监督和自监督视觉编码器上持续提高量化模型准确性的效果

### 29. Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation

研究团队发布了《Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation》论文，使用训练无关的预测特征缓存策略（基于SE(3)等变骨架预测求解器步骤间的中间隐藏状态），解决了流匹配模型分子几何生成时的推理计算成本高（需数百次网络评估）的瓶颈，达成了在保持样本质量下推理时间减半、最高提速3倍（轻微质量损失）且与其他优化结合达7倍提速的效果。

### 30. Prompt-Aware Scheduling for Low-Latency LLM Serving

University of Illinois, Chicago和Argonne发布了PARS论文，提出基于成对排序（margin ranking loss）的prompt-aware调度器近似最短作业优先（SJF），解决了传统FCFS调度的Head-of-Line阻塞问题，达成了LLM服务低延迟、跨模型泛化良好且性能显著提升（含推理工作负载）的效果。

### 31. PT 2 -LLM: POST-TRAINING TERNARIZATION FOR LARGE LANGUAGE MODELS

腾讯发布了PT²-LLM论文，使用不对称三元量化器及两阶段优化（ITF、AGA）与SSR策略，解决了LLM部署中内存和计算需求大及三元化在PTQ中面临的参数优化和量化困难问题，达成相比SOTA 2位PTQ方法内存成本更低且加速预填充与解码实现端到端加速的效果

### 32. QUANT-DLLM: POST-TRAINING EXTREME LOW-BIT QUANTIZATION FOR DIFFUSION LARGE LANGUAGE MODELS

上海交通大学发布了QUANT-DLLM论文，使用Quant-dLLM框架（含Masked Calibration Simulation、Data-aware Any-order Quantizer、Adaptive Blockwise Mixed Precision），解决了扩散大型语言模型（dLLMs）的2-bit后训练量化性能不佳问题，达成了在2-bit精度下比现有AR迁移PTQ方法更高准确率的效果。

### 33. QUANTDEMOIRE: QUANTIZATION WITH OUTLIER AWARE FOR IMAGE DEMOIR ÉING

上海交通大学发布了QUANTDEMOIRE论文，使用异常值感知量化器与频率感知校准策略的训练后量化框架，解决了现有去摩尔纹模型量化性能下降（因分布异常值和光滑区域表示弱化）的问题，达成了在W4A4上超过现有量化方法4dB以上、大幅减少参数和计算量并保持质量的效果。

### 34. QUANTIZATION RANGE ESTIMATION FOR CONVOLUTIONAL NEURAL NETWORKS A PREPRINT

Xidian University发布了量化范围估计相关论文，使用通过将范围估计建模为逐层局部损失最小化量化误差的优化问题并提出高效搜索算法应用于变换权重空间的技术，解决了低比特量化中保持模型精度的难题，达成了在ResNet系列及Inception-v3上8/6位量化几乎无top-1精度损失、4位精度显著提升且性能优于现有技术的效果。

### 35. Reactive Transformer (RxT) -Stateful Real-Time Processing for Event-Driven Reactive Language Models

研究团队发布了《Reactive Transformer (RxT)》论文，使用事件驱动的Reactive Transformer (RxT)架构（集成固定大小短期记忆系统，分离响应生成与记忆更新），解决了Transformer在对话AI中因无状态性和二次计算复杂度导致的长对话成本高、延迟大问题，达成对话成本从O(N²•T)降至线性O(N•T)、实现低延迟和恒定时间推理的效果。

### 36. ReTiDe: Real-Time Denoising for Energy-Efficient Motion Picture Processing with FPGAs

Trinity College Dublin发布了ReTiDe论文，使用基于FPGA的实时去噪结合量化（PTQ/QAT）技术，解决运动图像处理中实时性与能效的平衡问题，达成在BSD100数据集噪声强度45场景下有效去噪且提升能效的效果。

### 37. SDAKD: Student Discriminator Assisted Knowledge Distillation for Super-Resolution Generative Adversarial Networks

CERTH-ITI Thessaloniki发布了SDAKD论文，使用引入学生判别器的三阶段训练策略及整合适应的特征图蒸馏方法，解决了学生生成器与教师判别器容量不匹配导致的GAN压缩训练困难问题，达成了在GCFSR和Real-ESRGAN上性能优于基线及SOTA GAN知识蒸馏方法的效果

### 38. SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size

清华大学发布了SDQ-LLM论文，使用Sigma-Delta量化结合连续可调过采样率（OSR）及Hadamard权重平滑技术，解决了大型语言模型（LLMs）的计算和内存挑战，达成了在1-bit/1.58-bit极低比特量化下保留语言推理能力并实现高效高精度性能的效果

### 39. SELF SPECULATIVE DECODING FOR DIFFUSION LARGE LANGUAGE MODELS

华为技术有限公司与上海交通大学发布了SELF SPECULATIVE DECODING FOR DIFFUSION LARGE LANGUAGE MODELS论文，使用Self Speculative Decoding (SSD)技术（利用dLLM自身作为推测解码器和验证器，无需辅助模块，通过自起草机制和层次验证树在单次前向传播中验证多个位置预测），解决了当前并行解码方法偏离逐步解码导致的性能下降问题，达成了高达3.46倍加速且保持输出与逐步解码一致的效果。

### 40. SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling

Rutgers University - New Brunswick发布了SliceMoE论文，使用路由嵌入切片而非tokens的技术，解决了传统token-MoE的容量瓶颈、负载均衡问题和有限专业化，达成了比密集基线快1.7倍推理、比参数匹配token-MoE低12-18%困惑度及更好专家平衡的效果

### 41. Sliding Window Attention for Learned Video Compression

Friedrich-Alexander University发布了视频压缩相关论文，使用3D滑动窗口注意力（SWA）技术，解决了现有局部注意力机制的不规则感受野和计算冗余重叠窗口问题，达成了率失真性能提升（Bjøntegaard Delta-rate节省达18.6%）、解码器复杂度降低2.8倍及熵模型效率提升近3.5倍的效果。

### 42. Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs

东北大学等机构发布了关于小型语言模型（SLMs）在智能体系统中应用的综述论文，使用引导解码、工具执行及SLM-default+LLM-fallback系统，解决了大型语言模型在智能体工作负载中成本/延迟/能耗高及目标泛化与约束准确性不匹配的问题，达成SLMs匹配或超越LLMs性能且token成本降低10×-100×的效果。

### 43. Speak, Edit, Repeat: High-Fidelity Voice Editing and Zero-Shot TTS with Cross-Attentive Mamba

ITMO University发布了关于高保真语音编辑与零样本TTS的论文，使用交叉注意力Mamba架构，解决了语音编辑与零样本TTS合成中的自然度和一致性问题，达成了编辑语音与原始难以区分、零样本TTS性能超现有模型且内存成本降低约6倍的效果

### 44. SPEAR: Soft Prompt Enhanced Anomaly Recognition for Time Series Data

卡尔加里大学发布了SPEAR论文，使用软提示和量化技术，解决了传统方法处理变长时间序列及基于上下文异常的困难，达成了有效提升LLMs在时间序列异常检测下游任务性能的效果

### 45. Speculative Actions: A Lossless Framework for Faster Agentic Systems

哥伦比亚大学纽约发布了《Speculative Actions: A Lossless Framework for Faster Agentic Systems》论文，使用推测动作框架（通过更快模型预测可能动作实现多步并行执行），解决了AI代理因动作顺序执行、API调用耗时导致的执行慢瓶颈，达成了动作预测准确率高达55%、显著减少端到端延迟的效果。

### 46. StructPrune: Structured Global Pruning asymptotics with O( √ N ) GPU Memory

Emory University发布了StructPrune论文，使用分治策略分解全局剪枝为协调子问题的ADMM框架，解决了全局结构化剪枝高GPU内存需求与局部剪枝性能不足的问题，达成了GPU内存从O(N)降至O(√N)且匹配全局结构化剪枝困惑度的效果

### 47. UNIPRUNING: UNIFYING LOCAL METRIC AND GLOBAL FEEDBACK FOR SCALABLE SPARSE LLMS

复旦大学发布了UniPruning论文，使用统一局部度量与全局反馈的镜像下降优化训练后剪枝框架，解决了现有LLM剪枝效率与鲁棒性难以平衡的问题，在多个LLM家族和基准测试中达成更优的困惑度和零样本准确率。

## 论文详细信息

### 1. A.I.R.: ADAPTIVE, ITERATIVE, AND REASONING-BASED FRAME SELECTION FOR VIDEO QUESTION ANSWERING

**主要机构**: University of Central Florida, Weill Cornell Medicine
**作者数量**: 6人

**摘要**:
Effectively applying Vision-Language Models (VLMs) to Video Question Answering (VideoQA) hinges on selecting a concise yet comprehensive set of frames, as processing entire videos is computationally infeasible. However, current frame selection methods face a critical trade-off: approaches relying on lightweight similarity models, such as CLIP, often fail to capture the nuances of complex queries, resulting in inaccurate similarity scores that cannot reflect the authentic queryframe relevance, which further undermines frame selection. Meanwhile, methods that leverage a VLM for deeper analysis achieve higher accuracy but incur prohibitive computational costs. To address these limitations, we propose A.I.R., a training-free approach for Adaptive, Iterative, and Reasoning-based frame selection. We leverage a powerful VLM to perform deep, semantic analysis on complex queries, and this analysis is deployed within a cost-effective iterative loop that processes only a small batch of the most high-potential frames at a time. Extensive experiments on various VideoQA benchmarks demonstrate that our approach outperforms existing frame selection methods, significantly boosts the performance of the foundation VLM, and achieves substantial gains in computational efficiency over other VLM-based techniques.

### 2. Best of mini-N in-loop Sampling: A Contextual Quality Reward Model for Reliable and Efficient Best-of-N Sampling

**主要机构**: University of Mississippi
**作者数量**: 1人

**摘要**:
Modern preference alignment techniques, such as Best-of-N (BoN) sampling, rely on reward models trained with pairwise comparison data. While effective at learning relative preferences, this paradigm fails to capture a signal of response acceptability, leaving systems vulnerable to selecting the least bad of many unacceptable options. This is particularly problematic for hard prompts, where the risk of such false acceptances increases with the number of samples. In this paper, we address this critical reliability gap by introducing a new data collection and modeling framework. By augmenting preference data with an outside option, inspired by discrete choice models, we train a reward model that can distinguish not just what is better, but what is good enough. We leverage this capability to create an adaptive inference strategy, best of mini-N in-loop, which partitions the generation budget into sequential loops with a calibrated, early-exit condition. Our experiments show that when tuned as an alignment guardrail, it reduces reliability failures by 70%, and when tuned as an inference accelerator, it improves average inference speed by over 22% in IMDBsentiment setting. We thus provide a principled and flexible framework for practitioners to explicitly manage the trade-off between reliability and computational efficiency.

### 3. CoDA: Coding LM via Diffusion Adaptation

**主要机构**: AI Research * Core Contributors
**作者数量**: 15人

**摘要**:
Diffusion language models promise bidirectional context and infilling capabilities that autoregressive coders lack, yet practical systems remain heavyweight. We introduce CoDA, a 1.7B-parameter diffusion coder trained on TPU with a fully open-source training pipeline. CoDA pairs large-scale diffusion pre-training with code-centric mid-training and instruction tuning, enabling confidence-guided sampling that keeps inference latency competitive. On Humaneval, MBPP, and EvalPlus, CoDA-1.7B-Instruct matches or surpasses diffusion models up to 7B parameters. Our release includes model checkpoints, evaluation harnesses, and TPU training pipelines to accelerate research on lightweight diffusion-based coding assistants.

### 4. Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space

**主要机构**: 
**作者数量**: 5人

**摘要**:
Multi-headed Attention's (MHA) quadratic compute and linearly growing KV-cache make long-context transformers expensive to train and serve. Prior works such as Grouped Query Attention (GQA) and Multi-Latent Attention (MLA) shrink the cache, speeding decode, but leave compute, which determines prefill and training speed, largely unchanged. We introduce Compressed Convolutional Attention (CCA), a novel attention method which down-projects queries, keys, and values and performs the entire attention operation inside the shared latent space. This simple design dramatically cuts parameters, KV-cache, and FLOPs all at once by the desired compression factor. Because CCA is orthogonal to head-sharing, we combine the two to form Compressed Convolutional Grouped Query Attention (CCGQA), which further tightens the compute-bandwidth Pareto frontier so that users can tune compression toward either FLOP or memory limits without sacrificing quality. Experiments show that CCGQA consistently outperforms both GQA and MLA at equal KV-cache compression on dense and MoE models. Additionally, we show that CCGQA outperforms all other attention methods on MoE models with half the KV-cache of GQA and MLA, achieving an 8x KV-cache compression with no drop in performance compared to standard MHA. CCA and CCGQA also dramatically reduce the FLOP cost of attention which leads to substantially faster training and prefill than existing methods. On H100 GPUs, our fused CCA/CCGQA kernel reduces prefill latency by about 1.7× at a sequence length of 16k relative to MHA, and accelerates backward by about 1.3×.

### 5. Conditional pseudo-supervised contrast for data-Free knowledge distillation

**主要机构**: School of Computer Science and Technology, East China Normal Unversity
**作者数量**: 3人

**摘要**:
Data-free knowledge distillation (DFKD) is an effective manner to solve model compression and transmission restrictions while retaining privacy protection, which has attracted extensive attention in recent years. Currently, the majority of existing methods utilize a generator to synthesize images to support the distillation. Although the current methods have achieved great success, there are still many issues to be explored. Firstly, the outstanding performance of supervised learning in deep learning drives us to explore a pseudo-supervised paradigm on DFKD. Secondly, current synthesized methods cannot distinguish the distributions of different categories of samples, thus producing ambiguous samples that may lead to an incorrect evaluation by the teacher. Besides, current methods cannot optimize the category-wise diversity samples, which will hinder the student model learning from diverse samples and further achieving better performance. In this paper, to address the above limitations, we propose a novel learning paradigm, i.e., conditional pseudo-supervised contrast for data-free knowledge distillation (CPSC-DFKD). The primary innovations of CPSC-DFKD are: (1) introducing a conditional generative adversarial network to synthesize category-specific diverse images for pseudo-supervised learning, (2) improving the modules of the generator to distinguish the distributions of different categories, and (3) proposing pseudo-supervised contrastive learning based on teacher and student views to enhance diversity. Comprehensive experiments on three commonly-used datasets validate the performance lift of both the student and generator brought by CPSC-DFKD. The code is available at https://github.com/RoryShao/CPSC-DFKD.git

### 6. CONTEXTVLA: VISION-LANGUAGE-ACTION MODEL WITH AMORTIZED MULTI-FRAME CONTEXT

**主要机构**: 
**作者数量**: 9人

**摘要**:
Leveraging temporal context is crucial for success in partially observable robotic tasks. However, prior work in behavior cloning has demonstrated inconsistent performance gains when using multi-frame observations. In this paper, we introduce ContextVLA, a policy model that robustly improves robotic task performance by effectively leveraging multi-frame observations. Our approach is motivated by the key observation that Vision-Language-Action models (VLA), i.e., policy models built upon a Vision-Language Model (VLM), more effectively utilize multi-frame observations for action generation. This suggests that VLMs' inherent temporal understanding capability enables them to extract more meaningful context from multi-frame observations. However, the high dimensionality of video inputs introduces significant computational overhead, making VLA training and inference inefficient. To address this, ContextVLA compresses past observations into a single context token, allowing the policy to efficiently leverage temporal context for action generation. Our experiments show that ContextVLA consistently improves over single-frame VLAs and achieves the benefits of full multi-frame training but with reduced training and inference times.

### 7. Efficient Test-Time Scaling for Small Vision-Language Models EFFICIENT TEST-TIME SCALING FOR SMALL VISION-LANGUAGE MODELS

**主要机构**: Technical University of Denmark, Pioneer Center for AI
**作者数量**: 3人

**摘要**:
Small Vision-Language Models (VLMs) provide a computationally efficient alternative to larger models, at the cost of weaker generalization abilities and downstream task performance. These shortcomings could be addressed by test-time scaling techniques, but existing methods are typically computationally demanding, contradicting the resource-efficient design goals of small models. To address these limitations, we propose two novel and efficient test-time scaling strategies that leverage the model-internal features rather than external supervision: (i) Test-Time Augmentation (TTAug), which generates multiple augmented inputs and aggregates outputs at the token level without parameter updates, and (ii) Test-Time Adaptation (TTAdapt), which adapts model parameters during inference using consensus-based pseudolabels from TTAug. Through extensive experiments across nine benchmarks, we demonstrate consistent performance improvements while maintaining computational efficiency suitable for resource-constrained environments. The generality of our approach is demonstrated both within models at different scales and across different VLMs without additional tuning.

### 8. ERDE: Entropy-Regularized Distillation for Early-exit

**主要机构**: LIRIS, Université, UMR5205, CNRS, INSA Lyon
**作者数量**: 14人

**摘要**:
Although deep neural networks and in particular Convolutional Neural Networks have demonstrated state-of-the-art performance in image classification with relatively high efficiency, they still exhibit high computational costs, often rendering them impractical for real-time and edge applications. Therefore, a multitude of compression techniques have been developed to reduce these costs while maintaining accuracy. In addition, dynamic architectures have been introduced to modulate the level of compression at execution time, which is a desirable property in many resourcelimited application scenarios. The proposed method effectively integrates two well-established optimization techniques: early exits and knowledge distillation, where a reduced student early-exit model is trained from a more complex teacher early-exit model. The primary contribution of this research lies in the approach for training the student early-exit model. In comparison to the conventional Knowledge Distillation loss, our approach incorporates a new entropy-based loss for images where the teacher's classification was incorrect. The proposed method optimizes the trade-off between accuracy and efficiency, thereby achieving significant reductions in computational complexity without compromising classification performance. The validity of this approach is substantiated by experimental results on image classification datasets CIFAR10, CIFAR100 and SVHN, which further opens new research perspectives for Knowledge Distillation in other contexts.

### 9. EVOENGINEER: MASTERING AUTOMATED CUDA KERNEL CODE EVOLUTION WITH LARGE LANGUAGE MODELS

**主要机构**: City University of Hong Kong, Department of Computer Science
**作者数量**: 7人

**摘要**:
CUDA kernel optimization has become a critical bottleneck for AI performance, as deep learning training and inference efficiency directly depends on highly optimized GPU kernels. Despite the promise of Large Language Models (LLMs) for automating kernel optimization, this field suffers from a fragmented ecosystem of isolated and incomparable approaches with unclear problem formulations. Furthermore, general-purpose LLM code evolution methods cannot meet strict correctness requirements of CUDA kernel optimization. We address these fundamental challenges by first formalizing CUDA kernel optimization as a code optimization task with a clear objective, constraints, and evaluation metrics. We then establish the first systematic LLM-based code evolution framework, EvoEngineer, that provides guidance for designing and adapting optimization strategies to achieve a balance between performance and correctness. Finally, we implement a kernel optimization system based on this framework and conduct extensive experiments on 91 real-world CUDA kernels. Our results demonstrate that EvoEngineer achieves a principled balance between performance and correctness, with the highest averaged median speedup of 2.72× over baseline CUDA kernels and a code validity rate of 69.8%, outperforming existing methods on both dimensions. Our method achieves a maximum speedup of 36.75× among all operations over PyTorch kernels and delivers the highest speedup on 28 (56.0%) of 50 operations that achieve over 2× acceleration.

### 10. FINISH FIRST, PERFECT LATER: TEST-TIME TOKEN-LEVEL CROSS-VALIDATION FOR DIFFUSION LARGE LANGUAGE MODELS

**主要机构**: University of Illinois Urbana-Champaign, University of California San Diego
**作者数量**: 5人

**摘要**:
Diffusion large language models (dLLMs) have recently emerged as a promising alternative to autoregressive (AR) models, offering advantages such as accelerated parallel decoding and bidirectional context modeling. However, the vanilla decoding strategy in discrete dLLMs suffers from a critical limitation: once a token is accepted, it can no longer be revised in subsequent steps. As a result, early mistakes persist across iterations, harming both intermediate predictions and final output quality. To address this issue, we propose TOLERATOR (Token-Level Cross-Validation Refinement), a training-free decoding strategy that leverages cross-validation among predicted tokens. Unlike existing methods that follow a single progressive unmasking procedure, TOLERATOR introduces a two-stage process: (i) sequence fill-up and (ii) iterative refinement by remasking and decoding a subset of tokens while treating the remaining as context. This design enables previously accepted tokens to be reconsidered and corrected when necessary, leading to more reliable diffusion decoding outputs. We evaluate TOLERATOR on five standard benchmarks covering language understanding, code generation, and mathematics. Experiments show that our method achieves consistent improvements over the baselines under the same computational budget. These findings suggest that decoding algorithms are crucial to realizing the full potential of diffusion large language models. Code and data are publicly available.

### 11. Flexible and Efficient Spatio-Temporal Transformer for Sequential Visual Place Recognition

**主要机构**: 
**作者数量**: 4人

**摘要**:
Sequential Visual Place Recognition (Seq-VPR) leverages transformers to capture spatio-temporal features effectively; however, existing approaches prioritize performance at the expense of flexibility and efficiency. In practice, a transformer-based Seq-VPR model should be flexible to the number of frames per sequence (seq-length), deliver fast inference, and have low memory usage to meet real-time constraints. To our knowledge, no existing transformer-based Seq-VPR method achieves both flexibility and efficiency. To address this gap, we propose Adapt-STformer, a Seq-VPR method built around our novel Recurrent Deformable Transformer Encoder (Recurrent-DTE), which uses an iterative recurrent mechanism to fuse information from multiple sequential frames. This design naturally supports variable seq-lengths, fast inference, and low memory usage. Experiments on the Nordland, Oxford, and NuScenes datasets show that Adapt-STformer boosts recall by up to 17% while reducing sequence extraction time by 36% and lowering memory usage by 35% compared to the second-best baseline.

### 12. From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing

**主要机构**: University of California, Harvard University
**作者数量**: 5人

**摘要**:
Mixture-of-Experts (MoE) models can scale parameter capacity by routing each token to a subset of experts through a learned gate function. While conditional routing reduces training costs, it shifts the burden on inference memory: expert parameters and activations consume memory, limiting the number of experts per device. As tokens are routed, some experts become overloaded while others are underutilized. Because experts are mapped to GPUs, this imbalance translates directly into degraded system performance in terms of latency, throughput, and cost. We present LASER, a plug-and-play, inference-time routing algorithm that balances load while preserving accuracy. LASER adapts to the shape of the gate's score distribution. When scores provide a clear preference, it routes to the strongest experts; when scores are more uniform, it broadens the set of viable experts and routes to the least-loaded among them. Because LASER relies only on gate scores from a trained model, it integrates directly into existing MoE inference pipelines without retraining or finetuning. We evaluate LASER on Mixtral-8×7B and DeepSeek-MoE-16b-chat across four datasets (ARC-Easy, ARC-Challenge, MMLU, and GSM8K). LASER improves load balancing, translating into lower latency and higher throughput, while keeping accuracy changes negligible.

### 13. FT-MDT: Extracting Decision Trees from Medical Texts via a Novel Low-rank Adaptation Method

**主要机构**: University of Hong Kong, Stanford University, University of Chicage, Johns Hopkins University, Independent Researcher, HK, Carnegie Mellon University
**作者数量**: 7人

**摘要**:
Knowledge of the medical decision process, which can be modeled as medical decision trees (MDTs), is critical to building clinical decision support systems. However, current MDT construction methods rely heavily on time-consuming and laborious manual annotation. To address this challenge, we propose PI-LoRA (Path-Integrated LoRA), a novel lowrank adaptation method for automatically extracting MDTs from clinical guidelines and textbooks. We integrate gradient path information to capture synergistic effects between different modules, enabling more effective and reliable rank allocation. This framework ensures that the most critical modules receive appropriate rank allocations while less important ones are pruned, resulting in a more efficient and accurate model for extracting medical decision trees from clinical texts. Extensive experiments on medical guideline datasets demonstrate that our PI-LoRA method significantly outperforms existing parameter-efficient fine-tuning approaches for the Text2MDT task, achieving better accuracy with substantially reduced model complexity. The proposed method achieves state-of-the-art results while maintaining a lightweight architecture, making it particularly suitable for clinical decision support systems where computational resources may be limited.

### 14. Hybrid Architectures for Language Models: Systematic Analysis and Design Insights

**主要机构**: 
**作者数量**: 10人

**摘要**:
Recent progress in large language models demonstrates that hybrid architectures-combining selfattention mechanisms with structured state space models like Mamba-can achieve a compelling balance between modeling quality and computational efficiency, particularly for long-context tasks. While these hybrid models show promising performance, systematic comparisons of hybridization strategies and analyses on the key factors behind their effectiveness have not been clearly shared to the community. In this work, we present a holistic evaluation of hybrid architectures based on inter-layer (sequential) or intra-layer (parallel) fusion. We evaluate these designs from a variety of perspectives: language modeling performance, long-context capabilities, scaling analysis, and training and inference efficiency. By investigating the core characteristics of their computational primitive, we identify the most critical elements for each hybridization strategy and further propose optimal design recipes for both hybrid models. Our comprehensive analysis provides practical guidance and valuable insights for developing hybrid language models, facilitating the optimization of architectural configurations.

### 15. HYPERVLA: EFFICIENT INFERENCE IN VISION-LANGUAGE-ACTION MODELS VIA HYPERNETWORKS

**主要机构**: University of Oxford
**作者数量**: 6人

**摘要**:
Built upon language and vision foundation models with strong generalization ability and trained on large-scale robotic data, Vision-Language-Action (VLA) models have recently emerged as a promising approach to learning generalist robotic policies. However, a key drawback of existing VLAs is their extremely high inference costs. In this paper, we propose HyperVLA to address this problem. Unlike existing monolithic VLAs that activate the whole model during both training and inference, HyperVLA uses a novel hypernetwork (HN)-based architecture that activates only a small task-specific policy during inference, while still retaining the high model capacity needed to accommodate diverse multi-task behaviors during training. Successfully training an HN-based VLA is nontrivial so HyperVLA contains several key algorithm design features that improve its performance, including properly utilizing the prior knowledge from existing vision foundation models, HN normalization, and an action generation strategy. Compared to monolithic VLAs, HyperVLA achieves a similar or even higher success rate for both zero-shot generalization and few-shot adaptation, while significantly reducing inference costs. Compared to OpenVLA, a state-of-the-art VLA model, HyperVLA reduces the number of activated parameters at test time by 90×, and accelerates inference speed by 120×. Code is publicly available at https://github.com/MasterXiong/Hyper-VLA.

### 16. Learning Efficient Meshflow and Optical Flow from Event Cameras

**主要机构**: 
**作者数量**: 0人

**摘要**:
In this paper, we explore the problem of event-based meshflow estimation, a novel task that involves predicting a spatially smooth sparse motion field from event cameras. To start, we review the state-of-the-art in event-based flow estimation, highlighting two key areas for further research: i) the lack of meshflow-specific event datasets and methods, and ii) the underexplored challenge of event data density. First, we generate a large-scale High-Resolution Event Meshflow (HREM) dataset, which showcases its superiority by encompassing the merits of high resolution at 1280×720, handling dynamic objects and complex motion patterns, and offering both optical flow and meshflow labels. These aspects have not been fully explored in previous works. Besides, we propose Efficient Event-based MeshFlow (EEMFlow) network, a lightweight model featuring a specially crafted encoder-decoder architecture to facilitate swift and accurate meshflow estimation. Furthermore, we upgrade EEMFlow network to support dense event optical flow, in which a Confidence-induced Detail Completion (CDC) module is proposed to preserve sharp motion boundaries. We conduct comprehensive experiments to show the exceptional performance and runtime efficiency (30× faster) of our EEMFlow model compared to the recent state-of-the-art flow method. As an extension, we expand HREM into HREM+, a multi-density event dataset contributing to a thorough study of the robustness of existing methods across data with varying densities, and propose an Adaptive Density Module (ADM) to adjust the density of input event data to a more optimal range, enhancing the model's generalization ability. We empirically demonstrate that ADM helps to significantly improve the performance of EEMFlow and EEMFlow+ by 8% and 10%, respectively. Code and dataset are released at https://github.com/boomluo02/EEMFlowPlus.

### 17. Learning from All: Concept Alignment for Autonomous Distillation from Multiple Drifting MLLMs

**主要机构**: Faulty of Engineering and Information Technology, University of Technology, Australian Artificial Intelligence Institute (AAII)
**作者数量**: 3人

**摘要**:
This paper identifies a critical yet underexplored challenge in distilling from multimodal large language models (MLLMs): the reasoning trajectories generated by multiple drifting teachers exhibit concept drift, whereby their reasoning distributions evolve unpredictably and transmit biases to the student model, ultimately compromising its performance. To tackle this issue, we pioneer a theoretical connection between concept drift and knowledge distillation, casting the nonstationary reasoning dynamics from multiple MLLM teachers as next-token prediction of multi-stream reasoning trajectories. Guided by concept drift, we introduce the "learn-compare-critique" paradigm, culminating in autonomous preference optimization (APO). Under the active guidance of the teachers, the student model first learns and self-distils preferred thinking by comparing multiple teachers. It then engages in critical reflection over the drifting inference from teachers, performing concept alignment through APO, ultimately yielding a robust, consistent, and generalizable model. Extensive experiments demonstrate our superior performance of consistency, robustness and generalization within knowledge distillation. Besides, we also contributed a large-scale dataset CXR-MAX (Multi-teachers Alignment X-rays), comprising 170,982 distilled reasoning trajectories derived from publicly accessible MLLMs based on MIMIC-CXR. Our code and data are public at: https://anonymous.4open.science/r/ Autonomous-Distillation/.

### 18. LET FEATURES DECIDE THEIR OWN SOLVERS: HYBRID FEATURE CACHING FOR DIFFUSION TRANSFORMERS

**主要机构**: Shanghai Jiao Tong University
**作者数量**: 9人

**摘要**:
Diffusion Transformers offer state-of-the-art fidelity in image and video synthesis, but their iterative sampling process remains a major bottleneck due to the high cost of transformer forward passes at each timestep. To mitigate this, feature caching has emerged as a training-free acceleration technique that reuses or forecasts hidden representations. However, existing methods often apply a uniform caching strategy across all feature dimensions, ignoring their heterogeneous dynamic behaviors. Therefore, we adopt a new perspective by modeling hidden feature evolution as a mixture of ODEs across dimensions, and introduce HyCa, a Hybrid ODE solver inspired caching framework that applies dimension-wise caching strategies. HyCa achieves near-lossless acceleration across diverse domains and models, including 5.55× speedup on FLUX, 5.56× speedup on HunyuanVideo, 6.24× speedup on Qwen-Image and Qwen-Image-Edit without retraining.

### 19. MACE: A HYBRID LLM SERVING SYSTEM WITH COLOCATED SLO-AWARE CONTINUOUS RETRAINING ALIGNMENT

**主要机构**: 
**作者数量**: 4人

**摘要**:
Large language models (LLMs) deployed on edge servers are increasingly used in latency-sensitive applications such as personalized assistants, recommendation, and content moderation. However, the non-stationary nature of user data necessitates frequent retraining, which introduces a fundamental tension between inference latency and model accuracy under constrained GPU resources. Existing retraining strategies either delay model updates, over-commit resources to retraining, or overlook iteration-level retraining granularity. In this paper, we identify that iteration-level scheduling is crucial for adapting retraining frequency to model drift without violating servicelevel objectives (SLOs). We propose MACE, a hybrid LLM system that colocates concurrent inference (prefill, decode) and fine-tuning, with intelligent memory management to maximize task performance while promising inference throughput. MACE leverages the insight that not all model updates equally affect output alignment and allocates GPU cycles accordingly to balance throughput, latency, and update freshness. Our trace-driven evaluation shows that MACE matches or exceeds continuous retraining while reducing inference latency by up to 63% and maintaining throughput under resource constraints. Compared to periodic retraining, MACE improves latency breakdown across prefill, decode, and finetune stages, and sustains GPU utilization above 85% in NVIDIA AGX Orin. These results demonstrate that iteration-level hybrid scheduling is a promising direction for deploying LLMs with continual learning capabilities on edge platforms.

### 20. MambaCAFU: Hybrid Multi-Scale and Multi-Attention Model with Mamba-Based Fusion for Medical Image Segmentation

**主要机构**: Ho Chi Minh City Open University, University of the Basque Country UPV/EHU
**作者数量**: 4人

**摘要**:
In recent years, deep learning has demonstrated remarkable potential in achieving medical-expertlevel performance for segmenting complex medical imaging tissues and tumors. However, current models face several limitations. Many of these approaches are task-specific, with performance varying significantly across different modalities and anatomical regions. Moreover, achieving an optimal tradeoff between model complexity and performance remains an open challenge, especially in real-world clinical environments where both accuracy and efficiency are critical. To address these limitations, we propose a novel hybrid medical imaging segmentation architecture. Our model features a threebranch encoder that integrates Convolutional Neural Networks (CNNs), Transformers, and a Mambabased Attention Fusion (MAF) mechanism to leverage complementary strengths in capturing local, global, and long-range dependencies. A multi-scale attention-based CNN decoder is then employed to reconstruct fine-grained segmentation maps while preserving contextual consistency. Furthermore, we introduce a co-attention gate, which enhances feature selection by adaptively emphasizing relevant spatial and semantic information across different scales during both encoding and decoding phases. This mechanism facilitates improved feature interaction and cross-scale communication, leading to more precise and robust segmentation outcomes. Extensive experiments across multiple benchmark datasets demonstrate that our approach consistently outperforms existing state-of-the-art methods in terms of both accuracy and generalization, while maintaining computational complexity comparable to that of average models. By effectively balancing efficiency and effectiveness, our architecture represents a practical and scalable solution for diverse medical imaging segmentation tasks. The source codes and trained models will be made publicly available upon acceptance of the paper to encourage reproducibility and further research.

### 21. MECKD: Deep Learning-Based Fall Detection in Multilayer Mobile Edge Computing With Knowledge Distillation

**主要机构**: 
**作者数量**: 5人

**摘要**:
The rising aging population has increased the importance of fall detection (FD) systems as an assistive technology, where deep learning techniques are widely applied to enhance accuracy. FD systems typically use edge devices (EDs) worn by individuals to collect real-time data, which are transmitted to a cloud center (CC) or processed locally. However, this architecture faces challenges such as a limited ED model size and data transmission latency to the CC. Mobile edge computing (MEC), which allows computations at MEC servers deployed between EDs and CC, has been explored to address these challenges. We propose a multilayer MEC (MLMEC) framework to balance accuracy and latency. The MLMEC splits the architecture into stations, each with a neural network model. If front-end equipment cannot detect falls reliably, data are transmitted to a station with more robust back-end computing. The knowledge distillation (KD) approach was employed to improve front-end detection accuracy by allowing high-power back-end stations to provide additional learning experiences, enhancing precision while reducing latency and processing loads. Simulation results demonstrate that the KD approach improved accuracy by 11.65% on the SisFall dataset and 2.78% on the FallAllD dataset. The MLMEC with KD also reduced the data latency rate by 54.15% on the FallAllD dataset and 46.67% on the SisFall dataset compared to the MLMEC without KD. In summary, the MLMEC FD system exhibits improved accuracy and reduced latency.

### 22. Under review as a conference paper at ICLR 2026 MEMMAMBA: RETHINKING MEMORY PATTERNS IN STATE SPACE MODEL

**主要机构**: University of China Beijing, Institute of Artificial Intelligence, Renmin University of China Beijing, Shanghai University of Finance and Economics Shanghai, Shanghai Artificial Intelligence Laboratory Shanghai, School of Statistics Renmin
**作者数量**: 9人

**摘要**:
With the explosive growth of data, long-sequence modeling has become increasingly important in tasks such as natural language processing and bioinformatics. However, existing methods face inherent trade-offs between efficiency and memory. Recurrent neural networks suffer from gradient vanishing and explosion, making them hard to scale. Transformers can model global dependencies but are constrained by quadratic complexity. Recently, selective state-space models such as Mamba have demonstrated high efficiency with O(n) time and O(1) recurrent inference, yet their long-range memory decays exponentially. In this work, we conduct mathematical derivations and information-theoretic analysis to systematically uncover the memory decay mechanism of Mamba, answering a fundamental question: what is the nature of Mamba's long-range memory and how does it retain information? To quantify key information loss, we further introduce horizontal-vertical memory fidelity metrics that capture degradation both within and across layers. Inspired by how humans distill and retain salient information when reading long documents, we propose MemMamba, a novel architectural framework that integrates state summarization mechanism together with crosslayer and cross-token attention, which alleviates long-range forgetting while preserving linear complexity. MemMamba achieves significant improvements over existing Mamba variants and Transformers on long-sequence benchmarks such as PG19-PPL and Passkey Retrieval, while delivering a 48% speedup in inference efficiency. Both theoretical analysis and empirical results demonstrate that Mem-Mamba achieves a breakthrough in the complexity-memory trade-off, offering a new paradigm for ultra-long sequence modeling.

### 23. MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition

**主要机构**: Imperial College London, Imperial College London NatWest AI Research, Stavros Petridis Imperial College London NatWest AI Research
**作者数量**: 9人

**摘要**:
Large language models (LLMs) have recently shown strong potential in audiovisual speech recognition (AVSR), but their high computational demands and sensitivity to token granularity limit their practicality in resource-constrained settings. Token compression methods can reduce inference cost, but they require fixing a compression rate in advance and produce a single fixed-length output, offering no flexibility to balance information density and efficiency at inference time. Matryoshka representation learning (MRL) addresses this by enabling a single model to operate across multiple token granularities, allowing compression rates to be adjusted dynamically. However, current MRL-based methods treat each scale independently during training, limiting cross-scale generalization, robustness at high compression, and interpretability. To overcome these limitations, we propose MoME (Mixture of Matryoshka Experts), a novel framework that integrates sparse Mixture-of-Experts (MoE) into MRL-based LLMs for AVSR. MoME augments a frozen LLM with top-k routed and shared experts, allowing dynamic capacity allocation across scales and modalities. A shared router promotes consistent expert activation across granularities, enabling compressed sequences to benefit from representations learned at lower compression. Experiments on LRS2 and LRS3 demonstrate that MoME achieves state-of-the-art performance across AVSR, ASR, and VSR tasks, while requiring significantly fewer parameters and maintaining robustness under noise. MoME unifies the adaptability of MRL with the efficiency of MoE, offering a scalable and interpretable solution for resource-aware speech recognition.

### 24. MonitorVLM: A Vision-Language Framework for Safety Violation Detection in Mining Operations

**主要机构**: Laboratory for Computational Sensing and Robotics, School of Mechanical Engineering, University of Science and Technology Bei- jing, Johns Hopkins University
**作者数量**: 14人

**摘要**:
Industrial accidents, particularly in high-risk domains such as surface and underground mining, are frequently caused by unsafe worker behaviors. Traditional manual inspection remains labor-intensive, error-prone, and insufficient for large-scale, dynamic environments, highlighting the urgent need for intelligent and automated safety monitoring. In this paper, we present MonitorVLM, a novel vision-language framework designed to detect safety violations directly from surveillance video streams. MonitorVLM introduces three key innovations: (1) a domain-specific violation dataset comprising 9,000 visionquestion-answer (VQA) samples across 40 high-frequency mining regulations, enriched with augmentation and auxiliary detection cues; (2) a clause filter (CF) module that dynamically selects the Top-K most relevant clauses, reducing inference latency by 13.56% while maintaining accuracy; and (3) a behavior magnifier (BM) module that enhances worker regions to improve finegrained action recognition, yielding additional gains of 3.45% in precision and 8.62% in recall. Experimental results demonstrate that MonitorVLM significantly outperforms baseline visionlanguage models, achieving improvements of 22.01% in precision, 34.22% in recall, and 28.37% in F1 score over the 72B unfine-tuned baseline. A lightweight web-based interface further integrates MonitorVLM into practical workflows, enabling automatic violation reporting with video timestamping. This study highlights the potential of multimodal large models to enhance occupational safety monitoring in mining and beyond.

### 25. On Structured State-Space Duality

**主要机构**: Northwestern University, University of California, Center for Foundation Models and Generative AI, Ensemble AI
**作者数量**: 5人

**摘要**:
Structured State-Space Duality (SSD) [Dao & Gu, ICML 2024] is an equivalence between a simple Structured State-Space Model (SSM) and a masked attention mechanism. In particular, a state-space model with a scalar-times-identity state matrix is equivalent to a masked self-attention with a 1-semiseparable causal mask. Consequently, the same sequence transformation (model) has two algorithmic realizations: as a linear-time O(T) recurrence or as a quadratic-time O(T 2) attention. In this note, we formalize and generalize this duality: (i) we extend SSD from the scalar-identity case to general diagonal SSMs (diagonal state matrices); (ii) we show that these diagonal SSMs match the scalar case's training complexity lower bounds while supporting richer dynamics; (iii) we establish a necessary and sufficient condition under which an SSM is equivalent to 1-semiseparable masked attention; and (iv) we show that such duality fails to extend to standard softmax attention due to rank explosion. Together, these results tighten bridge between recurrent SSMs and Transformers, and widen the design space for expressive yet efficient sequence models.

### 26. OPTIMIZED MINIMAL 4D GAUSSIAN SPLATTING

**主要机构**: Seoul National University, Yonsei University, Sungkyunkwan University
**作者数量**: 10人

**摘要**:


### 27. POLYKAN: A POLYHEDRAL ANALYSIS FRAMEWORK FOR PROVABLE AND APPROXIMATELY OPTIMAL KAN COMPRESSION

**主要机构**: -Liverpool University Suzhou, School of Advanced Technology Xi'an Jiaotong
**作者数量**: 1人

**摘要**:
Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to traditional Multi-Layer Perceptrons (MLPs), offering enhanced interpretability and a solid mathematical foundation. However, their parameter efficiency remains a significant challenge for practical deployment. This paper introduces PolyKAN, a novel theoretical framework for KAN compression that provides formal guarantees on both model size reduction and approximation error. By leveraging the inherent piecewise polynomial structure of KANs, we formulate the compression problem as a polyhedral region merging task. We establish a rigorous polyhedral characterization of KANs, develop a complete theory of ϵ-equivalent compression, and design a dynamic programming algorithm that achieves approximately optimal compression under specified error bounds. Our theoretical analysis demonstrates that PolyKAN achieves provably near-optimal compression while maintaining strict error control, with guaranteed global optimality for univariate spline functions. This framework provides the first formal foundation for KAN compression with mathematical guarantees, opening new directions for the efficient deployment of interpretable neural architectures.

### 28. POST-TRAINING QUANTIZATION OF VISION ENCODERS NEEDS PREFIXING REGISTERS

**主要机构**: Dankook University 3 Google
**作者数量**: 6人

**摘要**:
Transformer-based vision encoders-such as CLIP-are central to multimodal intelligence, powering applications from autonomous web agents to robotic control. Since these applications often demand real-time processing of massive visual data, reducing the inference cost of vision encoders is critical. Post-training quantization offers a practical path, but remains challenging even at 8-bit precision due to massive-scale activations (i.e., outliers). In this work, we propose RegCache, a training-free algorithm to mitigate outliers in vision encoders, enabling quantization with significantly smaller accuracy drops. The proposed RegCache introduces outlier-prone yet semantically meaningless prefix tokens to the target vision encoder, which prevents other tokens from having outliers. Notably, we observe that outliers in vision encoders behave differently from those in language models, motivating two technical innovations: middle-layer prefixing and token deletion. Experiments show that our method consistently improves the accuracy of quantized models across both text-supervised and self-supervised vision encoders.

### 29. Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation

**主要机构**: 
**作者数量**: 5人

**摘要**:
Flow matching models generate high-fidelity molecular geometries but incur significant computational costs during inference, requiring hundreds of network evaluations. This inference overhead becomes the primary bottleneck when such models are employed in practice to sample large numbers of molecular candidates. This work discusses a training-free caching strategy that accelerates molecular geometry generation by predicting intermediate hidden states across solver steps. The proposed method operates directly on the SE(3)-equivariant backbone, is compatible with pretrained models, and is orthogonal to existing training-based accelerations and system-level optimizations. Experiments on the GEOM-Drugs dataset demonstrate that caching achieves a twofold reduction in wall-clock inference time at matched sample quality and a speedup of up to 3× compared to the base model with minimal sample quality degradation. Because these gains compound with other optimizations, applying caching alongside other general, lossless optimizations yield as much as a 7× speedup.

### 30. Prompt-Aware Scheduling for Low-Latency LLM Serving

**主要机构**: University of Illinois, Chicago University of Illinois, Chicago Argonne, National Laboratory University of Illinois
**作者数量**: 6人

**摘要**:
Efficient scheduling of large language model (LLM) inference tasks is essential for achieving low latency and high throughput, particularly with the growing use of reasoningcapable LLMs. Traditional strategies like First Come, First-Serve (FCFS) often suffer from Head-of-Line (HOL) blocking, where long-running tasks delay shorter ones queued behind them. In this paper, we introduce PARS, a prompt-aware LLM task scheduler that improves serving efficiency by approximating shortestjob-first (SJF) scheduling through pairwise ranking with margin ranking loss. PARS focuses on impactful scheduling decisions and seamlessly integrates into the state-of-the-art LLM serving system vLLM. It effectively predicts response-length-based task ordering, reducing latency with minimal overhead. Extensive experiments across multiple LLMs and real-world inference datasets show that PARS significantly improves performance, including for reasoning workloads. Furthermore, our cross-model evaluations demonstrate that the design generalizes well, enabling effective scheduling even when predictors are trained on different LLMs.

### 31. PT 2 -LLM: POST-TRAINING TERNARIZATION FOR LARGE LANGUAGE MODELS

**主要机构**: Tencent Hunyuan, Shanghai Jiao Tong University
**作者数量**: 9人

**摘要**:
Large Language Models (LLMs) have shown impressive capabilities across diverse tasks, but their large memory and compute demands hinder deployment. Ternarization has gained attention as a promising compression technique, delivering substantial size reduction and high computational efficiency. However, its potential in the post-training quantization (PTQ) setting remains underexplored, due to the challenge of training-free parameter optimization and the quantization difficulty posed by outliers and dispersed weights. To address these issues, we propose PT 2-LLM, a post-training ternarization framework tailored for LLMs. At its core is an Asymmetric Ternary Quantizer equipped with a two-stage refinement pipeline: (1) Iterative Ternary Fitting (ITF), which alternates between optimal ternary grid construction and flexible rounding to minimize quantization error, and (2) Activation-aware Grid Alignment (AGA), which further refines the ternary grid to better match full-precision outputs. In addition, we propose a plug-and-play Structural Similarity-based Reordering (SSR) strategy that leverages inter-column structural similarity to ease quantization and mitigate outlier effects, further enhancing overall performance. Extensive experiments demonstrate that PT 2-LLM delivers competitive performance against state-of-the-art (SOTA) 2-bit PTQ methods with lower memory cost, while also accelerating both prefill and decoding to achieve end-to-end speedup. The code and models will be available at https://github.com/XIANGLONGYAN/PT2-LLM.

### 32. QUANT-DLLM: POST-TRAINING EXTREME LOW-BIT QUANTIZATION FOR DIFFUSION LARGE LANGUAGE MODELS

**主要机构**: South China University of Technology, Shanghai Jiao Tong University
**作者数量**: 6人

**摘要**:
Diffusion large language models (dLLMs), which offer bidirectional context and flexible masked-denoising generation, are emerging as a compelling alternative to autoregressive (AR) LLMs. However, like AR LLMs, their model sizes continue to grow, motivating weight compression for deployment. Although post-training quantization (PTQ) is effective for AR LLMs, directly transferring it to dLLMs at 2-bit leads to unsatisfactory performance. To tackle these challenges, we propose Quant-dLLM, an ultra-low-bit PTQ framework tailored to dLLMs. Since maskeddenoising activations in dLLMs differ from the fully visible signals assumed by standard PTQ methods, we introduce Masked Calibration Simulation (MCS) to align calibration with the timestep-dependent masking, which yields more reliable calibrations. Moreover, we propose a Data-aware Any-order Quantizer (DAQ) that learns ultra-low-bit weight representations via an optimization algorithm. It performs iterative approximation guided by our simulated calibration data. In addition, under a strict 2-bit budget, we introduce Adaptive Blockwise Mixed Precision (ABMP), a sensitivity-based precision allocation scheme that adaptively assigns bit width across channel groups. When restricted to 2-bit precision, Quant-dLLM consistently achieves higher accuracy than state-of-the-art (SOTA) ARtransfer PTQ methods on dLLMs. The code and models will be available at https://github.com/ZTA2785/Quant-dLLM

### 33. QUANTDEMOIRE: QUANTIZATION WITH OUTLIER AWARE FOR IMAGE DEMOIR ÉING

**主要机构**: Central Media Technology Institute, Shanghai Jiao Tong University
**作者数量**: 7人

**摘要**:
Demoiréing aims to remove moiré artifacts that often occur in images. While recent deep learning-based methods have achieved promising results, they typically require substantial computational resources, limiting their deployment on edge devices. Model quantization offers a compelling solution. However, directly applying existing quantization methods to demoiréing models introduces severe performance degradation. The main reasons are distribution outliers and weakened representations in smooth regions. To address these issues, we propose QuantDemoire, a post-training quantization framework tailored to demoiréing. It contains two key components. First, we introduce an outlier-aware quantizer to reduce errors from outliers. It uses sampling-based range estimation to reduce activation outliers, and keeps a few extreme weights in FP16 with negligible cost. Second, we design a frequency-aware calibration strategy. It emphasizes low-and mid-frequency components during fine-tuning, which mitigates banding artifacts caused by low-bit quantization. Extensive experiments validate that our QuantDemoire achieves large reductions in parameters and computation while maintaining quality. Meanwhile, it outperforms existing quantization methods by over 4 dB on W4A4. Code is released at: https://github.com/zhengchen1999/QuantDemoire.

### 34. QUANTIZATION RANGE ESTIMATION FOR CONVOLUTIONAL NEURAL NETWORKS A PREPRINT

**主要机构**: Xidian University Xi'an, Department of Computer Science
**作者数量**: 4人

**摘要**:
Post-training quantization for reducing the storage of deep neural network models has been demonstrated to be an effective way in various tasks. However, low-bit quantization while maintaining model accuracy is a challenging problem. In this paper, we present a range estimation method to improve the quantization performance for post-training quantization. We model the range estimation into an optimization problem of minimizing quantization errors by layer-wise local loss. We prove this problem is locally convex and present an efficient search algorithm to find the optimal solution. We propose the application of the above search algorithm to the transformed weights space to do further improvement in practice. Our experiments demonstrate that our method outperforms state-of-the-art performance generally on top-1 accuracy for image classification tasks on the ResNet series models and Inception-v3 model. The experimental results show that the proposed method has almost no loss of top-1 accuracy in 8-bit and 6-bit settings for image classifications, and the accuracy of 4-bit quantization is also significantly improved. The code is available at https://github.com/codeiscommitting/REQuant.

### 35. Reactive Transformer (RxT) -Stateful Real-Time Processing for Event-Driven Reactive Language Models

**主要机构**: 
**作者数量**: 1人

**摘要**:
The Transformer architecture has become the de facto standard for Large Language Models (LLMs), demonstrating remarkable capabilities in language understanding and generation. However, its application in conversational AI is fundamentally constrained by its stateless nature and the quadratic computational complexity (O(L 2)) with respect to sequence length L. Current models emulate memory by reprocessing an ever-expanding conversation history with each turn, leading to prohibitive costs and latency in long dialogues. This paper introduces the Reactive Transformer (RxT), a novel architecture designed to overcome these limitations by shifting from a data-driven to an event-driven paradigm. RxT processes each conversational turn as a discrete event in real-time, maintaining context in an integrated, fixed-size Short-Term Memory (STM) system. The architecture features a distinct operational cycle where a generator-decoder produces a response based on the current query and the previous memory state, after which a memory-encoder and a dedicated Memory Attention network asynchronously update the STM with a representation of the complete interaction. This design fundamentally alters the scaling dynamics, reducing the total user-facing cost of a conversation from quadratic (O(N 2 • T)) to linear (O(N • T)) with respect to the number of interactions N. By decoupling response generation from memory updates, RxT achieves low latency, enabling truly real-time, stateful, and economically viable long-form conversations. We validated our architecture with a series of proof-of-concept experiments on synthetic data, demonstrating superior performance and constant-time inference latency compared to a baseline stateless model of comparable size.

### 36. ReTiDe: Real-Time Denoising for Energy-Efficient Motion Picture Processing with FPGAs

**主要机构**: Trinity College Dublin Dublin
**作者数量**: 8人

**摘要**:
Ground truth (b) Noisy (c) ReTiDe (F32) (d) ReTiDe (PTQ) (e) ReTiDe (QAT) Figure 1: In the BSD100 dataset, when the noise intensity is 45, the ground truth, noisy image and denoising results from the FP32 and the quantised ReTiDe.

### 37. SDAKD: Student Discriminator Assisted Knowledge Distillation for Super-Resolution Generative Adversarial Networks

**主要机构**: CERTH-ITI Thessaloniki
**作者数量**: 1人

**摘要**:
Generative Adversarial Networks (GANs) achieve excellent performance in generative tasks, such as image superresolution, but their computational requirements make difficult their deployment on resource-constrained devices. While knowledge distillation is a promising research direction for GAN compression, effectively training a smaller student generator is challenging due to the capacity mismatch between the student generator and the teacher discriminator. In this work, we propose Student Discriminator Assisted Knowledge Distillation (SDAKD), a novel GAN distillation methodology that introduces a student discriminator to mitigate this capacity mismatch. SDAKD follows a three-stage training strategy, and integrates an adapted feature map distillation approach in its last two training stages. We evaluated SDAKD on two well-performing super-resolution GANs, GCFSR and Real-ESRGAN. Our experiments demonstrate consistent improvements over the baselines and SOTA GAN knowledge distillation methods. The SDAKD source code will be made openly available upon acceptance of the paper.

### 38. SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size

**主要机构**: Beijing National Research Center for Information Science and Technology, Dept of Electronic Engineering, Tsinghua University, ORCID (Junhao Xia
**作者数量**: 5人

**摘要**:
Large language models (LLMs) face significant computational and memory challenges, making extremely low-bit quantization crucial for their efficient deployment. In this work, we introduce SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size, a novel framework that enables extremely low-bit quantization of LLMs while preserving their linguistic reasoning capabilities. A distinctive feature of SDQ-LLM is the continuous adjustability of the Over-Sampling Ratio (OSR), enabling dynamic adaptation to memory or VRAM constraints by selecting fractional OSR (e.g., 2.5×) for an optimal trade-off between model size and accuracy. SDQ-LLM uses upsampling combined with Sigma-Delta Quantizer to binarize or ternarize LLMs' weights, encoding high-precision parameters into 1-bit or 1.58-bit representations, replacing the multiplication operations within linear layers with addition. This approach significantly enhances inference efficiency under extremely low-bit quantization. To further reduce the loss of quantization precision, we incorporate Hadamard-based weight smoothing prior to quantization, improving the stability and robustness of the weight representations. Furthermore, to fully leverage the continuity of the OSR and reduce precision loss, recognizing the correlation between quantization sensitivity and weight variance, we propose a fine-grained, layer-and linear-wise OSR allocation strategy, MultiOSR. This strategy distributes OSR both across layers and within each layer, based on weight variance and parameter scale. Finally, extensive experiments on OPT and LLaMA model families demonstrate that SDQ-LLM achieves a more efficient and high-precision performance even under highly aggressive low-OSR settings. Our code is available at https://github.com/Dreamlittlecat/LLM-Quant-Factory.

### 39. SELF SPECULATIVE DECODING FOR DIFFUSION LARGE LANGUAGE MODELS

**主要机构**: Huawei Technologies Ltd, School of AI, Shanghai Jiao Tong University, Xidian University, University of Science and Technology of China
**作者数量**: 8人

**摘要**:
Diffusion-based Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive models, offering unique advantages through bidirectional attention and parallel generation paradigms. However, the generation results of current parallel decoding methods deviate from stepwise decoding, introducing potential performance degradation, which limits their practical deployment. To address this problem, we propose Self Speculative Decoding (SSD), a lossless inference acceleration method that leverages the dLLM itself as both speculative decoding drafter and verifier without auxiliary modules. SSD introduces a selfdrafting mechanism where the model generates predictions for multiple positions, then verifies them through hierarchical verification trees in a single forward pass. Unlike traditional speculative decoding that requires separate draft models, SSD eliminates model redundancy and memory overhead by exploiting the dLLM's inherent parallel prediction capability for multiple positions. This self-speculative approach allows the model to progressively verify and accept multiple tokens in a single forward pass. Our experiments demonstrate that SSD achieves up to 3.46× speedup while keeping the output identical to stepwise decoding on open source models such as LLaDA and Dream. Code will be made publicly available on GitHub.

### 40. SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling

**主要机构**: Rutgers University -New Brunswick
**作者数量**: 1人

**摘要**:
Mixture-of-Experts (MoE) layers scale transformers by routing tokens to a sparse subset of feed-forward experts. Token-level routing, however, assigns an entire semantic spectrum to each expert, creating capacity bottlenecks, load-balancing pathologies, and limited specialisation. We introduce SliceMoE, an architecture that routes contiguous slices of a token's hidden vector. A d-dimensional embedding is partitioned into S slices, and for each slice, a lightweight shared router predicts the top-k experts. Experts operate on their assigned slices independently, and outputs are reassembled , maintaining per-token FLOP efficiency. Because slices from different tokens interleave within an expert, utilisation is naturally smoother. We propose a slice-level capacity loss, cross-slice dropout, and efficient fused batched-GEMM kernels. Experiments on WikiText-103 language modelling, WMT En-De translation, and three text-classification datasets show SliceMoE attains up to 1.7x faster inference than dense baselines, 12-18% lower perplexity than parameter-matched token-MoE, and improved expert balance, with interpretable expertise over syntactic versus semantic sub-spaces.

### 41. Sliding Window Attention for Learned Video Compression

**主要机构**: Multimedia Communications and Signal Processing Friedrich-Alexander University
**作者数量**: 2人

**摘要**:
To manage the complexity of transformers in video compression, local attention mechanisms are a practical necessity. The common approach of partitioning frames into patches, however, creates architectural flaws like irregular receptive fields. When adapted for temporal autoregressive models, this paradigm, exemplified by the Video Compression Transformer (VCT), also necessitates computationally redundant overlapping windows. This work introduces 3D Sliding Window Attention (SWA), a patchless form of local attention. By enabling a decoderonly architecture that unifies spatial and temporal context processing, and by providing a uniform receptive field, our method significantly improves rate-distortion performance, achieving Bjøntegaard Delta-rate savings of up to 18.6 % against the VCT baseline. Simultaneously, by eliminating the need for overlapping windows, our method reduces overall decoder complexity by a factor of 2.8, while its entropy model is nearly 3.5 times more efficient. We further analyze our model's behavior and show that while it benefits from long-range temporal context, excessive context can degrade performance.

### 42. Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs

**主要机构**: Northeastern University, University of Southern
**作者数量**: 2人

**摘要**:
Recent evidence (indicates that Small Language Models (SLMs; ≲1-12B params, occasionally ~20B) are not only sufficient but often superior for agentic workloads such as retrievalaugmented generation (RAG), robust function calling, structured decoding, and programmatic tool use. NVIDIA argues that SLMs are the future of agentic AI and edge inference, emphasizing cost/latency/energy advantages and the role of guided decoding and tool execution in shifting the objective from open-ended generalization to schema-and API-constrained accuracy. We synthesize results across open and proprietary SLMs (e.g., Phi-4-Mini, Qwen-2.5-7B, Gemma-2-9B, Llama-3.2-1B/3B, Ministral-3B/8B, Apple ondevice ~3B, DeepSeek-R1-Distill 1.5-70B) and connect them to modern evaluation (BFCL V3/V4; StableToolBench) and serving stacks (vLLM/SGLang/TensorRT-LLM + XGrammar/Outlines). We formalize SLM-default, LLM-fallback systems with uncertainty-aware routing and verifiers, and propose engineering metrics (e.g., Cost per Successful task (CPS), schema validity, executable-call rate, p50/p95 latency, energy/request). Guided decoding and validator-first tool use allow SLMs to match or surpass LLMs at a 10×-100× lower token cost on today's APIs.

### 43. Speak, Edit, Repeat: High-Fidelity Voice Editing and Zero-Shot TTS with Cross-Attentive Mamba

**主要机构**: ITMO University
**作者数量**: 4人

**摘要**:
We introduce MAVE (Mamba with Cross-Attention for Voice Editing and Synthesis), a novel autoregressive architecture for text-conditioned voice editing and high-fidelity text-to-speech (TTS) synthesis, built on a cross-attentive Mamba backbone. MAVE achieves state-of-the-art performance in speech editing and very competitive results in zero-shot TTS, while not being explicitly trained on the latter task, outperforming leading autoregressive and diffusion models on diverse, real-world audio. By integrating Mamba for efficient audio sequence modeling with cross-attention for precise text-acoustic alignment, MAVE enables context-aware voice editing with exceptional naturalness and speaker consistency. In pairwise human evaluations on a random 40-sample subset of the RealEdit benchmark (400 judgments), 57.2% of listeners rated MAVE-edited speech as perceptually equal to the original, while 24.8% prefered the original and 18.0% MAVE-demonstrating that in the majority of cases edits are indistinguishable from the source. MAVE compares favorably with Voice-Craft and FluentSpeech both on pairwise comparisons and standalone mean opinion score (MOS) evaluations. For zero-shot TTS, MAVE exceeds VoiceCraft in both speaker similarity and naturalness, without requiring multiple inference runs or post-processing. Remarkably, these quality gains come with a significantly lower memory cost and approximately the same latency: MAVE requires ∼6× less memory than VoiceCraft during inference on utterances from the RealEdit database (mean duration: 6.21s, A100, FP16, batch size 1). Our results demonstrate that MAVE establishes a new standard for flexible, high-fidelity voice editing and synthesis through the synergistic integration of structured state-space modeling and cross-modal attention.

### 44. SPEAR: Soft Prompt Enhanced Anomaly Recognition for Time Series Data

**主要机构**: University of Calgary, Department of Electrical and Software Engineering
**作者数量**: 5人

**摘要**:
Time series anomaly detection plays a crucial role in a wide range of fields, such as healthcare and internet traffic monitoring. The emergence of large language models (LLMs) offers new opportunities for detecting anomalies in the ubiquitous time series data. Traditional approaches struggle with variable-length time series sequences and context-based anomalies. We propose Soft Prompt Enhanced Anomaly Recognition (SPEAR), a novel approach to leverage LLMs for anomaly detection with soft prompts and quantization. Our methodology involves quantizing and transforming the time series data into input embeddings and combining them with learnable soft prompt embeddings. These combined embeddings are then fed into a frozen LLM. The soft prompts are updated iteratively based on a cross-entropy loss, allowing the model to adapt to time series anomaly detection. The use of soft prompts helps adapt LLMs effectively to time series tasks, while quantization ensures optimal handling of sequences, as LLMs are designed to handle discrete sequences. Our experimental results demonstrate that soft prompts effectively increase LLMs' performance in downstream tasks regarding time series anomaly detection.

### 45. Speculative Actions: A Lossless Framework for Faster Agentic Systems

**主要机构**: Columbia University New York
**作者数量**: 6人

**摘要**:
Despite growing interest in AI agents across industry and academia, their execution in an environment is often slow, hampering training, evaluation, and deployment. For example, a game of chess between two state-of-the-art agents may take hours. A critical bottleneck is that agent behavior unfolds sequentially: each action requires an API call, and these calls can be time-consuming. Inspired by speculative execution in microprocessors and speculative decoding in LLM inference, we propose speculative actions, a lossless framework for general agentic systems that predicts likely actions using faster models, enabling multiple steps to be executed in parallel. We evaluate this framework across three agentic environments: gaming, e-commerce, web search, and a "lossy" extension for an operating systems environment. In all cases, speculative actions achieve substantial accuracy in next-action prediction (up to 55%), translating into significant reductions in end-to-end latency. Moreover, performance can be further improved through stronger guessing models, top-K action prediction, multi-step speculation, and uncertainty-aware optimization, opening a promising path toward deploying low-latency agentic systems in the real world.

### 46. StructPrune: Structured Global Pruning asymptotics with O( √ N ) GPU Memory

**主要机构**: Emory University
**作者数量**: 3人

**摘要**:
Pruning is critical for scaling large language models (LLMs). Global pruning achieves strong performance but requires O(N) memory, which is infeasible for billion-parameter models. Local pruning reduces GPU memory usage to that of a single layer by pruning layers independently, but it neglects inter-layer dependencies and often leads to suboptimal performance in high-sparsity regimes. Unlike unstructured pruning, structured pruning produces regular sparsity patterns that align well with GPU kernels and library optimizations, making it more hardware-efficient. However, structured pruning typically relies on global pruning, since structured patterns are more prone to severe performance degradation under local optimization. To jointly achieve structured pruning and the memory efficiency of local pruning, we propose a divide-and-conquer strategy that decomposes the global pruning problem into coordinated subproblems across different modules, each of which fits within limited GPU memory. Building on this idea, we design STRUPRUNE, an ADMM-based framework that integrates structured sparsity into the pruning process, combining the memory efficiency of local pruning with the hardware compatibility of structured methods. We derive a closed-form analytical solution for structured pruning masks that provides an explicit rule for layer-wise sparsity allocation, and further develop an energy-based asymptotic framework yielding a softmax-form allocation scheme that simplifies optimization while adapting to heterogeneous layer importance. Experiments demonstrate that STRUPRUNE matches the perplexity of global structured pruning while reducing memory cost from O(N) to O(√ N), enabling practical deployment at the billion-parameter scale.

### 47. UNIPRUNING: UNIFYING LOCAL METRIC AND GLOBAL FEEDBACK FOR SCALABLE SPARSE LLMS

**主要机构**: Laboratory * Equal contribution, Fudan University
**作者数量**: 5人

**摘要**:
Large Language Models (LLMs) achieve strong performance across diverse tasks but face prohibitive computational and memory costs. Pruning offers a promising path by inducing sparsity while preserving architectural flexibility. However, existing methods struggle to balance efficiency and robustness: local metric approaches prune layer by layer but often collapse under high sparsity, whereas global feedback methods enforce consistency at the cost of expensive weight updates or restrictive semi-structured formats. We present UniPruning, a unified post-training pruning framework that combines the speed of local saliency metrics with the stability of global coordination, enabled by a mirror descent based optimization, all without updating model weights. UniPruning leverages fast layer-wise scoring and a lightweight global controller to allocate a single sparsity budget, supporting both unstructured and semi-structured N :M pruning within one framework. After a brief calibration, it can generate pruning masks for arbitrary sparsity levels in one shot, and adapts seamlessly to hardware-aware constraints. Extensive experiments on multiple pretrained LLM families and standard benchmarks show that UniPruning consistently delivers competitive or superior perplexity and zero-shot accuracy. Ablation studies further highlight the importance of mirror descent and local saliency anchoring. Overall, UniPruning provides an efficient, principled, and scalable solution for sparsifying large-scale LLMs. Our code is available at: https://github.com/RainbowQTT/UniPruning.
