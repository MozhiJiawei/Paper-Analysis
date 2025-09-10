# AI推理加速技术论文分析报告
生成时间: 2025-09-10 19:43:57
分析论文数量: 11篇

## 论文技术简报

### 1. Accelerating Local AI on Consumer GPUs: A Hardware-Aware Dynamic Strategy for YOLOv10s

Leo University Tampa发布了加速消费级GPU上本地AI的论文，使用Two-Pass Adaptive Inference算法，解决了消费级硬件上目标检测器基准性能与实际可用性的差距问题，在COCO数据集上比PyTorch Early-Exit基线提速1.85倍，mAP损失仅5.51%。

### 2. ALICE: An Interpretable Neural Architecture for Generalization in Substitution Ciphers

普林斯顿大学发布了ALICE: An Interpretable Neural Architecture for Generalization in Substitution Ciphers论文，使用ALICE可解释神经架构（含编码器Transformer及基于Gumbel-Sinkhorn方法的双射解码头），解决了替换密码解密中神经网络在组合复杂领域的泛化问题（需从26!种可能映射中选择且无明确密码访问），达成了准确率和速度新突破，仅训练约1500个独特密码即泛化到未见过的密码（占可能密码空间的3.7×10-24）的效果。

### 3. Astra: A Multi-Agent System for GPU Kernel Performance Optimization

斯坦福大学发布了Astra论文，使用首个基于LLM的多智能体系统，解决了GPU内核优化需大量手动调优的挑战，达成了在SGLang内核上用零样本提示OpenAI o4-mini平均加速1.32倍的效果。

### 4. DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference

悉尼大学发布了DuoServe-MoE论文，使用双阶段专家预取与缓存调度策略（预填充阶段双流CUDA管道重叠预取与计算、解码阶段轻量级预测器预取激活专家），解决了MoE LLM推理中因专家权重多导致的GPU内存压力及预填充/解码阶段统一调度的次优延迟与内存使用问题，达成端到端延迟提升1.42-7.54倍、峰值内存仅为全模型大小15%的效果。

### 5. EXPLAINING HOW QUANTIZATION DISPARATELY SKEWS A MODEL

北卡罗来纳州立大学发布了《解释量化如何导致模型差异偏差》论文，使用混合精度量化感知训练（QAT）结合数据集采样和加权损失函数，解决了量化加剧的不同群体（尤其是少数群体）差异影响问题，达成了量化神经网络公平部署的效果。

### 6. MEGS 2 : MEMORY-EFFICIENT GAUSSIAN SPLATTING VIA SPHERICAL GAUSSIANS AND UNIFIED PRUNING

研究团队发布了MEGS 2论文，使用球形高斯和统一剪枝技术，解决了高斯 splatting 的内存效率问题，达成了内存高效的效果。

### 7. MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?

香港大学发布了MoE-Compression论文，使用误差有界的有损压缩算法（如SZ3、CuSZp），解决了MoE模型在有限GPU内存下的非激活专家数据传输开销问题，达成了分析不同层专家压缩误差对推理精度影响（浅层影响小、中层影响大、深层有时提升精度）的效果

### 8. Systematic Optimization of Open Source Large Language Models for Mathematical Reasoning

Sudhir Dhekane Dwarkadas J. Sanghvi College of Engineering发布了开源大语言模型数学推理系统优化论文，使用通过系统优化温度、推理步骤等参数的整体优化框架，解决了开源大语言模型数学推理的效率与性能优化问题，达成了计算成本降低29.4%、推理速度提升23.9%、优化成功率100%的效果

### 9. TOWARDS GENERALIZED ROUTING: MODEL AND AGENT ORCHESTRATION FOR ADAPTIVE AND EFFI-CIENT INFERENCE

中国移动研究院发布了TOWARDS GENERALIZED ROUTING论文，使用MoMA（Mixture of Models and Agents）框架（整合LLM与代理路由、精确意图识别、自适应路由策略及上下文感知状态机与动态掩蔽的代理选择），解决了用户查询多样时如何准确路由到合适执行单元以优化性能与效率的问题，达成了成本效率和可扩展性优于现有方法的效果。

### 10. Toward Lifelong-Sustainable Electronic-Photonic AI Systems via Extreme Efficiency, Reconfigurability, and Robustness

亚利桑那州立大学发布了关于终身可持续电子光子AI系统的论文，使用电子光子设计自动化（EPDA）与跨层协同设计技术，解决了传统电子平台在大规模AI下的能耗、带宽及扩展限制问题，达成了提升面积效率、可重构性和鲁棒性，实现终身可持续并减少隐含碳足迹的效果。

### 11. WORD2SPIKE: POISSON RATE CODING FOR ASSOCIA-TIVE MEMORIES AND NEUROMORPHIC ALGORITHMS

莱斯大学与耶鲁大学发布了Word2Spike论文，使用结合连续词嵌入和神经形态架构的泊松速率编码机制（通过泊松过程将多维词向量转换为尖峰吸引子状态并经BitNet b1.58量化），解决了神经形态系统中语义编码的高效性与抗噪声问题，达成保持97%语义相似度、100%重建精度及100%类比性能的效果。

## 论文详细信息

### 1. Accelerating Local AI on Consumer GPUs: A Hardware-Aware Dynamic Strategy for YOLOv10s

**主要机构**: Department of Electrical and Computer Engineering, Leo University Tampa, School of Computing and Information Sciences, Florida International University Miami, Department of Computer and Information Systems Saint
**作者数量**: 3人

**摘要**:
As local AI grows in popularity, there is a critical gap between the benchmark performance of object detectors and their practical viability on consumer-grade hardware. While models like YOLOv10s promise real-time speeds, these metrics are typically achieved on high-power, desktop-class GPUs. This paper reveals that on resource-constrained systems, such as laptops with RTX 4060 GPUs, performance is not computebound but is instead dominated by system-level bottlenecks, as illustrated by a simple bottleneck test. To overcome this hardware-level constraint, we introduce a Two-Pass Adaptive Inference algorithm, a model-independent approach that requires no architectural changes. This study mainly focuses on 'adaptive' inference strategies and undertakes a comparative analysis of architectural early-exit and resolution-adaptive routing, highlighting their respective trade-offs within a unified evaluation framework. The system uses a fast, low-resolution pass and only escalates to a high-resolution model pass when detection confidence is low. On a 5000-image COCO dataset, our method achieves a 1.85x speedup over a PyTorch Early-Exit baseline, with a modest mAP loss of 5.51%. This work provides a practical and reproducible blueprint for deploying high-performance, real-time AI on consumer-grade devices by shifting the focus from pure model optimization to hardware-aware inference strategies that maximize throughput.

### 2. ALICE: An Interpretable Neural Architecture for Generalization in Substitution Ciphers

**主要机构**: Princeton University Princeton
**作者数量**: 2人

**摘要**:
We present cryptogram solving as an ideal testbed for studying neural network generalization in combinatorially complex domains. In this task, models must decrypt text encoded with substitution ciphers, choosing from 26! possible mappings without explicit access to the cipher. We develop ALICE (an Architecture for Learning Interpretable Cryptogram dEcipherment): a simple encoder-only Transformer that sets a new state-of-the-art for both accuracy and speed on this decryption problem. Surprisingly, ALICE generalizes to unseen ciphers after training on only ∼1500 unique ciphers, a minute fraction (3.7 × 10-24) of the possible cipher space. To enhance interpretability, we introduce a novel bijective decoding head that explicitly models permutations via the Gumbel-Sinkhorn method, enabling direct extraction of learned cipher mappings. Through early exit analysis, we reveal how ALICE progressively refines its predictions in a way that appears to mirror common human strategies for this task: early layers employ frequency-based heuristics, middle layers form word structures, and final layers correct individual characters. Our architectural innovations and analysis methods extend beyond cryptograms to any domain with bijective mappings and combinatorial structure, offering new insights into neural network generalization and interpretability. * Equal contribution. Preprint. Under review.

### 3. Astra: A Multi-Agent System for GPU Kernel Performance Optimization

**主要机构**: Stanford University, Nanjing University, Shanghai Jiao, Tong University
**作者数量**: 8人

**摘要**:
GPU kernel optimization has long been a central challenge at the intersection of high-performance computing and machine learning. Efficient kernels are crucial for accelerating large language model (LLM) training and serving, yet attaining high performance typically requires extensive manual tuning. Compiler-based systems reduce some of this burden, but still demand substantial manual design and engineering effort. Recently, researchers have explored using LLMs for GPU kernel generation, though prior work has largely focused on translating high-level PyTorch modules into CUDA code. In this work, we introduce Astra, the first LLM-based multi-agent system for GPU kernel optimization. Unlike previous approaches, Astra starts from existing CUDA implementations extracted from SGLang, a widely deployed framework for serving LLMs, rather than treating PyTorch modules as the specification. Within Astra, specialized LLM agents collaborate through iterative code generation, testing, profiling, and planning to produce kernels that are both correct and high-performance. On kernels from SGLang, Astra achieves an average speedup of 1.32× using zero-shot prompting with OpenAI o4-mini. A detailed case study further demonstrates that LLMs can autonomously apply loop transformations, optimize memory access patterns, exploit CUDA intrinsics, and leverage fast math operations to yield substantial performance gains. Our work highlights multi-agent LLM systems as a promising new paradigm for GPU kernel optimization.

### 4. DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference

**主要机构**: School of Electrical and Computer Engineering, The University of Sydney
**作者数量**: 5人

**摘要**:
Large Language Models (LLMs) have demonstrated impressive performance across a wide range of deep learning tasks. Mixture of Experts (MoE) further enhances their capabilities by increasing model width through sparsely activated expert branches, which keeps inference computation efficient. However, the large number of expert weights introduces significant GPU memory pressure, especially in resource-constrained environments such as single-GPU servers. More importantly, MoE inference consists of two fundamentally different stages: a prefill stage where most experts are activated densely, and a decode stage where only a few experts are triggered sparsely. Treating these stages with a uniform scheduling strategy often leads to suboptimal latency and memory usage. To address this, we propose DuoServe-MoE, an inference serving system that explicitly separates prefill and decode stages and applies tailored expert scheduling strategies to each. In the prefill stage, DuoServe-MoE uses a two-stream CUDA pipeline that overlaps expert weight prefetching with the computation of non-MoE layers, limiting expert residency in GPU memory. In the decode stage, a lightweight layer-level predictor trained offline from activation traces is used to prefetch only the most likely activated experts, without requiring any changes to the model. Experiments on 4-bit Mixtral-8×7B and 8×22B models show that DuoServe-MoE improves endto-end latency by 1.42 to 7.54 times while keeping peak memory usage at only 15 percent of the full model size.

### 5. EXPLAINING HOW QUANTIZATION DISPARATELY SKEWS A MODEL

**主要机构**: Department of Computer Science, North Carolina State University Raleigh
**作者数量**: 2人

**摘要**:
Post Training Quantization (PTQ) is widely adopted due to its high compression capacity and speed with minimal impact on accuracy. However, we observed that disparate impacts are exacerbated by quantization, especially for minority groups. Our analysis explains that in the course of quantization there is a chain of factors attributed to a disparate impact across groups during forward and backward passes. We explore how the changes in weights and activations induced by quantization cause cascaded impacts in the network, resulting in logits with lower variance, increased loss, and compromised group accuracies. We extend our study to verify the influence of these impacts on group gradient norms and eigenvalues of the Hessian matrix, providing insights into the state of the network from an optimization point of view. To mitigate these effects, we propose integrating mixed precision Quantization Aware Training (QAT) with dataset sampling methods and weighted loss functions, therefore providing fair deployment of quantized neural networks.

### 6. MEGS 2 : MEMORY-EFFICIENT GAUSSIAN SPLATTING VIA SPHERICAL GAUSSIANS AND UNIFIED PRUNING

**主要机构**: 
**作者数量**: 9人

**摘要**:


### 7. MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?

**主要机构**: Department of Computer Science and Engineering, The University of Hong Kong Hong Kong, Stevens Institute of Technology, University of California, Department of Computer Science, Kong Polytechnic University Hong Kong, Division Argonne National Laboratory, Mathematics and Computer Science, LSCM R&D Center, Department of Computing Hong
**作者数量**: 11人

**摘要**:
With the widespread application of Mixture of Experts (MoE) reasoning models in the field of LLM learning, efficiently serving MoE models under limited GPU memory constraints has emerged as a significant challenge. Offloading the non-activated experts to main memory has been identified as an efficient approach to address such a problem, while it brings the challenges of transferring the expert between the GPU memory and main memory. We need to explore an efficient approach to compress the expert and analyze how the compression error affects the inference performance. To bridge this gap, we propose employing error-bounded lossy compression algorithms (such as SZ3 and CuSZp) to compress non-activated experts, thereby reducing data transfer overhead during MoE inference. We conduct extensive experiments across various benchmarks and present a comprehensive analysis of how compression-induced errors in different experts affect overall inference accuracy. The results indicate that experts in the shallow layers, which are primarily responsible for the attention mechanism and the transformation of input tokens into vector representations, exhibit minimal degradation in inference accuracy when subjected to bounded errors. In contrast, errors in the middle-layer experts, which are central to model reasoning, significantly impair inference accuracy. Interestingly, introducing bounded errors in the deep-layer experts, which are mainly responsible for instruction following and output integration, can sometimes lead to improvements in inference accuracy.

### 8. Systematic Optimization of Open Source Large Language Models for Mathematical Reasoning

**主要机构**: Sudhir Dhekane Dwarkadas J. Sanghvi College of Engineering
**作者数量**: 5人

**摘要**:
This paper presents a practical investigation into fine-tuning model parameters for mathematical reasoning tasks through experimenting with various configurations including randomness control, reasoning depth, and sampling strategies, careful tuning demonstrates substantial improvements in efficiency as well as performance. A holistically optimized framework is introduced for five state-of-the-art models on mathematical reasoning tasks, exhibiting significant performance boosts while maintaining solution correctness. Through systematic parameter optimization across Qwen2.5-72B, Llama-3.1-70B, DeepSeek-V3, Mixtral-8x22B, and Yi-Lightning, consistent efficiency gains are demonstrated with 100% optimization success rate. The methodology achieves an average 29.4% reduction in computational cost and 23.9% improvement in inference speed across all tested models. This framework systematically searches parameter spaces including temperature (0.1-0.5), reasoning steps (4-12), planning periods (1-4), and nucleus sampling (0.85-0.98), determining optimal configurations through testing on mathematical reasoning benchmarks. Critical findings show that lower temperature regimes (0.1-0.4) and reduced reasoning steps (4-6) consistently enhance efficiency without compromising accuracy. DeepSeek-V3 achieves the highest accuracy at 98%, while Mixtral-8x22B delivers the most costeffective performance at 361.5 tokens per accurate response. Key contributions include: (1) the first comprehensive optimization study for five diverse SOTA models in mathematical reasoning, (2) a standardized production-oriented parameter optimization framework, (3) discovery of universal optimization trends applicable across model architectures, and (4) production-ready configurations with extensive performance characterization.

### 9. TOWARDS GENERALIZED ROUTING: MODEL AND AGENT ORCHESTRATION FOR ADAPTIVE AND EFFI-CIENT INFERENCE

**主要机构**: China Mobile Research Institute, JIUTIAN Team
**作者数量**: 9人

**摘要**:
The rapid advancement of large language models (LLMs) and domain-specific AI agents has greatly expanded the ecosystem of AI-powered services. User queries, however, are highly diverse and often span multiple domains and task types, resulting in a complex and heterogeneous landscape. This diversity presents a fundamental routing challenge: how to accurately direct each query to an appropriate execution unit while optimizing both performance and efficiency. To address this, we propose MoMA (Mixture of Models and Agents), a generalized routing framework that integrates both LLM and agent-based routing. Built upon a deep understanding of model and agent capabilities, MoMA effectively handles diverse queries through precise intent recognition and adaptive routing strategies, achieving an optimal balance between efficiency and cost. Specifically, we construct a detailed training dataset to profile the capabilities of various LLMs under different routing model structures, identifying the most suitable tasks for each LLM. During inference, queries are dynamically routed to the LLM with the best cost-performance efficiency. We also introduce an efficient agent selection strategy based on a context-aware state machine and dynamic masking. Experimental results demonstrate that the MoMA router offers superior cost-efficiency and scalability compared to existing approaches.

### 10. Toward Lifelong-Sustainable Electronic-Photonic AI Systems via Extreme Efficiency, Reconfigurability, and Robustness

**主要机构**: Arizona State University
**作者数量**: 5人

**摘要**:
The relentless growth of large-scale artificial intelligence (AI) has created unprecedented demand for computational power, straining the energy, bandwidth, and scaling limits of conventional electronic platforms. Electronic-photonic integrated circuits (EPICs) have emerged as a compelling platform for nextgeneration AI systems, offering inherent advantages in ultra-high bandwidth, low latency, and energy efficiency for computing and interconnection. Beyond performance, EPICs also hold unique promises for sustainability. Fabricated in relaxed process nodes with fewer metal layers and lower defect densities, photonic devices naturally reduce embodied carbon footprint (CFP) compared to advanced digital electronic integrated circuits, while delivering orders-of-magnitude higher computing performance and interconnect bandwidth. To further advance the sustainability of photonic AI systems, we explore how electronic-photonic design automation (EPDA) and cross-layer co-design methodologies can amplify these inherent benefits. We present how advanced EPDA tools enable more compact layout generation, reducing both chip area and metal layer usage. We will also demonstrate how cross-layer device-circuit-architecture co-design unlocks new sustainability gains for photonic hardware: ultracompact photonic circuit designs that minimize chip area cost, reconfigurable hardware topology that adapts to evolving AI workloads, and intelligent resilience mechanisms that prolong lifetime by tolerating variations and faults. By uniting intrinsic photonic efficiency with EPDA-and co-design-driven gains in area efficiency, reconfigurability, and robustness, we outline a vision for lifelong-sustainable electronic-photonic AI systems. This perspective highlights how EPIC AI systems can simultaneously meet the performance demands of modern AI and the urgent imperative for sustainable computing.

### 11. WORD2SPIKE: POISSON RATE CODING FOR ASSOCIA-TIVE MEMORIES AND NEUROMORPHIC ALGORITHMS

**主要机构**: Department of Bioengineering Rice University Houston, Department of Computer Science, Yale University New Haven
**作者数量**: 4人

**摘要**:
Spiking neural networks offer a promising path toward energy-efficient, brainlike associative memory. This paper introduces Word2Spike, a novel rate coding mechanism that combines continuous word embeddings and neuromorphic architectures. We develop a one-to-one mapping that converts multi-dimensional word vectors into spike-based attractor states using Poisson processes. Using BitNet b1.58 quantization, we maintain 97% semantic similarity of continuous embeddings on SimLex-999 while achieving 100% reconstruction accuracy on 10,000 words from OpenAI's text-embedding-3-large. We preserve analogy performance (100% of original embedding performance) even under intentionally introduced noise, indicating a resilient mechanism for semantic encoding in neuromorphic systems. Next steps include integrating the mapping with spiking transformers and liquid state machines (resembling Hopfield Networks) for further evaluation.
