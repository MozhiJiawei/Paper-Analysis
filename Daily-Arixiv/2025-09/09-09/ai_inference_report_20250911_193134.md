# AI推理加速技术论文分析报告
生成时间: 2025-09-11 19:31:34
分析论文数量: 11篇

## 论文技术简报

### 1. Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism

研究团队发布了Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism论文，使用MoEpic系统及自适应专家分割机制与分治算法，解决了MoE模型推理时GPU内存需求大及推理速度慢的问题，达成了节省约一半GPU成本、推理延迟降低37.51%~65.73%的效果。

### 2. A 410 GFLOP/s, 64 RISC-V Cores, 204.8 GBps Shared-Memory Cluster in 12 nm FinFET with Systolic Execution Support for Efficient B5G/6G AI-Enhanced O-RAN

ETH Zürich发布了HeartStream论文，使用64核RISC-V共享内存集群及定制化架构（支持复杂指令、硬件脉动队列），解决了B5G/6G AI增强O-RAN的高效基带处理问题，达成了峰值410 GFLOP/s性能、204.8 GBps带宽及关键基带内核能效提升1.89×的效果

### 3. BcQLM: Efficient Vision-Language Understanding with Distilled Q-Gated Cross-Modal Fusion

Durham University发布了BcQLM论文，使用BreezeCLIP紧凑视觉语言编码器及蒸馏Q-Gated跨模态融合的轻量级框架，解决了多模态大语言模型在资源受限环境部署的计算成本高问题，达成总参数12亿、显著降低计算成本并实现与标准尺寸模型相当性能的效果。

### 4. Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video

上海交通大学发布了《Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video》论文，使用基于Transformer的架构结合低比特率向量量化信息瓶颈与去噪扩散模型的自监督框架，解决了视频动态运动与静态内容分离问题，达成了在真实世界谈话头视频上实现运动迁移和自回归运动生成并能推广到其他视频类型的效果。

### 5. Efficient Decoding Methods for Language Models on Encrypted Data

IBM Research发布了Efficient Decoding Methods for Language Models on Encrypted Data论文，使用CutMax算法及HE兼容的nucleus sampling方法，解决了语言模型在加密数据上解码计算昂贵的瓶颈问题，达成了延迟降低24×-35×、实现高效安全文本生成的效果。

### 6. EvolKV: Evolutionary KV Cache Compression for LLM Inference

ETH Zurich发布了EvolKV论文，使用进化搜索的自适应分层任务驱动KV缓存压缩框架，解决了现有启发式KV缓存压缩方法忽略层特征与任务性能相互作用导致的泛化能力下降问题，达成了在11个任务上优于基线、GSM8K提升7个百分点、代码补全仅用1.5%预算超全缓存性能的效果

### 7. Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism

中山大学发布了Hetis论文，使用细粒度动态并行设计（包括选择性并行计算密集型操作、头粒度动态分配Attention计算及在线负载调度策略），解决了异构GPU集群中LLM服务因内存容量与计算能力不匹配及模块性能差距导致的内存和计算效率低问题，达成吞吐量提升2.25×、延迟降低1.49×的效果

### 8. Reshaping the Forward-Forward Algorithm with a Similarity-Based Objective

帝国理工学院发布了《Reshaping the Forward-Forward Algorithm with a Similarity-Based Objective》论文，使用将Forward-Forward算法与相似性学习框架整合提出FAUST的技术，解决了Forward-Forward算法精度低及推理时需多次前向传播的效率问题，达成了显著提升精度、在CIFAR-10上用简单MLP架构准确率达56.22%接近反向传播基准的效果

### 9. Strategies for Improving Communication Efficiency

Abdullah University of Science and Technology Thuwal发布了Strategies for Improving Communication Efficiency论文，使用通信效率提升策略，解决了通信效率不足的问题，达成了通信效率改善的效果。

### 10. Too Helpful, Too Harmless, Too Honest or Just Right?

Macquarie University发布了Too Helpful, Too Harmless, Too Honest or Just Right?论文，使用TrinityX框架（含混合校准专家Mo-CaE及校准任务自适应路由机制），解决了LLMs对齐HHH原则时各维度单独优化导致的权衡与行为不一致及MoE路由校准差的问题，达成在Alpaca、Beaver-Tails、TruthfulQA基准上分别实现32.5% win rate、33.9% safety score、28.4% truthfulness的相对提升，并减少内存使用和推理延迟超40%的效果。

### 11. 

请提供论文的标题、主要机构和摘要信息以生成技术简报。

## 论文详细信息

### 1. Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism

**主要机构**: 
**作者数量**: 4人

**摘要**:
Mixture-of-Experts (MoE) has emerged as a promising architecture for modern large language models (LLMs). However, massive parameters impose heavy GPU memory (i.e., VRAM) demands, hindering the widespread adoption of MoE LLMs. Offloading the expert parameters to CPU RAM offers an effective way to alleviate the VRAM requirements for MoE inference. Existing approaches typically cache a small subset of experts in VRAM and/or dynamically prefetch experts from RAM during inference, leading to significant degradation in inference speed due to the poor cache hit rate and substantial expert loading latency. In this work, we propose MoEpic, an efficient MoE inference system with a novel expert split mechanism. Specifically, each expert is vertically divided into two segments: top and bottom. MoEpic caches the top segment of hot experts, so that more experts will be stored under the limited VRAM budget, thereby improving the cache hit rate. During each layer's inference, MoEpic predicts and prefetches the activated experts for the next layer. Since the top segments of cached experts are exempt from fetching, the loading time is reduced, which allows efficient transfer-computation overlap. Nevertheless, the performance of MoEpic critically depends on the cache configuration (i.e., each layer's VRAM budget and expert split ratio). To this end, we propose a divide-and-conquer algorithm based on fixed-point iteration for adaptive cache configuration. Extensive experiments on popular MoE LLMs demonstrate that MoEpic can save about half of the GPU cost, while lowering the inference latency by about 37.51%∼65.73% compared to the baselines.

### 2. A 410 GFLOP/s, 64 RISC-V Cores, 204.8 GBps Shared-Memory Cluster in 12 nm FinFET with Systolic Execution Support for Efficient B5G/6G AI-Enhanced O-RAN

**主要机构**: ETH Zürich † DEI, IIS, University of Bologna
**作者数量**: 5人

**摘要**:
We present HeartStream, a 64-RV-core shared-L1memory cluster (410 GFLOP/s peak performance and 204.8 GBps L1 bandwidth) for energy-efficient AI-enhanced O-RAN. The cores and cluster architecture are customized for baseband processing, supporting complex (16-bit real&imaginary) instructions: multi-ply&accumulate, division&square-root, SIMD instructions, and hardware-managed systolic queues, improving up to 1.89× the energy efficiency of key baseband kernels. At 800 MHz@0.8 V, HeartStream delivers up to 243 GFLOP/s on complex-valued wireless workloads. Furthermore, the cores also support efficient AI processing on received data at up to 72 GOP/s. HeartStream is fully compatible with base station power and processing latency limits: it achieves leading-edge software-defined PUSCH efficiency (49.6 GFLOP/s/W) and consumes just 0.68 W (645 MHz@0.65 V), within the 4 ms end-to-end constraint for B5G/6G uplink.

### 3. BcQLM: Efficient Vision-Language Understanding with Distilled Q-Gated Cross-Modal Fusion

**主要机构**: Durham University, Department of Computer Science
**作者数量**: 3人

**摘要**:
As multimodal large language models (MLLMs) advance, their large-scale architectures pose challenges for deployment in resource-constrained environments. In the age of large models, where energy efficiency, computational scalability and environmental sustainability are paramount, the development of lightweight and high-performance models is critical for real-world applications. As such, we propose a lightweight MLLM framework for end-to-end visual question answering. Our proposed approach centres on BreezeCLIP, a compact yet powerful vision-language encoder optimised for efficient multimodal understanding. With only 1.2 billion parameters overall, our model significantly reduces computational cost while achieving performance comparable to standard-size MLLMs. Experiments conducted on multiple datasets further validate its effectiveness in balancing accuracy and efficiency. The modular and extensible design enables generalisation to broader multimodal tasks. The proposed lightweight vision-language framework is denoted as BcQLM (BreezeCLIPenhanced Q-Gated Multimodal Language Model). It offers a promising path toward deployable MLLMs under practical hardware constraints. The source code is available at https://github.com/thico0224/BcQLM.

### 4. Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video

**主要机构**: Shanghai Jiao, Tong University
**作者数量**: 7人

**摘要**:
We propose a novel and general framework to disentangle video data into its dynamic motion and static content components. Our proposed method is a self-supervised pipeline with less assumptions and inductive biases than previous works: it utilizes a transformer-based architecture to jointly generate flexible implicit features for frame-wise motion and clip-wise content, and incorporates a low-bitrate vector quantization as an information bottleneck to promote disentanglement and form a meaningful discrete motion space. The bitrate-controlled latent motion and content are used as conditional inputs to a denoising diffusion model to facilitate self-supervised representation learning. We validate our disentangled representation learning framework on real-world talking head videos with motion transfer and auto-regressive motion generation tasks. Furthermore, we also show that our method can generalize to other types of video data, such as pixel sprites of 2D cartoon characters. Our work presents a new perspective on self-supervised learning of disentangled video representations, contributing to the broader field of video analysis and generation. † Joint first authors. The major part of the work was done when Qi Chen was an intern at MSRA.

### 5. Efficient Decoding Methods for Language Models on Encrypted Data

**主要机构**: IBM Research, Bar-Ilan University
**作者数量**: 5人

**摘要**:
Large language models (LLMs) power modern AI applications, but processing sensitive data on untrusted servers raises privacy concerns. Homomorphic encryption (HE) enables computation on encrypted data for secure inference. However, neural text generation requires decoding methods like argmax and sampling, which are non-polynomial and thus computationally expensive under encryption, creating a significant performance bottleneck. We introduce CutMax, an HE-friendly argmax algorithm that reduces ciphertext operations compared to prior methods, enabling practical greedy decoding under encryption. We also propose the first HE-compatible nucleus (topp) sampling method, leveraging CutMax for efficient stochastic decoding with provable privacy guarantees. Both techniques are polynomial, supporting efficient inference in privacypreserving settings. Moreover, their differentiability facilitates gradient-based sequencelevel optimization as a polynomial alternative to straight-through estimators. We further provide strong theoretical guarantees for CutMax, proving it converges globally to a unique twolevel fixed point, independent of the input values beyond the identity of the maximizer, which explains its rapid convergence in just a few iterations. Evaluations on realistic LLM outputs show latency reductions of 24×-35× over baselines, advancing secure text generation.

### 6. EvolKV: Evolutionary KV Cache Compression for LLM Inference

**主要机构**: ETH Zurich, University of Chinese Academy of Sciences, School of Advanced Interdisciplinary Sciences
**作者数量**: 2人

**摘要**:
Existing key-value (KV) cache compression methods typically rely on heuristics, such as uniform cache allocation across layers or static eviction policies, however, they ignore the critical interplays among layer-specific feature patterns and task performance, which can lead to degraded generalization. In this paper, we propose EvolKV, an adaptive framework for layerwise, task-driven KV cache compression that jointly optimizes the memory efficiency and task performance. By reformulating cache allocation as a multi-objective optimization problem, EvolKV leverages evolutionary search to dynamically configure layer budgets while directly maximizing downstream performance. Extensive experiments on 11 tasks demonstrate that our approach outperforms all baseline methods across a wide range of KV cache budgets on long-context tasks and surpasses heuristic baselines by up to 7 percentage points on GSM8K. Notably, EvolKV achieves superior performance over the full KV cache setting on code completion while utilizing only 1.5% of the original budget, suggesting the untapped potential in learned compression strategies for KV cache budget allocation.

### 7. Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism

**主要机构**: Sun Yat-sen University, SC '25, University of Macau Macau SAR
**作者数量**: 8人

**摘要**:
The significant resource demands in LLM serving prompts production clusters to fully utilize heterogeneous hardware by partitioning LLM models across a mix of high-end and low-end GPUs. However, existing parallelization approaches often struggle to scale efficiently in heterogeneous environments due to their coarse-grained and static parallelization strategies. In this paper, we introduce Hetis, a new LLM system tailored for heterogeneous GPU clusters. Hetis addresses two critical challenges: (1) memory inefficiency caused by the mismatch between memory capacity and computational power in heterogeneous devices, and (2) computational inefficiency arising from performance gaps across different LLM modules. To tackle these issues, Hetis employs a fine-grained and dynamic parallelism design. Specifically, it selectively parallelizes compute-intensive operations to reduce latency and dynamically distributes Attention computations to low-end GPUs at a head granularity, leveraging the distinct characteristics of each module. Additionally, Hetis features an online load dispatching policy that continuously optimizes serving performance by carefully balancing network latency, computational load, and memory intensity. Evaluation results demonstrate that Hetis can improve serving throughput by up to 2.25× and reduce latency by 1.49× compared to existing systems.

### 8. Reshaping the Forward-Forward Algorithm with a Similarity-Based Objective

**主要机构**: University of Auckland, Imperial College London
**作者数量**: 7人

**摘要**:
Backpropagation is the pivotal algorithm underpinning the success of artificial neural networks, yet it has critical limitations such as biologically implausible backward locking and global error propagation. To circumvent these constraints, the Forward-Forward algorithm was proposed as a more biologically plausible method that replaces the backward pass with an additional forward pass. Despite this advantage, the Forward-Forward algorithm significantly trails backpropagation in accuracy, and its optimal form exhibits low inference efficiency due to multiple forward passes required. In this work, the Forward-Forward algorithm is reshaped through its integration with similarity learning frameworks, eliminating the need for multiple forward passes during inference. This proposed algorithm is named Forward-forward Algorithm Unified with Similaritybased Tuplet loss (FAUST). Empirical evaluations on MNIST, Fashion-MNIST, and CIFAR-10 datasets indicate that FAUST substantially improves accuracy, narrowing the gap with backpropagation. On CIFAR-10, FAUST achieves 56.22% accuracy with a simple multi-layer perceptron architecture, approaching the backpropagation benchmark of 57.63% accuracy.

### 9. Strategies for Improving Communication Efficiency

**主要机构**: Abdullah University of Science and Technology Thuwal, Doctor of Philosophy King
**作者数量**: 1人

**摘要**:


### 10. Too Helpful, Too Harmless, Too Honest or Just Right?

**主要机构**: Macquarie University, School of Computing
**作者数量**: 4人

**摘要**:
Large Language Models (LLMs) exhibit strong performance across a wide range of NLP tasks, yet aligning their outputs with the principles of Helpfulness, Harmlessness, and Honesty (HHH) remains a persistent challenge. Existing methods often optimize for individual alignment dimensions in isolation, leading to trade-offs and inconsistent behavior. While Mixture-of-Experts (MoE) architectures offer modularity, they suffer from poorly calibrated routing, limiting their effectiveness in alignment tasks. We propose TrinityX, a modular alignment framework that incorporates a Mixture of Calibrated Experts (Mo-CaE) within the Transformer architecture. Trin-ityX leverages separately trained experts for each HHH dimension, integrating their outputs through a calibrated, task-adaptive routing mechanism that combines expert signals into a unified, alignment-aware representation. Extensive experiments on three standard alignment benchmarks-Alpaca (Helpfulness), Beaver-Tails (Harmlessness), and TruthfulQA (Honesty)-demonstrate that TrinityX outperforms strong baselines, achieving relative improvements of 32.5% in win rate, 33.9% in safety score, and 28.4% in truthfulness. In addition, TrinityX reduces memory usage and inference latency by over 40% compared to prior MoEbased approaches. Ablation studies highlight the importance of calibrated routing, and crossmodel evaluations confirm TrinityX's generalization across diverse LLM backbones. Our code is available at: https://github.com/g skgautam/TrinityX

### 11. 

**主要机构**: 
**作者数量**: 0人

**摘要**:

