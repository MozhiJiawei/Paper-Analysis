# AI推理加速技术论文分析报告
生成时间: 2025-09-16 16:52:16
分析论文数量: 8篇

## 论文技术简报

### 1. Adaptive Token Merging for Efficient Transformer Semantic Communication at the Edge

论文《Adaptive Token Merging for Efficient Transformer Semantic Communication at the Edge》简报生成失败

### 2. CODICODEC: UNIFYING CONTINUOUS AND DISCRETE COMPRESSED REPRESENTATIONS OF AUDIO

Sony Computer Science Laboratories与Queen Mary University of London发布了CODICODEC论文，使用Finite Scalar Quantization (FSQ)及FSQ-dropout技术，解决了现有音频自编码器需在连续嵌入与离散令牌间选择及高压缩比下保真度的挑战，达成了同时生成连续嵌入和离散令牌、并行解码质量更优且速度更快、相似比特率下重建音频质量优于现有连续和离散自编码器的效果。

### 3. Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching

上海交通大学发布了相关论文，使用Cluster-Driven Feature Caching（ClusCa）技术，解决了扩散Transformer的巨大计算成本问题，达成了在FLUX上4.96倍加速且ImageReward达99.49%（超过原模型0.51%）的效果

### 4. Dropping Experts, Recombining Neurons: Retraining-Free Pruning for Sparse Mixture-of-Experts LLMs

浙江大学发布了Dropping Experts, Recombining Neurons论文，使用DERN（神经元级专家片段重组、无需重训练的剪枝框架）技术，解决了稀疏混合专家大语言模型（SMoE LLMs）的高内存使用与部署挑战，达成了50%专家稀疏度下常识推理和MMLU基准性能较现有方法提升超5%、大幅减少专家数量和内存使用的效果

### 5. Efficient and Accurate Downfacing Visual Inertial Odometry

研究团队发布了《Efficient and Accurate Downfacing Visual Inertial Odometry》论文，使用优化量化SuperPoint、PX4FLOW、ORB特征跟踪方法并结合刚体运动模型适配RISC-V超低功耗并行SoC的流水线，解决了高精度VIO在微纳无人机等场景下计算需求与低功耗的平衡问题，达成了GAP9低功耗SoC上ORB跟踪器RMSE平均降低3.65倍、PX4FLOW低速下精度与ORB相当且运行时间更低的效果。

### 6. EFFICIENT LEARNED IMAGE COMPRESSION THROUGH KNOWLEDGE DISTILLATION *

研究团队发布了基于知识蒸馏的高效图像压缩论文，使用知识蒸馏技术，解决了现有神经网络图像压缩模型处理能力需求高、不适合资源受限平台实时使用的问题，达成了降低资源需求、在不同架构尺寸和图像质量/比特率权衡下有效并节省处理与能源资源的效果。

### 7. I-Segmenter: Integer-Only Vision Transformer for Efficient Semantic Segmentation

CEA与Université Paris-Saclay发布了I-Segmenter论文，使用全整数Vision Transformer框架（含λ-ShiftGELU激活函数及整数化计算图优化），解决了Vision Transformer语义分割模型在资源受限设备部署时的高内存占用、计算成本及低精度量化脆弱性问题，达成精度接近FP32基线（平均差距5.1%）、模型大小减少3.8×、推理加速1.2×的效果。

### 8. LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation

南京大学发布了LAVa论文，使用通过最小化Transformer残差流信息损失的统一框架实现层内动态头预算和跨层动态预算分配的KV Cache压缩策略，解决了现有KV Cache压缩方法启发式且缺乏动态预算分配的问题，在LongBench等多个基准测试中表现优越并保持顶级性能。

## 论文详细信息

### 1. Adaptive Token Merging for Efficient Transformer Semantic Communication at the Edge

**主要机构**: 
**作者数量**: 3人

**摘要**:
Large-scale transformers are central to modern semantic communication, yet their high computational and communication costs hinder deployment on resource-constrained edge devices. This paper introduces a training-free framework for adaptive token merging, a novel mechanism that compresses transformer representations at runtime by selectively merging semantically redundant tokens under per-layer similarity thresholds. Unlike prior fixed-ratio reduction, our approach couples merging directly to input redundancy, enabling data-dependent adaptation that balances efficiency and task relevance without retraining. We cast the discovery of merging strategies as a multi-objective optimization problem and leverage Bayesian optimization to obtain Pareto-optimal trade-offs between accuracy, inference cost, and communication cost. On ImageNet classification, we match the accuracy of the unmodified transformer with 30% fewer floating-point operations per second and under 20% of the original communication cost, while for visual question answering our method achieves performance competitive with the full LLaVA model at less than one-third of the compute and one-tenth of the bandwidth. Finally, we show that our adaptive merging is robust across varying channel conditions and provides inherent privacy benefits, substantially degrading the efficacy of model inversion attacks. Our framework provides a practical and versatile solution for deploying powerful transformer models in resource-limited edge intelligence scenarios.

### 2. CODICODEC: UNIFYING CONTINUOUS AND DISCRETE COMPRESSED REPRESENTATIONS OF AUDIO

**主要机构**: Queen Mary University of London, Sony Computer Science Laboratories
**作者数量**: 3人

**摘要**:
Efficiently representing audio signals in a compressed latent space is critical for latent generative modelling. However, existing autoencoders often force a choice between continuous embeddings and discrete tokens. Furthermore, achieving high compression ratios while maintaining audio fidelity remains a challenge. We introduce CoDiCodec, a novel audio autoencoder that overcomes these limitations by both efficiently encoding global features via summary embeddings, and by producing both compressed continuous embeddings at ~11 Hz and discrete tokens at a rate of 2.38 kbps from the same trained model, offering unprecedented flexibility for different downstream generative tasks. This is achieved through Finite Scalar Quantization (FSQ) and a novel FSQ-dropout technique, and does not require additional loss terms beyond the single consistency loss used for end-to-end training. CoDiCodec supports both autoregressive decoding and a novel parallel decoding strategy, with the latter achieving superior audio quality and faster decoding. CoDiCodec outperforms existing continuous and discrete autoencoders at similar bitrates in terms of reconstruction audio quality. Our work enables a unified approach to audio compression, bridging the gap between continuous and discrete generative modelling paradigms. Pretrained weights are available under [this link]. 1

### 3. Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching

**主要机构**: Shanghai Jiao Tong University Shanghai, Zhixin Zheng, University of Electronic Science and Technology of China Chengdu, Jiao Tong University Shanghai, Linfeng Zhang, Shaobo Wang Shanghai, Shandong University Qingdao
**作者数量**: 11人

**摘要**:
Diffusion transformers have gained significant attention in recent years for their ability to generate high-quality images and videos, yet still suffer from a huge computational cost due to their iterative denoising process. Recently, feature caching has been introduced to accelerate diffusion transformers by caching the feature computation in previous timesteps and reusing it in the following timesteps, which leverage the temporal similarity of diffusion models while ignoring the similarity in the spatial dimension. In this paper, we introduce Cluster-Driven Feature Caching (ClusCa) as an orthogonal and complementary perspective for previous feature caching. Specifically, ClusCa performs spatial clustering on tokens in each timestep, computes only one token in each cluster and propagates their information to all the other tokens, which is able to reduce the number of tokens by over 90%. Extensive experiments on DiT, FLUX and HunyuanVideo demonstrate its effectiveness in both text-to-image and text-to-video generation. Besides, it can be directly applied to any diffusion transformer without requirements for training. For instance, ClusCa achieves 4.96× acceleration on FLUX with an Im-ageReward of 99.49%, surpassing the original model by 0.51%. The code is available at https://github.com/Shenyi-Z/Cache4Diffusion. CCS Concepts • Computing methodologies → Computer vision.

### 4. Dropping Experts, Recombining Neurons: Retraining-Free Pruning for Sparse Mixture-of-Experts LLMs

**主要机构**: Shanghai Innovation Institute, Zhejiang University, Laboratory, Southeast University
**作者数量**: 9人

**摘要**:
Sparse Mixture-of-Experts (SMoE) architectures are widely used in large language models (LLMs) due to their computational efficiency. However, though only a few experts are activated for each token, SMoE still requires loading all expert parameters, leading to high memory usage and challenges in deployment. Previous work has tried to reduce the overhead by pruning and merging experts, but primarily focused on expert-level operations, leaving neuron-level structure underexplored. We propose DERN (Dropping Experts, Recombining Neurons), a task-agnostic and retraining-free framework for expert pruning and reconstruction. We observe that experts are often misaligned and contain semantic conflicts at the neuron level, which poses challenges for direct merging. To solve this, DERN works in three steps: it first prunes redundant experts using router statistics; then it decomposes them into neuron-level expert segments, assigning each segment to its most compatible retained expert; and finally, it merges segments within each retained expert to build a compact representation. Experiments on Mixtral, Qwen, and DeepSeek SMoE models show that DERN achieves over a 5% performance gains than previous methods on commonsense reasoning and MMLU benchmarks under 50% expert sparsity, without extra training. It also greatly reduces the number of experts and memory usage, making SMoE LLMs easier to deploy in practice.

### 5. Efficient and Accurate Downfacing Visual Inertial Odometry

**主要机构**: Efficient and Accurate Downfacing Visual Inertial Odometry
**作者数量**: 5人

**摘要**:
Visual Inertial Odometry (VIO) is a widely used computer vision method that determines an agent's movement through a camera and an IMU sensor. This paper presents an efficient and accurate VIO pipeline optimized for applications on micro-and nano-UAVs. The proposed design incorporates stateof-the-art feature detection and tracking methods (SuperPoint, PX4FLOW, ORB), all optimized and quantized for emerging RISC-V-based ultra-low-power parallel systems on chips (SoCs). Furthermore, by employing a rigid body motion model, the pipeline reduces estimation errors and achieves improved accuracy in planar motion scenarios. The pipeline's suitability for real-time VIO is assessed on an ultra-low-power SoC in terms of compute requirements and tracking accuracy after quantization. The pipeline, including the three feature tracking methods, was implemented on the SoC for real-world validation. This design bridges the gap between high-accuracy VIO pipelines that are traditionally run on computationally powerful systems and lightweight implementations suitable for microcontrollers. The optimized pipeline on the GAP9 low-power SoC demonstrates an average reduction in RMSE of up to a factor of 3.65x over the baseline pipeline when using the ORB feature tracker. The analysis of the computational complexity of the feature trackers further shows that PX4FLOW achieves on-par tracking accuracy with ORB at a lower runtime for movement speeds below 24 pixels/frame.

### 6. EFFICIENT LEARNED IMAGE COMPRESSION THROUGH KNOWLEDGE DISTILLATION *

**主要机构**: 
**作者数量**: 4人

**摘要**:
Learned image compression sits at the intersection of machine learning and image processing. With advances in deep learning, neural network-based compression methods have emerged. In this process, an encoder maps the image to a low-dimensional latent space, which is then quantized, entropycoded into a binary bitstream, and transmitted to the receiver. At the receiver end, the bitstream is entropy-decoded, and a decoder reconstructs an approximation of the original image. Recent research suggests that these models consistently outperform conventional codecs. However, they require significant processing power, making them unsuitable for real-time use on resource-constrained platforms, which hinders their deployment in mainstream applications. This study aims to reduce the resource requirements of neural networks used for image compression by leveraging knowledge distillation, a training paradigm where smaller neural networks, partially trained on the outputs of larger, more complex models, can achieve better performance than when trained independently. Our work demonstrates that knowledge distillation can be effectively applied to image compression tasks: i) across various architecture sizes, ii) to achieve different image quality/bit rate tradeoffs, and iii) to save processing and energy resources. This approach introduces new settings and hyperparameters, and future research could explore the impact of different teacher models, as well as alternative loss functions. Knowledge distillation could also be extended to transformer-based models. The code is publicly available at: https://github.com/FABallemand/PRIM

### 7. I-Segmenter: Integer-Only Vision Transformer for Efficient Semantic Segmentation

**主要机构**: CEA, Université Paris-Saclay
**作者数量**: 3人

**摘要**:
Vision Transformers (ViTs) have recently achieved strong results in semantic segmentation, yet their deployment on resource-constrained devices remains limited due to their high memory footprint and computational cost. Quantization offers an effective strategy to improve efficiency, but ViT-based segmentation models are notoriously fragile under low precision, as quantization errors accumulate across deep encoder-decoder pipelines. We introduce I-Segmenter, the first fully integer-only ViT segmentation framework. Building on the Segmenter architecture, I-Segmenter systematically replaces floating-point operations with integer-only counterparts. To further stabilize both training and inference, we propose λ-ShiftGELU, a novel activation function that mitigates the limitations of uniform quantization in handling long-tailed activation distributions. In addition, we remove the L2 normalization layer and replace bilinear interpolation in the decoder with nearest-neighbor upsampling, ensuring integeronly execution throughout the computational graph. Extensive experiments show that I-Segmenter achieves accuracy within a reasonable margin of its FP32 baseline (5.1% on average), while reducing model size by up to 3.8× and enabling up to 1.2× faster inference with optimized runtimes. Notably, even in oneshot PTQ with a single calibration image, I-Segmenter delivers competitive accuracy, underscoring its practicality for real-world deployment.

### 8. LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation

**主要机构**: Nanjing University, School of Artificial Intelligence, State Key Laboratory for Novel Software Technology
**作者数量**: 8人

**摘要**:
KV Cache is commonly used to accelerate LLM inference with long contexts, yet its high memory demand drives the need for cache compression. Existing compression methods, however, are largely heuristic and lack dynamic budget allocation. To address this limitation, we introduce a unified framework for cache compression by minimizing information loss in Transformer residual streams. Building on it, we analyze the layer attention output loss and derive a new metric to compare cache entries across heads, enabling layer-wise compression with dynamic head budgets. Additionally, by contrasting cross-layer information, we also achieve dynamic layer budgets. LAVa is the first unified strategy for cache eviction and dynamic budget allocation that, unlike prior methods, does not rely on training or the combination of multiple strategies. Experiments with benchmarks (LongBench, Needle-In-A-Haystack, Ruler, and InfiniteBench) demonstrate its superiority. Moreover, our experiments reveal a new insight: dynamic layer budgets are crucial for generation tasks (e.g., code completion), while dynamic head budgets play a key role in extraction tasks (e.g., extractive QA). As a fully dynamic compression method, LAVa consistently maintains top performance across task types. Our code is available at https://github.com/MGDDestiny/Lava.
