# AI推理加速技术论文分析报告
生成时间: 2025-10-15 19:44:32
分析论文数量: 28篇

## 论文技术简报

### 1. Accelerating LLM Inference with Precomputed Query Storage

Sungkyunkwan University发布了《Accelerating LLM Inference with Precomputed Query Storage》论文，使用StorInfer系统（通过预计算存储查询-响应对、LLM驱动生成器的自适应查询掩码与采样、磁盘向量数据库索引），解决了LLM推理在资源受限环境中的高延迟问题，达成了17.3%的延迟降低且无响应质量损失的效果。

### 2. AdaBlock-dLLM: SEMANTIC-AWARE DIFFUSION LLM INFERENCE VIA ADAPTIVE BLOCK SIZE

帝国理工学院发布了AdaBlock-dLLM论文，使用自适应块大小的语义感知扩散LLM推理调度器（训练-free、即插即用），解决了传统半自回归解码中固定块大小导致的延迟解码开销与过早解码错误问题，在相同吞吐量下准确率提升高达5.3%。

### 3. AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving

University of Chicago发布了AdaptCache论文，使用自适应有损KV缓存压缩系统（决定压缩算法、压缩率和设备放置），解决了LLM服务中KV缓存存储在DRAM和SSD时因SSD加载慢导致的高延迟问题，达成相同质量下延迟节省1.43-2.4倍、相同延迟下质量提升6-55%的效果。

### 4. ADAPTIVE TEST-TIME REASONING VIA REWARD-GUIDED DUAL-PHASE SEARCH

密歇根州立大学发布了ADAPTIVE TEST-TIME REASONING VIA REWARD-GUIDED DUAL-PHASE SEARCH论文，使用奖励引导的双阶段（规划-执行分离）搜索及动态预算分配技术，解决了现有树基搜索方法忽略任务规划-执行性质导致的推理探索低效问题，在数学推理和代码生成基准上达成提高准确性并减少冗余计算的效果。

### 5. Benchmarking Deep Learning Convolutions on Energy-constrained CPUs

CNRS与索邦大学发布了Benchmarking Deep Learning Convolutions on Energy-constrained CPUs论文，使用评估直接、GEMM-based及Winograd等先进卷积算法在多品牌现代CPU上性能的技术，解决了CPU深度学习卷积实现优化不足的问题，达成了GEMM算法与Nvidia AGX Orin结合在推理延迟和能耗间取得最佳权衡的效果

### 6. CAST: Continuous and Differentiable Semi-Structured Sparsity-Aware Training for Large Language Models

清华大学发布了CAST论文，使用连续可微的半结构化稀疏感知训练框架（含AdamS优化器、Weight Scaling和知识蒸馏），解决了大语言模型稀疏模式与权重优化分离的问题，在2:4稀疏下，LLaMA2-7B仅用2%预训练token实现困惑度增加0.09、零样本准确率提升0.36%，显著优于SOTA

### 7. Under review as a conference paper at ICLR 2026 CAT: POST-TRAINING QUANTIZATION ERROR REDUCTION VIA CLUSTER-BASED AFFINE TRANSFORMATION

University of Wurzburg发布了CAT论文，使用集群特定参数的仿射变换（Cluster-based Affine Transformation）技术，解决了低比特后训练量化（PTQ）中的精度下降问题，在ImageNet-1K上超过现有PTQ方法，W2A2 ResNet-18达53.18% Top-1准确率且作为插件提升现有基线超3%。

### 8. CIMNAS: A Joint Framework for Compute-In-Memory-Aware Neural Architecture Search

研究团队发布了CIMNAS论文，使用联合模型-量化-硬件协同优化的存内计算感知神经架构搜索框架，解决了CIM加速器软硬件参数手动调优不切实际的问题，达成了MobileNet上EDAP减少90.1×-104.5×、TOPS/W提升4.68×-4.82×，扩展至ResNet50时EDAP减少819.5×且无精度损失的效果

### 9. COLLABORATIVE COMPRESSION FOR LARGE-SCALE MOE DEPLOYMENT ON EDGE

Futurewei Technologies, Inc与Northeastern University发布了关于大规模MoE边缘部署的协同压缩论文，使用结合专家剪枝、混合精度量化和激活优化的协同压缩框架，解决了超大规模MoE模型因内存/存储需求大难以在资源受限边缘平台部署的问题，达成将DeepSeek-V3存储占用从1.3TB降至103GB，同时保持高输出质量和比传统均匀低位量化方法更好准确性的效果。

### 10. DISTILLATION OF LARGE LANGUAGE MODELS VIA CONCRETE SCORE MATCHING

KAIST发布了大语言模型蒸馏论文，使用Concrete Score Distillation（CSD）技术，解决了现有知识蒸馏中softmax模糊logit信息及直接logit蒸馏的解空间限制问题，达成了超过近期蒸馏目标、实现良好保真度-多样性权衡并与on-policy技术结合获互补增益的效果。

### 11. LEARNABLE PARALLEL DECODING FOR DLLMS

新加坡国立大学发布了LEARNABLE PARALLEL DECODING FOR DLLMS论文，使用dParallel方法（核心为确定性强制蒸馏），解决了扩散大型语言模型(dLLMs)并行解码中掩码token顺序确定性收敛的瓶颈问题，达成了在LLaDA-8B-Instruct模型上解码步骤从256减少至30（GSM8K，8.5×加速）和24（MBPP，10.5×加速）且保持性能的效果。

### 12. dVLA: DIFFUSION VISION-LANGUAGE-ACTION MODEL WITH MULTIMODAL CHAIN-OF-THOUGHT

上海交通大学、北京大学发布了dVLA论文，使用扩散视觉-语言-动作模型（dVLA）与多模态思维链技术，解决了VLA模型中跨模态推理及对新指令和物体泛化的问题，达成了LIBERO基准96.4%平均成功率的SOTA性能、真实机器人任务成功及推理速度提升约2倍的效果。

### 13. ENHANCING LINEAR ATTENTION WITH RESIDUAL LEARNING

香港大学、北京大学发布了《ENHANCING LINEAR ATTENTION WITH RESIDUAL LEARNING》论文，使用残差线性注意力（RLA）框架（含显式残差拟合机制及辅助循环状态累积残差误差校正）及残差Delta网络（RDN）（结合自适应门控和残差裁剪）技术，解决了线性注意力难以捕捉长程模式的表达瓶颈问题，达成了在语言建模和召回密集型评估中优于基线及其他线性注意力方法、缩小与标准Transformer差距并保留线性缩放的效果。

### 14. FAST-D LLM V 2: Efficient Block-Diffusion LLM

香港大学发布了FAST-D LLM V 2: Efficient Block-Diffusion LLM论文，使用块扩散机制、互补注意力掩码及分层缓存机制，解决了AR大语言模型顺序解码限制推理效率的问题，达成了仅需~1B tokens微调（训练数据减少500倍）、解码速度提升2.5倍且保持性能，在dLLMs中实现最先进效率的效果。

### 15. FlashOmni: A Unified Sparse Attention Engine for Diffusion Transformers FLASHOMNI: A UNIFIED SPARSE ATTENTION ENGINE FOR DIFFUSION TRANSFORMERS

爱丁堡大学和中国科学技术大学发布了FlashOmni论文，使用统一稀疏注意力引擎FlashOmni（通过灵活稀疏符号标准化稀疏策略表示并设计优化稀疏GEMMs），解决了现有稀疏加速方法因稀疏模式多样需定制化内核导致的通用性不足问题以缓解Diffusion Transformers部署的高计算需求瓶颈，达成了注意力和GEMM-Q接近线性加速、GEMM-O加速2.5×-3.8×（峰值达理论极限87.5%），助力Hunyuan模型（33K）端到端加速约1.5×且不降低视觉质量的效果

### 16. FROM MNIST TO IMAGENET: UNDERSTANDING THE SCALABILITY BOUNDARIES OF DIFFERENTIABLE LOGIC GATE NETWORKS

ETH Zürich发布了关于可微逻辑门网络(DLGNs)可扩展性的论文，通过研究输出层设计（包括温度调优和Group-Sum层），解决了DLGNs在大型多类数据集上的可扩展性问题（此前限于10类），实现了最多2000类的大规模分类。

### 17. HILBERTA: HILBERT ATTENTION FOR IMAGE GENERATION WITH DIFFUSION MODELS

纽约大学发布了HILBERTA论文，使用基于希尔伯特曲线的二维感知GPU高效稀疏注意力机制（通过重排图像token实现连续内存布局、跨层滑动调度及中央共享区域），解决了扩散模型中稀疏注意力难以平衡二维空间局部性与GPU效率的问题，达成了在Flux.1-dev上生成1024×1024图像时注意力加速2.3倍、2048×2048时达4.17倍且图像质量相当或超过基线的效果。

### 18. Interpret, prune and distill Donut : towards lightweight VLMs for VQA on documents

Université Paris Cité发布了《Interpret, prune and distill Donut : towards lightweight VLMs for VQA on documents》论文，使用利用机械可解释性驱动架构设计的知识蒸馏技术，解决了大型视觉语言模型在实时或资源受限应用中的高成本问题，达成了开发出轻量级模型Donut-MINT、减少推理时间和内存使用并在DocVQA基准上保持强性能的效果。

### 19. 

纽约大学发布了知识蒸馏相关论文，使用Procrustes距离和特征Gram矩阵的Frobenius范数作为蒸馏损失，解决了现有特征蒸馏方法（如投影均方损失、CKA）无法捕捉特征结构的问题，在BERT和OPT等语言模型的分类及指令跟随任务上蒸馏性能显著提升达2个百分点。

### 20. ON-THE-FLY ADAPTATION TO QUANTIZATION: CONFIGURATION-AWARE LORA FOR EFFICIENT FINE-TUNING OF QUANTIZED LLMS

南方科技大学、香港大学发布了相关论文，使用CoA-LoRA方法，解决了边缘设备异构能力下量化LLMs为不同配置重复微调的计算成本问题，达成了动态调整LoRA适配器至任意量化配置无需重复微调且性能相当或更优的效果

### 21. Parallax: Efficient LLM Inference Service over Decentralized Environment

新加坡国立大学发布了Parallax论文，使用两阶段调度器（模型分配与请求时GPU流水线选择），解决了分布式环境下GPU异构、网络带宽有限及动态可用性导致的LLM推理调度挑战，达成了相比分布式基线持续降低延迟并提高吞吐量的效果。

### 22. POST-TRAINING QUANTIZATION VIA RESIDUAL TRUNCATION AND ZERO SUPPRESSION FOR DIF-FUSION MODELS

庆熙大学发布了QuaRTZ论文，使用4位后训练量化技术（通过8位min-max量化处理离群值及前导零抑制压缩保留LSB），解决了扩散模型4位后训练量化中低幅度激活舍入误差导致纹理细节丢失的问题，达成了在FLUX.1-schnell上FID 6.98且优于SVDQuant的效果

### 23. Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation

华为诺亚方舟实验室发布了Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation论文，使用系统的后训练管道（含课程监督微调与离线策略知识蒸馏），解决了小语言模型预训练后性能不足、难以满足复杂任务需求的问题，达成了在十亿参数模型中最先进性能，且在严格硬件约束下泛化能力强、准确率有竞争力的效果。

### 24. SKIP-IT? THEORETICAL CONDITIONS FOR LAYER SKIPPING IN VISION-LANGUAGE MODELS

伊利诺伊大学发布了《SKIP-IT? THEORETICAL CONDITIONS FOR LAYER SKIPPING IN VISION-LANGUAGE MODELS》论文，使用信息和学习理论框架，解决了视觉语言模型（VLMs）层跳过何时有益的理解有限问题，达成了明确层跳过条件、实现更快推理且保持性能、避免模型退化的效果。

### 25. Training Matryoshka Mixture-of-Experts for Elastic Inference-Time Expert Utilization

上海交通大学、中国科学技术大学发布了Matryoshka MoE（M-MoE）论文，使用Matryoshka MoE训练框架（通过训练时系统改变激活专家数量形成粗到细专家结构），解决了标准MoE模型推理时改变激活专家数量导致性能急剧下降的问题，达成了单个模型实现弹性推理、不同专家数量下性能接近系列专家模型且训练成本大幅降低的效果。

### 26. UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression

大连理工大学发布了UniMMAD论文，使用MoE驱动的特征解压缩技术，解决了统一的多模态和多类别异常检测问题，实现了性能提升

### 27. Vocabulary Customization for Efficient Domain-Specific LLM Deployment

eBay Inc发布了Vocabulary Customization for Efficient Domain-Specific LLM Deployment论文，使用设计算法扩展预训练分词器并加入领域特定token以保证tokenization效率不降低的技术，解决了通用领域分词器因词汇不匹配导致特定领域文本token数量增加、处理速度下降的问题，达成了输入序列缩短达20%、减少推理延迟且保持预测质量的效果

### 28. VRWKV-EDITOR: REDUCING QUADRATIC COMPLEXITY IN TRANSFORMER-BASED VIDEO EDITING

Mohammed VI Polytechnic University and Sorbonne University发布了VRWKV-EDITOR论文，使用VRWKV-Editor模型（集成线性时空聚合模块，利用RWKV transformer的双向加权键值循环机制），解决了传统Transformer视频编辑模型因二次计算复杂度难以适应长时长、高分辨率视频的问题，达成相比现有扩散基视频编辑方法3.7倍加速和60%内存降低，同时保持帧一致性和文本对齐性能的效果

## 论文详细信息

### 1. Accelerating LLM Inference with Precomputed Query Storage

**主要机构**: Sungkyunkwan University
**作者数量**: 6人

**摘要**:
Large language model (LLM) inference often suffers from high latency, particularly in resource-constrained environments such as on-device or edge deployments. To address this challenge, we present StorInfer, a novel storage-assisted LLM inference system that accelerates response time by precomputing and storing predictable query-response pairs offline. When a user query semantically matches a precomputed query, StorInfer bypasses expensive GPU inference and instantly returns the stored response, significantly reducing latency and compute costs. To maximize coverage and effectiveness, StorInfer employs an LLM-driven generator that adaptively produces diverse and deduplicated queries based on a given knowledge base. This is achieved via two techniques: adaptive query masking, which prevents regeneration of similar queries, and adaptive sampling, which dynamically tunes generation parameters to promote semantic diversity. The resulting query-response pairs are embedded and indexed using a disk-backed vector database to enable fast, similaritybased retrieval at runtime. Using this approach, we generated 150K unique precomputed pairs (taking up to 830 MB of storage space), achieving up to 17.3% latency reduction with no loss in response quality. Our evaluation across multiple QA datasets demonstrates the practicality and scalability of storage-assisted inference, especially in scenarios with predictable query distributions. StorInfer highlights a promising direction in leveraging storage as a primary enabler for efficient, low-latency LLM deployment.

### 2. AdaBlock-dLLM: SEMANTIC-AWARE DIFFUSION LLM INFERENCE VIA ADAPTIVE BLOCK SIZE

**主要机构**: Institute of Science, Imperial College London
**作者数量**: 6人

**摘要**:
Diffusion-based large language models (dLLMs) are gaining attention for their inherent capacity for parallel decoding, offering a compelling alternative to autoregressive LLMs. Among various decoding strategies, blockwise semiautoregressive (semi-AR) approaches are widely adopted due to their natural support for KV caching and their favorable accuracy-speed trade-off. However, this paper identifies two fundamental limitations in the conventional semi-AR decoding approach that applies a fixed block size: i) late decoding overhead, where the unmasking of high-confidence tokens outside the current block is unnecessarily delayed, and ii) premature decoding error, where low-confidence tokens inside the current block are committed too early, leading to incorrect tokens. This paper presents the first systematic investigation challenging the fixed block size assumption in semi-AR decoding. Through a statistical analysis of confidence dynamics during the denoising process, we identify a volatility band (VB) region during dLLM decoding, which encodes local semantic structure and can be used to guide adaptive block sizing. Leveraging these insights, we introduce AdaBlock-dLLM, a training-free, plug-and-play scheduler that adaptively aligns block boundaries with semantic steps by adjusting block size during runtime. Extensive experiments across diverse benchmarks show that AdaBlock-dLLM achieves up to 5.3% accuracy improvement under the same throughput budget. Beyond inferencetime optimization, we hope our semantics-aware adaptive scheduling approach and confidence-based analysis will inspire future training strategies for dLLMs.

### 3. AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving

**主要机构**: University of Chicago
**作者数量**: 11人

**摘要**:
Large language model (LLM) applications often reuse previously processed context, such as chat history and documents, which introduces significant redundant computation. Existing LLM serving systems address such redundant computation by storing the KV caches of processed context and loading the corresponding KV cache when a new request reuses the context. Further, as these LLM applications scale, the total size of KV caches becomes excessively large and requires both DRAM and SSD for full storage. However, prior work that stores KV caches in DRAM and SSD suffers from high loading delays, as most KV cache hits come from SSD, which is slow to load. To increase the KV cache hit rate on DRAM, we identify lossy KV cache compression as a promising approach. We design a lossy compression system that decides the compression algorithm, compression rate and device placement for each KV cache entry to maximise DRAM hits and minimise loading delay without significantly degrading generation quality. Compared to various static compression baselines across three tasks, our system AdaptCache achieves 1.43-2.4 × delay savings at the same quality and 6-55% quality improvements at the same delay.

### 4. ADAPTIVE TEST-TIME REASONING VIA REWARD-GUIDED DUAL-PHASE SEARCH

**主要机构**: Michigan State University
**作者数量**: 11人

**摘要**:
Large Language Models (LLMs) have achieved significant advances in reasoning tasks. A key approach is tree-based search with verifiers, which expand candidate reasoning paths and use reward models to guide pruning and selection. Although effective in improving accuracy, these methods are not optimal in terms of efficiency: they perform simple decomposition on the reasoning process, but ignore the planning-execution nature of tasks such as math reasoning or code generation. This results in inefficient exploration of reasoning process. To address this, we propose a dual-phase test-time scaling framework that explicitly separates reasoning into planning and execution, and performs search over the two phases individually. Specifically, we decompose reasoning trajectories and develop reward models for each phase, enabling the search to explore and prune plans and executions separately. We further introduce a dynamic budget allocation mechanism that adaptively redistributes sampling effort based on reward feedback, allowing early stopping on confident steps and reallocation of computation to more challenging parts of the reasoning process. Experiments on both mathematical reasoning and code generation benchmarks demonstrate that our approach consistently improves accuracy while reducing redundant computation.

### 5. Benchmarking Deep Learning Convolutions on Energy-constrained CPUs

**主要机构**: CNRS, Sorbonne Université
**作者数量**: 4人

**摘要**:
This work evaluates state-of-the-art convolution algorithms for CPU-based deep learning inference. While most prior studies focus on GPUs or NPUs, CPU implementations remain relatively underoptimized. We benchmark direct, GEMM-based, and Winograd convolutions across modern CPUs from ARM ® , Intel ® , AMD ® , Apple ® , and Nvidia ® , considering both latency and energy efficiency. Our results highlight the key architectural factors that govern CPU efficiency for convolution operations, providing practical guidance for energy-aware embedded deployment. As a main results of this work, the Nvidia ® AGX Orin combined with the GEMM algorithm achieves the best trade-off between inference latency and energy consumption.

### 6. CAST: Continuous and Differentiable Semi-Structured Sparsity-Aware Training for Large Language Models

**主要机构**: Tsinghua University, THBI Lab, Institute for AI, BNRist Center, Department of Computer Science and Technology, Tsinghua-Bosch Joint ML Center
**作者数量**: 4人

**摘要**:
Sparsity-aware training is an effective approach for transforming large language models (LLMs) into hardwarefriendly sparse patterns, thereby reducing latency and memory consumption during inference. In this paper, we propose Continuous Adaptive Sparse Trainer (CAST), a fully continuous and differentiable sparsity-aware training framework for semi-structured (or "N:M") sparse models. Unlike previous approaches that optimize sparsity patterns and weights separately, CAST enables seamless joint optimization during training, while progressively transforming the model into the desired sparsity format. Specifically, CAST introduces three key components: 1) AdamS, a sparsity-aware optimizer that leverages adaptive L1 decay to promote uniform sparsification across all parameters; 2) Weight Scaling, a module designed to mitigate the magnitude reduction caused by decay while preserving desired sparsity patterns; 3) Knowledge Distillation, which employs the dense model as a self-teacher to enhance training efficiency. We evaluate CAST under 2:4 sparsity patterns across multiple model families, ranging from 125M to 13B parameters. Our results demonstrate significant improvements over previous state-of-the-art methods in both perplexity and zero-shot accuracy with minimal training resources. Notably, on LLaMA2-7B, our 2:4 sparse model achieves a negligible perplexity increase of 0.09 and a 0.36% gain in zero-shot accuracy compared to the dense model using only 2% of the original pretraining tokens. Additionally, we establish an accurate and robust empirical scaling law to predict sparse model performance given adequate training resources. Finally, we demonstrate the practical applicability of our sparse models by evaluating them under quantization and fine-tuning scenarios.

### 7. Under review as a conference paper at ICLR 2026 CAT: POST-TRAINING QUANTIZATION ERROR REDUCTION VIA CLUSTER-BASED AFFINE TRANSFORMATION

**主要机构**: CAIDAS & IFI University of Wurzburg John, Department of Intelligent Future Technologies (IFT), Computer Vision Lab, Mälardalen University Västerås
**作者数量**: 5人

**摘要**:
Post-Training Quantization (PTQ) reduces the memory footprint and computational overhead of deep neural networks by converting full-precision (FP) values into quantized and compressed data types. While PTQ is more cost-efficient than Quantization-Aware Training (QAT), it is highly susceptible to accuracy degradation under a low-bit quantization (LQ) regime (e.g., 2-bit). Affine transformation is a classical technique used to reduce the discrepancy between the information processed by a quantized model and that processed by its full-precision counterpart; however, we find that using plain affine transformation, which applies a uniform affine parameter set for all outputs, worsens the results in low-bit PTQ. To address this, we propose Cluster-based Affine Transformation (CAT), an errorreduction framework that employs cluster-specific parameters to align LQ outputs with FP counterparts. CAT refines LQ outputs with only a negligible number of additional parameters, without requiring fine-tuning of the model or quantization parameters. We further introduce a novel PTQ framework integrated with CAT. Experiments on ImageNet-1K show that this framework consistently outperforms prior PTQ methods across diverse architectures and LQ settings, achieving up to 53.18% Top-1 accuracy on W2A2 ResNet-18. Moreover, CAT enhances existing PTQ baselines by more than 3% when used as a plug-in. We plan to release our implementation alongside the publication of this paper.

### 8. CIMNAS: A Joint Framework for Compute-In-Memory-Aware Neural Architecture Search

**主要机构**: 
**作者数量**: 4人

**摘要**:
To maximize hardware efficiency and performance accuracy in Compute-In-Memory (CIM)-based neural network accelerators for Artificial Intelligence (AI) applications, cooptimizing both software and hardware design parameters is essential. Manual tuning is impractical due to the vast number of parameters and their complex interdependencies. To effectively automate the design and optimization of CIM-based neural network accelerators, hardware-aware neural architecture search (HW-NAS) techniques can be applied. This work introduces CIMNAS, a joint model-quantization-hardware optimization framework for CIM architectures. CIMNAS simultaneously searches across software parameters, quantization policies, and a broad range of hardware parameters, incorporating device-, circuit-, and architecture-level co-optimizations. CIMNAS experiments were conducted over a search space of 9.9×10 85 potential parameter combinations with the MobileNet model as a baseline and RRAM-based CIM architecture. Evaluated on the ImageNet dataset, CIMNAS achieved a reduction in energy-delay-area product (EDAP) ranging from 90.1× to 104.5×, an improvement in TOPS/W between 4.68× and 4.82×, and an enhancement in TOPS/mm 2 from 11.3× to 12.78× relative to various baselines, all while maintaining an accuracy of 73.81%. The adaptability and robustness of CIMNAS are demonstrated by extending the framework to support the SRAM-based ResNet50 architecture, achieving up to an 819.5× reduction in EDAP. Unlike other state-of-the-art methods, CIMNAS achieves EDAP-focused optimization without any accuracy loss, generating diverse softwarehardware parameter combinations for high-performance CIMbased neural network designs. The source code of CIMNAS is available at https://github.com/OlgaKrestinskaya/CIMNAS.

### 9. COLLABORATIVE COMPRESSION FOR LARGE-SCALE MOE DEPLOYMENT ON EDGE

**主要机构**: Futurewei Technologies, Inc, Northeastern University
**作者数量**: 9人

**摘要**:
The Mixture of Experts (MoE) architecture is an important method for scaling Large Language Models (LLMs). It increases model capacity while keeping computation cost low. However, the ultra-large MoE models still have hundreds of billions of parameters, requiring massive memory/storage and leading to difficulties for deployment on resource-constrained edge platforms. Pruning or quantization alone can hardly address the issue, because of the super-aggressive compression ratio with significantly degraded accuracy and output quality. To facilitate the deployment of ultra-large MoEs on edge platforms, we propose a collaborative compression framework by combining expert pruning, mixed-precision quantization, and activation optimization. It can effectively reduce the storage footprint of the ultra-large MoE DeepSeek-V3 from 1.3TB to 103GB, while preserving high output quality with better accuracy than traditional uniform low-bit quantization methods. To the best of our knowledge, we are the first to deploy a compressed model from the ultra-large DeepSeek-V3 on the platform with a strict 128GB total memory limit. Our comprehensive experiments on multiple benchmarks under various memory constraints demonstrate the effectiveness of our method with smaller model sizes and higher accuracy than uniform low-bit quantization methods. All our models are available at https://huggingface.co/bobchenyx/ DeepSeek-V3-0324-MLA-GGUF .

### 10. DISTILLATION OF LARGE LANGUAGE MODELS VIA CONCRETE SCORE MATCHING

**主要机构**: Korea Advanced Institute of Science and Technology (KAIST)
**作者数量**: 5人

**摘要**:
Large language models (LLMs) deliver remarkable performance but are costly to deploy, motivating knowledge distillation (KD) for efficient inference. Existing KD objectives typically match student and teacher probabilities via softmax, which blurs valuable logit information. While direct logit distillation (DLD) mitigates softmax smoothing, it fails to account for logit shift invariance, thereby restricting the solution space. We propose Concrete Score Distillation (CSD), a discrete score-matching objective that overcomes both softmax-induced smoothing and restrictions on the optimal solution set. We resolve the training instability and quadratic complexity of discrete score-matching in autoregressive LLMs, and the resulting CSD objective aligns relative logit differences across all vocabulary pairs between student and teacher with flexible weighting. We provide both modeseeking and mode-covering instances within our framework and evaluate CSD on task-agnostic instruction-following and task-specific distillation using GPT-2-1.5B, OpenLLaMA-7B, and GEMMA-7B-IT. Experiments show that CSD consistently surpasses recent KD objectives, achieves favorable fidelity-diversity trade-offs, and yields complementary gains when combined with on-policy techniques, demonstrating its scalability and effectiveness for LLM distillation.

### 11. LEARNABLE PARALLEL DECODING FOR DLLMS

**主要机构**: National University of Singapore
**作者数量**: 5人

**摘要**:
Diffusion large language models (dLLMs) have recently drawn considerable attention within the research community as a promising alternative to autoregressive generation, offering parallel token prediction and lower inference latency. Yet, their parallel decoding potential remains largely underexplored, as existing open-source models still require nearly token-length decoding steps to ensure performance. To address this, we introduce dParallel, a simple and effective method that unlocks the inherent parallelism of dLLMs for fast sampling. We identify that the key bottleneck to parallel decoding arises from the sequential certainty convergence for masked tokens. Building on this insight, we introduce the core of our approach: certainty-forcing distillation, a novel training strategy that distills the model to follow its original sampling trajectories while enforcing it to achieve high certainty on masked tokens more rapidly and in parallel. Extensive experiments across various benchmarks demonstrate that our method can dramatically reduce the number of decoding steps while maintaining performance. When applied to the LLaDA-8B-Instruct model, dParallel reduces decoding steps from 256 to 30 on GSM8K, achieving an 8.5× speedup without performance degradation. On the MBPP benchmark, it cuts decoding steps from 256 to 24, resulting in a 10.5× speedup while maintaining accuracy. Our code is available at https://github.com/czg1225/dParallel

### 12. dVLA: DIFFUSION VISION-LANGUAGE-ACTION MODEL WITH MULTIMODAL CHAIN-OF-THOUGHT

**主要机构**: Midea Group, Shanghai Jiaotong University, Peking University
**作者数量**: 9人

**摘要**:
Vision-Language-Action (VLA) models are emerging as a next-generation paradigm for robotics. We introduce dVLA, a diffusion-based VLA that leverages a multimodal chain-of-thought to unify visual perception, language reasoning, and robotic control in a single system. dVLA jointly optimizes perception, language understanding, and action under a single diffusion objective, enabling stronger cross-modal reasoning and better generalization to novel instructions and objects. For practical deployment, we mitigate inference latency by incorporating two acceleration strategies-a prefix attention mask and key-value (KV) caching-yielding up to ∼ 2× speedup at test-time inference. We evaluate dVLA in both simulation and the real world: on the LIBERO benchmark it achieves state-of-the-art performance with a 96.4% average success rate, consistently surpassing both discrete and continuous action policies; on a real Franka robot, it succeeds across a diverse task suite, including a challenging bin-picking task that requires multi-step planning, demonstrating robust real-world performance. Together, these results underscore the promise of unified diffusion frameworks for practical, high-performance VLA robotics.

### 13. ENHANCING LINEAR ATTENTION WITH RESIDUAL LEARNING

**主要机构**: The University of Hong, Peking University
**作者数量**: 5人

**摘要**:
Linear attention offers a linear-time alternative to self-attention but often struggles to capture long-range patterns. We revisit linear attention through a predictioncorrection lens and show that prevalent variants can be written as a combination of a historical prediction and a single-token correction, which creates an expressivity bottleneck. To address this bottleneck, we introduce Residual Linear Attention (RLA), a framework that equips linear attention with an explicit residual-fitting mechanism. RLA maintains an auxiliary recurrent state that learns to accumulate residual errors over time and correct the base prediction. We further instantiate a delta-rule version, Residual Delta Net (RDN), incorporating adaptive gating and residual clipping for enhanced correction control and stability. Our implementation leverages highly optimized linear attention kernels and preserves linear time and memory. Across language modeling and recall-intensive evaluations, RLA and RDN consistently outperform their respective baselines and other modern linear-attention methods, narrowing the gap to standard Transformers while retaining linear scaling.

### 14. FAST-D LLM V 2: Efficient Block-Diffusion LLM

**主要机构**: The University of Hong
**作者数量**: 12人

**摘要**:
Autoregressive (AR) large language models (LLMs) have achieved remarkable performance across a wide range of natural language tasks, yet their inherent sequential decoding limits inference efficiency. In this work, we propose Fast-dLLM v2, a carefully designed block diffusion language model (dLLM) that efficiently adapts pretrained AR models into dLLMs for parallel text generation-requiring only ∼1B tokens of fine-tuning. This represents a 500× reduction in training data compared to full-attention diffusion LLMs such as Dream (580B tokens), while preserving the original model's performance. Our approach introduces a novel training recipe that combines a block diffusion mechanism with a complementary attention mask, enabling blockwise bidirectional context modeling without sacrificing AR training objectives. To further accelerate decoding, we design a hierarchical caching mechanism: a block-level cache that stores historical context representations across blocks, and a sub-block cache that enables efficient parallel generation within partially decoded blocks. Coupled with our parallel decoding pipeline, Fast-dLLM v2 achieves up to 2.5× speedup over standard AR decoding without compromising generation quality. Extensive experiments across diverse benchmarks demonstrate that Fast-dLLM v2 matches or surpasses AR baselines in accuracy, while delivering state-of-the-art efficiency among dLLMs-marking a significant step toward the practical deployment of fast and accurate LLMs. Code and model will be publicly released.

### 15. FlashOmni: A Unified Sparse Attention Engine for Diffusion Transformers FLASHOMNI: A UNIFIED SPARSE ATTENTION ENGINE FOR DIFFUSION TRANSFORMERS

**主要机构**: University of Edinburgh, University of Science and Technology of China
**作者数量**: 6人

**摘要**:
Multi-Modal Diffusion Transformers (DiTs) demonstrate exceptional capabilities in visual synthesis, yet their deployment remains constrained by substantial computational demands. To alleviate this bottleneck, many sparsity-based acceleration methods have been proposed. However, their diverse sparsity patterns often require customized kernels for high-performance inference, limiting universality. We propose FlashOmni, a unified sparse attention engine compatible with arbitrary DiT architectures. FlashOmni introduces flexible sparse symbols to standardize the representation of a wide range of sparsity strategies, such as feature caching and block-sparse skipping. This unified abstraction enables the execution of diverse sparse computations within a single attention kernel. In addition, FlashOmni designs optimized sparse GEMMs for attention blocks, leveraging sparse symbols to eliminate redundant computations and further improve efficiency. Experiments demonstrate that FlashOmni delivers near-linear, closely matching the sparsity ratio speedup (1:1) in attention and GEMM-Q, and achieves 2.5×-3.8× acceleration in GEMM-O (max peaking at about 87.5% of the theoretical limit). Applied with a multi-granularity sparsity strategy, it enables the Hunyuan model (33K) to achieve about 1.5× end-to-end acceleration without degrading visual quality.

### 16. FROM MNIST TO IMAGENET: UNDERSTANDING THE SCALABILITY BOUNDARIES OF DIFFERENTIABLE LOGIC GATE NETWORKS

**主要机构**: ETH Zürich Zürich
**作者数量**: 4人

**摘要**:
Differentiable Logic Gate Networks (DLGNs) are a very fast and energy-efficient alternative to conventional feed-forward networks. With learnable combinations of logical gates, DLGNs enable fast inference by hardware-friendly execution. Since the concept of DLGNs has only recently gained attention, these networks are still in their developmental infancy, including the design and scalability of their output layer. To date, this architecture has primarily been tested on datasets with up to ten classes. This work examines the behavior of DLGNs on large multi-class datasets. We investigate its general expressiveness, its scalability, and evaluate alternative output strategies. Using both synthetic and real-world datasets, we provide key insights into the importance of temperature tuning and its impact on output layer performance. We evaluate conditions under which the Group-Sum layer performs well and how it can be applied to large-scale classification of up to 2000 classes.

### 17. HILBERTA: HILBERT ATTENTION FOR IMAGE GENERATION WITH DIFFUSION MODELS

**主要机构**: Department of Computer Science, New York University
**作者数量**: 5人

**摘要**:
Designing sparse attention for diffusion transformers requires reconciling twodimensional spatial locality with GPU efficiency, a trade-off that current methods struggle to achieve. Existing approaches enforce two-dimensional spatial locality but often incur uncoalesced memory access. We present HilbertA, a 2D-aware and GPUefficient sparse attention mechanism. HilbertA reorders image tokens along Hilbert curves to achieve a contiguous memory layout while preserving spatial neighborhoods, and employs a sliding schedule across layers to enable long-range information propagation without repeated or uncoalesced memory access. To further enhance cross-tile communication and positional awareness, HilbertA introduces a small central shared region. Implemented in Triton, HilbertA delivers comparable image quality with significant acceleration over prior methods on Flux.1-dev, demonstrating the feasibility of hardware-aligned two-dimensional sparse attention for high-resolution image generation. HilbertA delivers attention speedups of 2.3× when generating 1024×1024 images, and up to 4.17× at 2048×2048, while achieving image quality comparable to or surpassing baselines.

### 18. Interpret, prune and distill Donut : towards lightweight VLMs for VQA on documents

**主要机构**: Université Paris Cité, LIPADE, DIENS, École Normale Supérieure of Paris
**作者数量**: 3人

**摘要**:
Recent advances in Visually-rich Document Understanding rely on large Vision-Language Models like Donut, which perform documentlevel Visual Question Answering without Optical Character Recognition. Despite their effectiveness, these models are too costly for real-time or resource-constrained applications. We investigate model compression through knowledge distillation, training compact student models from a larger teacher. We leverage mechanistic interpretability to drive student architecture design within this framework. By analyzing internal computations, we identify essential subcomponents to retain, while having a clear view of which subcomponents should be approximated, skipped, or reparametrized based on their function. This approach yields Donut-MINT (Mechanistic Interpretability-based Network Trimming), a pruned Donut variant that reduces inference time and memory usage while maintaining strong performance on DocVQA, a standard benchmark for document Visual Question Answering. Our method reframes compression as circuit discovery, bridging interpretability research and practical Vision-Language Model deployment.

### 19. 

**主要机构**: New York University
**作者数量**: 4人

**摘要**:
Knowledge distillation is a common paradigm for transferring capabilities from larger models to smaller ones. While traditional distillation methods leverage a probabilistic divergence over the output of the teacher and student models, featurebased distillation methods often minimize variants of Euclidean norms between the hidden layer representations. The main goal is for the student to mimic the structure of the feature space of the teacher. In this work, we theoretically show that existing feature distillation methods, such as projection based mean squared loss or Centered Kernel Alignment (CKA), cannot capture the feature structure, even under zero loss. We then motivate the use of Procrustes distance and the Frobenius norm of Feature Gram Matrix, distances already common in the context of measuring representational alignment, as distillation losses. We show that feature distillation through our method showcases statistically significant improvement in distillation performance across language models families (BERT and OPT) in classification and instruction-following tasks by up to 2 percentage points, showcasing the potential of integrating feature geometry into existing distillation methods. 1

### 20. ON-THE-FLY ADAPTATION TO QUANTIZATION: CONFIGURATION-AWARE LORA FOR EFFICIENT FINE-TUNING OF QUANTIZED LLMS

**主要机构**: Southern University of Science and Technology, The University of Hong Kong
**作者数量**: 3人

**摘要**:
As increasingly large pre-trained models are released, deploying them on edge devices for privacy-preserving applications requires effective compression. Recent works combine quantization with the fine-tuning of high-precision LoRA adapters, which can substantially reduce model size while mitigating the accuracy loss from quantization. However, edge devices have inherently heterogeneous capabilities, while performing configuration-wise fine-tuning for every quantization setting is computationally prohibitive. In this paper, we propose CoA-LoRA, a method that dynamically adjusts the LoRA adapter to arbitrary quantization configurations (i.e., the per-layer bit-width choices of a pre-trained model) without requiring repeated fine-tuning. This is accomplished via a configuration-aware model that maps each configuration to its low-rank adjustments. The effectiveness of this model critically depends on the training configuration set, a collection of configurations chosen to cover different total bit-width budgets. However, constructing a high-quality configuration set is non-trivial. We therefore design a Pareto-based configuration search that iteratively optimizes the training configuration set, yielding more precise low-rank adjustments. Our experiments demonstrate that, unlike the state-of-the-art methods that require fine-tuning a separate LoRA adapter for each configuration, CoA-LoRA incurs no additional time cost while achieving comparable or even superior performance to those methods.

### 21. Parallax: Efficient LLM Inference Service over Decentralized Environment

**主要机构**: National University of Singapore
**作者数量**: 11人

**摘要**:
Deploying a large language model (LLM) inference service remains costly because centralized serving depends on specialized GPU clusters and high-bandwidth interconnects in datacenters. An appealing alternative is to leverage collaborative decentralized GPU pools. However, heterogeneity in GPU and limited interconnected network bandwidth, along with potentially dynamic availability, make efficient scheduling the central challenge in this scenario. In this paper, we present Parallax, a decentralized LLM serving system that turns a pool of heterogeneous GPUs into an efficient inference platform via a two-phase scheduler. Parallax decomposes planning into (i) model allocation, which places layers of each replica across diverse GPUs to jointly optimize latency and throughput under memory and link-bandwidth constraints, and (ii) request-time GPU pipeline selection, which stitches layers from different replicas into end-to-end execution chains that balance load and adapt to current conditions. We implement Parallax and evaluate it on open-source LLMs deployed over real volunteer nodes. Parallax consistently reduces latency and increases throughput relative to decentralized baselines, demonstrating that principled scheduling can make volunteer compute a practical, affordable substrate for LLM inference.

### 22. POST-TRAINING QUANTIZATION VIA RESIDUAL TRUNCATION AND ZERO SUPPRESSION FOR DIF-FUSION MODELS

**主要机构**: Department of Computer Science, Department of Electrical Engineering, Kyung Hee University Yongin-si, Department of Artificial Intelligence
**作者数量**: 4人

**摘要**:
Diffusion models achieve high-quality image generation but face deployment challenges due to their high computational requirements. Although 8-bit outlieraware Post-Training Quantization (PTQ) matches full-precision performance, extending PTQ to 4 bits remains challenging. Larger step sizes in 4-bit quantization amplify rounding errors in dense, low-magnitude activations, leading to the loss of fine-grained textures. We hypothesize that not only outliers but also small activations are critical for texture fidelity. To this end, we propose Quantization via Residual Truncation and Zero Suppression (QuaRTZ), a 4-bit PTQ scheme for diffusion models. QuaRTZ applies 8-bit min-max quantization for outlier handling and compresses to 4 bits via leading-zero suppression to retain LSBs, thereby preserving texture details. Our approach reduces rounding errors and improves quantization efficiency by balancing outlier preservation and LSB precision. Both theoretical derivations and empirical evaluations demonstrate the generalizability of QuaRTZ across diverse activation distributions. Notably, 4-bit QuaRTZ achieves an FID of 6.98 on FLUX.1-schnell, outperforming SVDQuant that requires auxiliary FP16 branches.

### 23. Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation

**主要机构**: Huawei Noah's Ark Lab
**作者数量**: 9人

**摘要**:
The rapid advancement of large language models (LLMs) has significantly advanced the capabilities of artificial intelligence across various domains. However, their massive scale and high computational costs render them unsuitable for direct deployment in resource-constrained edge environments. This creates a critical need for highperformance small models that can operate efficiently at the edge. Yet, after pre-training alone, these smaller models often fail to meet the performance requirements of complex tasks. To bridge this gap, we introduce a systematic post-training pipeline that efficiently enhances small model accuracy. Our post training pipeline consists of curriculum-based supervised fine-tuning (SFT) and offline on-policy knowledge distillation. The resulting instruction-tuned model achieves stateof-the-art performance among billion-parameter models, demonstrating strong generalization under strict hardware constraints while maintaining competitive accuracy across a variety of tasks. This work provides a practical and efficient solution for developing high-performance language models on Ascend edge devices.

### 24. SKIP-IT? THEORETICAL CONDITIONS FOR LAYER SKIPPING IN VISION-LANGUAGE MODELS

**主要机构**: Department of Electrical & Computer Engineering, University of Illinois, Department of Electrical & Computer Engineering Department of Mathematics, AI Innovation Institute Stony Brook University
**作者数量**: 10人

**摘要**:
Vision-language models (VLMs) achieve incredible performance across a wide range of tasks, but their large size makes inference costly. Recent work shows that selectively skipping VLM layers can improve efficiency with minimal performance loss or even performance improvements. However, this technique remains underused due to the limited understanding of when layer skipping is beneficial. In this paper, we develop a framework that uses information and learning theory to characterize the conditions under which layer skipping enhances efficiency without sacrificing performance. Motivated by these observations, we analyze the evolution of the VLM's hidden representations through the LLM backbone and show that layers with large redundancy as predicted by our framework coincide with those skipped by popular layer-skipping methods in practice, providing a unified theoretical scaffolding for multiple efficient inference techniques. Our experiments demonstrate that skipping such layers yields faster inference that preserves performance, and also show that applying skipping outside these conditions leads to model degradation.

### 25. Training Matryoshka Mixture-of-Experts for Elastic Inference-Time Expert Utilization

**主要机构**: Shanghai Jiao Tong University, University of Science and Technology of China, Xiamen University
**作者数量**: 9人

**摘要**:
Mixture-of-Experts (MoE) has emerged as a promising paradigm for efficiently scaling large language models without a proportional increase in computational cost. However, the standard training strategy of Top-K router prevents MoE models from realizing their full potential for elastic inference. When the number of activated experts is altered at inference time, these models exhibit precipitous performance degradation. In this work, we introduce Matryoshka MoE (M-MoE), a training framework that instills a coarse-to-fine structure directly into the expert ensemble. By systematically varying the number of activated experts during training, M-MoE compels the model to learn a meaningful ranking: top-ranked experts collaborate to provide essential, coarse-grained capabilities, while subsequent experts add progressively finer-grained detail. We explore this principle at multiple granularities, identifying a layer-wise randomization strategy as the most effective. Our experiments demonstrate that a single M-MoE model achieves remarkable elasticity, with its performance at various expert counts closely matching that of an entire suite of specialist models, but at only a fraction of the total training cost. This flexibility not only unlocks elastic inference but also enables optimizing performance by allocating different computational budgets to different model layers. Our work paves the way for more practical and adaptable deployments of large-scale MoE models.

### 26. UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression

**主要机构**: Dalian University of Technology, IIAU-Lab, X3000 Inspection Co
**作者数量**: 7人

**摘要**:


### 27. Vocabulary Customization for Efficient Domain-Specific LLM Deployment

**主要机构**: eBay Inc
**作者数量**: 5人

**摘要**:
When using an LLM to process text outside the training domain(s), an often overlooked factor is vocabulary mismatch, where the general-domain tokenizer fails to capture frequent domain-specific terms, leading to higher token fertility and thus a decrease in processing speed due to suboptimal sub-word splits. We address this limitation by augmenting the pretrained vocabulary with a set of domain-specific tokens. To this end, we design an algorithm that extends an existing tokenizer while guaranteeing it never decreases tokenization efficiency: every input sequence is segmented into at most the same number of tokens as before. Evaluated on real-world e-commerce use-cases, the augmented tokenizer significantly shortens input sequences by up to 20% and reduces inference latency on downstream tasks while preserving predictive quality. We further analyze secondary effects, such as the impact on forward pass speed and the rate at which the model adopts the newly introduced tokens, to illustrate the broader benefits of vocabulary adaptation.

### 28. VRWKV-EDITOR: REDUCING QUADRATIC COMPLEXITY IN TRANSFORMER-BASED VIDEO EDITING

**主要机构**: Morocco University Mohammed VI Polytechnic Rabat, International Artificial Intelligence Center, Morocco University Mohammed VI Polytechnic Sorbonne University
**作者数量**: 4人

**摘要**:
In light of recent progress in video editing, deep learning models focusing on both spatial and temporal dependencies have emerged as the primary method. However, these models suffer from the quadratic computational complexity of traditional attention mechanisms, making them difficult to adapt to long-duration and high-resolution videos. This limitation restricts their applicability in practical contexts such as real-time video processing. To tackle this challenge, we introduce a method to reduce both time and space complexity of these systems by proposing VRWKV-Editor, a novel video editing model that integrates a linear spatio-temporal aggregation module into video-based diffusion models. VRWKV-Editor leverages bidirectional weighted key-value recurrence mechanism of the RWKV transformer to capture global dependencies while preserving temporal coherence, achieving linear complexity without sacrificing quality. Extensive experiments demonstrate that the proposed method achieves up to 3.7× speedup and 60% lower memory usage compared to state-of-the-art diffusion-based video editing methods, while maintaining competitive performance in frame consistency and text alignment. Furthermore, a comparative analysis we conducted on videos with different sequence lengths confirms that the gap in editing speed between our approach and architectures with self-attention becomes more significant with long videos.
