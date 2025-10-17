# AI推理加速技术论文分析报告
生成时间: 2025-10-17 13:05:02
分析论文数量: 17篇

## 论文技术简报

### 1. Accelerated Feature Detectors for Visual SLAM: A Comparative Study of FPGA vs GPU

Nvidia与AMD发布了《Accelerated Feature Detectors for Visual SLAM: A Comparative Study of FPGA vs GPU》论文，使用FPGA与GPU加速的特征检测器比较研究技术，解决了视觉SLAM中特征检测耗时及功耗受限平台（如无人机）的加速问题，实现学习型检测器SuperPoint的FPGA实现较GPU性能提升3.1×、能效提升1.4×，且硬件加速可减少全局束调整调用而不牺牲精度。

### 2. BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure

澳门大学与深圳先进院发布了BanaServe论文，使用统一KV缓存和动态模块迁移技术，解决了分离式LLM服务中的平衡问题，达成了提升吞吐量和资源效率的效果

### 3. BREADCRUMBS REASONING: MEMORY-EFFICIENT REASONING WITH COMPRESSION BEACONS

哈佛大学、康奈尔大学发布了BREADCRUMBS REASONING论文，使用通过联合蒸馏和强化学习训练的专用令牌定期压缩生成KV缓存并驱逐压缩条目的技术，解决了大语言模型长上下文推理时Transformer键值缓存线性增长导致的内存和计算成本高的问题，达成了比无缓存压缩和无训练压缩技术更优的内存-精度帕累托前沿的效果。

### 4. CLOSING THE GAP BETWEEN TEXT AND SPEECH UNDERSTANDING IN LLMS

CNRS、Aix Marseille Université发布了CLOSING THE GAP BETWEEN TEXT AND SPEECH UNDERSTANDING IN LLMS论文，使用SALAD（结合跨模态蒸馏与目标合成数据的样本高效对齐方法）技术，解决了LLMs的文本-语音理解差距及现有方法数据效率低的问题，达成了3B/7B模型用少一个数量级公共语音数据训练，在知识、语言理解、推理基准上与强开源模型竞争性能的效果。

### 5. DistilCLIP-EEG: Enhancing Epileptic Seizure Detection Through Multi-modal Learning and Knowledge Distillation

常州大学发布了DistilCLIP-EEG论文，使用基于CLIP框架的多模态学习（整合EEG信号与文本描述）及知识蒸馏技术（含Conformer架构EEG编码器、BERT-LP提示学习），解决了现有癫痫检测依赖单模态EEG信号忽略多模态信息的问题，在TUSZ等数据集上准确率超97%、F1-score超0.94，学生模型参数和大小约为教师的58.1%，显著降低复杂度同时保持高性能。

### 6. Efficient Adaptive Transformer: An Empirical Study and Reproducible Framework

发布了《Efficient Adaptive Transformer: An Empirical Study and Reproducible Framework》论文，使用EAT开源框架（结合渐进式token剪枝、稀疏注意力和动态早期退出的统一架构），解决了延迟敏感应用中Transformer计算动态调整的技术组合效果研究问题，达成了在SST-2上准确率略高于优化的DISTILBERT基线并提供社区探索工具的效果。

### 7. End-to-End Multi-Modal Diffusion Mamba

University of Wisconsin-Milwaukee和China University of Petroleum-Beijing发布了End-to-End Multi-Modal Diffusion Mamba论文，使用基于Mamba的多步选择扩散模型及统一变分自编码器，解决了现有端到端多模态模型中编码器解码器分离阻碍联合表示学习的问题，在图像生成、VQA等多任务上显著优于现有端到端模型，并能与GPT-4V等SOTA模型竞争，验证了统一多模态处理的有效性和计算效率。

### 8. F-BFQ: Flexible Block Floating-Point Quantization Accelerator for LLMs

格拉斯哥大学发布了F-BFQ加速器论文，使用Flexible Block Floating-Point Quantization (F-BFQ)加速器技术，解决了BFP量化LLM加速器需支持不同BFP变体而无需重新配置的问题，达成了在AMD Kria板上平均推理时间较Arm NEON CPU快1.4倍、达到5.2 tokens/秒的效果

### 9. GatePro: Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

ByteDance Seed发布了GatePro论文，使用参数-free的专家选择优化方法（通过识别相似专家对引入局部竞争机制），解决了MoE模型中功能相似专家同时被选导致的冗余计算与有效容量受限问题，达成了增强专家多样性、使专家发展出更独特互补能力且无需额外参数可热插拔部署的效果。

### 10. MIRROR SPECULATIVE DECODING: BREAKING THE SERIAL BARRIER IN LLM INFERENCE

研究团队发布了MIRROR SPECULATIVE DECODING论文，使用Mirror-SD技术（结合异构加速器并行计算与推测流），解决了LLM推理中推测解码的延迟-接受率权衡问题，达成2.8×-5.8×墙时加速、较EAGLE3平均提升30%的效果

### 11. NEURORVQ: MULTI-SCALE EEG TOKENIZATION FOR GENERATIVE LARGE BRAINWAVE MODELS

帝国理工学院发布了NEURORVQ论文，使用多尺度EEG标记化、分层残差向量量化（RVQ）码本及相位和幅度感知损失函数，解决了现有EEG标记器无法保留高频动态导致重建保真度低的问题，达成了更低重建误差并在多种下游任务上优于现有大型脑波模型的效果。

### 12. NOSA: Native and Offloadable Sparse Attention

清华大学发布了NOSA: Native and Offloadable Sparse Attention论文，使用通过将token选择分解为查询感知和查询无关组件引入显式局部性约束的NOSA框架，解决了现有可训练稀疏注意力方法中KV缓存大小未减少导致GPU批处理大小受限和解码吞吐量低的问题，达成了保持近乎无损性能的同时解码吞吐量较基准（InfLLM-V2）提升达2.3倍的效果

### 13. SCOPE: SELECTIVE CROSS-MODAL ORCHESTRATION OF VISUAL PERCEPTION EXPERTS

蒙特利尔大学和英属哥伦比亚大学发布了SCOPE论文，使用Mixture-of-Encoders框架通过动态实例级路由选择专用视觉编码器（结合交叉注意力路由器与双熵正则化训练），解决了视觉语言模型多编码器效果递减且推理成本高的问题，达成1个共享+1个路由编码器性能优于4个额外编码器同时使用且计算量减少24-49%的效果。

### 14. Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation

Microsoft Research Asia与Peking University发布了Speculative Decoding（SpecDec）论文，使用Spec-Drafter（高效准确草稿模型）和Spec-Verification（可靠草稿验证方法）的推测执行技术，解决了自回归（AR）解码加速问题，达成了对Transformer架构约5倍加速且生成质量与束搜索解码相当的效果

### 15. TAMING THE FRAGILITY OF KV CACHE EVICTION IN LLM INFERENCE

中国科学技术大学发布了TAMING THE FRAGILITY OF KV CACHE EVICTION IN LLM INFERENCE论文，使用防御性聚合策略（两步线性时间方法控制最坏情况风险）及DefensiveKV、Layer-DefensiveKV方法，解决了KV缓存驱逐中稳定性假设脆弱导致的极端情况风险问题，在20%缓存大小下相比最强基线生成质量损失分别减少2.3倍和4.3倍。

### 16. Transformer-based Scalable Beamforming Optimization via Deep Residual Learning

相关机构发布了基于Transformer的可扩展波束赋形优化论文，使用基于L2O范式的多层Transformer残差学习及课程学习、半摊销学习、滑动窗口训练策略的无监督深度学习框架，解决了大规模MU-MISO信道下行波束赋形优化及动态环境下实时推理问题，达成了低中SNR优于现有基线、高SNR接近WMMSE性能且推理速度显著快于迭代和在线学习方法的效果。

### 17. UniMoE-Audio: Unified Speech and Music Generation with Dynamic-Capacity MoE

相关机构发布了UniMoE-Audio论文，使用动态容量MoE框架（含Top-P路由与混合专家设计）及三阶段训练课程，解决了语音和音乐生成孤立开发、任务冲突与数据不平衡问题，在主要基准上达成SOTA性能并缓解了联合训练的性能下降。

## 论文详细信息

### 1. Accelerated Feature Detectors for Visual SLAM: A Comparative Study of FPGA vs GPU

**主要机构**: 
**作者数量**: 2人

**摘要**:
Feature detection is a common yet time-consuming module in Simultaneous Localization and Mapping (SLAM) implementations, which are increasingly deployed on powerconstrained platforms, such as drones. Graphics Processing Units (GPUs) have been a popular accelerator for computer vision in general, and feature detection and SLAM in particular. On the other hand, System-on-Chips (SoCs) with integrated Field Programmable Gate Array (FPGA) are also widely available. This paper presents the first study of hardwareaccelerated feature detectors considering a Visual SLAM (V-SLAM) pipeline. We offer new insights by comparing the best GPU-accelerated FAST, Harris, and SuperPoint implementations against the FPGA-accelerated counterparts on modern SoCs (Nvidia Jetson Orin and AMD Versal). The evaluation shows that when using a non-learning-based feature detector such as FAST and Harris, their GPU implementations, and the GPU-accelerated V-SLAM can achieve better run-time performance and energy efficiency than the FAST and Harris FPGA implementations as well as the FPGAaccelerated V-SLAM. However, when considering a learningbased detector such as SuperPoint, its FPGA implementation can achieve better run-time performance and energy efficiency (up to 3.1× and 1.4× improvements, respectively) than the GPU implementation. The FPGA-accelerated V-SLAM can also achieve comparable run-time performance compared to the GPU-accelerated V-SLAM, with better FPS in 2 out of 5 dataset sequences. When considering the accuracy, the results show that the GPU-accelerated V-SLAM is more accurate than the FPGA-accelerated V-SLAM in general. Last but not least, the use of hardware acceleration for feature detection could further improve the performance of the V-SLAM pipeline by having the global bundle adjustment module invoked less frequently without sacrificing accuracy.

### 2. BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure

**主要机构**: Faculty of Science and Technology, University of Macau, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, State Key Lab of IOTSC, Southern University of Science and Technology, Alibaba Group Inc, HangZhou, AIOS Team
**作者数量**: 11人

**摘要**:
Large language models (LLMs) are increasingly deployed in AI infrastructure, driving the need for high throughput, resource efficient serving systems. Disaggregated LLM serving, which separates prompt prefill from auto-regressive decode, has emerged as a promising architecture by isolating their heterogeneous compute and memory demands. However, current disaggregated systems face three key lim

### 3. BREADCRUMBS REASONING: MEMORY-EFFICIENT REASONING WITH COMPRESSION BEACONS

**主要机构**: Harvard University, Cornell University
**作者数量**: 5人

**摘要**:
The scalability of large language models for long-context reasoning is severely constrained by the linear growth of their Transformer key-value cache, which incurs significant memory and computational costs. We posit that as a model generates reasoning tokens, the informational value of past generated tokens diminishes, creating an opportunity for compression. In this work, we propose to periodically compress the generation KV cache with a learned, special-purpose token and evict compressed entries. We train the model to perform this compression via a modified joint distillation and reinforcement learning (RL) framework. Our training method minimizes overhead over the conventional RL process, as it leverages RL outputs for distillation. Empirically, our method achieves a superior memory-accuracy Pareto frontier compared to both the model without cache compression and training-free compression techniques.

### 4. CLOSING THE GAP BETWEEN TEXT AND SPEECH UNDERSTANDING IN LLMS

**主要机构**: Université de Toulon, CNRS, Aix Marseille Université
**作者数量**: 8人

**摘要**:
Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts-and even cascaded pipelines-on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD-Sample-efficient Alignment with Learning through Active selection and cross-modal Distillationwhich combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broaddomain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora.

### 5. DistilCLIP-EEG: Enhancing Epileptic Seizure Detection Through Multi-modal Learning and Knowledge Distillation

**主要机构**: Changzhou Univer- sity, Department of Computing, Changzhou University, Xi'an JiaoTong- Liverpool University, Aliyun School of Big Data
**作者数量**: 7人

**摘要**:
Epilepsy is a prevalent neurological disorder marked by sudden, brief episodes of excessive neuronal activity caused by abnormal electrical discharges, which may lead to some mental disorders. Most existing deep learning methods for epilepsy detection rely solely on unimodal EEG signals, neglecting the potential benefits of multimodal information. To address this, we propose a novel multimodal model, DistilCLIP-EEG, based on the CLIP framework, which integrates both EEG signals and text descriptions to capture comprehensive features of epileptic seizures. The model involves an EEG encoder based on the Conformer architecture as a text encoder, the proposed Learnable BERT (BERT-LP) as prompt learning within the encoders. Both operate in a shared latent space for effective cross-modal representation learning. To enhance efficiency and adaptability, we introduce a knowledge distillation method where the trained DistilCLIP-EEG serves as a teacher to guide a more compact student model to reduce training complexity and time. On the TUSZ, AUBMC, and CHB-MIT datasets, both the teacher and student models achieved accuracy rates exceeding 97%. Across all datasets, the F1-scores were consistently above 0.94, demonstrating the robustness and reliability of the proposed framework. Moreover, the student model's parameter count and model size are approximately 58.1% of those of the teacher model, significantly reducing model complexity and storage requirements while maintaining high performance. These results highlight the potential of our proposed model for EEG-based epilepsy detection and establish a solid foundation for deploying lightweight models in resource-constrained settings.

### 6. Efficient Adaptive Transformer: An Empirical Study and Reproducible Framework

**主要机构**: 
**作者数量**: 1人

**摘要**:
The concept of an "Efficient Adaptive Transformer" (EAT) that dynamically adjusts its computation is promising for latency-sensitive applications. This paper introduces the EAT framework-a reproducible, open-source tool designed to investigate the interplay of progressive token pruning, sparse attention, and dynamic early exiting in a unified architecture. We present a fully automated benchmarking protocol to rigorously analyze their combined effect on GLUE tasks (SST-2, QQP, MNLI). Our empirical study on a 6-layer architecture reveals a complex performance trade-off, finding that this direct combination can increase latency. However, the framework shows potential by achieving a slightly higher accuracy on SST-2 than the optimized DISTILBERT baseline, suggesting the architecture's capacity for high performance. The primary contribution of this work is not a new state-of-the-art model, but the open-source framework and the empirical analysis itself, which we offer as a tool for the community to investigate more effective configurations. All code, training scripts, and analysis utilities are released to facilitate this exploration.

### 7. End-to-End Multi-Modal Diffusion Mamba

**主要机构**: University of Wisconsin-Milwaukee, China University of Petroleum-Beijing
**作者数量**: 4人

**摘要**:
Current end-to-end multi-modal models utilize different encoders and decoders to process input and output information. This separation hinders the joint representation learning of various modalities. To unify multi-modal processing, we propose a novel architecture called MDM (Multi-modal Diffusion Mamba). MDM utilizes a Mamba-based multistep selection diffusion model to progressively generate and refine modality-specific information through a unified variational autoencoder for both encoding and decoding. This innovative approach allows MDM to achieve superior performance when processing high-dimensional data, particularly in generating high-resolution images and extended text sequences simultaneously. Our evaluations in areas such as image generation, image captioning, visual question answering, text comprehension, and reasoning tasks demonstrate that MDM significantly outperforms existing end-toend models (MonoFormer, LlamaGen, and Chameleon etc.) and competes effectively with SOTA models like GPT-4V, Gemini Pro, and Mistral. Our results validate MDM's effectiveness in unifying multi-modal processes while maintaining computational efficiency, establishing a new direction for end-to-end multi-modal architectures.

### 8. F-BFQ: Flexible Block Floating-Point Quantization Accelerator for LLMs

**主要机构**: School of Computing Science, University of Glasgow
**作者数量**: 2人

**摘要**:
Large Language Models (LLMs) have become increasingly prominent for daily tasks, from improving sound-totext translation to generating additional frames for the latest video games. With the help of LLM inference frameworks, such as llama.cpp, which support optimizations such as KV-caching and quantization, it is now easier than ever to deploy LLMs on edge devices. Quantization is fundamental to enable LLMs on resource-constrained edge devices, and llama.cpp utilizes block floating point (BFP) quantization to drastically reduce the bit width of weights and input tensors, the memory footprint, and the computational power required to run LLMs. LLMs are typically quantized with mixed BFP quantization across the model layers to reduce the loss of model accuracy due to quantization. Therefore, to efficiently accelerate across the layers of BFP-quantized LLMs, specialized accelerators need to support different BFP variants without reconfiguration. To address this issue, we propose a Flexible Block Floating-Point Quantization (F-BFQ) accelerator, which can dynamically switch between two BFP quantization variants and perform matrix multiplication (MatMul) operations. Our initial F-BFQ accelerator design, deployed on the AMD Kria board, reduces inference time by 1.4× on average over the Arm NEON-based CPU execution across three BFP quantized LLMs while achieving 5.2 tokens per second (∼3.9 words per second).

### 9. GatePro: Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

**主要机构**: ByteDance Seed
**作者数量**: 10人

**摘要**:
Modern large language models leverage Mixture-of-Experts (MoE) architectures for efficient scaling, but face a critical challenge: functionally similar experts are often selected simultaneously, creating redundant computation and limiting effective model capacity. Existing auxiliary balance loss methods improve token distribution but fail to address the underlying expert diversity problem. We introduce GatePro, a novel parameter-free method that directly promotes expert selection diversity. GatePro identifies the most similar expert pairs and introduces localized competition mechanisms, preventing redundant expert co-activation while maintaining natural expert specialization. Our comprehensive evaluation demonstrates GatePro's effectiveness across model scales and benchmarks. Analysis demonstrates GatePro's ability to achieve enhanced expert diversity, where experts develop more distinct and complementary capabilities, avoiding functional redundancy. This approach can be deployed hot-swappable during any training phase without additional learnable parameters, offering a practical solution for improving MoE effectiveness.

### 10. MIRROR SPECULATIVE DECODING: BREAKING THE SERIAL BARRIER IN LLM INFERENCE

**主要机构**: 
**作者数量**: 6人

**摘要**:
Speculative decoding accelerates LLM inference with draft lookahead, but its effectiveness is bottlenecked by autoregressive draft generation: larger drafts improve acceptance yet also increase speculation latency overhead, capping speedup. Existing approaches such as Medusa, Hydra, EAGLE partially address draft inefficiency, but ultimately trade acceptance rates for reduced draft latency, or preserve acceptance at the cost of added overheads that limit scaling. Modern SoCs increasingly integrate heterogeneous accelerators, most commonly GPUs and NPUs with complementary throughput and efficiency characteristics, yet existing approaches are accelerator-agnostic and usually place both draft and target on the same type of device, which leaves cross-accelerator parallelism unused. We introduce Mirror Speculative Decoding (Mirror-SD), which breaks the latency-acceptance tradeoff by launching branch-complete rollouts from earlyexit signals in parallel with the target's suffix and by explicitly mapping computation across heterogeneous accelerators. In this design, the draft speculates forward token continuations for target to verify, while the target speculates correction paths for the draft, creating a bidirectional speculative process. To further reduce draft speculation latency overhead while preserving acceptance semantics, we pair Mirror-SD with speculative streaming (SS) so the draft emits multiple tokens per step. This dual strategy of combining parallel heterogeneous execution and SS pushes speculative decoding closer to its ideal regime of high acceptance while reducing speculation overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD consistently delivers realistic end-to-end gains, achieving 2.8×-5.8× wall-time speedups across diverse tasks representing 30% average relative improvement over the strongest baseline, EAGLE3.

### 11. NEURORVQ: MULTI-SCALE EEG TOKENIZATION FOR GENERATIVE LARGE BRAINWAVE MODELS

**主要机构**: Cogitat Ltd, Northeastern University London, Imperial College London
**作者数量**: 7人

**摘要**:
Electroencephalography (EEG) captures neural activity across multiple temporal and spectral scales, yielding signals that are rich but complex for representation learning. Recently, EEG foundation models trained to predict masked signal-tokens have shown promise for learning generalizable representations. However, their performance is hindered by their signal tokenization modules. Existing neural tokenizers fail to preserve high-frequency dynamics, limiting their ability to reconstruct EEG signals with high fidelity. We introduce NEURORVQ, a scalable Large Brainwave Model (LBM) centered on a codebook-based tokenizer. Our tokenizer integrates: (i) multi-scale feature extraction modules that capture the full frequency neural spectrum; (ii) hierarchical residual vector quantization (RVQ) codebooks for high-resolution encoding; and, (iii) an EEG signal phase-and amplitude-aware loss function for efficient training. This design enables efficient EEG compression while supporting accurate reconstruction across all frequency bands, leading to robust generative masked modeling. Our empirical results demonstrate that NEURORVQ achieves lower reconstruction error and outperforms existing LBMs on a variety of downstream tasks. More broadly, the NEURORVQ tokenizer establishes a strong prior for codebook-based general-purpose brainwave models, enabling advances in neural decoding, generative modeling and multimodal biosignal integration.

### 12. NOSA: Native and Offloadable Sparse Attention

**主要机构**: Tsinghua University, Department of Computer Science and Technology
**作者数量**: 4人

**摘要**:
Trainable sparse attention has emerged as a promising solution to address the decoding efficiency bottleneck of LLMs in long-context processing, significantly saving memory accesses while minimally impacting task performance. However, existing sparse attention methods leave a crucial limitation unresolved: the size of the key-value (KV) cache remains unreduced, which constrains on-GPU batch sizes and throttles decoding throughput, especially in large-scale batched inference. In this paper, we show that trainable sparse attention naturally exhibits strong locality in token selection across adjacent decoding steps, thereby enabling KV cache offloading without altering the underlying attention computation. However, the inherent locality remains insufficient to achieve efficient offloading, as the transfer of selected KV pairs between the CPU and GPU continues to dominate the overall decoding cost. Building on this insight, we present NOSA, a trainable sparse attention framework designed to natively support KV cache offloading. NOSA introduces explicit locality constraints by decomposing token selection into query-aware and queryagnostic components, thereby reducing KV transfers while preserving the same attention computation as used during training. We pretrain a 1B-parameter model with NOSA and conduct extensive benchmarks, showing that it preserves near-lossless performance while achieving up to a 2.3× improvement in decoding throughput compared with the vanilla trainable sparse attention baseline (InfLLM-V2).

### 13. SCOPE: SELECTIVE CROSS-MODAL ORCHESTRATION OF VISUAL PERCEPTION EXPERTS

**主要机构**: Université de Montréal, University of British Columbia, Universitat Autònoma de Barcelona, ServiceNow
**作者数量**: 13人

**摘要**:
Vision-language models (VLMs) benefit from multiple vision encoders, but naively stacking them yields diminishing returns while multiplying inference costs. We propose SCOPE, a Mixture-of-Encoders (MoEnc) framework that dynamically selects one specialized encoder per image-text pair via instance-level routing, unlike token-level routing in traditional MoE. SCOPE maintains a shared encoder and a pool of routed encoders. A lightweight router uses cross-attention between text prompts and shared visual features to select the optimal encoder from the routed encoders. To train this router, we introduce dual entropy regularization with auxiliary losses to balance datasetlevel load distribution with instance-level routing confidence. Remarkably, SCOPE with one shared plus one routed encoder outperforms models using all four extra encoders simultaneously, while reducing compute by 24-49%. This demonstrates that intelligent encoder selection beats brute-force aggregation, challenging the prevailing paradigm in multi-encoder VLMs.

### 14. Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation

**主要机构**: Microsoft Research Asia, National Key Laboratory for Multimedia Information Processing, Peking University
**作者数量**: 6人

**摘要**:
We propose Speculative Decoding (SpecDec), for the first time ever 1 , to formally study exploiting the idea of speculative execution to accelerate autoregressive (AR) decoding. Speculative Decoding has two innovations: Spec-Drafter-an independent model specially optimized for efficient and accurate drafting-and Spec-Verification-a reliable method for verifying the drafted tokens efficiently in the decoding paradigm. Experimental results on various seq2seq tasks including machine translation and abstractive summarization show our approach can achieve around 5× speedup for the popular Transformer architectures with comparable generation quality to beam search decoding, refreshing the impression that the draft-then-verify paradigm introduces only 1.4×∼2× speedup. In addition to the remarkable speedup, we also demonstrate 3 additional advantages of SpecDec, revealing its practical value for accelerating generative models in real-world applications. Our models and codes are available at https://github.com/ hemingkx/SpecDec.

### 15. TAMING THE FRAGILITY OF KV CACHE EVICTION IN LLM INFERENCE

**主要机构**: School of Biomedical Engineering, School of Computer Science, University of Science and Technology of China
**作者数量**: 5人

**摘要**:
Large language models have revolutionized natural language processing, yet their deployment remains hampered by the substantial memory and runtime overhead of the transformer's Key-Value cache. To mitigate this, recent methods employ a scoring-aggregation framework to evict unimportant cache entries, based on the "stability assumption"-that a fixed subset of entries remains consistently important during generation. However, prior work has largely focused on refining importance indicators for scoring, while defaulting to mean aggregation due to a faithful trust in the stability assumption. In this work, we argue that this underlying assumption is inherently fragile, making mean aggregation highly vulnerable in extreme cases. To counter this, we propose a simple yet elegant defensive aggregation strategy: a twostep, linear-time approach that controls worst-case risk, thereby defending against extreme cases with negligible computational overhead. Embodying this strategy, we propose a novel cache eviction method, DefensiveKV and its extension, Layer-DefensiveKV, which incorporates layer-wise budget allocation. Across seven task domains (18 datasets), our methods reduce generation quality loss by 2.3× and 4.3× respectively, versus the strongest baseline under a 20% cache size. These results set new performance benchmarks and pioneer a promising direction for optimizing cache eviction against underlying fragility through worst-case risk management. Our code is available at https://github.com/FFY0/DefensiveKV.

### 16. Transformer-based Scalable Beamforming Optimization via Deep Residual Learning

**主要机构**: 
**作者数量**: 3人

**摘要**:
We develop an unsupervised deep learning framework for downlink beamforming in large-scale MU-MISO channels. The model is trained offline, allowing real-time inference through lightweight feedforward computations in dynamic communication environments. Following the learning-to-optimize (L2O) paradigm, a multi-layer Transformer iteratively refines both channel and beamformer features via residual connections. To enhance training, three strategies are introduced: (i) curriculum learning (CL) to improve early-stage convergence and avoid local optima, (ii) semi-amortized learning to refine each Transformer block with a few gradient ascent steps, and (iii) sliding-window training to stabilize optimization by training only a subset of Transformer blocks at a time. Extensive simulations show that the proposed scheme outperforms existing baselines at low-to-medium SNRs and closely approaches WMMSE performance at high SNRs, while achieving substantially faster inference than iterative and online learning approaches.

### 17. UniMoE-Audio: Unified Speech and Music Generation with Dynamic-Capacity MoE

**主要机构**: 
**作者数量**: 16人

**摘要**:
Recent advances in unified multimodal models indicate a clear trend towards comprehensive content generation. However, the auditory domain remains a significant challenge, with music and speech often developed in isolation, hindering progress towards universal audio synthesis. This separation stems from inherent task conflicts and severe data imbalances, which impede the development of a truly unified audio generation model. To address this challenge, we propose UniMoE-Audio, a unified speech and music generation model within a novel Dynamic-Capacity Mixture-of-Experts (MoE) framework. Architecturally, UniMoE-Audio introduces a Top-P routing strategy for dynamic expert number allocation, and a hybrid expert design comprising routed experts for domain-specific knowledge, shared experts for domain-agnostic features, and null experts for adaptive computation skipping. To tackle data imbalance, we introduce a three-stage training curriculum: 1) Independent Specialist Training leverages original datasets to instill domain-specific knowledge into each "proto-expert" without interference; 2) MoE Integration and Warmup incorporates these specialists into the UniMoE-Audio architecture, warming up the gate module and shared expert using a subset of balanced dataset; and 3) Synergistic Joint Training trains the entire model end-to-end on the fully balanced dataset, fostering enhanced cross-domain synergy. Extensive experiments show that UniMoE-Audio not only achieves state-of-the-art performance on major speech and music generation benchmarks, but also demonstrates superior synergistic learning, mitigating the performance degradation typically seen in naive joint training. Our findings highlight the substantial potential of specialized MoE architecture and curated training strategies in advancing the field of universal audio generation. Homepage: https://mukioxun.github.io/Uni-MoE-site/home.html.
