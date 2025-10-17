# AI推理加速技术论文分析报告
生成时间: 2025-10-17 12:13:47
分析论文数量: 21篇

## 论文技术简报

### 1. AUTOENCODING-FREE CONTEXT COMPRESSION FOR LLMS VIA CONTEXTUAL SEMANTIC ANCHORS

东北大学发布了AUTOENCODING-FREE CONTEXT COMPRESSION FOR LLMS VIA CONTEXTUAL SEMANTIC ANCHORS论文，使用SAC（语义锚定压缩）技术（通过直接选择锚定token并聚合上下文信息到KV表示实现免自编码训练），解决了现有自编码任务压缩方法中优化目标与下游任务不匹配的问题，达成了在不同压缩比下优于现有方法、MRQA分布外评估5倍压缩时EM提升1个点且更高压缩比优势更明显的效果。

### 2. BALDWHISPER: FASTER WHISPER WITH HEAD SHEARING AND LAYER MERGING

CNRS发布了BALDWHISPER论文，使用低秩分解和特征蒸馏压缩嵌入、层合并的剪枝技术，解决了低资源语言下Whisper模型轻量化需大量重训练数据的问题，达成了保留90%性能、模型小48%且速度提升2.15倍的效果

### 3. Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation

字节跳动和中山大学发布了《Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation》论文，使用将扩散Transformer重构为MoE（混合专家模型）的Dense2MoE技术，解决了大型文本到图像扩散模型参数量大、效率低的问题，达成了在视觉质量接近12B FLUX.1 [dev]的同时，激活参数降至5.2B、4B、3.2B的高效生成效果。

### 4. dInfer: An Efficient Inference Framework for Diffusion Language Models

Ant Group发布了dInfer论文，使用模块化推理管道分解及算法创新与系统优化技术，解决了扩散语言模型（dLLMs）缺乏标准化高效推理框架的问题，达成在8×H800 GPU上batch size 1时HumanEval达1100+ tokens/秒、较Fast-dLLM提速10倍且较QWen2.5-3B（vLLM）提速2-3倍的效果。

### 5. Dynamic Mixture-of-Experts for Visual Autoregressive Model

帝国理工学院与阿姆斯特丹大学发布了Dynamic Mixture-of-Experts for Visual Autoregressive Model论文，使用动态混合专家路由及尺度感知阈值策略，解决了视觉自回归模型因分辨率增加时重复调用Transformer导致的计算冗余问题，达成了减少约20% FLOPs、提升约11%推理速度且保持图像质量与密集基线相当的效果

### 6. FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference

National Yang Ming Chiao Tung University发布了FLRC论文，使用细粒度最优秩分配与渐进式低秩解码技术，解决了LLM低秩压缩中均匀压缩率导致性能下降及解码表现差的问题，达成在摘要任务上ROUGE-L较SOTA低秩压缩方法提升17%的效果

### 7. FOLK: Fast Open-Vocabulary 3D Instance Segmentation via Label-guided Knowledge Distillation

西安交通大学、浙江大学发布了FOLK论文，使用标签引导知识蒸馏技术，解决了现有方法中2D遮挡噪声及推理速度慢的问题，达成了在ScanNet200数据集上AP50 35.7的SOTA性能且推理速度提升约6.0×到152.2×的效果

### 8. FREQCA: ACCELERATING DIFFUSION MODELS VIA FREQUENCY-AWARE CACHING

发布了FREQCA论文，使用频率感知缓存(FreqCa)技术（对低频分量基于相似性复用、高频分量用二阶Hermite插值器预测，并缓存累积残差特征(CRF)），解决了扩散模型推理成本高及传统特征缓存对不同频段动态适应性不足的问题，达成在FLUX.1-dev等模型上有效加速生成与编辑且内存占用减少99%的效果

### 9. HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidance

华为技术有限公司发布了HES-SQL论文，使用融合思维模式融合监督微调(SFT)与组相对策略优化(GRPO)的混合训练框架（含骨架完整性评分、查询延迟感知奖励及自蒸馏过程），解决了Text-to-SQL生成中语义准确性与计算效率平衡的问题，达成在BIRD和KaggleDBQA基准上执行准确率分别达79.14%和54.9%、效率较监督基线提升11%-20%的效果。

### 10. LINEARSR: UNLOCKING LINEAR ATTENTION FOR STABLE AND EFFICIENT IMAGE SUPER-RESOLUTION

上海交通大学发布了LINEARSR论文，使用线性注意力技术，解决了图像超分辨率中的稳定性与效率问题，达成了稳定且高效的超分辨率效果。

### 11. Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference

卡内基梅隆大学、上海交通大学发布了dLLM高效推理论文，使用MaskKV框架（含掩码查询引导评分机制和自适应缓存预算策略），解决了dLLMs双向注意力缓存内存占用大、限制长上下文处理的问题，达成了KV缓存压缩至<5%令牌仍保持94%性能、32k长度时加速31倍的效果

### 12. PhyDAE: Physics-Guided Degradation-Adaptive Experts for All-in-One Remote Sensing Image Restoration

哈尔滨工业大学发布了PhyDAE论文，使用物理引导的退化自适应专家技术（结合两阶段级联架构、残差流形投影器及频率感知退化分解器等），解决了遥感图像多种异质退化（如 haze、噪声、模糊、低光）及现有一体化恢复方法依赖隐式特征、缺乏物理建模的问题，在三种基准数据集上所有恢复任务性能超越现有方法，显著提升恢复质量并大幅降低参数量与计算复杂度，实现性能与效率最优平衡。

### 13. Recover-LoRA: Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation

发布了Recover-LoRA论文，使用轻量级、数据集无关的Recover-LoRA技术（通过合成数据和logit蒸馏在选择性层学习LoRA适配器以对齐退化模型与全精度模型），解决了语言模型因推理优化（如量化、剪枝、序列化等）导致的性能下降问题，达成了在MHA和GQA小型语言模型上恢复5-17%准确率的效果。

### 14. SQS: Bayesian DNN Compression through Sparse Quantized Sub-distributions

Purdue University发布了SQS论文，使用贝叶斯变分学习下的spike-and-slab先验与高斯混合模型（GMM）结合的统一剪枝和低比特量化框架，解决了现有深度学习模型压缩方法单独使用剪枝或量化导致的压缩率次优问题，达成了在ResNet、BERT-base、Llama3及Qwen2.5等模型上实现比现有方法更高的压缩率且保持相当性能的效果。

### 15. StreamingVLM: Real-Time Understanding for Infinite Video Streams STREAMINGVLM: REAL-TIME UNDERSTANDING FOR INFINITE VIDEO STREAMS

研究团队发布了StreamingVLM论文，使用维护紧凑KV缓存及短重叠视频块全注意力SFT策略的统一框架，解决了视觉语言模型理解无限视频流时的延迟与内存激增问题，达成Inf-Streams-Eval对GPT-4O mini 66.18%胜率、单H100 8 FPS实时性能及LongVideoBench+4.30等VQA能力提升。

### 16. The Enduring Dominance of Deep Neural Networks: A Critical Analysis of the Fundamental Limitations of Quantum Machine Learning and Spiking Neural Networks

佐治亚理工学院发布了关于深度神经网络主导地位的批判性分析论文，通过分析量子机器学习和脉冲神经网络的局限性，解决了QML与SNNs能否取代DNNs的问题，论证了DNNs仍是AI发展的主导实用范式。

### 17. TINY-R1V: LIGHTWEIGHT MULTIMODAL UNIFIED REASONING MODEL VIA MODEL MERGING

北京邮电大学、南洋理工大学发布了TINY-R1V论文，使用LIPO强化学习与AMM模型合并技术，解决了现有MLLMs推理效率低及轻量级模型推理能力不足的问题，达成了更快推理、更高准确性，在多模态推理任务中表现优异的效果。

### 18. UTILIZING DYNAMIC SPARSITY ON PRETRAINED DETR

伦敦大学发布了UTILIZING DYNAMIC SPARSITY ON PRETRAINED DETR论文，使用微门控稀疏化（MGS）技术，解决了transformer模型在视觉任务中的高效推理挑战，达成了85-95%激活稀疏性、保持甚至提升性能并显著减少计算的效果。

### 19. VALUE-STATE GATED ATTENTION FOR MITIGATING EXTREME-TOKEN PHENOMENA IN TRANSFORMERS

Ant Group与北京大学发布了相关论文，使用Value-State Gated Attention（VGA）技术，解决了Transformer中的极端token现象（如注意力陷阱和值状态耗竭），达成了显著缓解注意力陷阱形成、稳定值状态范数，提升性能、量化保真度及模型可解释性的效果。

### 20. 

曼彻斯特大学发布了研究趋势测量论文，使用手工构建的词典对标题和摘要进行归一化、短语保护及匹配以分配主题标签并挖掘细粒度线索的技术，解决了量化计算机视觉和机器学习领域研究趋势的问题，达成了成功量化多模态VLM激增、生成方法扩展、3D视频韧性发展等三大宏观转变并支持跨会议比较及方法可审计扩展的效果。

### 21. When to Reason: Semantic Router for vLLM

UC Berkeley、IBM Research发布了论文，使用语义路由器技术，解决了LLM推理时推理模式带来的高成本（延迟、token使用）问题，达成MMLU-Pro准确率提升10.2个百分点、延迟降低47.1%、token消耗减少48.5%的效果。

## 论文详细信息

### 1. AUTOENCODING-FREE CONTEXT COMPRESSION FOR LLMS VIA CONTEXTUAL SEMANTIC ANCHORS

**主要机构**: Northeastern University, NLP Lab, Kunming University of Science and Technology
**作者数量**: 8人

**摘要**:
Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability a priori. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.

### 2. BALDWHISPER: FASTER WHISPER WITH HEAD SHEARING AND LAYER MERGING

**主要机构**: CNRS, LORIA
**作者数量**: 3人

**摘要**:
Pruning large pre-trained transformers for low-resource languages is challenging, as it often requires massive retraining data to recover performance. For instance, Distill-Whisper prunes Whisper by 40% and retrains on 21,000 hours of speech, far beyond what is available for most languages. Can Whisper be made lighter and faster for edge devices in data-scarce settings? Focusing on Bambara with only 32h of speech-to-text data, we propose a new pruning recipe. Instead of vocabulary pruning, which is unsuitable due to frequent code-switching by Bambara speakers, we compress the embeddings with low-rank decomposition and feature distillation. Rather than removing layers, we merge them to limit performance loss. The final model preserves 90% of the original performance while being 48% smaller and 2.15x faster on a MacBook Air M1.

### 3. Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation

**主要机构**: ByteDance Seed Vision, Sun Yat-sen University
**作者数量**: 5人

**摘要**:
Figure 1. The visual comparison between the 12B FLUX.1 [dev] and our FLUX.1-MoE models. The second, third, and fourth rows correspond to FLUX.1-MoE-L, FLUX.1-MoE-M, and FLUX.1-MoE-S, with 5.2B, 4B, and 3.2B activated parameters, respectively. These are sparse MoE models distilled from FLUX.1 [dev]. All images in each column are generated from the same random noise.

### 4. dInfer: An Efficient Inference Framework for Diffusion Language Models

**主要机构**: Ant Group
**作者数量**: 23人

**摘要**:
Diffusion-based large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs, leveraging denoising-based generation to enable inherent parallelism. Even more and more open-sourced dLLM models emerge, yet their widespread adoption remains constrained by the lack of a standardized and efficient inference framework. We present dInfer, an efficient and extensible framework for dLLM inference. dInfer decomposes the inference pipeline into four modular components-model, diffusion iteration manager, decoding strategy, and KV-cache manager-and integrates novel algorithms for each component alongside system-level optimizations. Through this combination of algorithmic innovations and system enhancements, dInfer achieves substantial efficiency gains without compromising output quality on LLaDA-MoE. At batch size 1, it surpasses 1,100 tokens per second on HumanEval and averages over 800 tokens per second across six benchmarks on 8× H800 GPUs. Compared to prior systems, dInfer delivers a 10× speedup over Fast-dLLM while maintaining similar model performance. Even compared to the AR model (with a comparable number of activation parameters and performance) QWen2.5-3B, which is highly optimized with the latest vLLM inference engine, dInfer still delivers a 2-3× speedup. The implementation of dInfer is open-sourced at https://github.com/inclusionAI/dInfer.

### 5. Dynamic Mixture-of-Experts for Visual Autoregressive Model

**主要机构**: Imperial College London, UvA-Bosch Delta Lab, University of Amsterdam
**作者数量**: 3人

**摘要**:
Visual Autoregressive Models (VAR) offer efficient and high-quality image generation but suffer from computational redundancy due to repeated Transformer calls at increasing resolutions. We introduce a dynamic Mixture-of-Experts router integrated into VAR. The new architecture allows to trade compute for quality through scale-aware thresholding. This thresholding strategy balances expert selection based on token complexity and resolution, without requiring additional training. As a result, we achieve ∼ 20% fewer FLOPs, ∼ 11% faster inference and match the image quality achieved by the dense baseline.

### 6. FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference

**主要机构**: National Yang Ming Chiao Tung University
**作者数量**: 5人

**摘要**:
Although large language models (LLM) have achieved remarkable performance, their enormous parameter counts hinder deployment on resource-constrained hardware. Low-rank compression can reduce both memory usage and computational demand, but applying a uniform compression ratio across all layers often leads to significant performance degradation, and previous methods perform poorly during decoding. To address these issues, we propose the Finegrained Low-Rank Compressor (FLRC), which efficiently determines an optimal rank allocation for each layer, and incorporates progressive low-rank decoding to maintain text generation quality. Comprehensive experiments on diverse benchmarks demonstrate the superiority of FLRC, achieving up to a 17% improvement in ROUGE-L on summarization tasks compared to state-of-the-art low-rank compression methods, establishing a more robust and efficient framework to improve LLM inference.

### 7. FOLK: Fast Open-Vocabulary 3D Instance Segmentation via Label-guided Knowledge Distillation

**主要机构**: Xi'an Jiaotong University, Zhejiang University, Tongji University, Zhejiang Laboratory
**作者数量**: 6人

**摘要**:
Open-vocabulary 3D instance segmentation seeks to segment and classify instances beyond the annotated label space. Existing methods typically map 3D instances to 2D RGB-D images, and then employ vision-language models (VLMs) for classification. However, such a mapping strategy usually introduces noise from 2D occlusions and incurs substantial computational and memory costs during inference, slowing down the inference speed. To address the above problems, we propose a Fast Open-vocabulary 3D instance segmentation method via Label-guided Knowledge distillation (FOLK). Our core idea is to design a teacher model that extracts highquality instance embeddings and distills its open-vocabulary knowledge into a 3D student model. In this way, during inference, the distilled 3D model can directly classify instances from the 3D point cloud, avoiding noise caused by occlusions and significantly accelerating the inference process. Specifically, we first design a teacher model to generate a 2D CLIP embedding for each 3D instance, incorporating both visibility and viewpoint diversity, which serves as the learning target for distillation. We then develop a 3D student model that directly produces a 3D embedding for each 3D instance. During training, we propose a label-guided distillation algorithm to distill open-vocabulary knowledge from label-consistent 2D embeddings into the student model. FOLK conducted experiments on the ScanNet200 and Replica datasets, achieving state-of-the-art performance on the ScanNet200 dataset with an AP50 score of 35.7, while running approximately 6.0× to 152.2× faster than previous methods. All codes will be released after the paper is accepted.

### 8. FREQCA: ACCELERATING DIFFUSION MODELS VIA FREQUENCY-AWARE CACHING

**主要机构**: 
**作者数量**: 16人

**摘要**:
The application of diffusion transformers is suffering from their significant inference costs. Recently, feature caching has been proposed to solve this problem by reusing features from previous timesteps, thereby skipping computation in future timesteps. However, previous feature caching assumes that features in adjacent timesteps are similar or continuous, which does not always hold in all settings. To investigate this, this paper begins with an analysis from the frequency domain, which reveal that different frequency bands in the features of diffusion models exhibit different dynamics across timesteps. Concretely, low-frequency components, which decide the structure of images, exhibit higher similarity but poor continuity. In contrast, the high-frequency bands, which decode the details of images, show significant continuity but poor similarity. These interesting observations motivate us to propose Frequency-aware Caching (FreqCa) which directly reuses features of low-frequency components based on their similarity, while using a second-order Hermite interpolator to predict the volatile high-frequency ones based on its continuity. Besides, we further propose to cache Cumulative Residual Feature (CRF) instead of the features in all the layers, which reduces the memory footprint of feature caching by 99%. Extensive experiments on FLUX.1-dev, FLUX.1-Kontextdev, Qwen-Image, and Qwen-Image-Edit demonstrate its effectiveness in both generation and editing. Codes are available in the supplementary materials and will be released on GitHub.

### 9. HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidance

**主要机构**: GTS, Huawei Technologies Co
**作者数量**: 6人

**摘要**:
We present HES-SQL, a novel hybrid training framework that advances Text-to-SQL generation through the integration of thinking-mode-fused supervised fine-tuning (SFT) with Group Relative Policy Optimization (GRPO). Our approach introduces three key innovations: (1) a skeleton-completeness scoring mechanism that enhances preference alignment between generated queries and optimal SQL structures; (2) a query-latency-aware reward system that incentivizes the generation of computationally efficient SQL queries; (3) a selfdistillation process for thinking-mode completion that prevents degradation of the model's reasoning capabilities. This framework enables hybrid thinking models to switch between reasoning and non-reasoning modes while improving SQL query accuracy and execution efficiency. Experimental evaluation, conducted on MySQL 8.0 and SQLite 3.42 under controlled single-user conditions, demonstrates that HES-SQL achieves competitive performance with execution accuracies of 79.14% and 54.9% on the BIRD and KaggleDBQA benchmarks, respectively. Query latency is measured as the end-to-end execution time of generated queries on the DBMS, averaged over multiple runs to mitigate variance. Efficiency gains range from 11% to 20% relative to supervised baselines. Our results establish a new paradigm for Text-to-SQL systems that effectively balances semantic accuracy with computational efficiency through execution-informed reinforcement learning (RL). The proposed methodology has significant implications for developing robust natural language interfaces to databases and can be extended to broader structured generation tasks requiring both correctness and efficiency optimization.

### 10. LINEARSR: UNLOCKING LINEAR ATTENTION FOR STABLE AND EFFICIENT IMAGE SUPER-RESOLUTION

**主要机构**: Shanghai Jiao Tong University, Shanghai Artificial Intelligence Laboratory
**作者数量**: 9人

**摘要**:


### 11. Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference

**主要机构**: Shanghai Jiao Tong University, Carnegie Mellon University, Shanghai Artificial Intelligence Laboratory, School of Artificial Intelligence
**作者数量**: 7人

**摘要**:
Diffusion large language models (dLLMs) present a promising alternative to dominant autoregressive models (ARMs) by the ability of parallel decoding at the expense of substantial computation and memory costs. Specifically, the cache mechanism for bidirectional attention in dLLMs demands large memory footprint, restricting their ability to handle long contexts under resource-limited settings. Existing cache eviction strategies are designed for ARMs and ignore the unique characteristics of dLLMs, thus leading to unsatisfactory performance. To address these challenges, we introduce MaskKV, a training-free cache eviction framework tailored to dLLMs, focusing on the effect of mask tokens in dLLMs. MaskKV is built on two key innovations: (1) a maskquery guided scoring mechanism that leverages attention weights to identify and evict less critical prompt tokens for each head; (2) an adaptive cache budgeting strategy that improves efficiency by reducing allocation in intermediate layers and concentrating resources on prompt-preferring heads. On LLaDA with MaskKV, compressing the KV cache to only 256 pairs (less than 5% of tokens) retains 94% of the full-cache performance on LongBench and achieves up to 31 × acceleration at 32k prompt length. The code is publicly available as an open-source project. 1 * Equal contribution. † Corresponding author.

### 12. PhyDAE: Physics-Guided Degradation-Adaptive Experts for All-in-One Remote Sensing Image Restoration

**主要机构**: School of Electronic and Information Engineering, Harbin Institute of Technology
**作者数量**: 5人

**摘要**:
Remote sensing images inevitably suffer from various degradation factors during acquisition, including atmospheric interference, sensor limitations, and imaging conditions. These complex and heterogeneous degradations pose severe challenges to image quality and downstream interpretation tasks. Addressing limitations of existing all-in-one restoration methods that overly rely on implicit feature representations and lack explicit modeling of degradation physics, this paper proposes Physics-Guided Degradation-Adaptive Experts (PhyDAE). The method employs a two-stage cascaded architecture transforming degradation information from implicit features into explicit decision signals, enabling precise identification and differentiated processing of multiple heterogeneous degradations including haze, noise, blur, and low-light conditions. The model incorporates progressive degradation mining and exploitation mechanisms, where the Residual Manifold Projector (RMP) and Frequency-Aware Degradation Decomposer (FADD) comprehensively analyze degradation characteristics from manifold geometry and frequency perspectives. Physics-aware expert modules and temperature-controlled sparse activation strategies are introduced to enhance computational efficiency while ensuring imaging physics consistency. Extensive experiments on three benchmark datasets (MD-RSID, MD-RRSHID, and MDRS-Landsat) demonstrate that PhyDAE achieves superior performance across all four restoration tasks, comprehensively outperforming state-of-the-art methods. Notably, PhyDAE substantially improves restoration quality while achieving significant reductions in parameter count and computational complexity, resulting in remarkable efficiency gains compared to mainstream approaches and achieving optimal balance between performance and efficiency. Code is available at https://github.com/HIT-SIRS/PhyDAE.

### 13. Recover-LoRA: Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation

**主要机构**: 
**作者数量**: 3人

**摘要**:
Inference optimizations such as quantization, pruning, format and datatype conversion, model export, and serialization can lead to functional degradations in language model task performance. While most efforts on performance recovery for deployment focus on robust quantization techniques, we focus on recovering model accuracies from any sources that degrade model weights, such as improper model serialization. In this work, we propose Recover-LoRA, a lightweight and dataset agnostic method to recover accuracy in degraded models. Recover-LoRA uses synthetic data and logit distillation to learn LoRA adapters on selective layers that facilitate aligning the degraded model to its full precision model. We investigate the utility of Recover-LoRA across a diverse set of small language models (SLMs), including models with varying attention architectures, multi-head attention (MHA) and group-query attention (GQA), as well as several evaluation datasets. Our results show that Recover-LoRA recovers model accuracies by 5-17% on MHA and GQA SLMs.

### 14. SQS: Bayesian DNN Compression through Sparse Quantized Sub-distributions

**主要机构**: University of Texas at El Paso, Purdue University
**作者数量**: 4人

**摘要**:
Compressing large-scale neural networks is essential for deploying models on resource-constrained devices. Most existing methods adopt weight pruning or low-bit quantization individually, often resulting in suboptimal compression rates to preserve acceptable performance drops. We introduce a unified framework for simultaneous pruning and low-bit quantization via Bayesian variational learning (SQS), which achieves higher compression rates than prior baselines while maintaining comparable performance. The key idea is to employ a spike-and-slab prior to inducing sparsity and model quantized weights using Gaussian Mixture Models (GMMs) to enable low-bit precision. In theory, we provide the consistent result of our proposed variational approach to a sparse and quantized deep neural network. Extensive experiments on compressing ResNet, BERT-base, Llama3, and Qwen2.5 models show that our method achieves higher compression rates than a line of existing methods with comparable performance drops.

### 15. StreamingVLM: Real-Time Understanding for Infinite Video Streams STREAMINGVLM: REAL-TIME UNDERSTANDING FOR INFINITE VIDEO STREAMS

**主要机构**: 
**作者数量**: 7人

**摘要**:
Vision-language models (VLMs) could power real-time assistants and autonomous agents, but they face a critical challenge: understanding near-infinite video streams without escalating latency and memory usage. Processing entire videos with full attention leads to quadratic computational costs and poor performance on long videos. Meanwhile, simple sliding window methods are also flawed, as they either break coherence or suffer from high latency due to redundant recomputation. In this paper, we introduce StreamingVLM, a model designed for real-time, stable understanding of infinite visual input. Our approach is a unified framework that aligns training with streaming inference. During inference, we maintain a compact KV cache by reusing states of attention sinks, a short window of recent vision tokens, and a long window of recent text tokens. This streaming ability is instilled via a simple supervised fine-tuning (SFT) strategy that applies full attention on short, overlapped video chunks, which effectively mimics the inference-time attention pattern without training on prohibitively long contexts. For evaluation, we build Inf-Streams-Eval, a new benchmark with videos averaging over two hours that requires dense, per-second alignment between frames and text. On Inf-Streams-Eval, StreamingVLM achieves a 66.18% win rate against GPT-4O mini and maintains stable, real-time performance at up to 8 FPS on a single NVIDIA H100. Notably, our SFT strategy also enhances general VQA abilities without any VQA-specific fine-tuning, improving performance on LongVideoBench by +4.30 and OVOBench Realtime by +5.96.

### 16. The Enduring Dominance of Deep Neural Networks: A Critical Analysis of the Fundamental Limitations of Quantum Machine Learning and Spiking Neural Networks

**主要机构**: ¹College of Computing, Georgia Institute of Technology
**作者数量**: 1人

**摘要**:
Recent advancements in quantum machine learning (QML) and spiking neural networks (SNNs) have generated considerable excitement, promising exponential speedups and brain-like energy efficiency to revolutionize artificial intelligence (AI). However, this paper critically examines their fundamental limitations, arguing that they are unlikely to displace deep neural networks (DNNs) in the near term. QML struggles with adapting backpropagation due to unitary constraints, measurement-induced state collapse, barren plateaus, and high measurement overheads, exacerbated by the limitations of current noisy intermediate-scale quantum hardware, overfitting risks due to underdeveloped regularization techniques, and a fundamental misalignment with machine learning's generalization. SNNs face restricted representational bandwidth, struggling with long-range dependencies and semantic encoding in language tasks due to their discrete, spike-based processing, unlike the attention mechanisms of Transformers. Furthermore, the goal of faithfully emulating the brain might impose inherent inefficiencies like cognitive biases, limited working memory, and slow learning speeds. Even their touted energyefficient advantages are overstated; optimized DNNs with quantization can outperform SNNs in energy costs under realistic conditions. Finally, SNN training incurs high computational overhead from temporal unfolding. In contrast, DNNs leverage efficient backpropagation, robust regularization, and innovations in large reasoning models that shift scaling to inference-time compute, enabling self-improvement via reinforcement learning and search algorithms like Monte Carlo tree search while mitigating data scarcity. This superiority is evidenced by recent models such as xAI's Grok-4 Heavy, which advances state-of-theart performance, and gpt-oss-120b, which surpasses or approaches the performance of leading industry models like OpenAI's o3-mini and o4-mini despite its modest 120-billion-parameter size deployable on a single 80GB GPU. Furthermore, specialized application-specific integrated circuits, such as the Cerebras Wafer-Scale Engine, the Groq Language Processing Unit, and the Etched Sohu, amplify these efficiency gains. Ultimately, QML and SNNs may serve niche hybrid roles, but DNNs remain the dominant, practical paradigm for AI advancement.

### 17. TINY-R1V: LIGHTWEIGHT MULTIMODAL UNIFIED REASONING MODEL VIA MODEL MERGING

**主要机构**: Beijing University of Posts and Telecommunications, Zhongguancun Academy, Nanyang Technological University
**作者数量**: 6人

**摘要**:
Although Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse tasks, they encounter numerous challenges in terms of reasoning efficiency, such as large model size, overthinking, and compromised accuracy in lightweight scenarios. However, research on the reasoning capabilities of lightweight MLLMs is quite lacking. To this end, we propose Tiny-R1V, a novel lightweight 3B model that achieves faster inference and higher accuracy via a two-stage optimization, while unifying multimodal reasoning across multiple tasks and using fewer tokens. In the first stage, Tiny-R1V introduces Length-Informed Relative Policy Optimization (LIPO), a novel reinforcement learning method, to train each reasoning model. The LIPO is designed to dynamically adjusts advantages of responses within groups, that is, by prioritizing concise yet high-quality responses to encourage the generation of shorter and more accurate response. In the second stage, we propose Adaptive Model Merging (AMM), a training-free model merging method that merges multiple specialist models into a unified architecture. Specifically, AMM adaptively adjusts the weights of task vectors and robustly optimizes the merged vectors via a novel gradient projection regularization loss function, thus mitigating redundant conflicts between them. Extensive evaluations on ten widely-used reasoning benchmarks covering mathematics, structured data (charts, tables, documents), OCR, and general capabilities showcase the superior performance of Tiny-R1V, enabling lightweight models to excel in diverse multimodal reasoning tasks.

### 18. UTILIZING DYNAMIC SPARSITY ON PRETRAINED DETR

**主要机构**: CITEC, University of London, Department of Computer Science, Bielefeld University
**作者数量**: 3人

**摘要**:
Efficient inference with transformer-based models remains a challenge, especially in vision tasks like object detection. We analyze the inherent sparsity in the MLP layers of DETR and introduce two methods to exploit it without retraining. First, we propose Static Indicator-Based Sparsification (SIBS), a heuristic method that predicts neuron inactivity based on fixed activation patterns. While simple, SIBS offers limited gains due to the input-dependent nature of sparsity. To address this, we introduce Micro-Gated Sparsification (MGS), a lightweight gating mechanism trained on top of a pretrained DETR. MGS predicts dynamic sparsity using a small linear layer and achieves up to 85-95% activation sparsity. Experiments on the COCO dataset show that MGS maintains or even improves performance while significantly reducing computation. Our method offers a practical, input-adaptive approach to sparsification, enabling efficient deployment of pretrained vision transformers without full model retraining.

### 19. VALUE-STATE GATED ATTENTION FOR MITIGATING EXTREME-TOKEN PHENOMENA IN TRANSFORMERS

**主要机构**: Ant Group, WICT, Peking University
**作者数量**: 4人

**摘要**:
Large models based on the Transformer architecture are susceptible to extremetoken phenomena, such as attention sinks and value-state drains. These issues, which degrade model performance, quantization fidelity, and interpretability, arise from a problematic mutual reinforcement mechanism where the model learns an inefficient 'no-op' behavior by focusing attention on tokens with near-zero value states. In this paper, we propose Value-State Gated Attention (VGA), a simple, dedicated, and stable architectural mechanism for performing 'no-op' attention efficiently by directly breaking this cycle. VGA introduces a learnable, datadependent gate, computed directly from the value vectors (V), to modulate the output. Through a theoretical analysis of the underlying gradients, we show that gating the value-state with a function of itself is more effective at decoupling value and attention score updates than prior methods that gate on input embeddings. This creates a direct regulatory pathway that allows the model to suppress a token's contribution based on its emergent value representation. Our experiments demonstrate that VGA significantly mitigates the formation of attention sinks and stabilizes value-state norms, leading to improved performance, robust quantization fidelity, and enhanced model interpretability.

### 20. 

**主要机构**: School of Computer Science, The University of Manchester
**作者数量**: 1人

**摘要**:
We present a transparent, reproducible measurement of research trends across 26,104 accepted papers from CVPR, ICLR, and NeurIPS spanning 2023-2025. Titles and abstracts are normalized, phraseprotected, and matched against a hand-crafted lexicon to assign up to 35 topical labels and mine finegrained cues about tasks, architectures, training regimes, objectives, datasets, and co-mentioned modalities. The analysis quantifies three macro shifts: (1) a sharp rise of multimodal vision-language-LLM work, which increasingly reframes classic perception as instruction following and multi-step reasoning; (2) steady expansion of generative methods-with diffusion research consolidating around controllability, distillation, and speed; and (3) resilient 3D and video activity, with composition moving from NeRFs to Gaussian splatting and a growing emphasis on human-and agent-centric understanding. Within VLMs, parameter-efficient adaptation (prompting, adapters/LoRA) and lightweight vision-language bridges dominate; training practice shifts from building encoders from scratch to instruction tuning and finetuning strong backbones; contrastive objectives recede relative to cross-entropy/ranking and distillation. Cross-venue comparisons show CVPR's stronger 3D footprint and ICLR's highest VLM share, while reliability themes (efficiency, robustness) diffuse across areas. We release the lexicon and methodology to enable auditing and extension. Limitations include lexicon recall and abstract-only scope, but the longitudinal signals are consistent across venues and years.

### 21. When to Reason: Semantic Router for vLLM

**主要机构**: IBM Research Yorktown Heights, UC Berkeley, University of Chicago
**作者数量**: 10人

**摘要**:
Large Language Models (LLMs) demonstrate substantial accuracy gains when augmented with reasoning modes such as chain-of-thought and inference-time scaling. However, reasoning also incurs significant costs in inference latency and token usage, with environmental and financial impacts, which are unnecessary for many simple prompts. We present a semantic router that classifies queries based on their reasoning requirements and selectively applies reasoning only when beneficial. Our approach achieves a 10.2 percentage point improvement in accuracy on the MMLU-Pro benchmark while reducing response latency by 47.1% and token consumption by 48.5% compared to direct inference with vLLM. These results demonstrate that semantic routing offers an effective mechanism for striking a balance between accuracy and efficiency in open-source LLM serving systems.
