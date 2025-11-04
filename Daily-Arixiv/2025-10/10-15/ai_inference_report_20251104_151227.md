# AI推理加速技术论文分析报告
生成时间: 2025-11-04 15:12:27
分析论文数量: 27篇

## 论文技术简报

### 1. ATTENTION IS ALL YOU NEED FOR KV CACHE IN DIFFUSION LLMS

FPT AI Residency Hanoi、VILA Lab发布了ATTENTION IS ALL YOU NEED FOR KV CACHE IN DIFFUSION LLMS论文，使用Elastic-Cache自适应KV缓存刷新策略（结合注意力感知漂移测试与深度感知调度），解决了扩散LLMs解码时KV缓存计算冗余导致的高延迟问题，达成了显著加速（GSM8K 8.7×、长序列45.1×、HumanEval 4.8×）并保持高准确性与吞吐量提升（比现有方法高6.8×）的效果

### 2. A FREE LUNCH IN LLM COMPRESSION: REVISITING RETRAINING AFTER PRUNING

柏林工业大学发布了A FREE LUNCH IN LLM COMPRESSION: REVISITING RETRAINING AFTER PRUNING论文，使用在每个Transformer块内分别重构注意力和MLP组件的技术，解决了LLM剪枝后全重训练计算量大且传统重构方法效果不足的问题，达成了资源效率高、困惑度最佳且性能优于全重训练的效果。

### 3. Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production

发布了《Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production》论文，提出动态自主调整每个输入token计算步数的Catch Your Breath (CYB)损失函数，解决了语言模型计算资源与token复杂度自适应匹配问题，使基线模型需3倍训练数据才能达到其性能且能根据token复杂度调整处理时间。

### 4. CONTEXT-SELECTIVE STATE SPACE MODELS: FEEDBACK IS ALL YOU NEED

帕多瓦大学发布了CONTEXT-SELECTIVE STATE SPACE MODELS: FEEDBACK IS ALL YOU NEED论文，使用结合状态反馈的COFFEE新型时变状态空间模型，解决了Transformer的二次复杂度和长序列依赖处理难题及S6模块选择性仅依赖当前输入的局限，达成了在induction head任务上以少两个数量级参数和训练序列实现近完美准确率、MNIST上3585参数达97%准确率远超S6的效果。

### 5. Cortex: Workflow-Aware Resource Pooling and Scheduling for Agentic Serving

哥伦比亚大学发布了Cortex论文，使用阶段隔离策略（为智能体工作流各阶段配置专用资源池），解决了智能体工作流中的计算和内存阶段间干扰问题，达成了提升KV缓存利用率、提高吞吐量和实现更可预测性能的效果。

### 6. DRBD-Mamba for Robust and Efficient Brain Tumor Segmentation with Analytical Insights

研究团队发布了DRBD-Mamba论文，使用双分辨率双向Mamba（DRBD-Mamba）模型及空间填充曲线、门控融合模块和量化块技术，解决了现有Mamba模型在脑肿瘤分割中计算开销大及鲁棒性不足的问题，达成了肿瘤核心和增强肿瘤平均Dice分别提升1.16%和1.68%、15倍效率提升的效果。

### 7. DYNASPEC: CONTEXT-AWARE DYNAMIC SPECULATIVE SAMPLING FOR LARGE-VOCABULARY LANGUAGE MODELS A PREPRINT

University of Bath和Aalto University发布了DYNASPEC论文，使用上下文相关的动态短列表机制，解决了大词汇量语言模型投机解码中草稿模型的延迟瓶颈，达成了平均接受长度持续提升且更小短列表不降低接受度的效果

### 8. Efficient Seq2seq Coreference Resolution Using Entity Representations

爱丁堡大学信息学院发布了《Efficient Seq2seq Coreference Resolution Using Entity Representations》论文，使用通过提取和重组实体级令牌、丢弃大部分其他输入令牌的压缩表示技术，解决了seq2seq共指消解模型在增量场景（如对话）中的效率和灵活性不足问题，达成了在OntoNotes上压缩比达1.8且性能接近基线、在LitBank上超过SOTA的效果。

### 9. Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference

研究团队发布了Efficient Video Sampling (EVS)论文，使用通过识别剪枝连续帧中时间静态补丁以减少token冗余的EVS技术，解决了视觉语言模型(VLMs)处理长视频时因密集帧序列二次成本导致的token预算超限及延迟问题，达成了将LLM的time-to-first-token (TTFT)降低高达4倍且准确率损失最小的效果。

### 10. ENTROPY MEETS IMPORTANCE: A UNIFIED HEAD IMPORTANCE-ENTROPY SCORE FOR STABLE AND EFFICIENT TRANSFORMER PRUNING

韩国大学发布了关于Transformer剪枝的论文，使用统一头重要性-熵分数（HIES）技术，解决了现有基于头重要性分数（HIS）的剪枝方法忽略注意力模式多样性的局限，达成了模型质量提升15.2%、稳定性提升2.04倍的高效剪枝效果。

### 11. EXPERTISE NEED NOT MONOPOLIZE: ACTION-SPECIALIZED MIXTURE OF EXPERTS FOR VISION-LANGUAGE-ACTION LEARNING

上海交通大学、上海AI实验室发布了提出AdaMoE的论文，使用通过独立尺度适配器分离专家选择与权重的解耦技术的混合专家架构，解决了VLA模型扩展时资源效率与实时控制需求的平衡及充分利用预训练权重的问题，达成了LIBERO提升1.8%、RoboTwin提升9.3%及真实世界实验21.5%的性能提升。

### 12. FraQAT: Quantization Aware Training with Fractional bits

三星发布了FraQAT论文，使用分数位量化感知训练技术，解决了生成模型量化中训练时间与高保真度的平衡问题，达成大型文本到图像模型（W4A8量化）FiD分数较最先进技术降低16%的效果。

### 13. From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR

AMD发布了《From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR》论文，使用基于MLIR的MLIR-AIR开源编译器栈（含AIR dialect，提供异步和分层操作的结构化表示），解决了通用编译器因抽象并行性、局部性和同步而无法充分利用AMD NPUs等现代空间架构细粒度控制的问题，达成了矩阵乘法78.7%计算效率且性能接近手动优化实现、多头注意力约150行代码高效映射到空间硬件的效果。

### 14. Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention

Ant Group等机构发布了《Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention》论文，使用Minimal Test-Time Intervention（MTI）训练-free框架（含选择性CFG干预和轻量级负提示引导），解决了LLM推理中测试时扩展效率低的问题，达成了在通用、编码和STEM任务上提升推理准确性和稳定性（如Qwen3-8B-Base平均+1.35%、AIME2024 +5%）且保持高效的效果。

### 15. LightQANet: Quantized and Adaptive Feature Learning for Low-Light Image Enhancement

深圳大学、宁波诺丁汉大学发布了LightQANet论文，使用量化与自适应特征学习（含Light Quantization Module和Light-Aware Prompt Module）技术，解决了低光图像增强中因像素级信息退化导致的纹理恢复差、颜色不一致及伪影问题，在多个低光数据集上达成了state-of-the-art性能，提供更优的定性和定量结果。

### 16. LiteStage: Latency-aware Layer Skipping for Multi-stage Reasoning

首尔国立大学发布了LiteStage论文，使用阶段级离线搜索（分配最优层预算）结合在线基于置信度的生成提前退出的层跳过框架，解决了多阶段推理中小语言模型延迟增加及现有层跳过技术难以平衡效率与准确性的问题，达成在OBQA、CSQA、StrategyQA等基准上最高1.70×加速且准确率损失小于4.0%的效果

### 17. MACE: Mixture-of-Experts Accelerated Coordinate Encoding for Large-Scale Scene Localization and Rendering

研究团队发布了MACE论文，使用混合专家加速坐标编码（MACE）方法（结合门控网络和无辅助损失负载平衡策略），解决了大规模场景下高效定位与高质量渲染的计算成本高及单一网络容量限制问题，实现显著成本降低与更高精度，并在Cambridge测试集上仅需10分钟训练达成高质量渲染。

### 18. PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model

百度发布了PaddleOCR-VL论文，使用集成NaViT-style动态分辨率视觉编码器与ERNIE-4.5-0.3B语言模型的0.9B超紧凑视觉语言模型，解决了多语言文档解析及复杂元素识别问题，达成了页面级文档解析与元素级识别SOTA性能，显著超越现有方案且推理速度快。

### 19. Pruning Overparameterized Multi-Task Networks for Degraded Web Image Restoration

University of Thessaly Volos发布了Pruning Overparameterized Multi-Task Networks for Degraded Web Image Restoration论文，使用迭代剪枝策略（MIR-L模型，移除低幅度权重并重置剩余权重至初始值），解决了多任务图像恢复模型参数过多、计算低效的问题，达成了仅保留10%参数同时保持高图像恢复性能（处理去雨、去雾、去噪任务）的效果

### 20. REAP THE EXPERTS: WHY PRUNING PREVAILS FOR ONE-SHOT MOE COMPRESSION

University of Calgary和Cerebras Systems Inc发布论文，提出Routerweighted Expert Activation Pruning(REAP)技术（结合路由器门值与专家激活范数的修剪标准），解决了稀疏激活混合专家(SMoE)模型大参数内存开销及专家合并的“功能子空间坍缩”误差问题，在20B到1T参数模型生成任务中优于合并和其他修剪方法，50%压缩时近无损（如Qwen3-Coder-480B和Kimi-K2修剪50%专家仍近无损）。

### 21. REWIRING EXPERTS ON THE FLY: CONTINUOUS REROUTING FOR BETTER ONLINE ADAPTATION IN MIXTURE-OF-EXPERT MODELS

马克斯·普朗克智能系统研究所与中山大学发布了REWIRING EXPERTS ON THE FLY论文，使用数据-free在线测试时框架通过自监督基于已生成序列优化路由决策并以轻量级加法向量更新选定层路由logits的技术，解决了MoE模型部署中因分布偏移导致的次优路由决策问题，达成在HumanEval上提升5.5%、与自一致性结合平均提升6%的效果。

### 22. SCALEWEAVER: WEAVING EFFICIENT CONTROL-LABLE T2I GENERATION WITH MULTI-SCALE REFER-ENCE ATTENTION

中国科学技术大学发布了SCALEWEAVER论文，使用多尺度参考注意力技术，解决文本到图像生成中效率与可控性不足的问题，实现了高效且可控的文本到图像生成。

### 23. SHISHULM: LIGHTWEIGHT LANGUAGE MODEL WITH HYBRID DECODER-MLP ARCHITECTURE AND PAIRED WEIGHT SHARING

研究团队发布了ShishuLM论文，使用混合解码器-MLP架构与配对权重共享技术，解决了transformer模型内存和计算开销大的问题，达成了内存需求减少25%、训练和推理延迟提升40%的效果。

### 24. Vision Mamba for Permeability Prediction of Porous Media

斯坦福大学发布了Vision Mamba for Permeability Prediction of Porous Media论文，首次使用Vision Mamba作为骨干网络，解决了三维多孔介质渗透率预测中传统ViTs和CNN计算/内存效率低的问题，达成了相比ViTs和CNN具有计算/内存效率优势及良好预测性能的效果。

### 25. WeCKD: Weakly-supervised Chained Distillation Network for Efficient Multimodal Medical Imaging

研究团队发布了WeCKD论文，使用弱监督链式蒸馏网络（渐进式蒸馏链），解决了传统知识蒸馏在有限数据场景下的知识退化、监督效率低及依赖强教师或大标签数据集的问题，达成了在多模态医学影像上匹配甚至超越现有监督方法、泛化性良好且累积准确率较单一骨干模型提升高达+23%的效果。

### 26. WHAT LAYERS WHEN: LEARNING TO SKIP COMPUTE IN LLMS WITH RESIDUAL GATES

阿姆斯特丹大学与高通发布了《WHAT LAYERS WHEN: LEARNING TO SKIP COMPUTE IN LLMS WITH RESIDUAL GATES》论文，使用GateSkip残差流门控机制，解决了大型语言模型推理中计算资源浪费问题，达成在长文本推理节省15%计算并保持>90%准确率、指令微调模型近50%计算节省时匹配基线质量的效果。

### 27. 

相关机构发布了xLLM研究论文，使用针对多样化AI加速器深度优化的智能高效大语言模型推理框架，解决了主流推理框架在企业级服务中面临的混合动态工作负载、高可用性需求、分布式存储管理挑战及AI加速器利用率不足的瓶颈，实现了高性能大规模企业级服务。

## 论文详细信息

### 1. ATTENTION IS ALL YOU NEED FOR KV CACHE IN DIFFUSION LLMS

**主要机构**: FPT AI Residency Hanoi, VILA Lab
**作者数量**: 3人

**摘要**:
This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant MASK tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose Elastic-Cache, a training-free, architecture-agnostic strategy that jointly decides when to refresh (via an attention-aware drift test on the most-attended token) and where to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: 8.7× on GSM8K (256 tokens), 45.1× on longer sequences, and 4.8× on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput (6.8× on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.

### 2. A FREE LUNCH IN LLM COMPRESSION: REVISITING RETRAINING AFTER PRUNING

**主要机构**: Technische Universität Berlin, Institute of Mathematics, Zuse Institute, Department for AI in Society, Science, and Technology
**作者数量**: 5人

**摘要**:
While Neural Network pruning typically requires retraining the model to recover pruning-induced performance degradation, state-of-the-art Large Language Model (LLM) pruning methods instead solve a layer-wise mask selection and reconstruction problem on a small set of calibration data to avoid full retraining, as it is considered computationally infeasible for LLMs. Reconstructing single matrices in isolation has favorable properties, such as convexity of the objective and significantly reduced memory requirements compared to full retraining. In practice, however, reconstruction is often implemented at coarser granularities, e.g., reconstructing a whole transformer block against its dense activations instead of a single matrix. In this work, we study the key design choices when reconstructing or retraining the remaining weights after pruning. We conduct an extensive computational study on state-of-the-art GPT architectures, and report several surprising findings that challenge common intuitions about retraining after pruning. In particular, we observe a free lunch scenario: reconstructing attention and MLP components separately within each transformer block is nearly the most resourceefficient yet achieves the best perplexity. Most importantly, this Pareto-optimal setup achieves better performance than full retraining, despite requiring only a fraction of the memory. Furthermore, we demonstrate that simple and efficient pruning criteria such as Wanda can outperform much more complex approaches when the reconstruction step is properly executed, highlighting its importance. Our findings challenge the narrative that retraining should be avoided at all costs and provide important insights into post-pruning performance recovery for LLMs.

### 3. Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production

**主要机构**: 
**作者数量**: 3人

**摘要**:
We explore a class of supervised training objectives that allow a language model to dynamically and autonomously scale the number of compute steps used for each input token. For any token, the model can request additional compute steps by emitting a <DON'T KNOW> output. If the model is granted a delay, a specialized <PAUSE> token is inserted at the next input step, providing the model with additional compute resources to generate an output. The model can request multiple pauses. To train the model to use <DON'T KNOW> outputs judiciously and to calibrate its uncertainty, we frame the selection of each output token as a sequential-decision problem with a time cost. We refer to the class of methods as Catch Your Breath losses and we study three methods in this class: CYB-AP frames the model's task as anytime prediction, where an output may be required at any step and accuracy is discounted over time; CYB-VA is a variational approach that aims to maximize prediction accuracy subject to a specified distribution over stopping times; and CYB-DP imposes a penalty based on a computational budget. Through fine-tuning experiments, we determine a specific form of the loss that performs best. To cast the performance improvement in intuitive terms, a baseline (no pause) model needs 3× as much training data to match the CYB loss, and a model with pauses and a cross-entropy loss requires 2× as much data. We find that the CYB model requests additional steps when doing so improves accuracy, and the model adapts its processing time to token-level complexity and context. For example, it often pauses after plural nouns like patients and challenges but never pauses after the first token of contracted words like wasn and didn, and it shows high variability for ambiguous tokens like won, which could function as either a verb or part of a contraction.

### 4. CONTEXT-SELECTIVE STATE SPACE MODELS: FEEDBACK IS ALL YOU NEED

**主要机构**: University of Padova Padova, Department of Information Engineering
**作者数量**: 5人

**摘要**:
Transformers, powered by the attention mechanism, are the backbone of most foundation models, yet they suffer from quadratic complexity and difficulties in dealing with long-range dependencies in the input sequence. Recent work has shown that state space models (SSMs) provide an efficient alternative, with the S6 module at the core of the Mamba architecture achieving state-of-the-art results on long-sequence benchmarks. In this paper, we introduce the COFFEE (COntext From FEEdback) model, a novel time-varying SSM that incorporates state feedback to enable context-dependent selectivity, while still allowing for parallel implementation. Whereas the selectivity mechanism of S6 only depends on the current input, COFFEE computes it from the internal state, which serves as a compact representation of the sequence history. This shift allows the model to regulate its dynamics based on accumulated context, improving its ability to capture long-range dependencies. In addition to state feedback, we employ an efficient model parametrization that removes redundancies present in S6 and leads to a more compact and trainable formulation. On the induction head task, COFFEE achieves near-perfect accuracy with two orders of magnitude fewer parameters and training sequences compared to S6. On MNIST, COFFEE largely outperforms S6 within the same architecture, reaching 97% accuracy with only 3585 parameters. These results showcase the role of state feedback as a key mechanism for building scalable and efficient sequence models.

### 5. Cortex: Workflow-Aware Resource Pooling and Scheduling for Agentic Serving

**主要机构**: Columbia University, Columbia University Yeounoh Chung
**作者数量**: 2人

**摘要**:
We introduce Cortex, a prototype workflow-aware serving platform designed for agentic workloads. The core principle of Cortex is stage isolation: it provisions dedicated resource pools for each distinct stage of an agentic workflow. This simple yet powerful strategy mitigates inter-stage interference in compute and memory, leading to better KV cache utilization, higher throughput, and more predictable performance. By customizing resource allocation and scheduling within each distinct stage of agentic workflows, Cortex lays the groundwork for more advanced, agent-native serving paradigms, including malleable resource management, speculative execution of workflow branches, and a shared, multi-tiered cache for "agentic state.

### 6. DRBD-Mamba for Robust and Efficient Brain Tumor Segmentation with Analytical Insights

**主要机构**: 
**作者数量**: 4人

**摘要**:
Accurate brain tumor segmentation is significant for clinical diagnosis and treatment but remains challenging due to tumor heterogeneity. Mamba-based State Space Models have demonstrated promising performance. However, despite their computational efficiency over other neural architectures, they incur considerable overhead for this task due to their sequential feature computation across multiple spatial axes. Moreover, their robustness across diverse BraTS data partitions remains largely unexplored, leaving a critical gap in reliable evaluation. To address this, we first propose a dual-resolution bi-directional Mamba (DRBD-Mamba), an efficient 3D segmentation model that captures multi-scale long-range dependencies with minimal computational overhead. We leverage a spacefilling curve to preserve spatial locality during 3D-to-1D feature mapping, thereby reducing reliance on computationally expensive multi-axial feature scans. To enrich feature representation, we propose a gated fusion module that adaptively integrates forward and reverse contexts, along with a quantization block that improves robustness. We further propose five systematic folds on BraTS2023 for rigorous evaluation of segmentation techniques under diverse conditions and present analysis of common failure scenarios. On the 20% test set used by recent methods, our model achieves Dice improvements of 0.10% for whole tumor, 1.75% for tumor core, and 0.93% for enhancing tumor. Evaluations on the proposed systematic folds demonstrate that our model maintains competitive whole tumor accuracy while achieving clear average Dice gains of 1.16% for tumor core and 1.68% for enhancing tumor over existing state-of-the-art. Furthermore, our model achieves a 15x efficiency improvement while maintaining high segmentation accuracy, highlighting its robustness and computational advantage over existing methods.

### 7. DYNASPEC: CONTEXT-AWARE DYNAMIC SPECULATIVE SAMPLING FOR LARGE-VOCABULARY LANGUAGE MODELS A PREPRINT

**主要机构**: Deep Algorithms and Systems, University of Bath, Aalto University Espoo, Department of Computer Science
**作者数量**: 4人

**摘要**:
Speculative decoding (a.k.a. speculative sampling) has become a standard way to accelerate LLM inference: a small drafter proposes multiple tokens and a large target model verifies them once per speculation length. Recently, scaling of the LLM vocabulary has pushed the number of tokens to grow substantially. While verification over the full vocabulary leaves the target model largely unaffected, the O(|V |d) parameters in the drafter's output head become a latency bottleneck, slowing the entire pipeline. Contemporary methods (e.g., FR-Spec, VocabTrim) restrict the drafter's vocabulary to a fixed subset of the target model's vocabulary, ranked in descending order of token frequency. Although this reduces draft-time compute, it is brittle, since : (i) frequency lists are corpus-dependent and require retuning to generalize, and (ii) static shortlists suppress rare or domain-specific tokens, lowering the expected number of tokens per verification step. We propose DYNASPEC, a contextdependent dynamic shortlisting mechanism that is robust, speeds up drafting, and generalizes across diverse tasks. Concretely, we introduce lightweight, coarse-grained meta-classifiers that route contexts to a small number of token clusters; the union of the top-k selected clusters forms the drafter's shortlist, while verification retains the full vocabulary and exactness. The meta-classifier finishes its computation earlier than the drafter's hidden state generation by exploiting parallel execution of draft encoding and meta shortlisting on separate streams. On standard speculative-decoding benchmarks, we observe consistent gains in mean accepted length over fixed-shortlist baselines, while context-dependent selection enables smaller shortlists without degrading acceptance.

### 8. Efficient Seq2seq Coreference Resolution Using Entity Representations

**主要机构**: School of Informatics, University of Edinburgh
**作者数量**: 3人

**摘要**:
Seq2seq coreference models have introduced a new paradigm for coreference resolution by learning to generate text corresponding to coreference labels, without requiring task-specific parameters. While these models achieve new state-of-the-art performance, they do so at the cost of flexibility and efficiency. In particular, they do not efficiently handle incremental settings such as dialogue, where text must processed sequentially. We propose a compressed representation in order to improve the efficiency of these methods in incremental settings. Our method works by extracting and reorganizing entity-level tokens, and discarding the majority of other input tokens. On OntoNotes, our best model achieves just 0.6 CoNLL F1 points below a full-prefix, incremental baseline while achieving a compression ratio of 1.8. On LitBank, where singleton mentions are annotated, it passes state-of-the-art performance. Our results indicate that discarding a wide portion of tokens in seq2seq resolvers is a feasible strategy for incremental coreference resolution.

### 9. Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference

**主要机构**: 
**作者数量**: 12人

**摘要**:
Vision-language models (VLMs) have recently expanded from static image understanding to video reasoning, but their scalability is fundamentally limited by the quadratic cost of processing dense frame sequences. Long videos often exceed the token budget of modern language models, leading to severe context limitations and latency issues. We introduce Efficient Video Sampling (EVS), a simple, plug-and-play method for reducing token redundancy in videos by identifying and pruning temporally static patches-spatial regions that remain unchanged across consecutive frames. EVS preserves positional identity, requires no architectural changes or retraining. We show that EVS substantially reduces token count while maintaining semantic fidelity, enabling faster inference and longer input sequences. Applied at inference time, EVS reduces large language model (LLM) time-to-first-token (TTFT) by up to 4× with minimal accuracy loss. When combined with an uptraining phase using stochastic pruning rates, EVS yields models that are robust to varying compression levels and retain full performance under aggressive pruning. Extensive experiments demonstrate that EVS consistently improves efficiency-accuracy trade-offs, unlocking scalable video-language understanding without sacrificing quality.

### 10. ENTROPY MEETS IMPORTANCE: A UNIFIED HEAD IMPORTANCE-ENTROPY SCORE FOR STABLE AND EFFICIENT TRANSFORMER PRUNING

**主要机构**: Department of Computer Science and Engineering, School of Software, Korea University, Soongsil University
**作者数量**: 4人

**摘要**:
Transformer-based models have achieved remarkable performance in NLP tasks. However, their structural characteristics-multiple layers and attention heads-introduce efficiency challenges in inference and deployment. To address these challenges, various pruning methods have recently been proposed. Notably, gradient-based methods using Head Importance Scores (HIS) have gained traction for interpretability, efficiency, and ability to identify redundant heads. However, HIS alone has limitations as it captures only the gradient-driven contribution, overlooking the diversity of attention patterns. To overcome these limitations, we introduce a novel pruning criterion, HIES (Head Importance-Entropy Score), which integrates head importance scores with attention entropy, providing complementary evidence on per-head contribution. Empirically, HIES-based pruning yields up to 15.2% improvement in model quality and 2.04× improvement in stability over HIS-only methods, enabling substantial model compression without sacrificing either accuracy or stability. Code will be released upon publication.

### 11. EXPERTISE NEED NOT MONOPOLIZE: ACTION-SPECIALIZED MIXTURE OF EXPERTS FOR VISION-LANGUAGE-ACTION LEARNING

**主要机构**: Shanghai Jiao Tong University, Shanghai AI Laboratory, School of Computer Science, Tsinghua Shenzhen International Graduate School, School of Automation and Intelligent Sensing, Tsinghua University, AI Institute, Laboratory of Integrated Administration Technologies for Information Security, MoE key Lab of Artificial Intelligence
**作者数量**: 14人

**摘要**:
Vision-Language-Action (VLA) models are experiencing rapid development and demonstrating promising capabilities in robotic manipulation tasks. However, scaling up VLA models presents several critical challenges: (1) Training new VLA models from scratch demands substantial computational resources and extensive datasets. Given the current scarcity of robot data, it becomes particularly valuable to fully leverage well-pretrained VLA model weights during the scaling process. (2) Real-time control requires carefully balancing model capacity with computational efficiency. To address these challenges, We propose AdaMoE, a Mixture-of-Experts (MoE) architecture that inherits pretrained weights from dense VLA models, and scales up the action expert by substituting the feedforward layers into sparsely activated MoE layers. AdaMoE employs a decoupling technique that decouples expert selection from expert weighting through an independent scale adapter working alongside the traditional router. This enables experts to be selected based on task relevance while contributing with independently controlled weights, allowing collaborative expert utilization rather than winnertakes-all dynamics. Our approach demonstrates that expertise need not monopolize. Instead, through collaborative expert utilization, we can achieve superior performance while maintaining computational efficiency. AdaMoE consistently outperforms the baseline model across key benchmarks, delivering performance gains of 1.8% on LIBERO and 9.3% on RoboTwin. Most importantly, a substantial 21.5% improvement in real-world experiments validates its practical effectiveness for robotic manipulation tasks.

### 12. FraQAT: Quantization Aware Training with Fractional bits

**主要机构**: Samsung AI Center Cambridge
**作者数量**: 7人

**摘要**:
Figure 1: FraQAT is a Quantization aware Training (QAT) technique that grants generative models high fidelity at a fraction of training time required. Large text-to-image (T2I) models quantized with FraQAT (W4A8) achieve 16% lower FiD score than the state-of-the-art.

### 13. From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR

**主要机构**: Research and Advanced Development, AMD
**作者数量**: 1人

**摘要**:
search and Advanced Development, AMD, USA General-purpose compilers abstract away parallelism, locality, and synchronization, limiting their effectiveness on modern spatial architectures. As modern computing architectures increasingly rely on fine-grained control over data movement, execution order, and compute placement for performance, compiler infrastructure must provide explicit mechanisms for orchestrating compute and data to fully exploit such architectures. We introduce MLIR-AIR, a novel, open-source compiler stack built on MLIR that bridges the semantic gap between high-level workloads and fine-grained spatial architectures such as AMD's NPUs. MLIR-AIR defines the AIR dialect, which provides structured representations for asynchronous and hierarchical operations across compute and memory resources. AIR primitives allow the compiler to orchestrate spatial scheduling, distribute computation across hardware regions, and overlap communication with computation without relying on ad hoc runtime coordination or manual scheduling. We demonstrate MLIR-AIR's capabilities through two case studies: matrix multiplication and the multi-head attention block from the LLaMA 2 model. For matrix multiplication, MLIR-AIR achieves up to 78.7% compute efficiency and generates implementations with performance almost identical to state-of-the-art, hand-optimized matrix multiplication written using the lower-level, close-to-metal MLIR-AIE framework. For multi-head attention, we demonstrate that the AIR interface supports fused implementations using approximately 150 lines of code, enabling tractable expression of complex workloads with efficient mapping to spatial hardware. MLIR-AIR transforms high-level structured control flow into spatial programs that efficiently utilize the compute fabric and memory hierarchy of an NPU, leveraging asynchronous execution, tiling, and communication overlap through compiler-managed scheduling.

### 14. Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention

**主要机构**: Kuaishou Technology, Ant Group, AIML, HKUST(GZ)
**作者数量**: 8人

**摘要**:
Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +1.35% average improvement on eight benchmarks for Qwen3-8B-Base and +5% on AIME2024 using Qwen3-32B-Reasoning-while remaining highly efficient. The code can be found here.

### 15. LightQANet: Quantized and Adaptive Feature Learning for Low-Light Image Enhancement

**主要机构**: Shenzhen University, University of Nottingham Ningbo China, College of Computer Science and Software Engineering, School of Artificial Intelligence, School of Artificial Intel- ligence, School of Computer Science, School of Mathematics and Statistics, Xi'an Jiaotong-Liverpool University, Computer Vision Institute, Department of Computer Science, School of AI and Advanced Computing, Nanyang Technological University, Sichuan Normal University, Changsha University of Science and Technology, College of Computing and Data Science
**作者数量**: 15人

**摘要**:
Low-light image enhancement (LLIE) aims to improve illumination while preserving high-quality color and texture. However, existing methods often fail to extract reliable feature representations due to severely degraded pixel-level information under low-light conditions, resulting in poor texture restoration, color inconsistency, and artifact. To address these challenges, we propose LightQANet, a novel framework that introduces quantized and adaptive feature learning for lowlight enhancement, aiming to achieve consistent and robust image quality across diverse lighting conditions. From the static modeling perspective, we design a Light Quantization Module (LQM) to explicitly extract and quantify illumination-related factors from image features. By enforcing structured light factor learning, LQM enhances the extraction of light-invariant representations and mitigates feature inconsistency across varying illumination levels. From the dynamic adaptation perspective, we introduce a Light-Aware Prompt Module (LAPM), which encodes illumination priors into learnable prompts to dynamically guide the feature learning process. LAPM enables the model to flexibly adapt to complex and continuously changing lighting conditions, further improving image enhancement. Extensive experiments on multiple low-light datasets demonstrate that our method achieves state-of-the-art performance, delivering superior qualitative and quantitative results across various challenging lighting scenarios.

### 16. LiteStage: Latency-aware Layer Skipping for Multi-stage Reasoning

**主要机构**: Seoul National University
**作者数量**: 3人

**摘要**:
Multi-stage reasoning has emerged as an effective strategy for enhancing the reasoning capability of small language models by decomposing complex problems into sequential sub-stages. However, this comes at the cost of increased latency. We observe that existing adaptive acceleration techniques, such as layer skipping, struggle to balance efficiency and accuracy in this setting due to two key challenges: (1) stage-wise variation in skip sensitivity, and (2) the generation of redundant output tokens. To address these, we propose LiteStage, a latency-aware layer skipping framework for multi-stage reasoning. LiteStage combines a stage-wise offline search that allocates optimal layer budgets with an online confidence-based generation early exit to suppress unnecessary decoding. Experiments on three benchmarks, e.g., OBQA, CSQA, and StrategyQA, show that LiteStage achieves up to 1.70× speedup with less than 4.0% accuracy loss, outperforming prior training-free layer skipping methods. The code is available at https://github. com/beomseokg/LiteStage.

### 17. MACE: Mixture-of-Experts Accelerated Coordinate Encoding for Large-Scale Scene Localization and Rendering

**主要机构**: 
**作者数量**: 13人

**摘要**:
Efficient localization and high-quality rendering in large-scale scenes remain a significant challenge due to the computational cost involved. While Scene Coordinate Regression (SCR) methods perform well in small-scale localization, they are limited by the capacity of a single network when extended to large-scale scenes. To address these challenges, we propose the Mixed Expert-based Accelerated Coordinate Encoding method (MACE), which enables efficient localization and high-quality rendering in large-scale scenes. Inspired by the remarkable capabilities of MOE in large model domains, we introduce a gating network to implicitly classify and select subnetworks, ensuring that only a single sub-network is activated during each inference. Furtheremore, we present Auxiliary-Loss-Free Load Balancing (ALF-LB) strategy to enhance the localization accuracy on large-scale scene. Our framework provides a significant reduction in costs while maintaining higher precision, offering an efficient solution for large-scale scene applications. Additional experiments on the Cambridge test set demonstrate that our method achieves high-quality rendering results with merely 10 minutes of training.

### 18. PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model

**主要机构**: Baidu Inc, PaddlePaddle Team
**作者数量**: 18人

**摘要**:
In this report, we propose PaddleOCR-VL, a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful visionlanguage model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and elementlevel recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios.

### 19. Pruning Overparameterized Multi-Task Networks for Degraded Web Image Restoration

**主要机构**: University of Thessaly Volos, Dept. of Electrical and Computer Engineering
**作者数量**: 2人

**摘要**:
Image quality is a critical factor in delivering visually appealing content on web platforms. However, images often suffer from degradation due to lossy operations applied by online social networks (OSNs), negatively affecting user experience. Image restoration is the process of recovering a clean high-quality image from a given degraded input. Recently, multi-task (all-inone) image restoration models have gained significant attention, due to their ability to simultaneously handle different types of image degradations. However, these models often come with an excessively high number of trainable parameters, making them computationally inefficient. In this paper, we propose a strategy for compressing multi-task image restoration models. We aim to discover highly sparse subnetworks within overparameterized deep models that can match or even surpass the performance of their dense counterparts. The proposed model, namely MIR-L, utilizes an iterative pruning strategy that removes low-magnitude weights across multiple rounds, while resetting the remaining weights to their original initialization. This iterative process is important for the multi-task image restoration model's optimization, effectively uncovering "winning tickets" that maintain or exceed state-of-the-art performance at high sparsity levels. Experimental evaluation on benchmark datasets for the deraining, dehazing, and denoising tasks shows that MIR-L retains only 10% of the trainable parameters while maintaining high image restoration performance. Our code, datasets and pre-trained models are made publicly available at https://github.com/Thomkat/MIR-L.

### 20. REAP THE EXPERTS: WHY PRUNING PREVAILS FOR ONE-SHOT MOE COMPRESSION

**主要机构**: University of Calgary, Schulich School of Engineering, Cerebras Systems Inc
**作者数量**: 6人

**摘要**:
Sparsely-activated Mixture-of-Experts (SMoE) models offer efficient pre-training and low latency but their large parameter counts create significant memory overhead, motivating research into expert compression. Contrary to recent findings favouring expert merging on discriminative benchmarks, we demonstrate that expert pruning is a superior strategy for generative tasks. We prove that merging introduces an irreducible error by causing a "functional subspace collapse", due to the loss of the router's independent, input-dependent control over experts. Leveraging this insight, we propose Routerweighted Expert Activation Pruning (REAP), a novel pruning criterion that considers both router gate-values and expert activation norms. Across a diverse set of SMoE models ranging from 20B to 1T parameters, REAP consistently outperforms merging and other pruning methods on generative benchmarks, especially at 50% compression. Notably, our method achieves near-lossless compression on code generation and toolcalling tasks with Qwen3-Coder-480B and Kimi-K2, even after pruning 50% of experts.

### 21. REWIRING EXPERTS ON THE FLY: CONTINUOUS REROUTING FOR BETTER ONLINE ADAPTATION IN MIXTURE-OF-EXPERT MODELS

**主要机构**: Sun Yat-sen University, University of Tübingen, Max Planck Institute for Intelligent Systems, University of Surrey
**作者数量**: 6人

**摘要**:
Mixture-of-Experts (MoE) models achieve efficient scaling through sparse expert activation, but often suffer from suboptimal routing decisions due to distribution shifts in deployment. While existing test-time adaptation methods could potentially address these issues, they primarily focus on dense models and require access to external data, limiting their practical applicability to MoE architectures. However, we find that, instead of relying on reference data, we can optimize MoE expert selection on-the-fly based only on input context. As such, we propose a data-free, online test-time framework that continuously adapts MoE routing decisions during text generation without external supervision or data. Our method cycles between two phases: During the prefill stage, and later in regular intervals, we optimize the routing decisions of the model using self-supervision based on the already generated sequence. Then, we generate text as normal, maintaining the modified router until the next adaption. We implement this through lightweight additive vectors that only update router logits in selected layers, maintaining computational efficiency while preventing over-adaptation. The experimental results show consistent performance gains on challenging reasoning tasks while maintaining robustness to context shifts. For example, our method achieves a 5.5% improvement on HumanEval with OLMoE. Furthermore, owing to its plug-andplay property, our method naturally complements existing test-time scaling techniques, e.g., achieving 6% average gains when incorporated with self-consistency on DeepSeek-V2-Lite.

### 22. SCALEWEAVER: WEAVING EFFICIENT CONTROL-LABLE T2I GENERATION WITH MULTI-SCALE REFER-ENCE ATTENTION

**主要机构**: University of Science and Technology of China
**作者数量**: 6人

**摘要**:


### 23. SHISHULM: LIGHTWEIGHT LANGUAGE MODEL WITH HYBRID DECODER-MLP ARCHITECTURE AND PAIRED WEIGHT SHARING

**主要机构**: 
**作者数量**: 2人

**摘要**:
While the transformer architecture has achieved state-of-the-art performance on natural language processing tasks, these models impose substantial memory and computational overhead. Recent research has identified significant architectural redundancies within these models, presenting opportunities for optimization without compromising performance. Taking insights from research in AI interpretability and inference-time layer pruning, we introduce an efficient language model architecture, referred to as ShishuLM, which reduces both the parameter count and Key-Value (KV) cache requirements. Given the increasing importance of Small Language Models (SLMs) in agentic AI systems, we evaluate our approach on two SLMs of different scales. Our analysis reveals that for moderate-context scenarios, normalization coupled with attention computation is roughly linear with the input, enabling entire transformer blocks to be approximated through Multi-Layer Perceptrons (MLPs). Our results show that ShishuLM provides up to 25% reduction in memory requirements and up to 40% improvement in latency during both training and inference, compared to parent models. Our experimental and analytical findings provide insights towards building more efficient SLM architectures from a pre-training standpoint.

### 24. Vision Mamba for Permeability Prediction of Porous Media

**主要机构**: Stanford University
**作者数量**: 2人

**摘要**:
Vision Mamba has recently received attention as an alternative to Vision Transformers (ViTs) for image classification. The network size of Vision Mamba scales linearly with input image resolution, whereas ViTs scale quadratically, a feature that improves computational and memory efficiency. Moreover, Vision Mamba requires a significantly smaller number of trainable parameters than traditional convolutional neural networks (CNNs), and thus, they can be more memory efficient. Because of these features, we introduce, for the first time, a neural network that uses Vision Mamba as its backbone for predicting the permeability of three-dimensional porous media. We compare the performance of Vision Mamba with ViT and CNN models across multiple aspects of permeability prediction and perform an ablation study to assess the effects of its components on accuracy. We demonstrate in practice the aforementioned advantages of Vision Mamba over ViTs and CNNs in the permeability prediction of three-dimensional porous media. We make the source code publicly available to facilitate reproducibility and to enable other researchers to build on and extend this work. We believe the proposed framework has the potential to be integrated into large vision models in which Vision Mamba is used instead of ViTs.

### 25. WeCKD: Weakly-supervised Chained Distillation Network for Efficient Multimodal Medical Imaging

**主要机构**: 
**作者数量**: 8人

**摘要**:
Knowledge distillation (KD) has traditionally relied on a static teacher-student framework, where a large, well-trained teacher transfers knowledge to a single student model. However, these approaches often suffer from knowledge degradation, inefficient supervision, and reliance on either a very strong teacher model or large labeled datasets, which limits their effectiveness in realworld, limited-data scenarios. To address these, we present the first-ever Weakly-supervised Chain-based KD network (WeCKD) that redefines knowledge transfer through a structured sequence of interconnected models. Unlike conventional KD, it forms a progressive distillation chain, where each model not only learns from its predecessor but also refines the knowledge before passing it forward. This structured knowledge transfer further enhances feature learning, reduces data dependency, and mitigates the limitations of one-step KD. Each model in the distillation chain is trained on only a fraction of the dataset and demonstrates that effective learning can be achieved with minimal supervision. Extensive evaluations across four otoscopic imaging datasets demonstrate that it not only matches but in many cases surpasses the performance of existing supervised methods. Experimental results on two other datasets further underscore its generalization across diverse medical imaging modalities, including microscopic and magnetic resonance imaging. Furthermore, our evaluations resulted in cumulative accuracy gains of up to +23% over a single backbone trained on the same limited data, which highlights its potential for real-world adoption.

### 26. WHAT LAYERS WHEN: LEARNING TO SKIP COMPUTE IN LLMS WITH RESIDUAL GATES

**主要机构**: University of Amsterdam, University of Technology, Qualcomm-UvA Lab, FunAI Lab
**作者数量**: 4人

**摘要**:
We introduce GateSkip, a simple residual-stream gating mechanism that enables token-wise layer skipping in decoder-only LMs. Each Attention/MLP branch is equipped with a sigmoid-linear gate that condense the branch's output before it re-enters the residual stream. During inference we rank tokens by the gate values and skip low-importance ones using a per-layer budget. While early-exit or routerbased Mixture-of-Depths models are known to be unstable and need extensive retraining, our smooth, differentiable gates fine-tune stably on top of pretrained models. On long-form reasoning, we save up to 15% compute while retaining >90% of baseline accuracy. On instruction-tuned models we see accuracy gains at full compute and match baseline quality near 50% savings. The learned gates give insight into transformer information flow (e.g., BOS tokens act as anchors), and the method combines easily with quantization, pruning, and self-speculative decoding.

### 27. 

**主要机构**: 
**作者数量**: 55人

**摘要**:
We introduce xLLM, an intelligent and efficient Large Language Model (LLM) inference framework designed for high-performance, large-scale enterprise-grade serving, with deep optimizations for diverse AI accelerators. Current mainstream inference frameworks face practical challenges. On the one hand, enterprise-grade serving struggles with hybrid and dynamic workloads, strict demand for high availability of services, and distributed storage management. On the other hand, inference execution is bottlenecked by underutilized AI accelerators due to new paradigms of hardwares, model architectures and inference algorithms.
