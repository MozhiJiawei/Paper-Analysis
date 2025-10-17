# AI推理加速技术论文分析报告
生成时间: 2025-10-17 10:56:25
分析论文数量: 20篇

## 论文技术简报

### 1. AdaSwitch: Adaptive Switching Generation for Knowledge Distillation

百度公司发布了AdaSwitch论文，使用动态结合on-policy和off-policy生成（token级别）的技术，解决了现有知识蒸馏中off-policy与on-policy的权衡问题（训练-推理不匹配与低质量监督的矛盾），达成了持续提升小语言模型准确性且额外开销可接受的效果。

### 2. ASYNCSPADE: EFFICIENT TEST-TIME SCALING WITH ASYNCHRONOUS SPARSE DECODING

Johns Hopkins University和University of California发布了AsyncSpade论文，使用轻量级时间回归模块预测查询状态与异步解耦KV缓存过滤的框架，解决了LLM长CoT场景下KV-cache内存瓶颈及解码效率问题，达成在Qwen3等模型上TPOT较Quest降低20%+、较全注意力降低50%+且保持推理精度的效果。

### 3. DeepPrune: Parallel Scaling without Inter-trace Redundancy

清华大学发布了DeepPrune论文，使用基于专门判断模型（focal loss和过采样训练实现等价预测）与在线贪心聚类算法的动态剪枝框架，解决了并行推理中轨迹间冗余导致的计算效率低问题，达成了在多个基准上较传统共识采样减少80%以上token且精度保持在3个百分点内的效果。

### 4. FEWER WEIGHTS, MORE PROBLEMS: A PRACTICAL ATTACK ON LLM PRUNING

研究团队发布了FEWER WEIGHTS, MORE PROBLEMS: A PRACTICAL ATTACK ON LLM PRUNING论文，使用基于参数修剪可能性代理指标构造模型的技术，解决了LLM修剪被恶意利用导致修剪后展现恶意行为的安全隐患，在五种模型上经vLLM修剪后攻击成功率高达95.7%（越狱）、98.7%（良性指令拒绝）及99.5%（目标内容注入）。

### 5. FROM TOKENS TO LAYERS: REDEFINING STALL-FREE SCHEDULING FOR LLM SERVING WITH LAYERED PREFILL

研究团队发布了《为LLM服务重新定义无停滞调度：分层预填充》论文，使用分层预填充技术（以transformer层组为主要调度单元），解决了MoE模型中块预填充导致的冗余专家权重加载、内存流量增加及能耗膨胀问题，降低片外带宽需求，将TTFT提升70%、端到端延迟降低41%、每token能耗降低22%，并改善TTFT-TBT帕累托边界。

### 6. LARGE SCALE DIFFUSION DISTILLATION VIA SCORE-REGULARIZED CONTINUOUS-TIME CONSISTENCY

清华大学发布了LARGE SCALE DIFFUSION DISTILLATION VIA SCORE-REGULARIZED CONTINUOUS-TIME CONSISTENCY论文，使用分数正则化连续时间一致性模型（rCM）及并行兼容的FlashAttention-2 JVP内核，解决了连续时间一致性模型在大规模文本到图像和视频任务中的适用性问题及细节生成质量限制，达成在14B参数模型和5秒视频上匹配或超越SOTA的DMD2、多样性更优且1-4步生成加速15-50倍的效果。

### 7. LESS IS MORE: STRATEGIC EXPERT SELECTION OUTPERFORMS ENSEMBLE COMPLEXITY IN TRAFFIC FORECASTING

布达佩斯厄特沃什·罗兰大学发布了LESS IS MORE论文，使用TESTAM+框架（引入SpatioSemantic Expert整合物理道路拓扑与数据驱动特征相似性及战略专家选择）技术，解决了现有混合专家框架缺乏整合物理道路网络拓扑限制空间能力的问题，达成了在METR-LA和PEMS-BAY数据集上MAE分别降低1.3%和4.1%，且Identity + Adaptive配置比MegaCRN MAE降低11.5%、推理延迟减少53.1%的效果

### 8. LINVIDEO: A Post-Training Framework towards O(n) Attention in Efficient Video Generation

北航与香港科技大学发布了LINVIDEO论文，使用选择性迁移与随时分布匹配（ADM）目标的后训练框架，解决了视频扩散模型自注意力二次复杂度导致的高计算成本问题，达成1.25-2.00×加速且保持生成质量，4步模型更实现15.92×延迟降低。

### 9. LiveThinking: Enabling Real-Time Efficient Reasoning for AI-Powered Livestreaming via Reinforcement Learning

阿里巴巴发布了LiveThinking论文，使用两阶段优化框架（Rejection Sampling Fine-Tuning知识蒸馏与Group Relative Policy Optimization强化学习），解决了AI驱动直播中数字分身实时响应观众评论的高延迟问题，达成了计算成本降低30倍且在多跳推理数据集MuSiQue上性能超越670B教师模型的效果

### 10. Local MAP Sampling for Diffusion Models

加州大学河滨分校发布了Local MAP Sampling for Diffusion Models论文，使用Local MAP Sampling (LMAPS)框架（通过沿扩散轨迹迭代求解局部MAP子问题），解决了逆问题中优化-based扩散求解器缺乏清晰概率基础的问题，达成了在运动去模糊等任务上≥2dB增益、逆散射等基准>1.5dB提升的SOTA性能。

### 11. Multi-Task Pre-Finetuning of Lightweight Transformer Encoders for Text Classification and NER

三星研发 institute发布了轻量级Transformer编码器多任务预微调论文，使用基于任务主导LoRA模块的多任务预微调框架，解决了多任务预微调中冲突优化信号导致性能下降的问题，在21个下游任务上实现NER平均提升+0.8%、文本分类+8.8%，性能可比单独预微调且满足部署约束。

### 12. OBCACHE: OPTIMAL BRAIN KV CACHE PRUNING FOR EFFICIENT LONG-CONTEXT LLM INFERENCE

Duke University和University of Pennsylvania发布了OBCACHE论文，使用基于Optimal Brain Damage (OBD)理论的OBCACHE框架（将KV缓存驱逐建模为分层结构化剪枝问题，通过测量剪枝token对注意力输出的扰动量化token显著性并推导闭形式分数），解决了长上下文LLM推理中KV缓存内存开销大且现有方法未考虑对注意力输出真实影响的问题，达成了在LLaMA和Qwen模型上持续提升长上下文准确性的效果。

### 13. OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs

卡内基梅隆大学发布了OWL论文，使用基于LSTM的drafter（仅依赖最后token状态）、[SPEC]特殊token及混合解码算法等创新，解决了现有推测解码方法在长上下文输入时性能严重下降的问题，达成了在长上下文输入上接受长度较EAGLE3提升约5倍的效果。

### 14. Parallel Test-Time Scaling for Latent Reasoning Models

香港理工大学、中国科学技术大学发布了Parallel Test-Time Scaling for Latent Reasoning Models论文，使用不确定性启发的随机采样策略（蒙特卡洛Dropout和加性高斯噪声）及潜奖励模型（LatentRM），解决了潜推理模型因连续空间缺乏采样机制和概率信号无法受益于并行测试时缩放（TTS）的问题，达成了采样策略随计算量有效扩展并展示不同探索动态、LatentRM实现有效轨迹选择的效果

### 15. Work in Progress RCPU: ROTATION-CONSTRAINED ERROR COMPEN-SATION FOR STRUCTURED PRUNING OF A LARGE LANGUAGE MODEL

KDDI Research, Inc发布了RCPU论文，使用旋转约束补偿方法（结合旋转约束更新与方差感知重要性分数），解决了大型语言模型结构化剪枝时因校准数据有限导致的输出误差及过拟合问题，在LLaMA-7B上的WikiText-2和语言理解基准中达成了优于现有基线的困惑度和任务准确率。

### 16. RESPLAT: LEARNING RECURRENT GAUSSIAN SPLATS

ETH Zurich发布了ReSplat论文，使用前馈循环网络迭代优化3D高斯片（结合紧凑重建模型在16×下采样空间预测高斯），解决了稀疏视图设置下优化型3DGS的性能问题，达成较优化型3DGS快100×、较MVSplat等少16×高斯且渲染快4×的效果

### 17. SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation

Institute for Infocomm Research发布了论文SimCast，使用短期到长期知识蒸馏技术及加权MSE损失并整合到扩散框架CasCast，解决了降水临近预报的准确性问题，在SEVIR、HKO-7、MeteoNet数据集上达成平均CSI分数0.452、0.474、0.361且显著超过现有方法的效果。

### 18. STEPER: Step-wise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented Language Models

研究团队发布了STEPPER论文，使用分步知识蒸馏（结合分步监督与难度感知训练）技术，解决了现有知识蒸馏方法忽视多步骤中不同推理能力需求、阻碍多步检索增强框架迁移的问题，达成了在多跳QA基准上优于现有方法，8B模型性能接近70B教师模型的效果。

### 19. THINK JUST ENOUGH: SEQUENCE-LEVEL ENTROPY AS A CONFIDENCE SIGNAL FOR LLM REASONING

相关机构发布了《THINK JUST ENOUGH: SEQUENCE-LEVEL ENTROPY AS A CONFIDENCE SIGNAL FOR LLM REASONING》论文，使用基于序列级熵的置信度信号实现早期停止，解决了大语言模型推理任务中的token效率问题，达成了25-50%计算量节省同时保持任务准确率的效果。

### 20. Which Heads Matter for Reasoning? RL-Guided KV Cache Compression WHICH HEADS MATTER FOR REASONING? RL-GUIDED KV CACHE COMPRESSION

McGill University和Westlake University发布了《Which Heads Matter for Reasoning? RL-Guided KV Cache Compression》论文，使用RLKV（强化学习引导的推理关键头识别框架），解决了推理大语言模型解码时KV缓存开销大且现有压缩方法破坏推理完整性或错误压缩关键头的问题，达成了20-50%缓存减少下接近无损性能且优于基线方法的效果。

## 论文详细信息

### 1. AdaSwitch: Adaptive Switching Generation for Knowledge Distillation

**主要机构**: § City University of Hong Kong, † Baidu Inc, University of Science and Technology of China
**作者数量**: 8人

**摘要**:
Small language models (SLMs) are crucial for applications with strict latency and computational constraints, yet achieving high performance remains challenging. Knowledge distillation (KD) can transfer capabilities from large teacher models, but existing methods involve trade-offs: off-policy distillation provides high-quality supervision but introduces a training-inference mismatch, while on-policy approaches maintain consistency but rely on lowquality student outputs. To address these issues, we propose AdaSwitch, a novel approach that dynamically combines on-policy and off-policy generation at the token level. AdaSwitch allows the student to first explore its own predictions and then selectively integrate teacher guidance based on real-time quality assessment. This approach simultaneously preserves consistency and maintains supervision quality. Experiments on three datasets with two teacher-student LLM pairs demonstrate that AdaSwitch consistently improves accuracy, offering a practical and effective method for distilling SLMs with acceptable additional overhead.

### 2. ASYNCSPADE: EFFICIENT TEST-TIME SCALING WITH ASYNCHRONOUS SPARSE DECODING

**主要机构**: Johns Hopkins University, University of California, //github.com/UNITES-Lab/AsyncSpade, University of North Carolina
**作者数量**: 5人

**摘要**:
Test-time scaling (TTS) boosts LLM reasoning via long chain-of-thought (CoT), but the linear KV-cache growth amplifies the memory-bound bottleneck of LLM decoding. Query-aware page-level sparse decoding can achieve state-of-the-art performance under constrained FLOPs budgets, but is limited by both sequentialdependent page filtering and coarse-grained token selection, hampering serving efficiency and model performance on TTS tasks under high concurrency and long CoT scenarios (consuming even higher runtime than the forward pipeline itself). In this paper, we first find that the current-step query state can be accurately approximated in a unified manner from a short window of recent queries, enabling training-free query-aware sparsity without waiting in the decoding loop. We propose AsyncSpade, an asynchronous framework for efficient TTS built on two core components: (1) a novel lightweight temporal-regressive module that predicts the next-token query state; (2) an asynchronous and disaggregated framework that decouples the KV cache filtering from the auto-regressive decoding loop, overlapping the token-level KV selection with the forward inference computation through asynchronism. To our knowledge, AsyncSpade is the first to eliminate the sequential dependence without sacrificing model performance. We validate the effectiveness of AsyncSpade on common LLM serving setups with an A100 node, where AsyncSpade fully overlaps KV-cache operations with the inference pipeline, achieving theoretical optimal time-peroutput-token (TPOT). Specifically, AsyncSpade delivers over 20% reduction on TPOT compared to SoTA baseline (i.e. Quest) and at least 50% TPOT reduction compared to full attention on Qwen3-8B and Qwen3-32B models, while matching or surpassing their accuracy on various TTS benchmarks (AIME-24/25, GPQA-Diamond, MATH-500).

### 3. DeepPrune: Parallel Scaling without Inter-trace Redundancy

**主要机构**: ShanghaiTech University, Tsinghua University
**作者数量**: 5人

**摘要**:
Parallel scaling has emerged as a powerful paradigm to enhance reasoning capabilities in large language models (LLMs) by generating multiple Chain-of-Thought (CoT) traces simultaneously. However, this approach introduces significant computational inefficiency due to inter-trace redundancy-our analysis reveals that over 80% of parallel reasoning traces yield identical final answers, representing substantial wasted computation. To address this critical efficiency bottleneck, we propose DeepPrune, a novel framework that enables efficient parallel scaling through dynamic pruning. Our method features a specialized judge model trained with focal loss and oversampling techniques to accurately predict answer equivalence from partial reasoning traces which realizes 0.87 AUROC on equivalence prediction, combined with an online greedy clustering algorithm that dynamically prunes redundant paths while preserving answer diversity. Comprehensive evaluations across three challenging benchmarks (AIME 2024, AIME 2025, and GPQA) and multiple reasoning models demonstrate that DeepPrune achieves remarkable token reduction by over 80% compared to conventional consensus sampling on most cases, while maintaining competitive accuracy within 3 percentage points. Our work establishes a new standard for efficient parallel reasoning, making high-performance reasoning more efficient. Our code and data are here: https://deepprune.github.io.

### 4. FEWER WEIGHTS, MORE PROBLEMS: A PRACTICAL ATTACK ON LLM PRUNING

**主要机构**: 
**作者数量**: 6人

**摘要**:
Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to 95.7% for jailbreak, 98.7% for benign instruction refusal, and 99.5% for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.

### 5. FROM TOKENS TO LAYERS: REDEFINING STALL-FREE SCHEDULING FOR LLM SERVING WITH LAYERED PREFILL

**主要机构**: 
**作者数量**: 5人

**摘要**:
Large Language Model (LLM) inference in production must meet stringent service-level objectives for both time-to-first-token (TTFT) and time-between-token (TBT) while maximizing throughput under fixed compute, memory, and interconnect budgets. Modern serving systems adopt stall-free scheduling techniques such as chunked prefill, which splits long prompt processing along the token dimension and interleaves prefill with ongoing decode iterations. While effective at stabilizing TBT, chunked prefill incurs substantial overhead in Mixture-of-Experts (MoE) models: redundant expert weight loads increase memory traffic by up to 39% and inflate energy consumption. We propose layered prefill, a new scheduling paradigm that treats transformer layer groups as the primary scheduling unit. By vertically partitioning the model into contiguous layer groups and interleaving prefill and decode across the groups, layered prefill sustains stall-free decoding while eliminating chunk-induced MoE weight reloads. It reduces off-chip bandwidth demand, lowering TTFT by up to 70%, End-to-End latency by 41% and per-token energy by up to 22%. Evaluations show that layered prefill consistently improves the TTFT-TBT Pareto frontier over chunked prefill, reducing expert-load traffic and energy cost while maintaining stall-free decoding. Overall, shifting the scheduling axis from tokens to layers unlocks a new operating regime for high-efficiency, energy-aware LLM serving in co-located environments.

### 6. LARGE SCALE DIFFUSION DISTILLATION VIA SCORE-REGULARIZED CONTINUOUS-TIME CONSISTENCY

**主要机构**: Tsinghua University
**作者数量**: 10人

**摘要**:
This work represents the first effort to scale up continuous-time consistency distillation to general application-level image and video diffusion models. Although continuous-time consistency model (sCM) is theoretically principled and empirically powerful for accelerating academic-scale diffusion, its applicability to large-scale text-to-image and video tasks remains unclear due to infrastructure challenges in Jacobian-vector product (JVP) computation and the limitations of standard evaluation benchmarks. We first develop a parallelism-compatible FlashAttention-2 JVP kernel, enabling sCM training on models with over 10 billion parameters and high-dimensional video tasks. Our investigation reveals fundamental quality limitations of sCM in fine-detail generation, which we attribute to error accumulation and the "mode-covering" nature of its forward-divergence objective. To remedy this, we propose the score-regularized continuous-time consistency model (rCM), which incorporates score distillation as a long-skip regularizer. This integration complements sCM with the "mode-seeking" reverse divergence, effectively improving visual quality while maintaining high generation diversity. Validated on large-scale models (Cosmos-Predict2, Wan2.1) up to 14B parameters and 5-second videos, rCM matches or surpasses the state-of-the-art distillation method DMD2 on quality metrics while offering notable advantages in diversity, all without GAN tuning or extensive hyperparameter searches. The distilled models generate high-fidelity samples in only 1 ∼ 4 steps, accelerating diffusion sampling by 15× ∼ 50×. These results position rCM as a practical and theoretically grounded framework for advancing large-scale diffusion distillation.

### 7. LESS IS MORE: STRATEGIC EXPERT SELECTION OUTPERFORMS ENSEMBLE COMPLEXITY IN TRAFFIC FORECASTING

**主要机构**: Eötvös Loránd University Budapest, Department of Artificial Intelligence ELTE
**作者数量**: 3人

**摘要**:
Traffic forecasting is fundamental to intelligent transportation systems, enabling congestion mitigation and emission reduction in increasingly complex urban environments. While recent graph neural network approaches have advanced spatial-temporal modeling, existing mixture-of-experts frameworks like Time-Enhanced Spatio-Temporal Attention Model (TESTAM) lack explicit incorporation of physical road network topology, limiting their spatial capabilities. We present TESTAM+, an enhanced spatio-temporal forecasting framework that introduces a novel SpatioSemantic Expert integrating physical road topology with data-driven feature similarity through hybrid graph construction. TESTAM+ achieves significant improvements over TESTAM: 1.3% MAE reduction on METR-LA (3.10 vs. 3.14) and 4.1% improvement on PEMS-BAY (1.65 vs. 1.72). Through comprehensive ablation studies, we discover that strategic expert selection fundamentally outperforms naive ensemble aggregation. Individual experts demonstrate remarkable effectiveness: the Adaptive Expert achieves 1.63 MAE on PEMS-BAY, outperforming the original three-expert TESTAM (1.72 MAE), while the SpatioSemantic Expert matches this performance with identical 1.63 MAE. The optimal Identity + Adaptive configuration achieves an 11.5% MAE reduction compared to state-of-the-art MegaCRN on METR-LA (2.99 vs. 3.38), while reducing inference latency by 53.1% compared to the full four-expert TESTAM+. Our findings reveal that fewer, strategically designed experts outperform complex multi-expert ensembles, establishing new state-of-the-art performance with superior computational efficiency for real-time deployment.

### 8. LINVIDEO: A Post-Training Framework towards O(n) Attention in Efficient Video Generation

**主要机构**: Beihang University, Hong Kong University of Science and Technology, Sensetime Research
**作者数量**: 5人

**摘要**:
Video diffusion models (DMs) have enabled high-quality video synthesis. However, their computation costs scale quadratically with sequence length because self-attention has quadratic complexity. While linear attention lowers the cost, fully replacing quadratic attention requires expensive pretraining due to the limited expressiveness of linear attention and the complexity of spatiotemporal modeling in video generation. In this paper, we present LINVIDEO, an efficient data-free post-training framework that replaces a target number of self-attention modules with linear attention while preserving the original model's performance. First, we observe a significant disparity in the replaceability of different layers. Instead of manual or heuristic choices, we frame layer selection as a binary classification problem and propose selective transfer, which automatically and progressively converts layers to linear attention with minimal performance impact. Additionally, to overcome the ineffectiveness and even inefficiency of existing objectives in optimizing this challenge transfer process, we introduce an anytime distribution matching (ADM) objective that aligns the distributions of samples across any timestep along the sampling trajectory. This objective is highly efficient and recovers model performance. Extensive experiments show that our method achieves a 1.25-2.00× speedup while preserving generation quality, and our 4-step distilled model further delivers a 15.92× latency reduction with minimal visual quality drop.

### 9. LiveThinking: Enabling Real-Time Efficient Reasoning for AI-Powered Livestreaming via Reinforcement Learning

**主要机构**: Zhiwei Huang, Taobao & Tmall Group of Alibaba Beijing, Tmall Group of Alibaba Hangzhou, Wanqing Cui, Junfeng Ma, ROLL Team of Alibaba Hangzhou, Shaopan Xiong, Yazhi Guo
**作者数量**: 9人

**摘要**:
The real-time nature of e-commerce livestreaming requires digital avatars to respond immediately to viewer comments, ensuring conversational fluency and seamless user experience. This goes beyond typical chatbot functionality, requiring not only correctness and helpfulness but also ultra-low latency. This poses a significant challenge for state-of-the-art Retrieval-Augmented Generation (RAG) systems powered by Large Reasoning Models (LRMs), which often incur high inference delays despite their strong reasoning capabilities. To mitigate this latency-quality trade-off, we propose Live-Thinking, a two-stage optimization framework. First, we employ knowledge distillation via Rejection Sampling Fine-Tuning (RFT) to compress a large teacher LRM into a lightweight architecture, significantly reducing computational cost. While this drastically lowers inference cost, the distilled model inherits verbose reasoning paths that still exceed latency budgets. In the second stage, we apply reinforcement learning via Group Relative Policy Optimization (GRPO) to directly shorten the reasoning path, with a reward function that prioritizes response brevity while preserving correctness and helpfulness. LiveThinking demonstrates strong performance on both public benchmarks and real-world deployments. On the multi-hop reasoning dataset MuSiQue, our 30B MoE model (3B active) outperforms a 670B teacher in EM (+12.3) and F1 (+10.2), with 41% shorter responses and 95% lower decoding cost. Deployed on Taobao Live, LiveThinking reduces computational cost by 30× (12× from distillation and 2.5× from reasoning path compression),

### 10. Local MAP Sampling for Diffusion Models

**主要机构**: Vector Institute, University of California Riverside
**作者数量**: 3人

**摘要**:
Diffusion Posterior Sampling (DPS) provides a principled Bayesian approach to inverse problems by sampling from p(x 0 | y). However, in practice, the goal of inverse problem solving is not to cover the posterior but to recover the most accurate reconstruction, where optimization-based diffusion solvers often excel despite lacking a clear probabilistic foundation. We introduce Local MAP Sampling (LMAPS), a new inference framework that iteratively solving local MAP subproblems along the diffusion trajectory. This perspective clarifies their connection to global MAP estimation and DPS, offering a unified probabilistic interpretation for optimization-based methods. Building on this foundation, we develop practical algorithms with a probabilistically interpretable covariance approximation, a reformulated objective for stability and interpretability, and a gradient approximation for non-differentiable operators. Across a broad set of image restoration and scientific tasks, LMAPS achieves state-of-the-art performance, including ≥ 2 dB gains on motion deblurring, JPEG restoration, and quantization, and > 1.5 dB improvements on inverse scattering benchmarks.

### 11. Multi-Task Pre-Finetuning of Lightweight Transformer Encoders for Text Classification and NER

**主要机构**: Samsung R&D Institute
**作者数量**: 6人

**摘要**:
Deploying natural language processing (NLP) models on mobile platforms requires models that can adapt across diverse applications while remaining efficient in memory and computation. We investigate pre-finetuning strategies to enhance the adaptability of lightweight BERTlike encoders for two fundamental NLP task families: named entity recognition (NER) and text classification. While pre-finetuning improves downstream performance for each task family individually, we find that naïve multitask pre-finetuning introduces conflicting optimization signals that degrade overall performance. To address this, we propose a simple yet effective multi-task pre-finetuning framework based on task-primary LoRA modules, which enables a single shared encoder backbone with modular adapters. Our approach achieves performance comparable to individual pre-finetuning while meeting practical deployment constraint. Experiments on 21 downstream tasks show average improvements of +0.8% for NER and +8.8% for text classification, demonstrating the effectiveness of our method for versatile mobile NLP applications.

### 12. OBCACHE: OPTIMAL BRAIN KV CACHE PRUNING FOR EFFICIENT LONG-CONTEXT LLM INFERENCE

**主要机构**: Duke University, University of Pennsylvania, University of Electronic Science and Technology of China
**作者数量**: 4人

**摘要**:
Large language models (LLMs) with extended context windows enable powerful downstream applications but impose significant memory overhead, as caching all key-value (KV) states scales linearly with sequence length and batch size. Existing cache eviction methods address this by exploiting attention sparsity, yet they typically rank tokens heuristically using accumulated attention weights without considering their true impact on attention outputs. We propose Optimal Brain Cache (OBCACHE), a principled framework that formulates cache eviction as a layer-wise structured pruning problem. Building upon the Optimal Brain Damage (OBD) theory, OBCACHE quantifies token saliency by measuring the perturbation in attention outputs induced by pruning tokens, with closed-form scores derived for isolated keys, isolated values, and joint key-value pairs. Our scores account not only for attention weights but also for information from value states and attention outputs, thereby enhancing existing eviction strategies with output-aware signals. Experiments on LLaMA and Qwen models demonstrate that replacing the heuristic scores in existing works, which estimate token saliency across different query positions, with OBCACHE's output-aware scores consistently improves long-context accuracy. Our code is available at this link.

### 13. OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs

**主要机构**: Carnegie Mellon University, Snowflake AI Research, Seoul National University
**作者数量**: 6人

**摘要**:
Speculative decoding promises faster inference for large language models (LLMs), yet existing methods fail to generalize to real-world settings. Benchmarks typically assume short contexts (e.g., 2K tokens), whereas practical workloads involve long contexts. We find current approaches degrade severely with long contexts; for instance, EAGLE3 even slows down the generation speed by 0.81×. We address these limitations by releasing a new long-context benchmark (LongSpecBench) and introducing a novel model (OWL). OWL achieves about 5× higher acceptance length than EAGLE3 on long-context inputs through three innovations: (1) an LSTM-based drafter conditioned only on the last-token state, making it generalize to various lengths, (2) a special token [SPEC] in the verifier that produces richer representation for drafter, and (3) a hybrid algorithm combining both tree and non-tree decoding methods. We release all code and datasets to advance future research. 1

### 14. Parallel Test-Time Scaling for Latent Reasoning Models

**主要机构**: The Hong Kong Polytechnic University, University of Science and Technology of China, Shandong Jianzhu University, Harbin Institute of Technology
**作者数量**: 6人

**摘要**:
Parallel test-time scaling (TTS) is a pivotal approach for enhancing large language models (LLMs), typically by sampling multiple tokenbased chains-of-thought in parallel and aggregating outcomes through voting or search. Recent advances in latent reasoning, where intermediate reasoning unfolds in continuous vector spaces, offer a more efficient alternative to explicit Chain-of-Thought, yet whether such latent models can similarly benefit from parallel TTS remains open, mainly due to the absence of sampling mechanisms in continuous space, and the lack of probabilistic signals for advanced trajectory aggregation. This work enables parallel TTS for latent reasoning models by addressing the above issues. For sampling, we introduce two uncertainty-inspired stochastic strategies: Monte Carlo Dropout and Additive Gaussian Noise. For aggregation, we design a Latent Reward Model (LatentRM) trained with step-wise contrastive objective to score and guide latent reasoning. Extensive experiments and visualization analyses show that both sampling strategies scale effectively with compute and exhibit distinct exploration dynamics, while LatentRM enables effective trajectory selection. Together, our explorations open a new direction for scalable inference in continuous spaces. Code released at here.

### 15. Work in Progress RCPU: ROTATION-CONSTRAINED ERROR COMPEN-SATION FOR STRUCTURED PRUNING OF A LARGE LANGUAGE MODEL

**主要机构**: Mori Kurokawa AI Division KDDI Research, Inc
**作者数量**: 4人

**摘要**:
In this paper, we propose a rotation-constrained compensation method to address the errors introduced by structured pruning of large language models (LLMs). LLMs are trained on massive datasets and accumulate rich semantic knowledge in their representation space. In contrast, pruning is typically carried out with only a small amount of calibration data, which makes output mismatches unavoidable. Although direct least-squares fitting can reduce such errors, it tends to overfit to the limited calibration set, destructively modifying pretrained weights. To overcome this difficulty, we update the pruned parameters under a rotation constraint. This constrained update preserves the geometry of output representations (i.e., norms and inner products) and simultaneously realigns the pruned subspace with the original outputs. Furthermore, in rotation-constrained compensation, removing components that strongly contribute to the principal directions of the output makes error recovery difficult. Since input dimensions with large variance strongly affect these principal directions, we design a variance-aware importance score that ensures such dimensions are preferentially kept in the pruned model. By combining this scoring rule with rotation-constrained updates, the proposed method effectively compensates errors while retaining the components likely to be more important in a geometry-preserving manner. In the experiments, we apply the proposed method to LLaMA-7B and evaluate it on WikiText-2 and multiple language understanding benchmarks. The results demonstrate consistently better perplexity and task accuracy compared with existing baselines.

### 16. RESPLAT: LEARNING RECURRENT GAUSSIAN SPLATS

**主要机构**: Tübingen AI Center, University of Tübingen, ETH Zurich
**作者数量**: 4人

**摘要**:
ReSplat (Iter 3) GT Figure 1: Learning recurrent Gaussian splats in a feed-forward manner. We propose ReSplat, a feed-forward recurrent network that iteratively refines 3D Gaussian splats to improve sparse view settings where optimization-based 3DGS (Kerbl et al., 2023) struggles. As initialization (iteration 0), we introduce a compact reconstruction model that predicts Gaussians in a 16× subsampled space, producing 16× fewer Gaussians and 4× faster rendering than per-pixel MVSplat (Chen et al., 2024b) and DepthSplat (Xu et al., 2025). The reduced number of Gaussians makes subsequent refinement efficient. Compared to the optimization-based 3DGS, ReSplat is 100× faster thanks to its feedforward design, while still benefiting from iterative updates. Here we show results for 8 input views (512 × 960 resolution) on DL3DV (Ling et al., 2023) dataset; see Table 1 for detailed metrics.

### 17. SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation

**主要机构**: Institute for Infocomm Research
**作者数量**: 7人

**摘要**:
Precipitation nowcasting predicts future radar sequences based on current observations, which is a highly challenging task driven by the inherent complexity of the Earth system. Accurate nowcasting is of utmost importance for addressing various societal needs, including disaster management, agriculture, transportation, and energy optimization. As a complementary to existing non-autoregressive nowcasting approaches, we investigate the impact of prediction horizons on nowcasting models and propose SimCast, a novel training pipeline featuring a short-to-long term knowledge distillation technique coupled with a weighted MSE loss to prioritize heavy rainfall regions. Improved nowcasting predictions can be obtained without introducing additional overhead during inference. As SimCast generates deterministic predictions, we further integrate it into a diffusion-based framework named CasCast, leveraging the strengths from probabilistic models to overcome limitations such as blurriness and distribution shift in deterministic outputs. Extensive experimental results on three benchmark datasets validate the effectiveness of the proposed framework, achieving mean CSI scores of 0.452 on SEVIR, 0.474 on HKO-7, and 0.361 on MeteoNet, which outperforms existing approaches by a significant margin.

### 18. STEPER: Step-wise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented Language Models

**主要机构**: 
**作者数量**: 5人

**摘要**:
Answering complex real-world questions requires step-by-step retrieval and integration of relevant information to generate well-grounded responses. However, existing knowledge distillation methods overlook the need for different reasoning abilities at different steps, hindering transfer in multi-step retrieval-augmented frameworks. To address this, we propose Stepwise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented Language Models (STEPER). STE-PER employs step-wise supervision to align with evolving information and reasoning demands across stages. Additionally, it incorporates difficulty-aware training to progressively optimize learning by prioritizing suitable steps. Our method is adaptable to various multi-step retrieval-augmented language models, including those that use retrieval queries for reasoning paths or decomposed questions. Extensive experiments show that STEPER outperforms prior methods on multi-hop QA benchmarks, with an 8B model achieving performance comparable to a 70B teacher model.

### 19. THINK JUST ENOUGH: SEQUENCE-LEVEL ENTROPY AS A CONFIDENCE SIGNAL FOR LLM REASONING

**主要机构**: 
**作者数量**: 3人

**摘要**:
We introduce a simple, yet novel entropy-based framework to drive token efficiency in large language models during reasoning tasks. Our approach uses Shannon entropy from token-level logprobs as a confidence signal to enable early stopping, achieving 25-50% computational savings while maintaining task accuracy. Crucially, we demonstrate that entropy-based confidence calibration represents an emergent property of advanced post-training optimization present in modern reasoning models but notably absent in standard instruction-tuned and pre-trained models (Llama 3.3 70B). We show that the entropy threshold to stop reasoning varies from model to model but can be calculated easily in one shot using only a few examples from existing reasoning datasets. Our results indicate that advanced reasoning models often know that they've gotten a correct answer early on, and that this emergent confidence awareness can be exploited to save tokens and reduce latency.

### 20. Which Heads Matter for Reasoning? RL-Guided KV Cache Compression WHICH HEADS MATTER FOR REASONING? RL-GUIDED KV CACHE COMPRESSION

**主要机构**: McGill University, Westlake University
**作者数量**: 5人

**摘要**:
Reasoning large language models exhibit complex reasoning behaviors through the extended chain-of-thought generation, creating unprecedented Key-Value (KV) cache overhead during the decoding phase. Existing KV cache compression methods underperform on reasoning models: token-dropping methods break reasoning integrity by discarding critical information, while head-reallocating methods mistakenly compress reasoning-critical heads since they are designed for retrieval tasks, resulting in significant performance degradation as compression rates increase. We hypothesize that KV heads exhibit functional heterogeneity in reasoning models-some heads are critical for chain-of-thought consistency while others are compressible. To validate and exploit this insight, we propose RLKV, a novel reasoning-critical head identification framework, which uses reinforcement learning to directly optimize the relationship between each head's cache usage and reasoning quality. As RLKV produces rewards from actual generated samples during training, it naturally identifies heads relevant to reasoning behaviors. We then allocate full KV cache to these heads while applying compressed constant KV cache to others for efficient inference. Our experiments reveal that only a small fraction of attention heads is essential for reasoning, enabling our KV compression approach to outperform baseline methods while achieving 20-50% cache reduction with near lossless performance compared to uncompressed results.
