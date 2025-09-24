# AI推理加速技术论文分析报告
生成时间: 2025-09-24 10:35:51
分析论文数量: 14篇

## 论文技术简报

### 1. A1: ASYNCHRONOUS TEST-TIME SCALING VIA CON-FORMAL PREDICTION

华为诺亚方舟实验室发布了A1: Asynchronous Test-Time Scaling via Conformal Prediction论文，使用异步测试时扩展（A1）框架（通过优化算术强度识别同步瓶颈、在线校准实现异步推理及三阶段拒绝采样管道），解决了大语言模型测试时扩展中的同步开销、内存瓶颈和延迟问题，达成了56.7倍测试时扩展加速、4.14倍吞吐量提升，同时保持准确的拒绝率控制、降低延迟和内存开销且无精度损失的效果。

### 2. ADVERSARIAL DISTILLED RETRIEVAL-AUGMENTED GUARDING MODEL FOR ONLINE MALICIOUS INTENT DETECTION

研究团队发布了ADVERSARIAL DISTILLED RETRIEVAL-AUGMENTED GUARDING MODEL FOR ONLINE MALICIOUS INTENT DETECTION论文，使用两阶段对抗蒸馏检索增强框架（含对抗扰动检索增强训练教师模型、蒸馏至紧凑学生模型及在线更新知识库），解决了现有方法难以实时处理多样复杂用户查询的在线恶意意图检测问题，达成149M参数模型性能达WildGuard-7B的98.5%、分布外检测超GPT-4 3.3%和Llama-Guard-3-8B 9.5%且延迟降低5.6倍（300 QPS）的效果。

### 3. CROSS-MODAL KNOWLEDGE DISTILLATION FOR SPEECH LARGE LANGUAGE MODELS

腾讯公司发布了语音大语言模型跨模态知识蒸馏论文，使用跨模态知识蒸馏框架（利用文本-文本及语音-文本通道），解决了语音大语言模型中灾难性遗忘和模态不等价导致的文本知识与推理退化、语音查询性能下降问题，达成了保留文本知识、改善跨模态对齐并增强语音交互推理能力的效果

### 4. Delta Knowledge Distillation for Large Language Models

发布了Delta Knowledge Distillation for Large Language Models论文，使用Delta-KD技术，通过显式保留教师SFT期间引入的分布偏移∆让学生逼近最优表示空间，解决了传统token-level KD中教师和学生可能不共享相同最优表示空间的问题，在ROUGE指标上提升了学生性能并保留更多教师知识。

### 5. Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation

博洛尼亚大学发布了Depth AnyEvent论文，使用跨模态蒸馏范式，解决了基于事件的单目深度估计问题，达成了降低RMSE、提升深度估计精度的效果。

### 6. Fair-GPTQ: Bias-Aware Quantization for Large Language Models

École Centrale de Lyon发布了Fair-GPTQ论文，使用添加群体公平约束的量化技术，解决了现有量化方法导致大型语言模型偏见输出增加的问题，达成了减少模型不公平性同时保持至少90%基线准确率及内存和速度优势的效果。

### 7. Forecasting and Visualizing Air Quality from Sky Images with Vision-Language Models

Georgia State University等机构发布了Forecasting and Visualizing Air Quality from Sky Images with Vision-Language Models论文，使用结合统计纹理分析与监督学习的污染分类及视觉语言模型（VLM）引导的生成建模技术，解决了传统空气质量监测系统空间覆盖和可及性有限的问题，达成了有效估计污染水平并生成语义一致可视化以支持环境决策的效果

### 8. FURINA: FREE FROM UNMERGEABLE ROUTER VIA LINEAR AGGREGATION OF MIXED EXPERTS

复旦大学、腾讯发布了FURINA论文，使用自路由机制及专家线性聚合技术，解决了现有MoE-LoRA方法因离散路由器导致的无法整合到骨干模型及推理开销问题，达成了可完全合并到骨干模型、消除推理开销且性能优于标准LoRA并匹配现有MoE-LoRA的效果

### 9. HARNESS: Lightweight Distilled Arabic Speech Foundation Models

Qatar Computing Research Institute发布了HARNES论文，使用迭代自蒸馏与低秩近似技术，解决了大型预训练语音模型在资源有限环境部署难的问题，在阿拉伯语ASR、SER、DID任务上优于HuBERT和XLS-R，实现SOTA或可比性能，成为轻量级强大替代方案。

### 10. MEANFLOWSE: ONE-STEP GENERATIVE SPEECH ENHANCEMENT VIA CONDITIONAL MEAN FLOW

厦门大学发布了MEANFLOWSE论文，使用条件平均流学习有限区间平均速度并结合雅可比向量积(JVP)推导局部训练目标的技术，解决了实时生成式语音增强中的多步推理瓶颈，达成单步生成即可实现强清晰度、保真度和感知质量且计算成本远低于多步基线模型的效果。

### 11. Q-ROAR: Outlier-Aware Rescaling for RoPE Position Interpolation in Quantized Long-Context LLMs

加州大学发布了Q-ROAR论文，使用RoPE感知的仅权重稳定技术（将RoPE维度分组为频率带并搜索每带尺度），解决了结合RoPE位置插值与后训练量化导致的量化长上下文LLM精度下降问题，达成了恢复0.7%标准任务准确率、GovReport困惑度降低超10%并保持短上下文性能及现有推理兼容性的效果。

### 12. SparseDoctor: Towards Efficient Chat Doctor with Mixture of Experts Enhanced Large Language Models

香港理工大学与香港珠海学院发布了SparseDoctor论文，使用对比学习增强的LoRA-MoE架构（含自动路由与专家记忆队列机制），解决了传统医疗LLM微调训练成本高的问题，在CMB、CMExam、CMMLU-Med医疗基准上持续优于HuatuoGPT系列等强基线。

### 13. Under review as a conference paper TABLEDART: DYNAMIC ADAPTIVE MULTI-MODAL ROUTING FOR TABLE UNDERSTANDING

The University of Queensland发布了TableDART论文，使用轻量级MLP门控网络动态选择多模态路径及跨模态知识整合agent的训练高效框架，解决了表格理解中语义与结构信息建模的冗余冲突及MLLM微调成本高的问题，达成在七个基准上超越最强开源模型平均4.02%的效果。

### 14. Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization

研究团队发布了Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization论文，提出robust-kbench基准与智能代理框架（含PyTorch转CUDA内核、进化元生成优化及LLM验证器），解决了现有CUDA内核基准漏洞多、多样性不足及低级别优化关注少的问题，实现了CUDA内核性能优于PyTorch、能融合操作且验证器准确分类错误内核的效果。

## 论文详细信息

### 1. A1: ASYNCHRONOUS TEST-TIME SCALING VIA CON-FORMAL PREDICTION

**主要机构**: Huawei Noah's Ark Lab, Independent Researcher, The University of Hong Kong
**作者数量**: 14人

**摘要**:
Large language models (LLMs) benefit from test-time scaling, but existing methods face significant challenges, including severe synchronization overhead, memory bottlenecks, and latency, especially during speculative decoding with long reasoning chains. We introduce A1 (Asynchronous Test-Time Scaling), a statistically guaranteed adaptive inference framework that addresses these challenges. A1 refines arithmetic intensity to identify synchronization as the dominant bottleneck, proposes an online calibration strategy to enable asynchronous inference, and designs a three-stage rejection sampling pipeline that supports both sequential and parallel scaling. Through experiments on the MATH, AMC23, AIME24, and AIME25 datasets, across various draft-target model families, we demonstrate that A1 achieves a remarkable 56.7x speedup in test-time scaling and a 4.14x improvement in throughput, all while maintaining accurate rejection-rate control, reducing latency and memory overhead, and no accuracy loss compared to using target model scaling alone. These results position A1 as an efficient and principled solution for scalable LLM inference. We have released the code at https://github.com/menik1126/asynchronous-test-time-scaling.

### 2. ADVERSARIAL DISTILLED RETRIEVAL-AUGMENTED GUARDING MODEL FOR ONLINE MALICIOUS INTENT DETECTION

**主要机构**: 
**作者数量**: 22人

**摘要**:
With the deployment of Large Language Models (LLMs) in interactive applications, online malicious intent detection has become increasingly critical. However, existing approaches fall short of handling diverse and complex user queries in real time. To address these challenges, we introduce ADRAG (Adversarial Distilled Retrieval-Augmented Guard), a two-stage framework for robust and efficient online malicious intent detection. In the training stage, a high-capacity teacher model is trained on adversarially perturbed, retrieval-augmented inputs to learn robust decision boundaries over diverse and complex user queries. In the inference stage, a distillation scheduler transfers the teacher's knowledge into a compact student model, with a continually updated knowledge base collected online. At deployment, the compact student model leverages top-K similar safety exemplars retrieved from the online-updated knowledge base to enable both online and real-time malicious query detection. Evaluations across ten safety benchmarks demonstrate that ADRAG, with 149M parameters model, achieves 98.5% of WildGuard-7B's performance, surpasses GPT-4 by 3.3% and Llama-Guard-3-8B by 9.5% on out-of-distribution detection, while simultaneously delivering up to 5.6× lower latency at 300 queries per second (QPS) in real-time applications.

### 3. CROSS-MODAL KNOWLEDGE DISTILLATION FOR SPEECH LARGE LANGUAGE MODELS

**主要机构**: Tencent Corporation, Tencent Ethereal Audio Lab, College of Computer Science, TMCC, Nankai University
**作者数量**: 6人

**摘要**:
In this work, we present the first systematic evaluation of catastrophic forgetting and modality inequivalence in speech large language models, showing that introducing speech capabilities can degrade knowledge and reasoning even when inputs remain textual, and performance further decreases with spoken queries. To address these challenges, we propose a cross-modal knowledge distillation framework that leverages both text-to-text and speech-to-text channels to transfer knowledge from a text-based teacher model to a speech LLM. Extensive experiments on dialogue and audio understanding tasks validate the effectiveness of our approach in preserving textual knowledge, improving cross-modal alignment, and enhancing reasoning in speech-based interactions.

### 4. Delta Knowledge Distillation for Large Language Models

**主要机构**: 
**作者数量**: 5人

**摘要**:
Knowledge distillation (KD) is a widely adopted approach for compressing large neural networks by transferring knowledge from a large teacher model to a smaller student model. In the context of large language models, tokenlevel KD, typically minimizing the KL divergence between student output distribution π s and teacher output distribution π t , has shown strong empirical performance. However, prior work typically assumes π t and π s share the same optimal representation space-a premise that may not hold in many cases. To solve this problem, we propose Delta Knowledge Distillation (Delta-KD), a novel extension of tokenlevel KD that encourages the student to approximate an optimal representation space by explicitly preserving the distributional shift ∆ introduced during the teacher's supervised finetuning (SFT). Empirical results on ROUGE metrics demonstrate that Delta-KD substantially improves student performance while preserving more of the teacher's knowledge. 1

### 5. Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation

**主要机构**: Advanced Research Center on Electronic System (ARCES), University of Bologna, Department of Computer Science and Engineering (DISI)
**作者数量**: 5人

**摘要**:
Figure 1. DepthAnyEvent-R in action. The first column shows the input frame (used only for distillation) and the corresponding event visualization. The other three columns present depth estimation results from different approaches: E2Depth [15], our DepthAnyEvent-R, and our DepthAnyEvent-R trained with our distillation approach. The top row shows the estimated depth maps while the bottom row depicts their corresponding RMSE visualizations.

### 6. Fair-GPTQ: Bias-Aware Quantization for Large Language Models

**主要机构**: Université, École Centrale de Lyon LIRIS, UMR 5205, CNRS, Université Lumière Lyon
**作者数量**: 6人

**摘要**:
High memory demands of generative language models have drawn attention to quantization, which reduces computational cost, memory usage, and latency by mapping model weights to lower-precision integers. Approaches such as GPTQ effectively minimize input-weight product errors during quantization; however, recent empirical studies show that they can increase biased outputs and degrade performance on fairness benchmarks, and it remains unclear which specific weights cause this issue. In this work, we draw new links between quantization and model fairness by adding explicit group-fairness constraints to the quantization objective and introduce Fair-GPTQ, the first quantization method explicitly designed to reduce unfairness in large language models. The added constraints guide the learning of the rounding operation toward less-biased text generation for protected groups. Specifically, we focus on stereotype generation involving occupational bias and discriminatory language spanning gender, race, and religion. Fair-GPTQ has minimal impact on performance, preserving at least 90% of baseline accuracy on zero-shot benchmarks, reduces unfairness relative to a half-precision model, and retains the memory and speed benefits of 4-bit quantization. We also compare the performance of Fair-GPTQ with existing debiasing methods and find that it achieves performance on par with the iterative null-space projection debiasing approach on racial-stereotype benchmarks. Overall, the results validate our theoretical solution to the quantization problem with a group-bias term, highlight its applicability for reducing group bias at quantization time in generative models, and demonstrate that our approach can further be used to analyze channel-and weight-level contributions to fairness during quantization.

### 7. Forecasting and Visualizing Air Quality from Sky Images with Vision-Language Models

**主要机构**: Savannah College of Art and Design, Georgia State University
**作者数量**: 3人

**摘要**:
Air pollution remains a critical threat to public health and environmental sustainability, yet conventional monitoring systems are often constrained by limited spatial coverage and accessibility. This paper proposes an AI-driven agent that predicts ambient air pollution levels from sky images and synthesizes realistic visualizations of pollution scenarios using generative modeling. Our approach combines statistical texture analysis with supervised learning for pollution classification, and leverages vision-language model (VLM)-guided image generation to produce interpretable representations of air quality conditions. The generated visuals simulate varying degrees of pollution, offering a foundation for user-facing interfaces that improve transparency and support informed environmental decision-making. These outputs can be seamlessly integrated into intelligent applications aimed at enhancing situational awareness and encouraging behavioral responses based on real-time forecasts. We validate our method using a dataset of urban sky images and demonstrate its effectiveness in both pollution level estimation and semantically consistent visual synthesis. The system design further incorporates human-centered user experience principles to ensure accessibility, clarity, and public engagement in air quality forecasting. To support scalable and energyefficient deployment, future iterations will incorporate a green CNN architecture enhanced with FPGA-based incremental learning, enabling real-time inference on edge platforms.

### 8. FURINA: FREE FROM UNMERGEABLE ROUTER VIA LINEAR AGGREGATION OF MIXED EXPERTS

**主要机构**: Fudan University, Interactive Entertainment Group Tencent Inc, Shandong University, Northeastern University
**作者数量**: 6人

**摘要**:
The Mixture of Experts (MoE) paradigm has been successfully integrated into Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning (PEFT), delivering performance gains with minimal parameter overhead. However, a key limitation of existing MoE-LoRA methods is their reliance on a discrete router, which prevents the integration of the MoE components into the backbone model. This results in persistent computational overhead and increased system complexity during inference. To overcome this, we propose FURINA, a novel Free from Unmergeable Router framework based on the LINear Aggregation of experts. FURINA eliminates the router by introducing a Self-Routing mechanism. This is achieved through three core innovations: (1) decoupled learning of the direction and magnitude for LoRA adapters, (2) a shared learnable magnitude vector for consistent activation scaling, and (3) an expert selection loss that encourages divergent expert activation. The proposed mechanism leverages the angular similarity between the input and each adapter's directional component to activate experts, which are then scaled by the shared magnitude vector. This design allows the output norm to naturally reflect the importance of each expert, thereby enabling dynamic, router-free routing. The expert selection loss further sharpens this behavior by encouraging sparsity and aligning it with standard MoE activation patterns. We also introduce a shared expert within the MoE-LoRA block that provides stable, foundational knowledge. To the best of our knowledge, FURINA is the first router-free, MoE-enhanced LoRA method that can be fully merged into the backbone model, introducing zero additional inference-time cost or complexity. Extensive experiments demonstrate that FURINA not only significantly outperforms standard LoRA but also matches or surpasses the performance of existing MoE-LoRA methods, while eliminating the extra inference-time overhead of MoE. We plan to open-source the code upon publication.

### 9. HARNESS: Lightweight Distilled Arabic Speech Foundation Models

**主要机构**: Qatar Computing Research Institute, HBKU
**作者数量**: 2人

**摘要**:
Large pre-trained speech models excel in downstream tasks but their deployment is impractical for resource-limited environments. In this paper, we introduce HArnESS, the first Arabic-centric self-supervised speech model family, designed to capture Arabic speech nuances. Using iterative selfdistillation, we train large bilingual HArnESS (HL) SSL models and then distill knowledge into compressed student models (HS, HST), preserving Arabic-specific representations. We use low-rank approximation to further compact the teacher's discrete supervision into shallow, thin models. We evaluate HArnESS on Arabic ASR, Speaker Emotion Recognition (SER), and Dialect Identification (DID), demonstrating effectiveness against HuBERT and XLS-R. With minimal fine-tuning, HArnESS achieves SOTA or comparable performance, making it a lightweight yet powerful alternative for real-world use. We release our distilled models and findings to support responsible research and deployment in low-resource settings.

### 10. MEANFLOWSE: ONE-STEP GENERATIVE SPEECH ENHANCEMENT VIA CONDITIONAL MEAN FLOW

**主要机构**: Xiamen University, School of Electronic Science and Engineering, School of Informatics
**作者数量**: 6人

**摘要**:
Multistep inference is a bottleneck for real-time generative speech enhancement because flow and diffusion-based systems learn an instantaneous velocity field and therefore rely on iterative ordinary differential equation (ODE) solvers. We introduce MeanFlowSE, a conditional generative model that learns the average velocity over finite intervals along a trajectory. Using a Jacobian-vector product (JVP) to instantiate the MeanFlow identity, we derive a local training objective that directly supervises finite-interval displacement while remaining consistent with the instantaneous-field constraint on the diagonal. At inference, MeanFlowSE performs single-step generation via a backward-in-time displacement, removing the need for multistep solvers; an optional few-step variant offers additional refinement. On VoiceBank-DEMAND, the single-step model achieves strong intelligibility, fidelity, and perceptual quality with substantially lower computational cost than multistep baselines. The method requires no knowledge distillation or external teachers, providing an efficient, high-fidelity framework for real-time generative speech enhancement. The proposed method is open-sourced at https://github.com/liduojia1/MeanFlowSE.

### 11. Q-ROAR: Outlier-Aware Rescaling for RoPE Position Interpolation in Quantized Long-Context LLMs

**主要机构**: University of California
**作者数量**: 2人

**摘要**:
Extending LLM context windows is crucial for long range tasks. RoPE-based position interpolation (PI) methods like linear and frequency-aware scaling extend input lengths without retraining, while post-training quantization (PTQ) enables practical deployment. We show that combining PI with PTQ degrades accuracy due to coupled effects long context aliasing, dynamic range dilation, axis grid anisotropy, and outlier shifting that induce position-dependent logit noise. We provide the first systematic analysis of PI plus PTQ and introduce two diagnostics: Interpolation Pressure (per-band phase scaling sensitivity) and Tail Inflation Ratios (outlier shift from short to long contexts). To address this, we propose Q-ROAR, a RoPE-aware, weight-only stabilization that groups RoPE dimensions into a few frequency bands and performs a small search over per-band scales for WQ, WK , with an optional symmetric variant to preserve logit scale. The diagnostics guided search uses a tiny long-context dev set and requires no fine-tuning, kernel, or architecture changes. Empirically, Q-ROAR recovers up to 0.7% accuracy on standard tasks and reduces GovReport perplexity by more than 10%, while preserving short-context performance and compatibility with existing inference stacks.

### 12. SparseDoctor: Towards Efficient Chat Doctor with Mixture of Experts Enhanced Large Language Models

**主要机构**: Department of Computing, Hong Kong Chu Hai College, The Hong Kong Polytechnic University, Department of Computer Science
**作者数量**: 8人

**摘要**:
Large language models (LLMs) have achieved great success in medical question answering and clinical decision-making, promoting the efficiency and popularization of the personalized virtual doctor in society. However, the traditional fine-tuning strategies on LLM require the updates of billions of parameters, substantially increasing the training cost, including the training time and utility cost. To enhance the efficiency and effectiveness of the current medical LLMs and explore the boundary of the representation capability of the LLMs on the medical domain, apart from the traditional fine-tuning strategies from the data perspective (i.e., supervised fine-tuning or reinforcement learning from human feedback), we instead craft a novel sparse medical LLM named SparseDoctor armed with contrastive learning enhanced LoRA-MoE (low rank adaptation-mixture of experts) architecture. To this end, the crafted automatic routing mechanism can scientifically allocate the computational resources among different LoRA experts supervised by the contrastive learning. Additionally, we also introduce a novel expert memory queue mechanism to further boost the efficiency of the overall framework and prevent the memory overflow during training. We conduct comprehensive evaluations on three typical medical benchmarks: CMB, CMExam, and CMMLU-Med. Experimental results demonstrate that the proposed LLM can consistently outperform the strong baselines such as the HuatuoGPT series.

### 13. Under review as a conference paper TABLEDART: DYNAMIC ADAPTIVE MULTI-MODAL ROUTING FOR TABLE UNDERSTANDING

**主要机构**: University of Notre, The University of Queensland, Griffith University
**作者数量**: 7人

**摘要**:
Modeling semantic and structural information from tabular data remains a core challenge for effective table understanding. Existing Table-as-Text approaches flatten tables for large language models (LLMs), but lose crucial structural cues, while Table-as-Image methods preserve structure yet struggle with fine-grained semantics. Recent Table-as-Multimodality strategies attempt to combine textual and visual views, but they (1) statically process both modalities for every querytable pair within a large multimodal LLMs (MLLMs), inevitably introducing redundancy and even conflicts, and (2) depend on costly fine-tuning of MLLMs. In light of this, we propose TableDART, a training-efficient framework that integrates multimodal views by reusing pretrained single-modality models. TableDART introduces a lightweight 2.59M-parameter MLP gating network that dynamically selects the optimal path (either Text-only, Image-only, or Fusion) for each table-query pair, effectively reducing redundancy and conflicts from both modalities. In addition, we propose a novel agent to mediate cross-modal knowledge integration by analyzing outputs from text-and image-based models, either selecting the best result or synthesizing a new answer through reasoning. This design avoids the prohibitive costs of full MLLM fine-tuning. Extensive experiments on seven benchmarks show that TableDART establishes new state-of-the-art performance among open-source models, surpassing the strongest baseline by an average of 4.02%. The code is available at: https://anonymous.4open.science/r/TableDART-C52B.

### 14. Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization

**主要机构**: 
**作者数量**: 3人

**摘要**:
Recent advances in large language models (LLMs) demonstrate their effectiveness in scaling test-time compute for software engineering tasks. However, these approaches often focus on high-level solutions, with limited attention to optimizing low-level CUDA kernel implementations. Additionally, existing kernel generation benchmarks suffer from exploitable loopholes and insufficient diversity in testing conditions, hindering true generalization assessment. To address these limitations, we introduce robust-kbench, a new benchmark for rigorous evaluation of kernel performance and correctness across varied scenarios. Furthermore, we present a comprehensive agentic framework that automates CUDA kernel discovery, verification, and optimization. This pipeline enables frontier LLMs to translate torch code to CUDA kernels and iteratively improve their runtime within our robust evaluation setting. Our sequential workflow first translates PyTorch code into equivalent CUDA kernels. It then optimizes their runtime using a novel evolutionary metageneration procedure tailored to the CUDA ecosystem, guided by LLM-based verifiers for correctness and efficient filtering. Evaluated on robust-kbench, our approach produces CUDA kernels outperforming torch implementations for practical applications, including forward and backward passes. It can fuse operations and deploy various runtime optimization strategies. The verifier workflow accurately classifies incorrect kernels, enhancing hardware verification efficiency.
