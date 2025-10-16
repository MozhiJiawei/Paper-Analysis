# AI推理加速技术论文分析报告
生成时间: 2025-10-16 13:12:25
分析论文数量: 25篇

## 论文技术简报

### 1. Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM

University of California发布了Activation-Informed Pareto-Guided Low-Rank Compression论文，使用Pareto-Guided Singular Value Decomposition (PGSVD)技术，解决了LLM/VLM部署中的内存和计算挑战，达成了相同压缩水平下更高准确性和推理加速的效果

### 2. AMAQ: Adaptive Mixed-bit Activation Quantization for Collaborative Parameter Efficient Fine-tuning

UC Irvine发布了AMAQ: Adaptive Mixed-bit Activation Quantization for Collaborative Parameter Efficient Fine-tuning论文，使用Adaptive Mixed-bit Activation Quantization (AMAQ) 与 Parameter-efficient Split Learning技术，解决了大型语言模型协作式服务器-客户端分布式训练中的通信效率和计算开销问题，达成了有效平衡低资源设备上协作训练效率和性能的效果

### 3. Barbarians at the Gate: How AI is Upending Systems Research

UC Berkeley发布了Barbarians at the Gate: How AI is Upending Systems Research论文，使用AI-Driven Research for Systems（ADRS）方法，通过生成-验证-优化解决方案并利用基于真实系统/模拟器的可靠性能验证器，解决了系统研究中性能导向算法的设计问题，达成了发现优于人类设计的算法（如5.0×运行时提升或26%成本降低）的效果。

### 4. CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credits

上海交通大学、浙江大学发布了CreditDecoding论文，使用Trace Credit概念及CreditDecoding算法，解决了扩散大语言模型并行解码中因重复掩蔽导致的冗余迭代问题，达成了较LLaDA-8B-Ins 5.48×加速和0.48性能提升、较LLaDA-MoE-Ins 4.11×加速和0.15性能提升的效果

### 5. Discretized Quadratic Integrate-and-Fire Neuron Model for Deep Spiking Neural Networks

University of Novi Sad发布了Discretized Quadratic Integrate-and-Fire Neuron Model for Deep Spiking Neural Networks论文，使用首个针对高性能深度脉冲神经网络的离散化二次整合发放（QIF）神经元模型，解决了LIF神经元表达能力不足及复杂神经元模型训练不稳定的问题，达成了性能优于最先进基于LIF方法的效果

### 6. H1IB-KV: Hybrid One-Bit Caches for Memory-Efficient Large Language Model Inference

Rutgers University发布了H1IB-KV论文，使用键向量1位二进制草图+值向量4位量化的混合1位KV缓存技术，解决了大型语言模型长上下文推理时KV缓存内存受限问题，达成70亿参数模型处理8k token上下文缓存内存减少70倍（至60MB以下）且轻量微调后匹配全精度性能的效果。

### 7. IMPROVING CHAIN-OF-THOUGHT EFFICIENCY FOR AUTOREGRESSIVE IMAGE GENERATION

Meta Superintelligence Labs和Cornell University发布了IMPROVING CHAIN-OF-THOUGHT EFFICIENCY FOR AUTOREGRESSIVE IMAGE GENERATION论文，使用改进链式思维效率的技术，解决了自回归图像生成中链式思维效率低的问题，达成了提升自回归图像生成效率的效果。

### 8. InstaGeo: Compute-Efficient Geospatial Machine Learning from Data to Deployment

InstaDeep AI发布了InstaGeo论文，使用端到端地理空间机器学习框架（结合自动化数据处理与任务特定模型蒸馏），解决了地理空间基础模型缺乏专用数据管道及微调模型尺寸过大的问题，达成模型缩小8倍、减少碳排放、作物分割mIoU提升12个百分点至60.65%且数据到部署周期缩短至一天内的效果。

### 9. LANTERN: Scalable Distillation of Large Language Models for Job-Person Fit and Explanation

LinkedIn发布了LANTERN论文，使用针对人岗匹配的LLM多级知识蒸馏框架（含多目标建模的编码器解码器），解决了LLM在该任务中输出质量低、推理延迟及扩展性差的问题，达成提升人岗匹配和解释任务指标，申请率+0.24%、合格申请+0.28%的效果

### 10. LATENT SPEECH-TEXT TRANSFORMER

Johns Hopkins University和Meta Superintelligence Labs发布了Latent Speech-Text Transformer（LST）论文，使用动态聚合语音令牌为潜在语音补丁的技术，解决了语音令牌过长导致的模态计算不平衡、对齐困难及缩放慢问题，在语音转语音和文本转文本基准上优于传统方法，HellaSwag任务中语音准确率计算控制训练提升6.5%、数据控制训练提升5.3%并改善文本性能。

### 11. LIGHTCACHE: MEMORY-EFFICIENT, TRAINING-FREE ACCELERATION FOR VIDEO GENERATION

Microsoft Research发布了LIGHTCACHE论文，使用异步缓存交换、特征分块和潜在变量切片解码的阶段特定策略，解决了视频生成中缓存加速导致的内存激增问题，达成了更快推理速度、更低内存使用且质量下降在可接受范围的效果。

### 12. MIXTURE OF NEURON EXPERTS

清华大学深圳国际研究生院、上海交通大学发布了《MIXTURE OF NEURON EXPERTS》论文，使用神经元粒度专家选择（通过每个专家内top-k选择）技术，解决了MoE模型参数利用率低、推理效率不高的问题，达成了激活50% MoE层参数时与传统MoE性能相当，参数数量相等时始终优于传统MoE，提升参数利用率和推理效率的效果

### 13. OptiFLIDS: Optimized Federated Learning for Energy-Efficient Intrusion Detection in IoT

Ibn Tofail University发布了OptiFLIDS论文，使用本地训练剪枝技术与定制化聚合方法，解决了联邦学习在IoT中的数据异质性及高能耗计算成本问题，达成了保持强检测性能同时提升能源效率的效果。

### 14. Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

哥伦比亚大学、印第安纳大学发布了《Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting》论文，使用数据移动预测技术，解决了MoE架构大型语言模型在多单元服务系统中因随机专家选择导致的数据移动开销过大问题，达成了在DeepSeek V3和Qwen3上分别实现6.3倍和4.0倍平均加速的效果。

### 15. PATTERNKV: FLATTENING KV REPRESENTATION EX-PANDS QUANTIZATION HEADROOM

北京理工大学、小红书公司发布了PatternKV论文，使用模式对齐残差量化方案，解决了自回归大模型KV缓存的内存带宽瓶颈及量化时分布不平坦导致的精度下降问题，达成2-bit增益、4-bit精度下降仅0.08%、测试时扩展精度平均提升10%的效果。

### 16. Rasterized Steered Mixture of Experts for Efficient 2D Image Regression

研究团队发布了《Rasterized Steered Mixture of Experts for Efficient 2D Image Regression》论文，使用结合光栅化高斯核渲染与边缘感知门控机制的光栅化优化策略，解决了Steered Mixture of Experts回归框架计算成本高的问题，达成了显著加快参数更新、提升内存效率并支持原生超分辨率和图像去噪的效果。

### 17. Shaken or Stirred? An Analysis of MetaFormer's Token Mixing for Medical Imaging

Ulm University与University of Lübeck发布了MetaFormer的token mixing在医学影像中的分析论文，通过系统分析池化、卷积、注意力基token mixers在医学影像分类与分割任务中的应用，解决了MetaFormer在医学影像中token mixers研究稀缺且缺乏比较的问题，结果显示分类任务低复杂度token mixers足够且预训练权重有用，分割任务卷积token mixers的局部归纳偏置关键，分组卷积为优选并减少运行时和参数。

### 18. SYN-DIAG: AN LLM-BASED SYNERGISTIC FRAMEWORK FOR GENERALIZABLE FEW-SHOT FAULT DIAGNOSIS ON THE EDGE

北京航空航天大学发布了SYN-DIAG论文，使用基于LLM的云边协同框架（含视觉-语义协同、内容感知推理及知识蒸馏轻量化），解决了工业故障诊断中数据稀缺及资源受限环境下大型AI模型部署困难的问题，达成了1-shot和跨工况诊断性能优于现有方法、边缘模型大小减少83%且延迟降低50%（性能接近云端）的效果

### 19. TFM Dataset: A Novel Multi-task Dataset and Integrated Pipeline for Automated Tear Film Break-Up Segmentation

研究团队发布了TFM Dataset相关论文，使用首个多任务泪膜分析数据集TFM Dataset、基于MobileOnemini骨干和重参数化技术的TF-Net模型及集成实时 pipeline TF-Collab，解决了自动泪膜破裂（TFBU）分割因缺乏标注数据集和集成解决方案的挑战，达成了准确性与计算效率的良好平衡，建立了TFM分割子集基准性能。

### 20. The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models

研究团队发布了《The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models》论文，使用知识蒸馏技术研究数据量对小模型代码推理能力的影响，解决了蒸馏数据量与小模型代码推理性能关系不明确的问题，发现了“代码推理谷”现象，即性能随数据量增加先降后升并呈超对数线性增长。

### 21. TINY BUT MIGHTY: A SOFTWARE-HARDWARE CO-DESIGN APPROACH FOR EFFICIENT MULTIMODAL IN-FERENCE ON BATTERY-POWERED SMALL DEVICES

威斯康星大学麦迪逊分校发布了TINY BUT MIGHTY论文，使用软硬件协同设计推理框架NANOMIND（将大模型拆分为模块化组件并映射到理想加速器，结合模块级动态卸载和低比特计算内核），解决了大型多模态模型在电池供电小型设备上因整体式运行未充分利用异构加速器导致的高延迟和资源效率低问题，达成能源消耗降低42.3%、GPU内存使用减少11.2%，并实现设备运行LlaVA-OneVision近半天和LLaMA-3-8B语音交互近20.8小时的效果

### 22. UNTANGLING COMPONENT IMBALANCE IN HYBRID LINEAR ATTENTION CONVERSION METHODS

伦敦大学学院与诺亚方舟实验室发布了关于混合线性注意力转换组件不平衡的论文，使用混合线性转换与SWA、Hedge-CATs及Scheduled Sliding-window Dropout技术，解决了混合方法中线性组件被绕过、过度依赖滑动窗口softmax的问题，达成了保持计算效率同时恢复大部分基础模型性能并确保线性注意力真正采用的效果。

### 23. VATTENTION: VERIFIED SPARSE ATTENTION

加州大学伯克利分校发布了VATTENTION: VERIFIED SPARSE ATTENTION论文，使用统一top-k和采样并提供(ϵ, δ)近似精度保证的稀疏注意力机制vAttention，解决了现有稀疏注意力缺乏近似质量保证导致实际部署受限的问题，显著提升稀疏注意力质量并在20倍稀疏度下匹配全模型质量。

### 24. VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization

中国科学院、小米公司发布了VecInfer论文，使用异常值抑制向量量化（通过平滑和Hadamard变换抑制key缓存异常值及优化CUDA核融合计算与反量化）技术，解决了LLM推理中KV缓存内存开销大且现有VQ方法在超低比特下性能下降的问题，达成了2比特量化性能接近全精度，Llama-3.1-8B上196k序列长度时大批次自注意力计算加速2.7倍、单批次端到端延迟减少8.3倍的效果

### 25. VER: VISION EXPERT TRANSFORMER FOR ROBOT LEARNING VIA FOUNDATION DISTILLATION AND DYNAMIC ROUTING

北京大学与UNC-Chapel Hill发布了VER论文，使用视觉专家Transformer通过基础模型蒸馏构建视觉专家库并动态路由轻量级网络，解决了单个视觉基础模型跨任务通用性不足及蒸馏整合时特征选择不灵活、需全量重训练的问题，达成了17个机器人任务上的SOTA性能，提升了跨任务通用性与参数效率。

## 论文详细信息

### 1. Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM

**主要机构**: University of California
**作者数量**: 7人

**摘要**:
Large language models (LLM) and visionlanguage models (VLM) have achieved stateof-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.

### 2. AMAQ: Adaptive Mixed-bit Activation Quantization for Collaborative Parameter Efficient Fine-tuning

**主要机构**: UC Irvine, VMware Research
**作者数量**: 6人

**摘要**:
Large Language Models (LLMs) are scaling rapidly, creating significant challenges for collaborative server-client distributed training, particularly in terms of communication efficiency and computational overheads. To address these challenges, we implement Parameter-efficient Split Learning, which effectively balances efficiency and performance for collaborative training on low-resource devices.

### 3. Barbarians at the Gate: How AI is Upending Systems Research

**主要机构**: UC Berkeley
**作者数量**: 17人

**摘要**:
Artificial Intelligence (AI) is starting to transform the research process as we know it by automating the discovery of new solutions. Given a task, the typical AIdriven approach is (i) to generate a set of diverse solutions, and then (ii) to verify these solutions and select one that solves the problem. Crucially, this approach assumes the existence of a reliable verifier, i.e., one that can accurately determine whether a solution solves the given problem. We argue that systems research, long focused on designing and evaluating new performance-oriented algorithms, is particularly well-suited for AI-driven solution discovery. This is because system performance problems naturally admit reliable verifiers: solutions are typically implemented in real systems or simulators, and verification reduces to running these software artifacts against predefined workloads and measuring performance. We term this approach as AI-Driven Research for Systems (ADRS), which iteratively generates, evaluates, and refines solutions. Using OpenEvolve, an existing opensource ADRS instance, we present case studies across diverse domains, including multi-region cloud scheduling, load balancing for Mixture-of-Experts inference, LLM-based SQL queries, and transaction scheduling. In multiple instances, ADRS discovers algorithms that outperform state-of-the-art human designs (e.g., achieving up to 5.0× runtime improvements or 26% cost reductions). We distill best practices for guiding algorithm evolution, from prompt design to evaluator construction, for existing frameworks. We then discuss the broader implications for the systems community: as AI assumes a central role in algorithm design, we argue that human researchers will increasingly focus on problem formulation and strategic guidance. Our results highlight both the disruptive potential and the urgent need to adapt systems research practices in the age of AI.

### 4. CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credits

**主要机构**: Westlake University, Shanghai Jiao, Zhejiang University, Tong University
**作者数量**: 9人

**摘要**:
Diffusion large language models (dLLMs) generate text through iterative denoising steps, achieving parallel decoding by denoising only high-confidence positions at each step. However, existing approaches often repetitively remask tokens due to initially low confidence scores, leading to redundant iterations and limiting overall acceleration. Through the analysis of dLLM decoding traces, we observe that the model often determines the final prediction for a token several steps before the decoding step. To leverage this historical information and avoid redundant steps, we introduce the concept of Trace Credit, which quantifies each token's convergence potential by accumulating historical logits. Furthermore, we propose CreditDecoding, a training-free parallel decoding algorithm that accelerates the confidence convergence of correct but underconfident tokens by fusing current logits with Trace Credit. This process significantly reduces redundant iterations and enhances decoding robustness. On eight benchmarks, CreditDecoding achieves a 5.48× speedup and a 0.48 performance improvement over LLaDA-8B-Ins, and a 4.11× speedup with a 0.15 performance improvement over LLaDA-MoE-Ins. Importantly, CreditDecoding scales effectively to long sequences and is orthogonal to mainstream inference optimizations, making it a readily integrable and versatile solution.

### 5. Discretized Quadratic Integrate-and-Fire Neuron Model for Deep Spiking Neural Networks

**主要机构**: STAM Center, University of Novi Sad Novi Sad, Faculty of Technical Sciences, Center for Advanced Studies and Systems of Recife, Arizona State University
**作者数量**: 7人

**摘要**:
Spiking Neural Networks (SNNs) have emerged as energy-efficient alternatives to traditional artificial neural networks, leveraging asynchronous and biologically inspired neuron dynamics. Among existing neuron models, the Leaky Integrate-and-Fire (LIF) neuron has become widely adopted in deep SNNs due to its simplicity and computational efficiency. However, this efficiency comes at the expense of expressiveness, as LIF dynamics are constrained to linear decay at each timestep. In contrast, more complex models, such as the Quadratic Integrate-and-Fire (QIF) neuron, exhibit richer, nonlinear dynamics but have seen limited adoption due to their training instability. On that note, we propose the first discretization of the QIF neuron model tailored for high-performance deep spiking neural networks and provide an in-depth analysis of its dynamics. To ensure training stability, we derive an analytical formulation for surrogate gradient windows directly from our discretizations' parameter set, minimizing gradient mismatch. We evaluate our method on CIFAR-10, CIFAR-100, ImageNet, and CIFAR-10 DVS, demonstrating its ability to outperform state-of-the-art LIF-based methods. These results establish our discretization of the QIF neuron as a compelling alternative to LIF neurons for deep SNNs, combining richer dynamics with practical scalability.

### 6. H1IB-KV: Hybrid One-Bit Caches for Memory-Efficient Large Language Model Inference

**主要机构**: Rutgers University New Brunswick, Department of Computer Science
**作者数量**: 1人

**摘要**:
Autoregressive decoding in large language models (LLMs) requires caching a growing list of past key-value (KV) pairs, making long-context inference a memory-bound problem. While recent methods have explored quantizing the cache, evicting tokens, or using binary sketches for keys (e.g., Loki), these approaches often provide an incomplete solution by leaving one component (like values) uncompressed or by discarding context information. This paper introduces the Hybrid One-Bit KV Cache (H1B-KV), a comprehensive compression scheme that radically reduces memory usage without sacrificing context. H1B-KV represents each key vector using a 1-bit binary sketch, enabling hardware-friendly bitwise attention, and further compresses value vectors using 4-bit quantization. This holistic, hybrid approach allows a 7-billion parameter LLM to handle an 8k-token context with under 60 MB of cache memory-a 70x reduction. We demonstrate that after a lightweight finetuning, H1B-KV matches full-precision performance not only on perplexity benchmarks but also on complex downstream tasks like mathematical reasoning (GSM8K), multi-task understanding (MMLU), and code generation (HumanEval). Our results show H1B-KV significantly outperforms leading quantization (KIVD, token eviction (SparseLLM), and key-only sketching (Loki) methods in quality-per-byte, establishing it as a robust solution for deploying LLMs in memory-constrained environments.

### 7. IMPROVING CHAIN-OF-THOUGHT EFFICIENCY FOR AUTOREGRESSIVE IMAGE GENERATION

**主要机构**: Stony Brook University, Meta Superintelligence Labs, Cornell University
**作者数量**: 15人

**摘要**:


### 8. InstaGeo: Compute-Efficient Geospatial Machine Learning from Data to Deployment

**主要机构**: 
**作者数量**: 6人

**摘要**:
multispectral imagery from Landsat 8-9, Sentinel-2 and similar missions has spurred the rise of geospatial foundation models (GFMs) fine-tuned for various downstream humanitarian and societal tasks. However, the deployment of these models faces two major obstacles: (i) the lack of dedicated geospatial data pipelines and (ii) the prohibitive size of fine-tuned models. These challenges arise because published GFMs currently do not include pipelines for generating input data from raw satellite imagery, and models derived for each downstream task retain the architecture and complexity of the pre-trained GFM encoder. To address these challenges, we introduce InstaGeo, an open-source, endto-end geospatial machine learning framework that combines automated data curation for converting raw satellite imagery into a model-ready format; a model component with task-specific distillation for transforming large GFMs into compute-efficient models whose complexity matches task difficulty; and a component for deploying models as interactive web-map application for operational use. Using InstaGeo, we faithfully reproduced the datasets that underlie three published studies from scratch and trained models with performance differences of-0.73 percentage point (pp) mean intersection-over-union (mIoU) for flood mapping,-0.20 pp mIoU for multitemporal crop segmentation, and +1.79 pp mIoU for desert locust breeding ground prediction. Our task-specific distilled models are compute efficient and up to 8× smaller than those obtained via standard fine-tuning, significantly reducing inference FLOPs and thereby CO 2 emissions with minimal performance degradation. Due to the ease of use of InstaGeo's data pipeline, we curated a larger crop segmentation dataset, achieving a new state-of-the-art mIoU of 60.65 %, an improvement of 12 pp over the previous baseline. Finally, we showcase how InstaGeo can significantly accelerate the data-to-deployment cycle, allowing a user to move from data preparation to model deployment within a single working day. By unifying data preparation, model compression, and visual analysis within a single open-source framework, InstaGeo transforms researchgrade GFMs into practical, low-carbon tools suitable for real-time, large-scale environmental monitoring. This unified approach has the potential to shift Earth Observation (EO) research from model-centric competition toward data-quality and application-centric innovation. We release the source code of InstaGeo, along with datasets, model checkpoints, and bash scripts used for data curation and model training, at https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git.

### 9. LANTERN: Scalable Distillation of Large Language Models for Job-Person Fit and Explanation

**主要机构**: LANTERN: Scalable Distillation of Large Language Models for Job-Person Fit and Explanation. In . ACM, LinkedIn
**作者数量**: 22人

**摘要**:
Large language models (LLMs) have achieved strong performance across a wide range of natural language processing tasks. However, deploying LLMs at scale for domain-specific applications-such as job-person fit and explanation in job-seeking platforms-introduces distinct challenges. At LinkedIn, the job-person fit task requires analyzing a candidate's public profile against job requirements to produce both a fit assessment and a detailed explanation. Directly applying open-source or finetuned LLMs to this task often fails to yield high-quality, actionable feedback due to the complexity of the domain and the need for structured outputs. Moreover, the large size of these models leads to high inference latency and limits scalability, making them unsuitable for online use. To address these challenges, we introduce LANTERN, a novel LLM knowledge distillation framework tailored specifically for job-person fit tasks. LANTERN involves modeling over multiple objectives, an encoder model for classification purpose, and a decoder model for explanation purpose. To better distill the knowledge from a strong blackbox teacher model to multiple downstream models, LANTERN incorporates multi-level knowledge distillation that integrates both data and logit-level insights. In addition to introducing the knowledge distillation framework, we share our insights on post-training techniques and prompt engineering, both of which are crucial for successfully adapting LLMs to domain-specific downstream tasks. Extensive experimental results demonstrate that LANTERN significantly improves task-specific metrics for both job-person fit and explanation. Online evaluations further confirm its effectiveness, showing measurable gains in job seeker engagement, including +0.24% increase in apply rate and a +0.28% increase in qualified applications. CCS Concepts • Computing methodologies → Natural language processing.

### 10. LATENT SPEECH-TEXT TRANSFORMER

**主要机构**: Johns Hopkins University, Center for Language and Speech Processing, Meta Superintelligence Labs
**作者数量**: 11人

**摘要**:
Auto-regressive speech-text models are typically pre-trained on a large number of interleaved sequences of text tokens and raw speech encoded as speech tokens using vector quantization. These models have demonstrated state-of-the-art performance in speech-to-speech understanding and generation benchmarks, together with promising scaling laws, primarily enabled by the representational alignment between text and speech. Nevertheless, they suffer from shortcomings, partly owing to the disproportionately longer sequences of speech tokens in contrast to textual tokens. This results in a large compute imbalance between modalities during pre-training as well as during inference, and a potential hindrance to effectively aligning speech and text, ultimately translating to several orders of magnitude slower scaling laws. We introduce the Latent Speech-Text Transformer (LST), which makes pre-training speech-text models more data-efficient by dynamically and inexpensively aggregating speech tokens into latent speech patches. These patches serve as higher-level units that can either align with corresponding textual units to aid capability transfer or even encapsulate common speech sequences like silences to be more compute-efficient. We show that LST outperforms vanilla approaches on speech-to-speech as well as text-to-text benchmarks in both data-and compute-controlled settings, the former indicating more effective representational alignment and the latter indicating steeper scaling laws for speech-text models. On HellaSwag story completion, LST achieves 6.5% absolute gain in speech accuracy under compute-controlled training and 5.3% under data-controlled training, while also improving text performance. We will release our models, code, and the evaluation data to facilitate further research.

### 11. LIGHTCACHE: MEMORY-EFFICIENT, TRAINING-FREE ACCELERATION FOR VIDEO GENERATION

**主要机构**: Microsoft Research, University of Tulsa, Northeastern University, Clemson University, The University of Arizona
**作者数量**: 8人

**摘要**:
Training-free acceleration has emerged as an advanced research area in video generation based on diffusion models. The redundancy of latents in diffusion model inference provides a natural entry point for acceleration. In this paper, we decompose the inference process into the encoding, denoising, and decoding stages, and observe that cache-based acceleration methods often lead to substantial memory surges in the latter two stages. To address this problem, we analyze the characteristics of inference across different stages and propose stagespecific strategies for reducing memory consumption: 1) Asynchronous Cache Swapping. 2) Feature chunk. 3) Slicing latents to decode. At the same time, we ensure that the time overhead introduced by these three strategies remains lower than the acceleration gains themselves. Compared with the baseline, our approach achieves faster inference speed and lower memory usage, while maintaining quality degradation within an acceptable range. The Code is available at https://github.com/NKUShaw/LightCache.

### 12. MIXTURE OF NEURON EXPERTS

**主要机构**: Xiamen University, Tsinghua Shenzhen International Graduate School, Shanghai Jiao Tong University, School of Informatics, Tsinghua University
**作者数量**: 9人

**摘要**:
In this work, We first explore whether the parameters activated by the MoE layer remain highly sparse at inference. We perform a sparsification study on several representative MoE models. For each expert, we rank parameters by the magnitude of their activations from the gate projection and progressively prune the activated subset. Pruning up to 60% of parameters within that subset causes only negligible task-performance degradation; substantial drops occur only after more than 90% are removed. We further decompose experts into neuron granular MoE and visualize their activation values, finding that most neuron activations are near zero. This observation motivates us to select only high-activation neuron experts during pretraining. Based on this insight, we propose Mixture of Neuron Experts (MoNE). MoNE achieve neuron granular expert select by only applying a simple top-k selection within each expert, incurs negligible latency, and requires no additional routing parameters or inter-expert communication. Extensive experiments demonstrate that MoNE matches traditional MoE performance while activating only 50% of the MoE-layer parameters, and it consistently outperforms traditional MoE when compared at equal numbers of activated parameters. These results suggest that MoNE is a practical approach to improving parameter utilization and inference efficiency in MoE-like models.

### 13. OptiFLIDS: Optimized Federated Learning for Energy-Efficient Intrusion Detection in IoT

**主要机构**: Ibn Tofail University, Faculty of Sciences, Laboratory of Research in Informatics (LaRI), College of Computing, University Mohammed VI Polytechnic
**作者数量**: 3人

**摘要**:
In critical IoT environments, such as smart homes and industrial systems, effective Intrusion Detection Systems (IDS) are essential for ensuring security. However, developing robust IDS solutions remains a significant challenge. Traditional machine learning-based IDS models typically require large datasets, but data sharing is often limited due to privacy and security concerns. Federated Learning (FL) presents a promising alternative by enabling collaborative model training without sharing raw data. Despite its advantages, FL still faces key challenges, such as data heterogeneity (non-IID data) and high energy and computation costs, particularly for resourceconstrained IoT devices. To address these issues, this paper proposes OptiFLIDS, a novel approach that applies pruning techniques during local training to reduce model complexity and energy consumption. It also incorporates a customized aggregation method to better handle pruned models that differ due to non-IID data distributions. Experiments conducted on three recent IoT IDS datasets, TON_IoT, X-IIoTID, and IDS-IoT2024, demonstrate that OptiFLIDS maintains strong detection performance while improving energy efficiency, making it wellsuited for deployment in real-world IoT environments.

### 14. Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

**主要机构**: Indiana University, Columbia University
**作者数量**: 7人

**摘要**:
Large Language Models (LLMs) with Mixture of Experts (MoE) architectures achieve remarkable performance improvements, but their random expert selection mechanism introduces significant data movement overhead that becomes the dominant bottleneck in multi-unit serving systems. To forecast the patterns underlying this data movement, we conduct comprehensive data-movement-centric profiling across three state-of-the-art large-scale MoE models (200B-671B) using over 24,000 requests spanning diverse workloads. With the resulting 150GB+ trace files, we perform systematic analysis from both temporal and spatial perspectives and distill six key insights to guide the design of diverse future serving systems. Taking wafer-scale GPUs as a case study, we demonstrate that minor architectural modifications leveraging our insights achieve substantial performance gains, delivering 6.3× and 4.0× average speedups on DeepSeek V3 and Qwen3, respectively. Our work provides the first comprehensive data-centric analysis of MoE models at scale. Our profiling traces and analysis results are publicly available at {https://huggingface.co/datasets/core12345/MoE_expert_-selection_trace. We will also release our simulation framework shortly to facilitate future research in this area.

### 15. PATTERNKV: FLATTENING KV REPRESENTATION EX-PANDS QUANTIZATION HEADROOM

**主要机构**: Beijing Institute of Technology, Xiaohongshu Inc, School of Computer Science
**作者数量**: 11人

**摘要**:
KV cache in autoregressive LLMs eliminates redundant recomputation but has emerged as the dominant memory and bandwidth bottleneck during inference, notably with long contexts and test-time scaling. KV quantization is a key lever for reducing cache cost, but accuracy drops sharply as the native KV distribution lacks flatness and thus maintains a wide quantization range. Prior work focuses on isolating outliers, which caps their error but fails to flatten the overall distribution, leaving performance fragile under low-bit settings. In this work, we show that the K cache maintains a stable structure that evolves gradually with context, while the V cache carries latent semantic regularities. Building on these insights, we propose PatternKV, a pattern-aligned residual quantization scheme. It mines representative pattern vectors online, aligns each KV vector to its nearest pattern, and quantizes only the residual. This reshaping of the KV distribution flattens the quantization target and narrows its range, thereby improving the fidelity of low-bit KV quantization. Across long-context and test-time scaling settings on multiple backbones, PatternKV delivers consistent 2-bit gains, with a 0.08% average 4-bit drop relative to FP16, improves test-time scaling accuracy by 10% on average, and raises throughput by 1.4× while supporting 1.25× larger batches.

### 16. Rasterized Steered Mixture of Experts for Efficient 2D Image Regression

**主要机构**: 
**作者数量**: 3人

**摘要**:
The Steered Mixture of Experts regression framework has demonstrated strong performance in image reconstruction, compression, denoising, and super-resolution. However, its high computational cost limits practical applications. This work introduces a rasterization-based optimization strategy that combines the efficiency of rasterized Gaussian kernel rendering with the edge-aware gating mechanism of the Steered Mixture of Experts. The proposed method is designed to accelerate two-dimensional image regression while maintaining the model's inherent sparsity and reconstruction quality. By replacing global iterative optimization with a rasterized formulation, the method achieves significantly faster parameter updates and more memory-efficient model representations. In addition, the proposed framework supports applications such as native super-resolution and image denoising, which are not directly achievable with standard rasterized Gaussian kernel approaches. The combination of fast rasterized optimization with the edgeaware structure of the Steered Mixture of Experts provides a new balance between computational efficiency and reconstruction fidelity for two-dimensional image processing tasks.

### 17. Shaken or Stirred? An Analysis of MetaFormer's Token Mixing for Medical Imaging

**主要机构**: Ulm University, University of Lübeck, Medical Informatics, Medical Systems Biology
**作者数量**: 3人

**摘要**:
The generalization of the Transformer architecture via MetaFormer has reshaped our understanding of its success in computer vision. By replacing self-attention with simpler token mixers, MetaFormer provides strong baselines for vision tasks. However, while extensively studied on natural image datasets, its use in medical imaging remains scarce, and existing works rarely compare different token mixers, potentially overlooking more suitable designs choices. In this work, we present the first comprehensive study of token mixers for medical imaging. We systematically analyze pooling-, convolution-, and attention-based token mixers within the MetaFormer architecture on image classification (global prediction task) and semantic segmentation (dense prediction task). Our evaluation spans eight datasets covering diverse modalities and common challenges in the medical domain. Given the prevalence of pretraining from natural images to mitigate medical data scarcity, we also examine transferring pretrained weights to new token mixers. Our results show that, for classification, low-complexity token mixers (e.g. grouped convolution or pooling) are sufficient, aligning with findings on natural images. Pretrained weights remain useful despite the domain gap introduced by the new token mixer. For segmentation, we find that the local inductive bias of convolutional token mixers is essential. Grouped convolutions emerge as the preferred choice, as they reduce runtime and parameter count compared to standard convolutions, while the MetaFormer's channel-MLPs already provide the necessary cross-channel interactions. Our code is available on GitHub.

### 18. SYN-DIAG: AN LLM-BASED SYNERGISTIC FRAMEWORK FOR GENERALIZABLE FEW-SHOT FAULT DIAGNOSIS ON THE EDGE

**主要机构**: Beihang University, Hangzhou International Innovation Institute of Beihang University
**作者数量**: 3人

**摘要**:
Industrial fault diagnosis faces the dual challenges of data scarcity and the difficulty of deploying large AI models in resource-constrained environments. This paper introduces Syn-Diag, a novel cloudedge synergistic framework that leverages Large Language Models to overcome these limitations in few-shot fault diagnosis. Syn-Diag is built on a three-tiered mechanism: 1) Visual-Semantic Synergy, which aligns signal features with the LLM's semantic space through cross-modal pre-training; 2) Content-Aware Reasoning, which dynamically constructs contextual prompts to enhance diagnostic accuracy with limited samples; and 3) Cloud-Edge Synergy, which uses knowledge distillation to create a lightweight, efficient edge model capable of online updates via a shared decision space. Extensive experiments on six datasets covering different CWRU and SEU working conditions show that Syn-Diag significantly outperforms existing methods, especially in 1-shot and cross-condition scenarios. The edge model achieves performance comparable to the cloud version while reducing model size by 83% and latency by 50%, offering a practical, robust, and deployable paradigm for modern intelligent diagnostics.

### 19. TFM Dataset: A Novel Multi-task Dataset and Integrated Pipeline for Automated Tear Film Break-Up Segmentation

**主要机构**: 
**作者数量**: 7人

**摘要**:
Tear film break-up (TFBU) analysis is critical for diagnosing dry eye syndrome, but automated TFBU segmentation remains challenging due to the lack of annotated datasets and integrated solutions. This paper introduces the Tear Film Multitask (TFM) Dataset, the first comprehensive dataset for multitask tear film analysis, comprising 15 high-resolution videos (totaling 6,247 frames) annotated with three vision tasks: framelevel classification ('clear', 'closed', 'broken', 'blur'), Placido Ring detection, and pixel-wise TFBU area segmentation. Leveraging this dataset, we first propose TF-Net, a novel and efficient baseline segmentation model. TF-Net incorporates a MobileOnemini backbone with re-parameterization techniques and an enhanced feature pyramid network to achieve a favorable balance between accuracy and computational efficiency for real-time clinical applications. We further establish benchmark performance on the TFM segmentation subset by comparing TF-Net against several state-of-the-art medical image segmentation models. Furthermore, we design TF-Collab, a novel integrated real-time pipeline that synergistically leverages models trained on all three tasks of the TFM dataset. By sequentially orchestrating frame classification for BUT determination, pupil region localization for input standardization, and TFBU segmentation, TF-Collab fully automates the analysis. Experimental results demonstrate the effectiveness of the proposed TF-Net and TF-Collab, providing a foundation for future research in ocular surface diagnostics. Our code and the TFM datasets are available at https://github.com/glory-wan/TF-Net

### 20. The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models

**主要机构**: 
**作者数量**: 5人

**摘要**:
Distilling the thinking traces of a Large Language Model (LLM) with reasoning capabilities into a smaller model has been proven effective. Yet, there is a scarcity of work done on how model performances scale with the quantity of distillation data. In this work, we study the scaling trend of distilling competitive coding skills on two small non-reasoning LLMs. We validate the hypothesis that there is a valley of code reasoning: downstream performance on competitive coding first drops as data quantity increases, then it steadily increases in a sharper-than-log-linear fashion. Having identified the trend, we further fine-tune the models at two different distillation stages on the same data to ground conclusions on their respective learning phases. We learn that across stages in the low and medium-low data regimes, small models benefit significantly from easier coding questions than from harder ones. We also find that, surprisingly, the correctness of outputs in training data makes no difference to distillation outcomes. Our work represents a step forward in understanding the training dynamics of code reasoning distillation outside intuition. We are open-sourcing dataset splits used for all our experiments at https://collinear.ai/valley-of-reasoning-data .

### 21. TINY BUT MIGHTY: A SOFTWARE-HARDWARE CO-DESIGN APPROACH FOR EFFICIENT MULTIMODAL IN-FERENCE ON BATTERY-POWERED SMALL DEVICES

**主要机构**: University of Wisconsin -Madison
**作者数量**: 8人

**摘要**:
Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware-software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular "bricks" (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on-device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3% and GPU memory usage by 11.2%. This enables a battery-powered device to run LlaVA-OneVision with a camera for nearly half a day and LLaMA-3-8B for voice interactions up to almost 20.8 hours.

### 22. UNTANGLING COMPONENT IMBALANCE IN HYBRID LINEAR ATTENTION CONVERSION METHODS

**主要机构**: University College London, Department of Computer Science, Noah's Ark Lab
**作者数量**: 7人

**摘要**:
Transformers' quadratic computational complexity limits their scalability despite remarkable performance. While linear attention reduces this to linear complexity, pre-training such models from scratch remains, in most cases, prohibitively expensive. Recent post-training linearisation methods convert pre-trained Transformers to linear models efficiently, often using hybrid approaches that combine linear attention with sliding-window softmax. We identify a critical flaw: existing hybrid methods inadvertently bypass the linear component, relying almost entirely on SWA. Component-level diagnostics reveal this previously undetected behaviour stems from overlooked evaluation practices on common-sense benchmarks. We propose three solutions to ensure balanced component usage: (i) inference-time hybridisation of linear-only conversions with sliding-window softmax; (ii) Hedge-CATs, combining attention-weight transfer with targeted LoRA fine-tuning; and (iii) Scheduled Sliding-window Dropout (SSD), which stochastically suppresses the softmax branch during training to prevent component collapse. Our methods maintain computational efficiency while recovering most base model performance and ensuring genuine linear attention adoption, restoring the validity of performance attributions in hybrid conversions.

### 23. VATTENTION: VERIFIED SPARSE ATTENTION

**主要机构**: Electrical Engineering and Computer Sciences, University of California
**作者数量**: 9人

**摘要**:
State-of-the-art sparse attention methods for reducing decoding latency fall into two main categories: approximate top-k (and its extension, top-p) and recently introduced sampling-based estimation. However, these approaches are fundamentally limited in their ability to approximate full attention: they fail to provide consistent approximations across heads and query vectors and, most critically, lack guarantees on approximation quality, limiting their practical deployment. We observe that top-k and random sampling are complementary: top-k performs well when attention scores are dominated by a few tokens, whereas random sampling provides better estimates when attention scores are relatively uniform. Building on this insight and leveraging the statistical guarantees of sampling, we introduce vAttention, the first practical sparse attention mechanism with user-specified (ϵ, δ) guarantees on approximation accuracy (thus, "verified"). These guarantees make vAttention a compelling step toward practical, reliable deployment of sparse attention at scale. By unifying top-k and sampling, vAttention outperforms both individually, delivering a superior quality-efficiency trade-off. Our experiments show that vAttention significantly improves the quality of sparse attention (e.g., ∼4.5 percentage points for Llama-3.1-8B-Inst and Deepseek-R1-Distill-Llama-8B on RULER-HARD), and effectively bridges the gap between full and sparse attention (e.g., across datasets, it matches full model quality with upto 20x sparsity). We also demonstrate that it can be deployed in reasoning scenarios to achieve fast decoding without compromising model quality (e.g., vAttention achieves full model quality on AIME2024 at 10x sparsity with up to 32K token generations). Code is open-sourced at https://github.com/xAlg-ai/sparse-attention-hub.

### 24. VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization

**主要机构**: MiLM Plus, Chinese Academy of Sciences, Xiaomi Inc, Institute of Information Engineering
**作者数量**: 7人

**摘要**:
The Key-Value (KV) cache introduces substantial memory overhead during large language model (LLM) inference. Although existing vector quantization (VQ) methods reduce KV cache usage and provide flexible representational capacity across bit-widths, they suffer severe performance degradation at ultra-low bit-widths due to key cache outliers that hinder effective codebook utilization. To address this challenge, we propose VecInfer, a novel VQ method for aggressive KV cache compression while enabling efficient inference. By applying smooth and Hadamard transformations, VecInfer suppresses outliers in the key cache, enabling the codebook to comprehensively cover the original data distribution and thereby reducing quantization difficulty. To facilitate efficient deployment, we design an optimized CUDA kernel that fuses computation with dequantization to minimize memory access overhead. Extensive evaluations demonstrate that VecInfer consistently outperforms existing quantization baselines across both longcontext understanding and mathematical reasoning tasks. With only 2-bit quantization, VecInfer achieves performance comparable to full precision, while delivering up to 2.7× speedup in large-batch self-attention computation and 8.3× reduction in single-batch endto-end latency on Llama-3.1-8B with a 196k sequence length.

### 25. VER: VISION EXPERT TRANSFORMER FOR ROBOT LEARNING VIA FOUNDATION DISTILLATION AND DYNAMIC ROUTING

**主要机构**: Stony Brook University, UNC-Chapel Hill, Peking University, University of Hong
**作者数量**: 13人

**摘要**:
Pretrained vision foundation models (VFMs) advance robotic learning via rich visual representations, yet individual VFMs typically excel only in specific domains, limiting generality across tasks. Distilling multiple VFMs into a unified representation for policy can mitigate this limitation but often yields inflexible task-specific feature selection and requires costly full retraining to incorporate robot-domain knowledge. We propose VER, a Vision Expert transformer for Robot learning. During pretraining, VER distills multiple VFMs into a vision expert library. It then fine-tunes only a lightweight routing network (fewer than 0.4% of parameters) to dynamically select task-relevant experts from the pretrained library for downstream robot tasks. We further introduce Patchwise Expert Routing with Curriculum Top-K Annealing to improve both flexibility and precision of dynamic expert selection. Moreover, VER supports parameter-efficient finetuning for scalable expert utilization and adaptive robot-domain knowledge integration. Across 17 diverse robotic tasks and multiple policy heads, VER achieves state-of-the-art performance. We find that VER reduces large-norm outliers in task-irrelevant regions (e.g., background) and concentrates on task-critical regions. Visualizations and codes can be found in https://yixiaowang7.github.io/ver_page/.
