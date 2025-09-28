# AI推理加速技术论文分析报告
生成时间: 2025-09-28 14:53:58
分析论文数量: 15篇

## 论文技术简报

### 1. ADAPTIVE GUIDANCE SEMANTICALLY ENHANCED VIA MULTIMODAL LLM FOR EDGE-CLOUD OBJECT DETECTION

中国科学院计算技术研究所发布了自适应引导语义增强边缘云目标检测论文，使用多模态大语言模型（MLLM）的自适应引导语义增强边缘云协同方法（含指令微调生成场景描述、自适应映射机制及边缘云置信度选择），解决了传统目标检测在低光、严重遮挡等复杂场景因缺乏高层语义理解导致的性能下降问题，达成了在低光和高遮挡场景下延迟降低超79%、计算成本减少70%且保持检测精度的效果。

### 2. Are We Scaling the Right Thing? A System Perspective on Test-Time Scaling

Microsoft Research发布了测试时缩放（TTS）的系统视角研究论文，使用系统驱动视角与整体系统感知评估技术，解决了现有TTS方法仅关注计算最优而忽略系统最优的问题，揭示了当前方法局限性并呼吁转向系统感知评估以捕获推理时缩放定律的本质。

### 3. A Comprehensive Evaluation of YOLO-based Deer Detection Performance on Edge Devices

密歇根州立大学发布了A Comprehensive Evaluation of YOLO-based Deer Detection Performance on Edge Devices论文，使用YOLOv8/v9/v10/v11等系列模型综合评估及边缘设备部署技术并引入公开数据集，解决了农业中鹿入侵导致的经济损失及传统方法不足、缺乏专用数据集和边缘可行性研究的问题，达成了小模型（如YOLOv11n）在边缘设备上高准确率（AP@.5>0.85）和实时效率（FPS>30）的效果

### 4. A Versatile Foundation Model for AI-enabled Mammogram Interpretation

中山大学发布了A Versatile Foundation Model for AI-enabled Mammogram Interpretation论文，使用通用基础模型技术，解决了AI辅助乳腺X光片解读问题，达成了提升解读性能与通用性的效果。

### 5. CapStARE: Capsule-based Spatiotemporal Architecture for Robust and Efficient Gaze Estimation

EHU发布了CapStARE论文，使用胶囊基时空架构（集成ConvNeXt骨干、注意力路由胶囊形成及双GRU解码器），解决了凝视估计的鲁棒性与效率问题，达成了在多个数据集上的最先进性能（如ETH-XGaze 3.36°、MPIIFaceGaze 2.65°）并保持实时推理（<10ms）的效果

### 6. Choosing to Be Green: Advancing Green AI via Dynamic Model Selection

佛罗伦萨大学发布了Choosing to Be Green: Advancing Green AI via Dynamic Model Selection论文，使用基于动态模型选择的Green AI方法（含动态模型级联与路由），解决了AI系统高耗能的环境问题并最小化精度损失，达成了高达约25%的能源节省且保留最耗能模型约95%精度的效果。

### 7. FRAME-STACKED LOCAL TRANSFORMERS FOR EFFICIENT MULTI-CODEBOOK SPEECH GENERATION

相关机构发布了FRAME-STACKED LOCAL TRANSFORMERS FOR EFFICIENT MULTI-CODEBOOK SPEECH GENERATION论文，使用帧堆叠局部Transformer技术（含自回归和MaskGIT-based架构），解决了多码本语音生成中并行预测因码本依赖导致的保真度低问题，达成了不降低感知质量的同时提升速度并提供解码策略选择指南的效果。

### 8. Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM Inference

阿里巴巴集团与复旦大学发布了Gyges论文，使用动态跨实例并行转换技术（含页友好头中心布局、专用权重填充及转换感知调度器），解决了LLM推理中请求动态性（尤其是上下文长度变化）导致的并行策略权衡（用TP提升上下文长度但降低吞吐量）问题，达成吞吐量提升1.75×-6.57×的效果

### 9. OmniScene: Attention-Augmented Multimodal 4D Scene Understanding for Autonomous Driving

清华大学发布了OmniScene论文，使用OmniVLM视觉语言模型、师生架构知识蒸馏及分层融合策略，解决了自动驾驶系统依赖深度3D重建而非类人化4D场景理解的问题，在nuScenes数据集多任务超越SOTA，VQA性能提升21.40%

### 10. Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators

印度科学学院发布了相关论文，使用时间和新型能量屋顶线模型，解决了边缘加速器及其功耗模式性能行为缺乏系统性研究的问题，达成了优化DNN推理延迟和能耗、实现高达15%能耗降低且推理时间几乎无损失的效果。

### 11. Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment

首尔国立大学发布了Q-Palette论文，使用分数比特量化器(Q-Palette)及混合方案量化框架，解决了LLM中不规则权重分布导致的量化困难及内存受限场景下的部署效率问题，达成了近最优量化性能以提升LLM部署效率。

### 12. RoboSSM: Scalable In-context Imitation Learning via State-Space Models

德克萨斯大学发布了RoboSSM论文，使用基于状态空间模型(SSM)的Longhorn技术，解决了现有基于Transformer的上下文模仿学习(ICIL)方法存在计算限制及处理长提示时表现不佳的问题，在LIBERO基准上实现了对不同数量演示的有效外推、未见过任务的高性能及长时场景的稳健性。

### 13. SIM-COT: SUPERVISED IMPLICIT CHAIN-OF-THOUGHT

复旦大学发布了SIM-COT论文，使用引入步骤级监督的辅助解码器（训练时对齐隐式token与显式推理步骤，推理时移除）的技术，解决了隐式CoT方法计算预算增加时训练不稳定（潜在表示同质化）的问题，达成了提升Coconut在GPT-2上+8.2%、CODI在LLaMA-3.1 8B上+3.0%，并超过显式CoT基线2.1%且token效率高2.3倍的效果

### 14. Smaller is Better: Enhancing Transparency in Vehicle AI Systems via Pruning

Toyota InfoTech Labs发布了“Smaller is Better: Enhancing Transparency in Vehicle AI Systems via Pruning”论文，使用修剪技术，解决了车辆AI系统中事后解释质量和可靠性不足的问题，达成了显著提升解释的可理解性与忠实性并提高模型效率的效果。

### 15. 

请提供论文的标题、主要机构和摘要等具体信息，以便生成符合要求的技术简报。

## 论文详细信息

### 1. ADAPTIVE GUIDANCE SEMANTICALLY ENHANCED VIA MULTIMODAL LLM FOR EDGE-CLOUD OBJECT DETECTION

**主要机构**: Institute of Computing Technology, Chinese Academy of Sciences
**作者数量**: 4人

**摘要**:
Traditional object detection methods face performance degradation challenges in complex scenarios such as low-light conditions and heavy occlusions due to a lack of high-level semantic understanding. To address this, this paper proposes an adaptive guidance-based semantic enhancement edge-cloud collaborative object detection method leveraging Multimodal Large Language Models (MLLM), achieving an effective balance between accuracy and efficiency. Specifically, the method first employs instruction fine-tuning to enable the MLLM to generate structured scene descriptions. It then designs an adaptive mapping mechanism that dynamically converts semantic information into parameter adjustment signals for edge detectors, achieving real-time semantic enhancement. Within an edgecloud collaborative inference framework, the system automatically selects between invoking cloud-based semantic guidance or directly outputting edge detection results based on confidence scores. Experiments demonstrate that the proposed method effectively enhances detection accuracy and efficiency in complex scenes. Specifically, it can reduce latency by over 79% and computational cost by 70% in low-light and highly occluded scenes while maintaining accuracy.

### 2. Are We Scaling the Right Thing? A System Perspective on Test-Time Scaling

**主要机构**: University of Central Florida, Microsoft Research
**作者数量**: 5人

**摘要**:
Test-time scaling (TTS) has recently emerged as a promising direction to exploit the hidden reasoning capabilities of pre-trained large language models (LLMs). However, existing scaling methods narrowly focus on the compute-optimal Paretofrontier, ignoring the simple fact that compute-optimal is not always system-optimal. In this work, we propose a system-driven perspective on TTS, analyzing how reasoning models scale against practical metrics, such as latency and cost-pertoken. By evaluating the impact of popular optimizations such as tensor parallelism and speculative decoding, our preliminary analysis reveals the limitations of current methods and calls for a paradigm shift toward holistic, system-aware evaluations that capture the true essence of scaling laws at inference time.

### 3. A Comprehensive Evaluation of YOLO-based Deer Detection Performance on Edge Devices

**主要机构**: Department of Agricultural and Biological Engineering, Department of Electrical and Computer Engineering, Michigan State University, Mississippi State University, Department of Wildlife, Fisheries and Aquaculture, Department of Plant and Soil Science
**作者数量**: 7人

**摘要**:
The escalating economic losses in agriculture due to deer intrusion, estimated to be in the hundreds of millions of dollars annually in the U.S., highlight the inadequacy of traditional mitigation strategies such as fencing, use of repellents, and conventional scare tactics. These methods are often laborintensive, costly, and ineffective for modern farming systems. To address this challenge, there is a critical need for intelligent, autonomous solutions capable of real-time wildlife detection and deterrence. At the core of such systems lies accurate and efficient deer detection. But the progress in this field is impeded by a significant gap in the literature, mainly the lack of a domain-specific, practical dataset and limited study on the viability of deer detection systems on edge devices. Addressing this gap, this study presents a comprehensive evaluation of state-of-the-art deep learning models for deer detection in challenging real-world scenarios. The contributions of this work are threefold. First, we introduce a curated, publicly available dataset of 3,095 annotated images with boundingbox annotations of deer, derived from the Idaho Cameratraps project, representing challenging realworld scenarios. Second, we provide an extensive comparative analysis of 12 model variants across four recent YOLO architectures(v8, v9, v10, and v11). Finally, we benchmarked performance on a high-end NVIDIA RTX 5090 GPU and evaluated on two representative edge computing platforms: the CPU-based Raspberry Pi 5 and the GPU-accelerated NVIDIA Jetson AGX Xavier to assess feasibility for real-world field deployment. Results show that the real-time detection performance is not feasible in Raspberry Pi without hardware-specific model optimization, while NVIDIA Jetson provides greater than 30 FPS with GPU-accelerated inference on 's' and 'n' series models. This study also reveals that smaller, architecturally advanced models such as YOLOv11n, YOLOv8s, and YOLOv9s offer the optimal balance of high accuracy (AP@.5 > 0.85) and computational efficiency (FPS > 30). To support further research, both the source code and datasets are publicly available at https://github.com/WinnerBishal/track-the-deer.

### 4. A Versatile Foundation Model for AI-enabled Mammogram Interpretation

**主要机构**: Sun Yat-Sen University, Shenzhen People's Hospital, Macau University of Science and Technology, Breast Tumor Centre, Chongqing University, Sun Yat-sen Memorial Hospital, The Third Affiliated Hospital, Chongqing Key Laboratory of Bio-perception and Multimodal Intelligent Information Processing, The Hong Kong University of Science and Technology, Department of Computer Science and Engineering, Phase I Clinical Trial Centre, Institute for AI in Medicine and Faculty of Medicine, South China University of Technology, Sun Yat-sen University, Department of Radiology, The Hong Kong University of Science and Technology (Guangzhou), Data Science and Analytics Thrust, Guangzhou First People's Hospital, Guangdong-Hong Kong Joint Laboratory for RNA Medicine, Guangdong Provincial Key Laboratory of Malignant Tumor Epigenetics and Gene Regulation, Department of Medical Oncology
**作者数量**: 23人

**摘要**:


### 5. CapStARE: Capsule-based Spatiotemporal Architecture for Robust and Efficient Gaze Estimation

**主要机构**: Computer Science and Artificial Intelligence, EHU
**作者数量**: 3人

**摘要**:
We introduce CapsStARE, a capsule-based spatiotemporal architecture for gaze estimation that integrates a ConvNeXt backbone, capsule formation with attention routing, and dual GRU decoders specialized for slow and rapid gaze dynamics. This modular design enables efficient part-whole reasoning and disentangled temporal modeling, achieving state-of-the-art performance on ETH-XGaze (3.36 •) and MPIIFaceGaze (2.65 •) while maintaining real-time inference (< 10 ms). The model also generalizes well to unconstrained conditions in Gaze360 (9.06 •) and human-robot interaction scenarios in RT-GENE (4.76 •), outperforming or matching existing methods with fewer parameters and greater interpretability. These results demonstrate that CapsStARE offers a practical and robust solution for real-time gaze estimation in interactive systems. The related code and results for this article can be found on: https://github.com/toukapy/capsStare

### 6. Choosing to Be Green: Advancing Green AI via Dynamic Model Selection

**主要机构**: University of Florence, European University of Rome
**作者数量**: 2人

**摘要**:
Artificial Intelligence is increasingly pervasive across domains, with ever more complex models delivering impressive predictive performance. This fast technological advancement however comes at a concerning environmental cost, with state-of-the-art models-particularly deep neural networks and large language models-requiring substantial computational resources and energy. In this work, we present the intuition of Green AI dynamic model selection, an approach based on dynamic model selection that aims at reducing the environmental footprint of AI by selecting the most sustainable model while minimizing potential accuracy loss. Specifically, our approach takes into account the inference task, the environmental sustainability of available models, and accuracy requirements to dynamically choose the most suitable model. Our approach presents two different methods, namely Green AI dynamic model cascading and Green AI dynamic model routing. We demonstrate the effectiveness of our approach via a proof of concept empirical example based on a real-world dataset. Our results show that Green AI dynamic model selection can achieve substantial energy savings (up to ≈25%) while substantially retaining the accuracy of the most energy greedy solution (up to ≈95%). As conclusion, our preliminary findings highlight the potential that hybrid, adaptive model selection strategies withhold to mitigate the energy demands of modern AI systems without significantly compromising accuracy requirements.

### 7. FRAME-STACKED LOCAL TRANSFORMERS FOR EFFICIENT MULTI-CODEBOOK SPEECH GENERATION

**主要机构**: 
**作者数量**: 9人

**摘要**:
Speech generation models based on large language models (LLMs) typically operate on discrete acoustic codes, which differ fundamentally from text tokens due to their multicodebook structure. At each timestep, models must predict N codebook entries jointly, introducing dependencies that challenge simple parallel prediction approaches. Parallel prediction assumes independence among codebooks, yielding efficient decoding but often at the cost of reduced fidelity. To address this, hierarchical strategies employ a local transformer (LT) to refine predictions and capture intra-timestep dependencies. In this work, we systematically investigate two LT architectures: an autoregressive transformer that generates codebooks sequentially, and a MaskGIT-based transformer that performs iterative masked prediction. Both designs further enable frame stacking, where the primary transformer predicts multiple frames jointly, and the LT decodes their codebooks, offering improvements in speed without compromising perceptual quality. Through extensive analysis, we characterize the tradeoffs between parallel and iterative sampling strategies across different throughput and quality regimes. Finally, we propose practical guidelines for selecting decoding strategies based on deployment priorities such as computational efficiency and synthesis fidelity 1 .

### 8. Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM Inference

**主要机构**: Xin Wang, Fudan University and Alibaba Group Shanghai, Alibaba Group Hangzhou, Yu Guan, Fudan University Shanghai, Xue Li
**作者数量**: 9人

**摘要**:
Efficiently processing the dynamics of requests, especially the context length variance, is important in Large Language Model (LLM) serving scenarios. However, there is an intrinsic trade-off: while leveraging parallelism strategies, such as Tensor Parallelism (TP), can coordinate multiple GPUs to accommodate larger context lengths, it inevitably results in degraded overall throughput. In this paper, we propose Cross-Instance Parallelism Transformation (Gyges), which adaptively adjusts the parallelism strategies of running instances to align with the dynamics of incoming requests. We design (1) a page-friendly, headercentric layout to accelerate KV cache transformations; (2) dedicated weight padding to accelerate model weight transformations; and (3) a transformation-aware scheduler to cooperatively schedule requests and parallelism transformations, optimizing the overall performance. Evaluations using real-world traces show that Gyges improves throughput by 1.75×-6.57× compared to state-of-the-art solutions.

### 9. OmniScene: Attention-Augmented Multimodal 4D Scene Understanding for Autonomous Driving

**主要机构**: 
**作者数量**: 6人

**摘要**:
Human vision is capable of transforming twodimensional observations into an egocentric three-dimensional scene understanding, which underpins the ability to translate complex scenes and exhibit adaptive behaviors. This capability, however, is still lacking in current autonomous driving systems, where mainstream approaches largely rely on depth-based 3D reconstruction rather than true scene understanding. To address this limitation, we propose a novel human-like framework called OmniScene. First, we introduce the OmniScene Vision-Language Model (OmniVLM), a vision-language framework that integrates multi-view and temporal perception for holistic 4D scene understanding. Then, harnessing a teacher-student OmniVLM architecture and knowledge distillation, we embed textual representations into 3D instance features for semantic supervision, enriching feature learning, and explicitly capturing human-like attentional semantics. These feature representations are further aligned with human driving behaviors, forming a more human-like perception-understanding-action architecture. In addition, we propose a Hierarchical Fusion Strategy (HFS) to address imbalances in modality contributions during multimodal integration. Our approach adaptively calibrates the relative significance of geometric and semantic features at multiple abstraction levels, enabling the synergistic use of complementary cues from visual and textual modalities. This learnable dynamic fusion enables a more nuanced and effective exploitation of heterogeneous information. We evaluate OmniScene comprehensively on the nuScenes dataset, benchmarking it against over ten state-of-the-art models across various tasks. Our approach consistently achieves superior results, establishing new benchmarks in perception, prediction, planning, and visual question answering. Notably, OmniScene yields a remarkable 21.40% improvement in visual question answering (VQA) performance, highlighting its robust multimodal reasoning capabilities. Project Link: https://github.com/ocean-luna/OmniScene.

### 10. Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators

**主要机构**: Indian Institute of Science, Department of Computational and Data Sciences
**作者数量**: 6人

**摘要**:
Edge accelerators such as Nvidia Jetsons are becoming an integral part of the computing continuum, and are often used for DNN inferencing and training. Nvidia Jetson edge devices have 2000+ CUDA cores within a 70W power envelope and offer 1000s of power modes to customize CPU, GPU and memory frequencies. Their widely varying power-performance trade-offs can be exploited for energy and power-constrained deployments. While data-driven methods to predict the power and latency of DNN workloads for edge devices exist, there is a lack of principled study to understand why edge accelerators and their power modes perform the way they do. We develop a time roofline and a novel energy roofline model for the Jetson Orin AGX for diverse power modes, and couple it with an analytical model of the compute (FLOP) and memory access (bytes) for DNN inference workloads to analyze them from first principles. These reveal unique, sometimes counter-intuitive, insights into the power and performance behavior of DNN workloads on edge accelerators, e.g., the default power mode MAXN is not the most energy efficient and time efficiency implies energy efficiency for all power modes. We also extend our analytical roofline models to DNN training. Finally, we apply these methods to tune the power mode (and hence the roofline) of the edge device to optimize the latency and energy for DNN inference, with up to 15% lower energy and minimal degradation in inference time.

### 11. Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment

**主要机构**: Seoul National University
**作者数量**: 2人

**摘要**:
We study weight-only post-training quantization (PTQ), which quantizes the weights of a large language model (LLM) without retraining, using little or no calibration data. Weight-only PTQ is crucial for reducing the memory footprint and latency of LLM inference, especially in memory-bound, small-batch inference scenarios, such as personalized inference on edge devices. Despite its importance, irregular weight distributions with heavy-tailed outliers in LLMs complicate quantization, recently motivating rotation-based methods that transform weights into near-Gaussian distributions, which are more regular with fewer outliers, thereby reducing quantization error. In this work, we first derive the information-theoretically optimal bit allocation for Gaussianized weights under given bit budgets, revealing that fine-grained fractional-bit quantizers approaching the Gaussian distortion-rate bound are essential to achieve near-optimal quantization performance. To bridge this theoretical insight and practical implementation, we introduce Q-Palette, a versatile collection of fractional-bit quantizers that range from trellis-coded quantizers offering near-optimal distortion to simpler vector and scalar quantizers optimized for faster inference, all efficiently implemented with optimized CUDA kernels across various bitwidths. Furthermore, leveraging Q-Palette as a foundational component, we propose a novel mixed-scheme quantization framework, jointly optimizing quantizer choices and layer fusion decisions given resource constraints. The code is available at https://github.com/snu-mllab/Q-Palette.

### 12. RoboSSM: Scalable In-context Imitation Learning via State-Space Models

**主要机构**: The University of Texas
**作者数量**: 7人

**摘要**:
In-context imitation learning (ICIL) enables robots to learn tasks from prompts consisting of just a handful of demonstrations. By eliminating the need for parameter updates at deployment time, this paradigm supports fewshot adaptation to novel tasks. However, recent ICIL methods rely on Transformers, which have computational limitations and tend to underperform when handling longer prompts than those seen during training. In this work, we introduce RoboSSM, a scalable recipe for in-context imitation learning based on state-space models (SSM). Specifically, RoboSSM replaces Transformers with Longhorn-a state-of-the-art SSM that provides linear-time inference and strong extrapolation capabilities, making it well-suited for long-context prompts. We evaluate our approach on the LIBERO benchmark and compare it against strong Transformer-based ICIL baselines. Experiments show that RoboSSM extrapolates effectively to varying numbers of in-context demonstrations, yields high performance on unseen tasks, and remains robust in longhorizon scenarios. These results highlight the potential of SSMs as an efficient and scalable backbone for ICIL. Our code is available at https://github.com/youngjuY/RoboSSM.

### 13. SIM-COT: SUPERVISED IMPLICIT CHAIN-OF-THOUGHT

**主要机构**: Laboratory, Fudan University
**作者数量**: 8人

**摘要**:
Implicit Chain-of-Thought (CoT) methods offer a token-efficient alternative to explicit CoT reasoning in Large Language Models (LLMs), but a persistent performance gap has limited their adoption. We identify a core latent instability issue when scaling the computational budget of implicit CoT: as the number of reasoning tokens increases, training often becomes unstable and collapses. Our analysis shows that this instability arises from latent representations becoming homogeneous and losing semantic diversity, caused by insufficient step-level supervision in current implicit CoT methods. To address this, we propose SIM-CoT, a plug-and-play training module that introduces step-level supervision to stabilize and enrich the latent reasoning space. SIM-CoT employs an auxiliary decoder during training to align each implicit token with its corresponding explicit reasoning step, ensuring latent states capture distinct and meaningful information. The auxiliary decoder is removed at inference, preserving the efficiency of implicit CoT with no added overhead. It also provides interpretability by projecting each latent token onto an explicit reasoning vocabulary, enabling per-step visualization and diagnosis. SIM-CoT significantly improves both in-domain accuracy and out-of-domain stability of implicit CoT methods, boosting Coconut by +8.2% on GPT-2 and CODI by +3.0% on LLaMA-3.1 8B. It further surpasses the explicit CoT baseline on GPT-2 by 2.1% with 2.3× greater token efficiency, while closing the performance gap on larger models like LLaMA-3.1 8B.

### 14. Smaller is Better: Enhancing Transparency in Vehicle AI Systems via Pruning

**主要机构**: Toyota InfoTech Labs, Rochester Institute of Technology
**作者数量**: 5人

**摘要**:
Connected and autonomous vehicles continue to heavily rely on AI systems, where transparency and security are critical for trust and operational safety. Post-hoc explanations provide transparency to these black-box like AI models but the quality and reliability of these explanations is often questioned due to inconsistencies and lack of faithfulness in representing model decisions. This paper systematically examines the impact of three widely used training approaches, namely natural training, adversarial training, and pruning, affect the quality of post-hoc explanations for traffic sign classifiers. Through extensive empirical evaluation, we demonstrate that pruning significantly enhances the comprehensibility and faithfulness of explanations (using saliency maps). Our findings reveal that pruning not only improves model efficiency but also enforces sparsity in learned representation, leading to more interpretable and reliable decisions. Additionally, these insights suggest that pruning is a promising strategy for developing transparent deep learning models, especially in resource-constrained vehicular AI systems.

### 15. 

**主要机构**: 
**作者数量**: 0人

**摘要**:

