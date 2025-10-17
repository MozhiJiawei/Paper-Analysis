# AI推理加速技术论文分析报告
生成时间: 2025-10-17 12:38:26
分析论文数量: 17篇

## 论文技术简报

### 1. APCE: Adaptive Progressive Context Expansion for Long Context Processing

LG Electronics发布了APCE论文，使用APCE技术，解决了长序列处理中内存占用高和ContextRot导致性能下降的问题，达成了用50%-70%输入序列实现相当或更优摘要性能并提升内存效率的效果。

### 2. COMPODISTILL: ATTENTION DISTILLATION FOR COMPOSITIONAL REASONING IN MULTIMODAL LLMS

KAIST发布了COMPODISTILL论文，使用CompoDistill框架（通过明确对齐学生与教师的视觉注意力），解决了现有知识蒸馏方法难以有效将教师多模态大语言模型的视觉感知能力蒸馏给学生（因视觉注意力错位）的问题，达成了在组合推理任务上显著提升性能、保持视觉问答任务良好表现且具有可泛化性的效果。

### 3. DISTAR: DIFFUSION OVER A SCALABLE TOKEN AUTOREGRESSIVE REPRESENTATION FOR SPEECH GENERATION

字节跳动发布了DISTAR论文，使用在离散RVQ码空间中结合自回归语言模型与掩码扩散模型的框架，解决了现有零样本TTS在分布偏移下脆弱及可控性有限的问题，达成了超越最先进系统、提升鲁棒性/自然度/说话人风格一致性并保持输出多样性的效果

### 4. DRIVEVLA-W0: WORLD MODELS AMPLIFY DATA SCALING LAW IN AUTONOMOUS DRIVING

中国科学院自动化研究所（CASIA）发布了DRIVEVLA-W0论文，使用世界模型技术，解决了自动驾驶中数据缩放律的优化问题，达成了增强数据缩放律的效果。

### 5. DR.LLM: DYNAMIC LAYER ROUTING IN LLMS

NAVER AI Lab发布了DR.LLM论文，使用基于蒙特卡洛树搜索显式监督训练的轻量级每层路由器实现动态层路由技术，解决了LLM处理简单查询计算浪费及复杂任务灵活性不足的问题，达成准确率提升3.4%p同时平均每示例节省5层计算，并在跨域任务中仅0.85%准确率下降的效果。

### 6. Dual Learning with Dynamic Knowledge Distillation and Soft Alignment for Partially Relevant Video Retrieval

相关团队发布了部分相关视频检索（PRVR）论文，使用双学习框架与动态知识蒸馏（DL-DKD++）技术，解决未修剪长时长视频的部分相关视频检索问题，在TVR、ActivityNet和Charades-STA数据集上达成最先进性能。

### 7. EVOLUTION OF META'S LLAMA MODELS AND PARAMETER-EFFICIENT FINE-TUNING OF LARGE LANGUAGE MODELS: A SURVEY

悉尼科技大学等机构发布了关于Meta LLAMA模型演变及参数高效微调方法的综述论文，系统综述LLAMA系列（7B-288B参数）及LoRA等五种PEFT方法，解决大语言模型微调参数效率低的问题，实现参数节省并使微调模型性能超过更大基线，成功应用于法律医疗等领域。

### 8. FLASHVSR: TOWARDS REAL-TIME DIFFUSION-BASED STREAMING VIDEO SUPER-RESOLUTION

香港大学、清华大学发布了FlashVSR论文，使用三阶段蒸馏流水线、局部约束稀疏注意力和小型条件解码器技术，解决了扩散模型在视频超分辨率中高延迟、计算量大、超高分辨率泛化差的挑战，达成在单个A100 GPU上768×1408视频~17 FPS的实时性能，较先前单步扩散VSR模型提速约12倍并实现超高分辨率可靠扩展。

### 9. LiteVPNet: A Lightweight Network for Video Encoding Control in Quality-Critical Applications

Trinity College Dublin发布了LiteVPNet论文，使用轻量级神经网络结合低复杂度特征预测量化参数，解决了质量关键应用中视频编码的精确质量控制与能效需求，达成平均VMAF误差低于1.2点、87%测试样本误差在2点内（现有方法约61%）的效果

### 10. MoBiLE: Efficient Mixture-of-Experts Inference on Consumer GPU with Mixture of Big Little Experts

清华大学发布了MoBiLE论文，使用混合大小专家的即插即用卸载框架，解决了MoE推理中CPU-GPU带宽瓶颈及现有预取方法训练开销大、细粒度专家分割效果差的问题，在消费级GPU系统上实现1.60×-1.72×加速，精度损失可忽略。

### 11. MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics

University of Artificial Intelligence和Mohamed bin Zayed发布了MosaicDiff论文，使用免训练结构化剪枝（反映预训练动态）技术，解决扩散模型加速问题，实现扩散模型高效加速。

### 12. On the Use of Hierarchical Vision Foundation Models for Low-Cost Human Mesh Recovery and Pose Estimation

东京都立大学发布了On the Use of Hierarchical Vision Foundation Models for Low-Cost Human Mesh Recovery and Pose Estimation论文，使用分层视觉基础模型技术，解决了低成本下的人体网格恢复与姿态估计问题，达成了高效的人体网格恢复与姿态估计效果。

### 13. PAGS: PRIORITY-ADAPTIVE GAUSSIAN SPLATTING FOR DYNAMIC DRIVING SCENES

哈尔滨工业大学发布了PAGS（Priority-Adaptive Gaussian Splatting）论文，使用语义引导的剪枝与正则化策略及优先级驱动的渲染管道技术，解决了动态驾驶场景重建中因语义无关设计导致资源均匀分配而产生的保真度与计算成本权衡问题，达成提升安全关键对象重建质量、减少训练时间且渲染速度超350 FPS的效果。

### 14. RETHINKING KNOWLEDGE DISTILLATION: A DATA DEPENDENT REGULARISER WITH A NEGATIVE ASYMMETRIC PAYOFF

Queen Mary University of London发布了重新思考知识蒸馏的研究论文，使用量化压缩能力、功能角度知识转移分析及假设检验等方法，解决了知识蒸馏作为压缩机制的功能影响理解不足的问题，揭示其更像数据依赖的正则化器且存在负知识不对称转移引发安全问题。

### 15. SMEC:Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression

阿里巴巴发布了SMEC论文，使用包含Sequential Matryoshka Representation Learning (SMRL)、Adaptive Dimension Selection (ADS)和Selectable Cross-batch Memory (S-XBM)模块的SMEC框架，解决了高维嵌入的计算复杂度和存储需求问题，在BEIR数据集上，压缩后的LLM2Vec嵌入（256维）性能较Matryoshka-Adaptor提升1.1点、较Search-Adaptor提升2.7点。

### 16. UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering

相关机构发布了UniGS论文，使用统一几何感知高斯溅射技术，解决了多模态渲染难题，提升了渲染质量与几何一致性。

### 17. ZERO-SHOT CFC: FAST REAL-WORLD IMAGE DENOISING BASED ON CROSS-FREQUENCY CONSISTENCY

北京工业大学发布了ZERO-SHOT CFC论文，使用基于跨频率一致性的损失和超轻量网络技术，解决了现有零样本去噪方法训练时间长、依赖噪声分布假设的问题，达成了在计算效率和去噪性能上优于其他SOTA零样本方法的效果。

## 论文详细信息

### 1. APCE: Adaptive Progressive Context Expansion for Long Context Processing

**主要机构**: LG Electronics
**作者数量**: 6人

**摘要**:
Deploying useful Long-Context Transformer Models (LCTMs) requires addressing two key challenges: (1) A growing memory footprint due to quadratic self-attention and linear KV-cache scaling in memory as sequence length increases; (2) the ContextRot phenomena where empirical evidence suggests that transformer architecture's performance degrades with increasing context length. Given the shared dependency on the input, a natural question arises: "Can we surgically select the most important input chunks for processing to synergistically (a) reduce the memory footprint, and (b) mitigate the ContextRot effects?" In this paper, we answer this question in the affirmative for long-context summarization tasks. We propose APCE as a context-aware solution to select the most important input chunks through low-dimensional semantic similarity matching with the current query. By directly operating on the input, APCE decouples from strict dependency on underlying hardware or CUDA environments, promising a compatible solution scalable to different deployment systems. Our empirical evaluations have demonstrated superior or on-par summarization performance for APCE compared to the full dense baseline using a fraction (50%-70%) of the input sequence resulting in KV-cache and self-attention memory efficiency improvements. We hope our findings inspire further research on context-aware efficiency solutions for LCTMs geared towards other relevant long-context tasks.

### 2. COMPODISTILL: ATTENTION DISTILLATION FOR COMPOSITIONAL REASONING IN MULTIMODAL LLMS

**主要机构**: KAIST
**作者数量**: 4人

**摘要**:
Recently, efficient Multimodal Large Language Models (MLLMs) have gained significant attention as a solution to their high computational complexity, making them more practical for real-world applications. In this regard, the knowledge distillation (KD) approach has emerged as a promising alternative, which transfers the rich visual and linguistic knowledge from a larger model (teacher) to a smaller model (student). However, we observe that existing KD methods struggle to effectively distill the teacher MLLM's rich visual perception abilities to the student, a challenge that has been largely overlooked in previous studies. Through a systematic analysis, we identify visual attention misalignment between student and teacher as the main cause of this issue. Based on this insight, we propose Com-poDistill, a novel KD framework that explicitly aligns the student's visual attention with that of the teacher to enhance the student's visual perception abilities. Our extensive experiments show that CompoDistill significantly improves performance on compositional reasoning tasks that require visual perception abilities while maintaining strong performance on visual question answering tasks, as done in existing studies. Furthermore, CompoDistill demonstrates effectiveness with a more advanced backbone, highlighting its generalizability.

### 3. DISTAR: DIFFUSION OVER A SCALABLE TOKEN AUTOREGRESSIVE REPRESENTATION FOR SPEECH GENERATION

**主要机构**: ByteDance Inc, Shanghai Jiao Tong University, X-LANCE Lab, School of Computer Science
**作者数量**: 11人

**摘要**:
Recent attempts to interleave autoregressive (AR) sketchers with diffusion-based refiners over continuous speech representations have shown promise, but they remain brittle under distribution shift and offer limited levers for controllability. We introduce DISTAR, a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an AR language model with a masked diffusion model, without forced alignment or a duration predictor. Concretely, DISTAR drafts block-level RVQ tokens with an AR language model and then performs parallel masked-diffusion infilling conditioned on the draft to complete the next block, yielding long-form synthesis with blockwise parallelism while mitigating classic AR exposure bias. The discrete code space affords explicit control at inference: DISTAR produces high-quality audio under both greedy and sample-based decoding using classifier-free guidance, supports trade-offs between robustness and diversity, and enables variable bit-rate and controllable computation via RVQ layer pruning at test time. Extensive experiments and ablations demonstrate that DISTAR surpasses state-ofthe-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency, while maintaining rich output diversity. Audio samples are provided on https://anonymous.4open.science/w/DiSTAR_demo.

### 4. DRIVEVLA-W0: WORLD MODELS AMPLIFY DATA SCALING LAW IN AUTONOMOUS DRIVING

**主要机构**: Institute of Automation, NLPR, Chinese Academy of Sciences (CASIA), Yinwang Intelligent Technology Co. Ltd
**作者数量**: 13人

**摘要**:


### 5. DR.LLM: DYNAMIC LAYER ROUTING IN LLMS

**主要机构**: NAVER AI Lab, University of Tübingen
**作者数量**: 7人

**摘要**:
Large Language Models (LLMs) process every token through all layers of a transformer stack, causing wasted computation on simple queries and insufficient flexibility for harder ones that need deeper reasoning. Adaptive-depth methods can improve efficiency, but prior approaches rely on costly inference-time search, architectural changes, or large-scale retraining, and in practice often degrade accuracy despite efficiency gains. We introduce Dr.LLM, Dynamic routing of Layers for LLMs, a retrofittable framework that equips pretrained models with lightweight per-layer routers deciding to skip, execute, or repeat a block. Routers are trained with explicit supervision: using Monte Carlo Tree Search (MCTS), we derive high-quality layer configurations that preserve or improve accuracy under a compute budget. Our design, windowed pooling for stable routing, focal loss with class balancing, and bottleneck MLP routers, ensures robustness under class imbalance and long sequences. On ARC (logic) and DART (math), Dr.LLM improves accuracy by up to +3.4%p while saving 5 layers per example on average. Routers generalize to out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval) with only 0.85% accuracy drop while retaining efficiency, and outperform prior routing methods by up to +7.7%p. Overall, Dr.LLM shows that explicitly supervised routers retrofit frozen LLMs for budgetaware, accuracy-driven inference without altering base weights.

### 6. Dual Learning with Dynamic Knowledge Distillation and Soft Alignment for Partially Relevant Video Retrieval

**主要机构**: 
**作者数量**: 8人

**摘要**:
Almost all previous text-to-video retrieval works ideally assume that videos are pre-trimmed with short durations containing solely text-related content. However, in practice, videos are typically untrimmed in long durations with much more complicated background content. Therefore, in this paper, we focus on the more practical yet challenging task of Partially Relevant Video Retrieval (PRVR), which aims to retrieve partially relevant untrimmed videos with the given query. To tackle this task, we propose a novel framework that distills generalization knowledge from a powerful large-scale visionlanguage pre-trained model and transfers it to a lightweight, task-specific PRVR network. Specifically, we introduce a Dual Learning framework with Dynamic Knowledge Distillation (DL-DKD++), where a large teacher model provides supervision to a compact dual-branch student network. The student model comprises two branches: an inheritance branch that absorbs transferable knowledge from the teacher, and an exploration branch that learns task-specific information from the PRVR dataset to address domain gaps. To further enhance learning, we incorporate a dynamic soft-target construction mechanism. By replacing rigid hard-target supervision with adaptive soft targets that evolve during training, our method enables the model to better capture the fine-grained, partial relevance between videos and queries. Experiment results demonstrate that our proposed model achieves state-of-the-art performance on TVR, ActivityNet, and Charades-STA datasets for PRVR. The code is available at https://github.com/HuiGuanLab/DL-DKD.

### 7. EVOLUTION OF META'S LLAMA MODELS AND PARAMETER-EFFICIENT FINE-TUNING OF LARGE LANGUAGE MODELS: A SURVEY

**主要机构**: Torrens University, University of Technology Sydney, Tehran University, TU Dortmund University, University of Kurdistan Sanandaj, Queen Mary University, University of Kurdistan Hewler
**作者数量**: 11人

**摘要**:
This review surveys the rapid evolution of Meta AI's LLaMA (Large Language Model Meta AI) series-from LLaMA 1 through LLaMA 4 and the specialized parameter-efficient fine-tuning (PEFT) methods developed for these models. We first describe the LLaMA family of foundation models (7B-65B to 288B parameters), their architectures (including native multimodal and Mixtureof-Experts variants), and key performance characteristics. We then describe and discuss the concept of PEFT, which adapts large pre-trained models by updating only a small subset of parameters, and review five PEFT methods that have been applied to LLaMA: LoRA (Low-Rank Adaptation), LLaMA-Adapter V1 and V2, LLaMA-Excitor, and QLoRA (Quantized LoRA). We discuss each method's mechanism, parameter savings, and example application to LLaMA (e.g., instruction tuning, multimodal tasks). We provide structured discussion and analysis of model and adapter architectures, parameter counts, and benchmark results (including examples where fine-tuned LLaMA models outperform larger baselines). Finally, we examine real-world use cases where LLaMA-based models and PEFT have been successfully applied (e.g., legal and medical domains), and we discuss ongoing challenges and future research directions (such as scaling to even larger contexts and improving robustness). This survey paper provides a one-stop resource for ML researchers and practitioners interested in LLaMA models and efficient fine-tuning strategies.

### 8. FLASHVSR: TOWARDS REAL-TIME DIFFUSION-BASED STREAMING VIDEO SUPER-RESOLUTION

**主要机构**: University of Hong Kong, Tsinghua University, Shanghai Artificial Intelligence Laboratory, The Chinese
**作者数量**: 7人

**摘要**:
Diffusion models have recently advanced video restoration, but applying them to real-world video super-resolution (VSR) remains challenging due to high latency, prohibitive computation, and poor generalization to ultra-high resolutions. Our goal in this work is to make diffusion-based VSR practical by achieving efficiency, scalability, and real-time performance. To this end, we propose FlashVSR, the first diffusion-based one-step streaming framework towards realtime VSR. FlashVSR runs at ∼17 FPS for 768 × 1408 videos on a single A100 GPU by combining three complementary innovations: (i) a train-friendly threestage distillation pipeline that enables streaming super-resolution, (ii) localityconstrained sparse attention that cuts redundant computation while bridging the train-test resolution gap, and (iii) a tiny conditional decoder that accelerates reconstruction without sacrificing quality. To support large-scale training, we also construct VSR-120K, a new dataset with 120k videos and 180k images. Extensive experiments show that FlashVSR scales reliably to ultra-high resolutions and achieves state-of-the-art performance with up to ∼ 12× speedup over prior one-step diffusion VSR models. We will release the code, pretrained models, and dataset to foster future research in efficient diffusion-based VSR at https://zhuang2002.github.io/FlashVSR.

### 9. LiteVPNet: A Lightweight Network for Video Encoding Control in Quality-Critical Applications

**主要机构**: Department of Electronic and Electrical Engineering, Trinity College Dublin, Sigmedia Group
**作者数量**: 2人

**摘要**:
In the last decade, video workflows in the cinema production ecosystem have presented new use cases for video streaming technology. These new workflows, e.g. in Onset Virtual Production, present the challenge of requiring precise quality control and energy efficiency. Existing approaches to transcoding often fall short of these requirements, either due to a lack of quality control or computational overhead. To fill this gap, we present a lightweight neural network (LiteVPNet) for accurately predicting Quantisation Parameters for NVENC AV1 encoders that achieve a specified VMAF score. We use low-complexity features including bitstream characteristics, video complexity measures, and CLIP-based semantic embeddings. Our results demonstrate that LiteVPNet achieves mean VMAF errors below 1.2 points across a wide range of quality targets. Notably, LiteVPNet achieves VMAF errors within 2 points for over 87% of our test corpus, c.f. ≈61% with state-of-the-art methods. LiteVPNet's performance across various quality regions highlights its applicability for enhancing high-value content transport and streaming for more energy-efficient, high-quality media experiences.

### 10. MoBiLE: Efficient Mixture-of-Experts Inference on Consumer GPU with Mixture of Big Little Experts

**主要机构**: Tsinghua University, BNRist
**作者数量**: 8人

**摘要**:
Mixture-of-Experts (MoE) models have recently demonstrated exceptional performance across a diverse range of applications. The principle of sparse activation in MoE models facilitates an offloading strategy, wherein active experts are maintained in GPU HBM, while inactive experts are stored in CPU DRAM. The efficacy of this approach, however, is fundamentally constrained by the limited bandwidth of the CPU-GPU interconnect. To mitigate this bottleneck, existing approaches have employed prefetching to accelerate MoE inference. These methods attempt to predict and prefetch the required experts using specially trained modules. Nevertheless, such techniques are often encumbered by significant training overhead and have shown diminished effectiveness on recent MoE models with fine-grained expert segmentation. In this paper, we propose MoBiLE, a plug-and-play offloadingbased MoE inference framework with mixture of big-little experts. It reduces the number of experts for unimportant tokens to half for acceleration while maintaining full experts for important tokens to guarantee model quality. Further, a dedicated fallback and prefetching mechanism is designed for switching between little and big experts to improve memory efficiency. We evaluate MoBiLE on four typical modern MoE architectures and challenging generative tasks. Our results show that MoBiLE achieves a speedup of 1.60× to 1.72× compared to the baseline on a consumer GPU system, with negligible degradation in accuracy. Index Terms-large language model, mixture-of-experts, offloading, algorithm-system co-design.

### 11. MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics

**主要机构**: University of Artificial Intelligence {Bowei.Guo, Shengkun.Tang, Mohamed bin Zayed
**作者数量**: 4人

**摘要**:


### 12. On the Use of Hierarchical Vision Foundation Models for Low-Cost Human Mesh Recovery and Pose Estimation

**主要机构**: Tokyo Metropolitan University
**作者数量**: 2人

**摘要**:


### 13. PAGS: PRIORITY-ADAPTIVE GAUSSIAN SPLATTING FOR DYNAMIC DRIVING SCENES

**主要机构**: Harbin Institute of Technology
**作者数量**: 5人

**摘要**:
Reconstructing dynamic 3D urban scenes is crucial for autonomous driving, yet current methods face a stark trade-off between fidelity and computational cost. This inefficiency stems from their semantically agnostic design, which allocates resources uniformly, treating static backgrounds and safety-critical objects with equal importance. To address this, we introduce Priority-Adaptive Gaussian Splatting (PAGS), a framework that injects task-aware semantic priorities directly into the 3D reconstruction and rendering pipeline. PAGS introduces two core contributions: (1) Semantically-Guided Pruning and Regularization strategy, which employs a hybrid importance metric to aggressively simplify non-critical scene elements while preserving fine-grained details on objects vital for navigation. (2) Priority-Driven Rendering pipeline, which employs a prioritybased depth pre-pass to aggressively cull occluded primitives and accelerate the final shading computations. Extensive experiments on the Waymo and KITTI datasets demonstrate that PAGS achieves exceptional reconstruction quality, particularly on safety-critical objects, while significantly reducing training time and boosting rendering speeds to over 350 FPS.

### 14. RETHINKING KNOWLEDGE DISTILLATION: A DATA DEPENDENT REGULARISER WITH A NEGATIVE ASYMMETRIC PAYOFF

**主要机构**: Queen Mary University of London London, UKRI Safe and Trustd AI Imperial and King', King's
**作者数量**: 5人

**摘要**:
Knowledge distillation is often considered a compression mechanism when judged on the resulting student's accuracy and loss, yet its functional impact is poorly understood. In this work, we quantify the compression capacity of knowledge distillation and the resulting knowledge transfer from a functional perspective, decoupling compression from architectural reduction, which provides an improved understanding of knowledge distillation. We employ hypothesis testing, controls, and random control distillation to understand knowledge transfer mechanisms across data modalities. To rigorously test the breadth and limits of our analyses, we explore multiple distillation variants and analyse distillation scaling laws across model sizes. Our findings demonstrate that, while there is statistically significant knowledge transfer in some modalities and architectures, the extent of this transfer is less pronounced than anticipated, even under conditions designed to maximise knowledge sharing. Notably, in cases of significant knowledge transfer, we identify a consistent and severe asymmetric transfer of negative knowledge to the student, raising safety concerns in knowledge distillation applications. Across 12 experimental setups, 9 architectures, and 7 datasets, our findings show that knowledge distillation functions less as a compression mechanism and more as a data-dependent regulariser with a negative asymmetric payoff. * Equal Contribution.

### 15. SMEC:Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression

**主要机构**: Taobao & Tmall Group of Alibaba Beijing, Tmall Group of Alibaba Hangzhou
**作者数量**: 4人

**摘要**:
Large language models (LLMs) generate highdimensional embeddings that capture rich semantic and syntactic information. However, high-dimensional embeddings exacerbate computational complexity and storage requirements, thereby hindering practical deployment. To address these challenges, we propose a novel training framework named Sequential Matryoshka Embedding Compression (SMEC). This framework introduces the Sequential Matryoshka Representation Learning(SMRL) method to mitigate gradient variance during training, the Adaptive Dimension Selection (ADS) module to reduce information degradation during dimension pruning, and the Selectable Cross-batch Memory (S-XBM) module to enhance unsupervised learning between high-and low-dimensional embeddings. Experiments on image, text, and multimodal datasets demonstrate that SMEC achieves significant dimensionality reduction while maintaining performance. For instance, on the BEIR dataset, our approach improves the performance of compressed LLM2Vec embeddings (256 dimensions) by 1.1 points and 2.7 points compared to the Matryoshka-Adaptor and Search-Adaptor models, respectively.

### 16. UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering

**主要机构**: 
**作者数量**: 7人

**摘要**:


### 17. ZERO-SHOT CFC: FAST REAL-WORLD IMAGE DENOISING BASED ON CROSS-FREQUENCY CONSISTENCY

**主要机构**: School of Information Science and Technology, Beijing University of Technology
**作者数量**: 3人

**摘要**:
Zero-shot denoisers address the dataset dependency of deep-learning-based denoisers, enabling the denoising of unseen single images. Nonetheless, existing zero-shot methods suffer from long training times and rely on the assumption of noise independence and a zero-mean property, limiting their effectiveness in real-world denoising scenarios where noise characteristics are more complicated. This paper proposes an efficient and effective method for real-world denoising, the Zero-Shot denoiser based on Cross-Frequency Consistency (ZSCFC), which enables training and denoising with a single noisy image and does not rely on assumptions about noise distribution. Specifically, image textures exhibit position similarity and content consistency across different frequency bands, while noise does not. Based on this property, we developed cross-frequency consistency loss and an ultralight network to realize image denoising. Experiments on various real-world image datasets demonstrate that our ZSCFC outperforms other state-of-the-art zero-shot methods in terms of computational efficiency and denoising performance.
