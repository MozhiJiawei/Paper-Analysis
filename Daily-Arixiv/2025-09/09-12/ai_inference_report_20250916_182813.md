# AI推理加速技术论文分析报告
生成时间: 2025-09-16 18:28:13
分析论文数量: 22篇

## 论文技术简报

### 1. AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models

浦项科技大学（POSTECH）发布了《AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models》论文，使用AMQ框架（整合搜索空间剪枝、量化代理、质量预测器及迭代搜索更新策略的AutoML技术），解决了大语言模型在严格内存约束下因组合搜索空间过大（超10^100种配置）导致传统黑盒优化不可行、难以找到最佳性能模型的问题，达成了高效探索质量-效率空间、达到帕累托前沿、获得紧凑且高性能大语言模型的效果。

### 2. AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs

Wadhwani School of Data Science and Artificial Intelligence发布了AQUA论文，使用基于查询向量幅度的注意力近似策略（通过SVD离线计算通用投影矩阵、在线动态选择稀疏维度），解决了注意力二次复杂度导致的LLM长上下文计算与内存瓶颈，达成了在Llama-3.1-8B上减少25%注意力点积计算且性能影响统计不显著、并减少KV缓存内存的效果

### 3. A GPU-Accelerated RAG-Based Telegram Assistant for Supporting Parallel Processing Students

Ben-Gurion University of the Negev发布了基于GPU加速的RAG系统的论文，该系统采用量化Mistral-7B Instruct模型并部署为Telegram机器人，解决了学生课后持续、按需学术帮助的需求，达成了实时个性化课程响应且通过GPU加速显著降低推理延迟、实现消费级硬件实用部署的效果。

### 4. Difficulty-Aware Agent Orchestration in LLM-Powered Workflows

使用DAAO动态框架（含VAE难度估计、模块化算子分配器及成本-性能感知LLM路由器），解决了现有多智能体框架静态工作流导致简单查询过度处理、复杂查询性能不足及忽略异构LLMs效率-性能权衡的问题，达成了在六个基准测试中准确性和推理效率优于现有多智能体系统的效果。

### 5. Dynamic Adaptive Shared Experts with Grouped Multi-Head Attention Mixture of Experts

研究团队发布了《Dynamic Adaptive Shared Experts with Grouped Multi-Head Attention Mixture of Experts》论文，使用Grouped Multi-Head Attention (GMHA)、Dual-Scale Shared Expert Structure (DSSE)及分层Adaptive Dynamic Routing (ADR)机制的DASG-MoE混合模型，解决了现有MoE架构Transformer在计算效率、长程依赖捕捉及专家资源分配动态适应性上的不足，在多个长序列基准数据集上优于最先进模型。

### 6. EfficientUICoder: Efficient MLLM-based UI Code Generation via Input and Output Token Compression

香港中文大学、南洋理工大学发布了EfficientUICoder论文，使用基于输入输出token压缩的EfficientUICoder框架（含元素布局感知压缩、区域感知精炼及自适应重复抑制），解决了UI2Code任务中输入图像与输出代码token冗余导致的高计算开销及生成代码过长无效问题，达成55%-60%压缩率且不影响网页质量，计算成本降低44.9%、推理时间减少48.8%。

### 7. Enriched text-guided variational multimodal knowledge distillation network for automated diagnosis of plaque in 3D carotid artery MRI

研究团队发布了颈动脉3D MRI斑块易损性自动化诊断论文，使用变分多模态知识蒸馏（VMD）技术，解决了利用有限标注数据和放射学报告提升未标注3D MRI图像诊断准确性的挑战，达成了增强诊断网络对未标注3D MRI图像准确性的效果。

### 8. GAPRUNE: GRADIENT-ALIGNMENT PRUNING FOR DOMAIN-AWARE EMBEDDINGS

香港科技大学发布了GAPRUNE论文，使用结合Fisher信息衡量领域重要性与通用领域梯度对齐评估参数行为的Domain Alignment Importance (DAI)评分的梯度对齐剪枝框架，解决了现有剪枝方法未区分通用语义与领域特定模式导致的次优剪枝决策问题，在FinMTEB和ChemTEB领域基准上，50%稀疏度下单次剪枝保持性能在2.5%内且优于基线，100步重训练后分别提升+4.51%和+1.73%。

### 9. Investigating the Lottery Ticket Hypothesis for Variational Quantum Circuits

研究团队发布了将彩票 ticket 假设（LTH）应用于变分量子电路（VQCs）的论文，使用LTH技术，解决了VQCs的贫瘠高原现象，达成了弱LTH下保留26.0%参数、强LTH下二进制VQC用45%权重实现100%准确率，减少参数同时保留性能以缓解贫瘠高原的效果。

### 10. Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction

哈尔滨工业大学发布了Judge Q论文，使用通过软令牌列表训练可学习查询（仅微调嵌入层）的技术，解决了KV缓存驱逐中现有方法过度关注局部信息、忽略全局信息导致的性能下降问题，达成了在LongBench上提升约1点、RULER上超过3点的效果

### 11. Know What You Don't Know: Selective Prediction for Early Exit DNNs

IIT Bombay发布了《Know What You Don't Know: Selective Prediction for Early Exit DNNs》论文，使用SPEED方法（结合选择性预测与延迟分类器），解决了早期退出DNNs因过度自信导致的不可靠问题，达成了减少50%错误预测并实现2.05×加速的效果

### 12. LIGHTWEIGHT METADATA-AWARE MIXTURE-OF-EXPERTS MASKED AUTOENCODER FOR EARTH OBSERVATION

European Centre for Medium-Range Weather Forecasts发布了LIGHTWEIGHT METADATA-AWARE MIXTURE-OF-EXPERTS MASKED AUTOENCODER FOR EARTH OBSERVATION论文，使用Metadata-aware Mixture-of-Experts Masked Autoencoder (MoE-MAE)技术（结合稀疏专家路由与地理时间条件，仅2.5M参数），解决了现有大型地球观测基础模型计算昂贵、限制可访问性与下游复用的问题，达成了以小参数实现与更大架构竞争的性能，提升迁移和标签效率且在无显式元数据数据集上仍具竞争力的效果。

### 13. NEUROGAZE-DISTILL: BRAIN-INFORMED DISTILLATION AND DEPRESSION-INSPIRED GEOMETRIC PRIORS FOR ROBUST FACIAL EMOTION RECOGNITION

东华大学发布了NEUROGAZE-DISTILL论文，使用神经信息蒸馏（含静态V/A原型网格）与抑郁启发几何先验（D-Geo）的跨模态蒸馏技术，解决了面部表情识别模型因面部外观间接有偏导致的跨数据集泛化能力差的问题，达成了提升鲁棒性、在域内及跨数据集上表现更优且无需复杂架构即可部署的效果

### 14. OPTIMAL BRAIN RESTORATION FOR JOINT QUANTIZA-TION AND SPARSIFICATION OF LLMS

研究团队发布了《OPTIMAL BRAIN RESTORATION FOR JOINT QUANTIZATION AND SPARSIFICATION OF LLMS》论文，使用Optimal Brain Restoration (OBR)训练无关框架，解决了量化与稀疏化联合时权重分布要求冲突（量化需紧凑范围、剪枝需高方差）的问题，达成了LLMs激进W4A4KV4量化与50%稀疏度、相比FP16密集基线高达4.72倍加速和6.4倍内存减少的效果。

### 15. RESOURCE-AWARE NEURAL NETWORK PRUNING USING GRAPH-BASED REINFORCEMENT LEARNING A PREPRINT

University of Antwerp -imec发布了RESOURCE-AWARE NEURAL NETWORK PRUNING USING GRAPH-BASED REINFORCEMENT LEARNING论文，使用基于图的强化学习（整合图表示、GAT编码器、二进制动作空间和CMDP框架）技术，解决了传统剪枝依赖手工启发式和局部优化导致的次优性能与低效策略问题，达成了在CIFAR-10、CIFAR-100和ImageNet上超越传统技术的SOTA效果，学习任务特定剪枝策略识别功能冗余连接。

### 16. SA-UNETV2: RETHINKING SPATIAL ATTENTION U-NET FOR RETINAL VESSEL SEGMENTATION

江西师范大学发布了SA-UNetv2论文，使用在所有跳跃连接中注入跨尺度空间注意力并采用加权BCE+MCC损失的技术，解决了SA-UNet在跳跃连接中注意力利用不足及前景-背景不平衡问题，达成了在DRIVE和STARE数据集上实现SOTA性能且模型轻量（1.2MB内存、0.26M参数）、CPU推理仅需1秒的效果。

### 17. Spec-LLaVA: Accelerating Vision-Language Models with Dynamic Tree-Based Speculative Decoding

研究团队发布了Spec-LLaVA论文，使用动态树结构的推测解码技术，解决了视觉语言模型自回归推理慢限制实时应用部署的问题，达成了在LLaVA-1.5（7B、13B）上实现高达3.28倍解码加速且生成质量无损失的效果。

### 18. SpeCa: Accelerating Diffusion Transformers with Speculative Feature Caching

清华大学发布了SpeCa论文，使用基于推测特征缓存的“预测-验证”加速框架（含推测采样和样本自适应计算分配），解决了扩散模型时间依赖性无法并行化及去噪步骤计算密集的问题，实现FLUX 6.34×加速（质量损失5.5%）、DiT 7.3×加速（保真度保持）等显著加速效果。

### 19. Under review as a conference paper SPECVLM: FAST SPECULATIVE DECODING IN VISION-LANGUAGE MODELS

西安交通大学与AMD发布了SpecVLM论文，使用弹性视觉压缩器和在线logit蒸馏协议，解决了视觉语言模型投机解码中预填充阶段视觉令牌计算与内存膨胀问题，达成在LLaVA和MMMU上2.5-2.9倍端到端加速且无损解码的效果

### 20. SVR-GS: Spatially Variant Regularization for Probabilistic Masks in 3D Gaussian Splatting

Dolby Laboratories发布了SVR-GS论文，使用空间变体正则化技术，解决了3D Gaussian Splatting中现有掩码修剪全局均值正则化与局部像素/光线重建损失不匹配的问题，达成平均减少高斯数量1.79×（vs MaskGS）和5.63×（vs 3DGS）、PSNR仅分别下降0.50dB和0.40dB的效果。

### 21. Tenma: Robust Cross-Embodiment Robot Manipulation with Diffusion Transformer

北京大学、清华大学发布了Tenma论文，使用跨实体归一化器、联合状态-时间编码器及扩散动作解码器等轻量化扩散Transformer技术，解决了结合Transformer策略与扩散模型在轻量级跨实体学习中的挑战，达成在分布内平均成功率88.95%、远超基线18.12%的效果。

### 22. ToMA: Token Merge with Attention for Diffusion Models

纽约大学发布了ToMA论文，使用通过子模优化选择多样token、GPU友好矩阵操作实现merge/unmerge及利用潜在局部性和序列冗余的Token Merge with Attention技术，解决了扩散模型中transformers因二次注意力复杂度及现有token reduction方法GPU低效操作导致的扩展性限制和理论加速抵消问题，达成了SDXL/Flux生成延迟降低24%/23%（DINO ∆ < 0.07）且优于先前方法的效果。

## 论文详细信息

### 1. AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models

**主要机构**: Graduate School of Artificial Intelligence, University of Science and Technology (POSTECH), Department of Convergence IT Engineering Pohang
**作者数量**: 5人

**摘要**:
To enable broader deployment of Large Language Models (LLMs), it is essential to identify the best-performing model under strict memory constraints. We present AMQ, Automated Mixed-Precision Weight-Only Quantization, a framework that assigns layer-wise quantization bit-widths to optimally balance model quality and memory usage. However, the combinatorial search space, with over 10 100 possible configurations, makes conventional black-box optimization infeasible. AMQ overcomes this challenge through four key innovations: (1) search space pruning using prior knowledge to exclude unpromising configurations, (2) quantization proxy to bypass costly format conversions during search, (3) quality predictor to minimize evaluation overhead, and (4) iterative search-and-update strategy for fast and stable convergence. By integrating these components, AMQ efficiently explores the quality-efficiency landscape, reaching the Pareto frontier and yielding LLMs that are both compact and high-performing. Our code is available at https://github.com/dlwns147/amq.

### 2. AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs

**主要机构**: Department of Electrical Engineering, Wadhwani School of Data Science and Artificial Intelligence, Centre for Responsible AI, Indian Institute of Technology Madras
**作者数量**: 3人

**摘要**:
The quadratic complexity of the attention mechanism remains a fundamental barrier to scaling Large Language Models (LLMs) to longer contexts, creating a critical bottleneck in both computation and memory. To address this, we introduce AQUA (Attention via QUery mAgnitudes) a novel and versatile approximation strategy that significantly reduces the cost of attention with a graceful performance trade-off. Our method operates in two phases: an efficient offline step where we compute a universal, language agnostic projection matrix via SVD on a calibration dataset, and an online inference step where we project query and key vectors and dynamically select a sparse subset of dimensions based on the query's magnitude. We provide a formal theoretical analysis of AQUA, establishing the breakeven point at which it becomes more computationally efficient than standard attention. Our empirical evaluations on state-of-the-art models like Llama-3.1-8B demonstrate that a 25% reduction in the attention dot-product computation can be achieved with a statistically insignificant impact on performance across a wide range of benchmarks. We further showcase the versatility of AQUA by demonstrating its ability to synergistically accelerate existing token eviction methods like H2O and to directly reduce KV-cache memory size. By offering a controllable knob to balance efficiency and accuracy, AQUA provides a practical and powerful tool for making large-scale LLM inference more accessible and sustainable.

### 3. A GPU-Accelerated RAG-Based Telegram Assistant for Supporting Parallel Processing Students

**主要机构**: Ben-Gurion University of the Negev Beer Sheva
**作者数量**: 1人

**摘要**:
This project addresses a critical pedagogical need: offering students continuous, on-demand academic assistance beyond conventional reception hours. I present a domain-specific Retrieval-Augmented Generation (RAG) system powered by a quantized Mistral-7B Instruct model[4] and deployed as a Telegram bot[9]. The assistant enhances learning by delivering real-time, personalized responses aligned with the "Introduction to Parallel Processing" course materials [1]. GPU acceleration significantly improves inference latency, enabling practical deployment on consumer hardware. This approach demonstrates how consumer GPUs can enable affordable, private, and effective AI tutoring for HPC education.

### 4. Difficulty-Aware Agent Orchestration in LLM-Powered Workflows

**主要机构**: 
**作者数量**: 7人

**摘要**:
Large Language Model (LLM)-based agentic systems have shown strong capabilities across various tasks. However, existing multi-agent frameworks often rely on static or tasklevel workflows, which either over-process simple queries or underperform on complex ones, while also neglecting the efficiency-performance trade-offs across heterogeneous LLMs. To address these limitations, we propose Difficulty-Aware Agentic Orchestration (DAAO), a dynamic framework that adapts workflow depth, operator selection, and LLM assignment based on the difficulty of each input query. DAAO comprises three interdependent modules: a variational autoencoder (VAE) for difficulty estimation, a modular operator allocator, and a cost-and performance-aware LLM router. By leveraging heterogeneous LLMs and dynamically tailoring workflows, DAAO enables fine-grained, query-specific reasoning strategies. DAAO outperforms prior multi-agent systems in both accuracy and inference efficiency across six benchmarks. We will release our code and implementation details upon publication.

### 5. Dynamic Adaptive Shared Experts with Grouped Multi-Head Attention Mixture of Experts

**主要机构**: 
**作者数量**: 5人

**摘要**:
Transformer models based on the Mixture of Experts (MoE) architecture have made significant progress in long-sequence modeling, but existing models still have shortcomings in computational efficiency and the ability to capture long-range dependencies, especially in terms of the dynamic adaptability of expert resource allocation. In this paper, we propose a Dynamic Adaptive Shared Expert and Grouped Multi-Head Attention Hybrid Model (DASG-MoE) to enhance long-sequence modeling capabilities by integrating three modules. First, we employ the Grouped Multi-Head Attention (GMHA) mechanism to effectively reduce the computational complexity of long sequences. By parallel processing through sequence grouping, local sliding window attention, and feature aggregation, we address long-range dependency issues and the model's lack of generalization for local information. Second, we design a Dual-Scale Shared Expert Structure (DSSE), where shallow experts use lightweight computations to quickly respond to low-dimensional features, while deep experts process high-dimensional complex semantics through pre-training transfer and post-training optimization, achieving a dynamic balance between efficiency and accuracy. Third, we propose a hierarchical Adaptive Dynamic Routing (ADR) mechanism that dynamically selects expert levels based on feature complexity and task requirements, and optimizes resource allocation through a local expert activation strategy. Experiments on multiple long-sequence benchmark datasets demonstrate that our DASG-MoE model outperforms state-of-the-art models.

### 6. EfficientUICoder: Efficient MLLM-based UI Code Generation via Input and Output Token Compression

**主要机构**: ZHONGYI ZHANG, The Chinese University of Hong Kong, Nanyang Technological University, Huazhong University of Science and Technology, YUXUAN WAN, YINTONG HUO *, Singapore Management University
**作者数量**: 16人

**摘要**:
Multimodal Large Language Models have demonstrated exceptional performance in UI2Code tasks, significantly enhancing website development efficiency. However, these tasks incur substantially higher computational overhead than traditional code generation due to the large number of input image tokens and extensive output code tokens required. Our comprehensive study identifies significant redundancies in both image and code tokens that exacerbate computational complexity and hinder focus on key UI elements, resulting in excessively lengthy and often invalid HTML files. We propose EfficientUICoder, a compression framework for efficient UI code generation with three key components. First, Element and Layout-aware Token Compression preserves essential UI information by detecting element regions and constructing UI element trees. Second, Region-aware Token Refinement leverages attention scores to discard low-attention tokens from selected regions while integrating high-attention tokens from unselected regions. Third, Adaptive Duplicate Token Suppression dynamically reduces repetitive generation by tracking HTML/CSS structure frequencies and applying exponential penalties. Extensive experiments show EfficientUICoderachieves a 55%-60% compression ratio without compromising webpage quality and delivers superior efficiency improvements: reducing computational cost by 44.9%, generated tokens by 41.4%, prefill time by 46.6%, and inference time by 48.8% on 34B-level MLLMs. Code is available at https://github.com/WebPAI/EfficientUICoder. CCS Concepts: • Software and its engineering → Automatic programming; • Computing methodologies → Artificial intelligence.

### 7. Enriched text-guided variational multimodal knowledge distillation network for automated diagnosis of plaque in 3D carotid artery MRI

**主要机构**: 
**作者数量**: 8人

**摘要**:
Multimodal learning has attracted much attention in recent years due to its ability to effectively utilize data features from a variety of different modalities. Diagnosing the vulnerability of atherosclerotic plaques directly from carotid 3D MRI images is relatively challenging for both radiologists and conventional 3D vision networks. In clinical practice, radiologists assess patient conditions using a multimodal approach that incorporates various imaging modalities and domain-specific expertise, paving the way for the creation of multimodal diagnostic networks. In this paper, we have developed an effective strategy to leverage radiologists' domain knowledge to automate the diagnosis of carotid plaque vulnerability through Variation inference and Multimodal knowledge Distillation (VMD). This method excels in harnessing cross-modality prior knowledge from limited image annotations and radiology reports within training data, thereby enhancing the diagnostic network's accuracy for unannotated 3D MRI images. We conducted in-depth experiments on the dataset collected in-house and verified the effectiveness of the VMD strategy we proposed.

### 8. GAPRUNE: GRADIENT-ALIGNMENT PRUNING FOR DOMAIN-AWARE EMBEDDINGS

**主要机构**: The Hong Kong University of Science and Technology
**作者数量**: 2人

**摘要**:
Domain-specific embedding models have shown promise for applications that require specialized semantic understanding, such as coding agents and financial retrieval systems, often achieving higher performance gains than general models. However, state-of-the-art embedding models are typically based on LLMs, which contain billions of parameters, making deployment challenging in resourceconstrained environments. Model compression through pruning offers a promising solution, but existing pruning methods treat all parameters uniformly, failing to distinguish between general semantic representations and domain-specific patterns, leading to suboptimal pruning decisions. Thus, we propose GAPrune, a pruning framework that addresses this challenge by considering both domain importance and preserving general linguistic foundation. Our method uses Fisher Information to measure importance and general-domain gradient alignment to assess parameter behavior, then combines these signals using our Domain Alignment Importance (DAI) scoring. Lower DAI scores indicate that the parameter is either less important for the domain task or creates conflicts between domain and general objectives. Experiments on two domain benchmarks, FinMTEB and ChemTEB, show that GAPrune maintains performance within 2.5% of dense models in oneshot pruning at 50% sparsity, while outperforming all baselines. With retraining in 100 steps, GAPrune achieves +4.51% improvement on FinMTEB and +1.73% on ChemTEB, demonstrating that our pruning strategy not only preserves but enhances domain-specific capabilities. Our findings demonstrate that principled pruning strategies can achieve model compression and enhanced domain specialization, providing the research community with a new approach for development 1 .

### 9. Investigating the Lottery Ticket Hypothesis for Variational Quantum Circuits

**主要机构**: 
**作者数量**: 6人

**摘要**:
Quantum computing is an emerging field in computer science that has seen considerable progress in recent years, especially in machine learning. By harnessing the principles of quantum physics, it can surpass the limitations of classical algorithms. However, variational quantum circuits (VQCs), which rely on adjustable parameters, often face the barren plateau phenomenon, hindering optimization. The Lottery Ticket Hypothesis (LTH) is a recent concept in classical machine learning that has led to notable improvements in parameter efficiency for neural networks. It states that within a large network, a smaller, more efficient subnetwork, or "winning ticket," can achieve comparable performance, potentially circumventing plateau challenges. In this work, we investigate whether this idea can apply to VQCs. We show that the weak LTH holds for VQCs, revealing winning tickets that retain just 26.0% of the original parameters. For the strong LTH, where a pruning mask is learned without any training, we discovered a winning ticket in a binary VQC, achieving 100% accuracy with only 45% of the weights. These findings indicate that LTH may mitigate barren plateaus by reducing parameter counts while preserving performance, thus enhancing the efficiency of VQCs in quantum machine learning tasks.

### 10. Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction

**主要机构**: Research Center for Social Computing and Interactive Robotics, Harbin Institute of Technology
**作者数量**: 7人

**摘要**:
Large language models (LLMs) utilize key-value (KV) cache to store historical information during sequence processing. The size of KV cache grows linearly as the length of the sequence extends, which seriously affects memory usage and decoding efficiency. Current methods for KV cache eviction typically utilize the last window from the pre-filling phase as queries to compute the KV importance scores for eviction. Although this scheme is simple to implement, it tends to overly focus on local information, potentially leading to the neglect or omission of crucial global information. To mitigate this issue, we propose Judge Q, a novel training method which incorporates a soft token list. This method only tunes the model's embedding layer at a low training cost. By concatenating the soft token list at the end of the input sequence, we train these tokens' attention map to the original input sequence to align with that of the actual decoded tokens. In this way, the queries corresponding to the soft tokens can effectively capture global information and better evaluate the importance of the keys and values within the KV cache, thus maintaining decoding quality when KV cache is evicted. Under the same eviction budget, our method exhibits less performance degradation compared to existing eviction approaches. We validate our approach through experiments conducted on models such as Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, using benchmarks including LongBench, RULER, and Needle-in-a-Haystack. Results indicate an improvement of approximately 1 point on the LongBench and over 3 points on RULER. This proposed methodology can be seamlessly integrated into existing open-source models with minimal training overhead, thereby enhancing performance in KV cache eviction scenarios.

### 11. Know What You Don't Know: Selective Prediction for Early Exit DNNs

**主要机构**: Department of IEOR, IIT Bombay
**作者数量**: 2人

**摘要**:
Inference latency and trustworthiness of Deep Neural Networks (DNNs) are the bottlenecks in deploying them in critical applications like sensitive tasks. Early Exit (EE) DNNs overcome the latency issues by allowing samples to exit from intermediary layers if they attain 'high' confidence scores on the predicted class. However, the DNNs are known to exhibit overconfidence, which can lead to many samples exiting early and render EE strategies untrustworthy. We use Selective Prediction (SP) to overcome this issue by checking the 'hardness' of the samples rather than just relying on the confidence score alone. We propose SPEED, a novel approach that uses Deferral Classifiers (DCs) at each layer to check the hardness of samples before performing EEs. Specifically, the DCs identify if a sample is hard to predict at an intermediary layer, leading to hallucination, and defer it to an expert. Early detection of hard samples for inference prevents the wastage of computational resources and improves trust by deferring the hard samples to the expert. We demonstrate that EE aided with SP improves both accuracy and latency. Our method minimizes the risk of wrong prediction by 50% with a speedup of 2.05× as compared to the final layer. The anonymized source code is available at https://github.com/Div290/SPEED

### 12. LIGHTWEIGHT METADATA-AWARE MIXTURE-OF-EXPERTS MASKED AUTOENCODER FOR EARTH OBSERVATION

**主要机构**: European Centre for Medium-Range Weather Forecasts Bonn
**作者数量**: 1人

**摘要**:
Recent advances in Earth Observation have focused on large-scale foundation models. However, these models are computationally expensive, limiting their accessibility and reuse for downstream tasks. In this work, we investigate compact architectures as a practical pathway toward smaller general-purpose EO models. We propose a Metadata-aware Mixture-of-Experts Masked Autoencoder (MoE-MAE) with only 2.5M parameters. The model combines sparse expert routing with geotemporal conditioning, incorporating imagery alongside latitude/longitude and seasonal-daily cyclic encodings. We pretrain the MoE-MAE on the BigEarthNet-Landsat dataset and evaluate embeddings from its frozen encoder using linear probes. Despite its small size, the model competes with much larger architectures, demonstrating that metadata-aware pretraining improves transfer and label efficiency. To further assess generalization, we evaluate on the EuroSAT-Landsat dataset, which lacks explicit metadata, and still observe competitive performance compared to models with hundreds of millions of parameters. These results suggest that compact, metadata-aware MoE-MAEs are an efficient and scalable step toward future EO foundation models. Model's code and weights are available on this repository https://github.com/AlbughdadiM/geo-moe-mae.

### 13. NEUROGAZE-DISTILL: BRAIN-INFORMED DISTILLATION AND DEPRESSION-INSPIRED GEOMETRIC PRIORS FOR ROBUST FACIAL EMOTION RECOGNITION

**主要机构**: Donghua University, North China Electric Power University, Department of Computer Science, College of Information and Intelligent Science
**作者数量**: 8人

**摘要**:
Facial emotion recognition (FER) models trained only on pixels often fail to generalize across datasets because facial appearance is an indirect-and biased-proxy for underlying affect. We present NeuroGaze-Distill, a cross-modal distillation framework that transfers brain-informed priors into an image-only FER student via static Valence-Arousal (V/A) prototypes and a depressioninspired geometric prior (D-Geo). A teacher trained on EEG topographic maps from DREAMER and MAHNOB-HCI produces a consolidated 5×5 V/A prototype grid that is frozen and reused; no EEG-face pairing and no non-visual signals at deployment are required. The student (ResNet-18/50) is trained on FERPlus with conventional CE/KD and two lightweight regularizers: (i) Proto-KD (cosine) aligns student features to the static prototypes; (ii) D-Geo softly shapes the embedding geometry in line with affective findings often reported in depression research (e.g., anhedonia-like contraction in high-valence regions). We evaluate both within-domain (FERPlus validation) and cross-dataset protocols (AffectNet-mini; optional CK+), reporting standard 8-way scores alongside present-only Macro-F1 and balanced accuracy to fairly handle label-set mismatch. Ablations attribute consistent gains to prototypes and D-Geo, and favor 5×5 over denser grids for stability. The method is simple, deployable, and improves robustness without architectural complexity. * First author. † This work was completed while the author was affiliated with the School of Computer Science and Technology. Note on the name. "NeuroGaze-Distill" emphasizes neuro-informed distillation. Gaze heatmaps are optional and may be disabled in the final experiments; the "Gaze" term survives to reflect the broader privileged-signal design.

### 14. OPTIMAL BRAIN RESTORATION FOR JOINT QUANTIZA-TION AND SPARSIFICATION OF LLMS

**主要机构**: 
**作者数量**: 4人

**摘要**:
Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72× speedup and 6.4× memory reduction compared to the FP16-dense baseline.

### 15. RESOURCE-AWARE NEURAL NETWORK PRUNING USING GRAPH-BASED REINFORCEMENT LEARNING A PREPRINT

**主要机构**: University of Antwerp -imec, University of Antwerp, CoSysLab -Faculty of Applied Engineering, IDLab -Faculty of Applied Engineering
**作者数量**: 4人

**摘要**:
This paper presents a novel approach to neural network pruning by integrating a graph-based observation space into an AutoML framework to address the limitations of existing methods. Traditional pruning approaches often depend on hand-crafted heuristics and local optimization perspectives, which can lead to suboptimal performance and inefficient pruning strategies. Our framework transforms the pruning process by introducing a graph representation of the target neural network that captures complete topological relationships between layers and channels, replacing the limited layer-wise observation space with a global view of network structure. The core innovations include a Graph Attention Network (GAT) encoder that processes the network's graph representation and generates a rich embedding. Additionally, for the action space we transition from continuous pruning ratios to fine-grained binary action spaces which enables the agent to learn optimal channel importance criteria directly from data, moving away from predefined scoring functions. These contributions are modelled within a Constrained Markov Decision Process (CMDP) framework, allowing the agent to make informed pruning decisions while adhering to resource constraints such as target compression rates. For this, we design a self-competition reward system that encourages the agent to outperform its previous best performance while satisfying the defined constraints. We demonstrate the effectiveness of our approach through extensive experiments on benchmark datasets including CIFAR-10, CIFAR-100, and ImageNet. The experiments show that our method consistently outperforms traditional pruning techniques, showing state-of-the-art results while learning task-specific pruning strategies that identify functionally redundant connections beyond simple weight magnitude considerations.

### 16. SA-UNETV2: RETHINKING SPATIAL ATTENTION U-NET FOR RETINAL VESSEL SEGMENTATION

**主要机构**: School of Artificial Intelligence, Department of Applied Mathematics and Computer Science, Jiangxi Normal University, Technical University of Denmark
**作者数量**: 5人

**摘要**:
Retinal vessel segmentation is essential for early diagnosis of diseases such as diabetic retinopathy, hypertension, and neurodegenerative disorders. Although SA-UNet introduces spatial attention in the bottleneck, it underuses attention in skip connections and does not address the severe foreground-background imbalance. We propose SA-UNetv2, a lightweight model that injects cross-scale spatial attention into all skip connections to strengthen multi-scale feature fusion and adopts a weighted Binary Cross-Entropy (BCE) + Matthews Correlation Coefficient (MCC) loss to improve robustness to class imbalance. On the public DRIVE and STARE datasets, SA-UNetv2 achieves state-of-the-art performance with only 1.2MB memory and 0.26M parameters (less than 50% of SA-UNet), and 1 second CPU inference on 592×592×3 images, demonstrating strong efficiency and deployability in resource-constrained, CPU-only settings. The code is available at github.com/clguo/SA-UNetv2.

### 17. Spec-LLaVA: Accelerating Vision-Language Models with Dynamic Tree-Based Speculative Decoding

**主要机构**: 
**作者数量**: 7人

**摘要**:
Vision-Language Models (VLMs) enable powerful multimodal reasoning but suffer from slow autoregressive inference, limiting their deployment in real-time applications. We introduce Spec-LLaVA, a system that applies speculative decoding to accelerate VLMs without sacrificing output quality. Spec-LLaVA pairs a lightweight draft VLM with a large target model: the draft speculates future tokens, which the target verifies in parallel, allowing multiple tokens to be generated per step. To maximize efficiency, we design a dynamic tree-based verification algorithm that adaptively expands and prunes speculative branches using draft model confidence. On MS COCO out-of-domain images, Spec-LLaVA achieves up to 3.28× faster decoding on LLaVA-1.5 (7B, 13B) with no loss in generation quality. This work presents a lossless acceleration framework for VLMs using dynamic tree-structured speculative decoding, opening a path toward practical real-time multimodal assistants. Importantly, the lightweight draft model design makes the framework amenable to resource-constrained or on-device deployment settings.

### 18. SpeCa: Accelerating Diffusion Transformers with Speculative Feature Caching

**主要机构**: Tsinghua University Beijing, Shandong University Weihai, National University of Singapore Singapore Linfeng Zhang † Shanghai Jiao Tong University Shanghai, University of Electronic Science and Technology of China Chengdu, Shanghai Jiao Tong University Shanghai, The Hong Kong University of Science and Technology (Guangzhou) Guangzhou
**作者数量**: 13人

**摘要**:
Diffusion models have revolutionized high-fidelity image and video synthesis, yet their computational demands remain prohibitive for real-time applications. These models face two fundamental challenges: strict temporal dependencies preventing parallelization, and computationally intensive forward passes required at each denoising step. Drawing inspiration from speculative decoding in large language models, we present SpeCa, a novel "Forecast-then-verify" acceleration framework that effectively addresses both limitations. SpeCa's core innovation lies in introducing Speculative Sampling to diffusion models, predicting intermediate features for subsequent timesteps based on fully computed reference timesteps. Our approach implements a parameter-free verification mechanism that efficiently evaluates prediction reliability, enabling real-time decisions to accept or reject each prediction while incurring negligible computational overhead. Furthermore, SpeCa introduces sampleadaptive computation allocation that dynamically modulates resources based on generation complexity-allocating reduced computation for simpler samples while preserving intensive processing for complex instances. Experiments demonstrate 6.34× acceleration on FLUX with minimal quality degradation (5.5% drop), 7.3× speedup on DiT while preserving generation fidelity, and 79.84% VBench score at 6.1× acceleration for HunyuanVideo. The verification mechanism incurs minimal overhead (1.67%-3.5% of full

### 19. Under review as a conference paper SPECVLM: FAST SPECULATIVE DECODING IN VISION-LANGUAGE MODELS

**主要机构**: Institute of Artificial Intelligence and Robotics, Advanced Micro Devices, Inc, Xi'an Jiaotong University
**作者数量**: 7人

**摘要**:
Speculative decoding is a powerful way to accelerate autoregressive large language models (LLMs), but directly porting it to vision-language models (VLMs) faces unique systems constraints: the prefill stage is dominated by visual tokens whose count scales with image resolution and video length, inflating both compute and memory-especially the key-value (KV) cache. We study speculative decoding for VLMs and introduce SpecVLM, a practical system that (1) establishes a strong EAGLE-2-style baseline, EagleVLM, delivering 1.5-2.3× end-toend speedups over full autoregressive inference, and (2) further accelerates VLM inference with an elastic visual compressor that adaptively selects among pruning, pooling, convolution, and resampler primitives to balance FLOPs/parameters and accuracy per input. To avoid costly offline distillation corpora, we propose an online-logit distillation protocol that trains the draft model with onthe-fly teacher logits and penultimate features using a combined cross-entropy and Smooth L1 objective, eliminating storage and preprocessing while remaining compute-efficient. This protocol reveals a training-time scaling effect: longer online training monotonically increases the draft model's average accepted length, improving speculative efficiency. Empirically, SpecVLM achieves additional acceleration, culminating in 2.5-2.9× end-to-end speedups within 5 epochs across LLaVA and MMMU, consistently over resolutions and task difficulties, while preserving the target model's output distribution (lossless decoding). Our code is available at https://github.com/haiduo/SpecVLM.

### 20. SVR-GS: Spatially Variant Regularization for Probabilistic Masks in 3D Gaussian Splatting

**主要机构**: Department of Computer Science and Software Engineering, Department of Electrical, Dolby Laboratories, School of Information Technology, The University of Western Australia, Murdoch University, Electronics and Computer Engineering
**作者数量**: 6人

**摘要**:
3D Gaussian Splatting (3DGS) enables fast, highquality novel view synthesis but relies on densification followed by pruning to optimize the number of Gaussians. Existing mask-based pruning, such as MaskGS, regularizes the global mean of the mask, which is misaligned with the local perpixel (per-ray) reconstruction loss that determines image quality along individual camera rays. This paper introduces SVR-GS, a spatially variant regularizer that renders a per-pixel spatial mask from each Gaussian's effective contribution along the ray, thereby applying sparsity pressure where it matters: on lowimportance Gaussians. We explore three spatial-mask aggregation strategies, implement them in CUDA, and conduct a gradient analysis to motivate our final design. Extensive experiments on Tanks&Temples, Deep Blending, and Mip-NeRF360 datasets demonstrate that, on average across the three datasets, the proposed SVR-GS reduces the number of Gaussians by 1.79 × compared to MaskGS and 5.63 × compared to 3DGS, while incurring only 0.50 dB and 0.40 dB PSNR drops, respectively. These gains translate into significantly smaller, faster, and more memory-efficient models, making them well-suited for real-time applications such as robotics, AR/VR, and mobile perception. The code will be released upon publication.

### 21. Tenma: Robust Cross-Embodiment Robot Manipulation with Diffusion Transformer

**主要机构**: ZhiCheng AI, Peking University, Tsinghua University
**作者数量**: 6人

**摘要**:
Scaling Transformer policies and diffusion models has advanced robotic manipulation, yet combining these techniques in lightweight, cross-embodiment learning settings remains challenging. We study design choices that most affect stability and performance for diffusion-transformer policies trained on heterogeneous, multimodal robot data, and introduce Tenma, a lightweight diffusion-transformer for bi-manual arm control. Tenma integrates multiview RGB, proprioception, and language via a cross-embodiment normalizer that maps disparate state/action spaces into a shared latent space; a Joint State-Time encoder for temporally aligned observation learning with inference speed boosts; and a diffusion action decoder optimized training stability and learning capacity. Across benchmarks and under matched compute, Tenma achieves an average success rate of 88.95% in-distribution and maintains strong performance under object and scene shifts, substantially exceeding baseline policies whose best in-distribution average is 18.12%. Despite using moderate data scale, Tenma delivers robust manipulation and generalization, indicating the great potential for multimodal and cross-embodiment learning strategies for further augmenting the capacity of transformer-based imitation learning policies.

### 22. ToMA: Token Merge with Attention for Diffusion Models

**主要机构**: New York University, Yuxuan Xia, Department of Computer Science
**作者数量**: 4人

**摘要**:
Diffusion models excel in high-fidelity image generation but face scalability limits due to transformers' quadratic attention complexity. Plug-andplay token reduction methods like ToMeSD and ToFu reduce FLOPs by merging redundant tokens in generated images but rely on GPU-inefficient operations (e.g., sorting, scattered writes), introducing overheads that negate theoretical speedups when paired with optimized attention implementations (e.g., FlashAttention). To bridge this gap, we propose Token Merge with Attention (ToMA), an off-the-shelf method that redesigns token reduction for GPU-aligned efficiency, with three key contributions: 1) a reformulation of token merge as a submodular optimization problem to select diverse tokens; 2) merge/unmerge as an attention-like linear transformation via GPUfriendly matrix operations; and 3) exploiting latent locality and sequential redundancy (pattern reuse) to minimize overhead. ToMA reduces SDXL/Flux generation latency by 24%/23% (with DINO ∆ < 0.07), outperforming prior methods. This work bridges the gap between theoretical and practical efficiency for transformers in diffusion.
