# AI推理加速技术论文分析报告
生成时间: 2025-09-24 11:31:49
分析论文数量: 15篇

## 论文技术简报

### 1. CETUS: Causal Event-Driven Temporal Modeling With Unified Variable-Rate Scheduling

发布了CETUS论文，使用Variable-Rate Spatial Event Mamba架构（直接处理原始事件流、轻量级因果空间邻域编码器、Mamba状态空间模型及自适应速率控制器），解决了事件相机处理中窗口延迟与计算成本高的问题，达成窗口延迟和推理延迟的最优平衡效果。

### 2. Comprehensive Evaluation of CNN-Based Audio Tagging Models on Resource-Constrained Devices

Instituto Tecnologico de Informatica (ITI)发布了基于CNN的音频标签模型在资源受限设备上的综合评估论文，使用多种CNN架构（含PANNs 1D/2D模型、ConvNeXt改编模型、MobileNetV3及CNN9/CNN13）的综合评估、ONNX格式转换及24小时连续推理评估技术，解决了资源受限设备上CNN音频标签模型部署的计算效率与热管理问题，达成了保持一致推理延迟并有效管理热行为的效果。

### 3. CSMoE: An Efficient Remote Sensing Foundation Model with Soft Mixture-of-Experts

柏林工业大学发布了CSMoE论文，使用Soft混合专家机制与主题气候描述符驱动采样策略，解决了现有遥感基础模型计算复杂度高或表征能力有限的问题，达成了计算效率提升超两倍且保持或提升表征性能的效果

### 4. 

[请补充主要机构]发布了[请补充论文标题]论文，使用[请补充核心技术]技术，解决了[请补充核心问题]问题，达成了[请补充关键效果]效果

### 5. DIVE WITH GRT: DENSE VIDEO UNDERSTANDING WITH GATED RESIDUAL TOKENIZATION

普林斯顿大学发布了DIVE WITH GRT论文，使用Gated Residual Tokenization (GRT)技术，解决了高帧率视频理解中逐帧标记化的计算成本与标记冗余问题，在DIVE基准上优于更大VLLM基线且随FPS增加持续改进，实现高效高FPS视频理解

### 6. DSPC: DUAL-STAGE PROGRESSIVE COMPRESSION FRAMEWORK FOR EFFICIENT LONG-CONTEXT REASONING

浙江工业大学与A*STAR Amazon Binjiang人工智能研究院发布了DSPC论文，使用无训练的两阶段渐进压缩框架，解决了长提示导致的计算成本高问题，在Longbench数据集FewShot任务中用3×少的token达49.17性能，超SOTA的LongLLMLingua 7.76。

### 7. Improving 3D Gaussian Splatting Compression by Scene-Adaptive Lattice Vector Quantization

研究团队发布了《通过场景自适应晶格矢量量化改进3D高斯溅射压缩》论文，使用场景自适应晶格矢量量化（SALVQ）技术，解决了3D高斯溅射（3DGS）数据量大需压缩且现有方法依赖均匀标量量化（USQ）导致压缩效率不足的问题，达成了在最小修改和计算开销下提升率失真（R-D）性能，支持动态调整晶格密度以适应多比特率目标并减少训练时间和内存消耗的效果

### 8. InfraMind: A Novel Exploration-based GUI Agentic Framework for Mission-critical Industrial Management

南洋理工大学发布了InfraMind论文，使用探索式GUI代理框架（整合系统搜索式探索、内存驱动规划等创新模块），解决了工业管理中不熟悉元素理解、精度效率、状态定位等挑战，在开源和商业DCIM平台上任务成功率和操作效率优于现有框架。

### 9. MemGS: Memory-Efficient Gaussian Splatting for Real-Time SLAM

研究团队发布了MemGS: Memory-Efficient Gaussian Splatting for Real-Time SLAM论文，使用体素空间几何相似性合并冗余3D高斯基元和Patch-Grid点采样初始化3D高斯基元技术，解决了嵌入式平台SLAM中3D高斯基元冗余导致的GPU内存限制与性能质量权衡问题，达成了降低GPU内存使用且不影响运行时性能、提升渲染质量的效果。

### 10. MOCHA: Multi-modal Objects-aware Cross-arcHitecture Alignment

三星英国研发中心发布了MOCHA论文，使用对象级多模态语义知识蒸馏技术（通过翻译模块将学生特征映射到联合空间并结合双目标损失），解决了将大型视觉语言教师模型的区域级语义高效转移到轻量级纯视觉目标检测器学生模型且无需修改教师或推理时文本输入的问题，达成了在少样本个性化检测基准上平均分数提升+10.1、性能与更大多模态模型相当的效果

### 11. 

缺少论文标题、主要机构和摘要信息，无法生成简报。请提供具体论文信息后再次尝试。

### 12. 

上海人工智能实验室发布了SAIL-VL2论文，使用大规模数据筛选管道、渐进式训练框架（含SAIL-ViT预训练及SFT-RL混合范式）与稀疏MoE架构，解决了全面的多模态理解与推理问题，达成在2B/8B参数规模上多模态基准SOTA，106个数据集表现优异，MMMU和Math-Vista等推理基准达SOTA，2B模型在OpenCompass 4B以下开源模型中排名第一的效果。

### 13. Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency

KTH Royal Institute of Technology发布了Slim-SC论文，使用基于思想层面链间相似性的逐步剪枝技术，解决了Self-Consistency(SC)技术计算开销大的问题，达成减少推理延迟达45%、KVC使用达26%且保持或提高准确性的效果。

### 14. SpecDiff : Accelerating Diffusion Model Inference with Self-Speculation

上海交通大学发布了SpecDiff论文，使用自推测范式的无训练多级特征缓存策略（含基于自推测信息的特征选择和多级分类算法），解决了现有特征缓存方法依赖历史信息导致的精度和速度受限问题，在Stable Diffusion 3/3.5/FLUX上比RFlow平均加速2.80×、2.74×、3.17×且质量损失可忽略，克服了速度-精度权衡瓶颈。

### 15. Where Do Tokens Go? Understanding Pruning Behaviors in STEP at High Resolutions

CNRS发布了论文《Where Do Tokens Go? Understanding Pruning Behaviors in STEP at High Resolutions》，使用STEP混合令牌缩减框架（核心为dCTS轻量级CNN策略网络及编码器早期退出，结合动态补丁合并与令牌剪枝），解决了ViT在语义分割中的高计算与内存成本问题，达成了计算复杂度降低4倍、推理速度提升1.7倍，精度下降不超过2.0%的效果。

## 论文详细信息

### 1. CETUS: Causal Event-Driven Temporal Modeling With Unified Variable-Rate Scheduling

**主要机构**: 
**作者数量**: 7人

**摘要**:
Event cameras capture asynchronous pixel-level brightness changes with microsecond temporal resolution, offering unique advantages for high-speed vision tasks. Existing methods often convert event streams into intermediate representations such as frames, voxel grids, or point clouds, which inevitably require predefined time windows and thus introduce window latency. Meanwhile, pointwise detection methods face computational challenges that prevent real-time efficiency due to their high computational cost. To overcome these limitations, we propose the Variable-Rate Spatial Event Mamba, a novel architecture that directly processes raw event streams without intermediate representations. Our method introduces a lightweight causal spatial neighborhood encoder to efficiently capture local geometric relations, followed by Mamba-based state space models for scalable temporal modeling with linear complexity. During inference, a controller adaptively adjusts the processing speed according to the event rate, achieving an optimal balance between window latency and inference latency.

### 2. Comprehensive Evaluation of CNN-Based Audio Tagging Models on Resource-Constrained Devices

**主要机构**: Instituto Tecnologico de Informatica (ITI)
**作者数量**: 5人

**摘要**:
Convolutional Neural Networks (CNNs) have demonstrated exceptional performance in audio tagging tasks. However, deploying these models on resource-constrained devices like the Raspberry Pi poses challenges related to computational efficiency and thermal management. In this paper, a comprehensive evaluation of multiple convolutional neural network (CNN) architectures for audio tagging on the Raspberry Pi is conducted, encompassing all 1D and 2D models from the Pretrained Audio Neural Networks (PANNs) framework, a ConvNeXt-based model adapted for audio classification, as well as MobileNetV3 architectures. In addition, two PANNs-derived networks, CNN9 and CNN13, recently proposed, are also evaluated. To enhance deployment efficiency and portability across diverse hardware platforms, all models are converted to the Open Neural Network Exchange (ONNX) format. Unlike previous works that focus on a single model, our analysis encompasses a broader range of architectures and involves continuous 24-hour inference sessions to assess performance stability. Our experiments reveal that, with appropriate model selection and optimization, it is possible to maintain consistent inference latency and manage thermal behavior effectively over extended periods. These findings provide valuable insights for deploying audio tagging models in real-world edge computing scenarios.

### 3. CSMoE: An Efficient Remote Sensing Foundation Model with Soft Mixture-of-Experts

**主要机构**: 
**作者数量**: 3人

**摘要**:
Self-supervised learning (SSL) through masked autoencoders (MAEs) has recently attracted great attention for remote sensing (RS) foundation model (FM) development, enabling improved representation learning across diverse sensors and downstream tasks. However, existing RS FMs often either suffer from substantial computational complexity during both training and inference or exhibit limited representational capacity. These issues restrict their practical applicability in RS. To address this limitation, we propose an adaptation for enhancing the efficiency of RS FMs by integrating the Soft mixture-ofexperts (MoE) mechanism into the FM. The integration of Soft MoEs into the FM allows modality-specific expert specialization alongside shared cross-sensor representation learning. To demonstrate the effectiveness of our adaptation, we apply it on the Cross-Sensor Masked Autoencoder (CSMAE) model, resulting in the Cross-Sensor Mixture-of-Experts (CSMoE) model. In addition, we introduce a thematic-climatic descriptor-driven sampling strategy for the construction of a representative and diverse training set to train our CSMoE model. Extensive experiments on scene classification, semantic segmentation, and content-based image retrieval (CBIR) demonstrate that our adaptation yields a reduction in computational requirements while maintaining or improving representational performance. Compared to state-ofthe-art RS FMs, CSMoE achieves a superior trade-off between representational capacity, accuracy, and computational efficiency. On average, CSMoE achieves more than twice the computational efficiency of existing RS FMs, while maintaining competitive performance across all experiments. These results highlight the effectiveness of the proposed adaptation for creating scalable and computationally efficient RS FMs. The associated code for the model and the training set creation, as well as the pretrained model weights, will be available at https://git.tu-berlin.de/rsim/ csmoe.

### 4. 

**主要机构**: 
**作者数量**: 0人

**摘要**:


### 5. DIVE WITH GRT: DENSE VIDEO UNDERSTANDING WITH GATED RESIDUAL TOKENIZATION

**主要机构**: University of Maryland, Princeton University, Northeastern University
**作者数量**: 5人

**摘要**:
High temporal resolution is essential for capturing fine-grained details in video understanding. However, current video large language models (VLLMs) and evaluation benchmarks predominantly rely on low-frame-rate sampling, such as uniform sampling or frame selection, which discards dense temporal information. This compromise is primarily made to avoid the high computational cost of tokenizing every frame, which leads to redundant computation during frame-level tokenization and a linear increase in token count as video length grows. Such a trade-off stems from engineering constraints in existing video understanding systems that rely on frame selection and sampling. Yet, for tasks such as lecture or educational video comprehension, where information is distributed across nearly every frame, this compromise becomes a major limitation. These tasks require frame-by-frame reasoning and fine-grained temporal alignment, and current approaches discourage progress on high-frame-rate datasets or models. To address this gap, we introduce the novel task of Dense Video Understanding, which aims to enable video comprehension at high frame rates. Our goal is to reduce the tokenization time of high-FPS videos and minimize the token overhead incurred by dense frame sampling. This lack of dense modeling also affects current benchmarks, whose question-answer pairs are often designed around slowly changing content, making them insufficient for evaluating fine-grained temporal understanding. To this end, we propose the first benchmark specifically tailored for dense video understanding: DIVE (Dense Information Video Evaluation). To overcome inefficiencies in frame-wise tokenization, we propose Gated Residual Tokenization (GRT), a two-stage token acceleration and reduction framework that operates both during and after tokenization, addressing inefficiencies at the inter-tokenization and intra-tokenization levels, respectively: First, Motion-Compensated Inter-Gated Tokenization applies pixel-level motion estimation and a gating mechanism during tokenization to identify and skip static regions, encoding only the moving patches. This results in sub-linear growth in both tokenization time and token count. Second, Semantic-Scene Intra-Tokenization Merging performs content-level token merging across static regions within a scene, further reducing redundancy while preserving dynamic semantic content. Extensive experiments on the DIVE benchmark show that our methods not only outperform larger VLLM baselines but also consistently improve as FPS increases. These results underscore the importance of preserving dense temporal information and demonstrate that GRT enables scalable, efficient high-FPS video understanding.

### 6. DSPC: DUAL-STAGE PROGRESSIVE COMPRESSION FRAMEWORK FOR EFFICIENT LONG-CONTEXT REASONING

**主要机构**: A*STAR Amazon Binjiang Institute of Artificial Intelligence, Institute of Cyberspace Security, Zhejiang University of Technology
**作者数量**: 6人

**摘要**:
Large language models (LLMs) have achieved remarkable success in many natural language processing (NLP) tasks. To achieve more accurate output, the prompts used to drive LLMs have become increasingly longer, which incurs higher computational costs. To address this prompt inflation problem, prompt compression has been proposed. However, most existing methods require training a small auxiliary model for compression, incurring a significant amount of additional computation. To avoid this, we propose a twostage, training-free approach, called Dual-Stage Progressive Compression (DSPC). In the coarse-grained stage, semanticrelated sentence filtering removes sentences with low semantic value based on TF-IDF. In the fine-grained stage, token importance is assessed using attention contribution, crossmodel loss difference, and positional importance, enabling the pruning of low-utility tokens while preserving semantics. We validate DSPC on LLaMA-3.1-8B-Instruct and GPT-3.5-Turbo under a constrained token budget and observe consistent improvements. For instance, in the FewShot task of the Longbench dataset, DSPC achieves a performance of 49.17 by using only 3× fewer tokens, outperforming the best state-of-the-art baseline LongLLMLingua by 7.76.

### 7. Improving 3D Gaussian Splatting Compression by Scene-Adaptive Lattice Vector Quantization

**主要机构**: 
**作者数量**: 3人

**摘要**:
3D Gaussian Splatting (3DGS) is rapidly gaining popularity for its photorealistic rendering quality and real-time performance, but it generates massive amounts of data. Hence compressing 3DGS data is necessary for the cost effectiveness of 3DGS models. Recently, several anchor-based neural compression methods have been proposed, achieving good 3DGS compression performance. However, they all rely on uniform scalar quantization (USQ) due to its simplicity. A tantalizing question is whether more sophisticated quantizers can improve the current 3DGS compression methods with very little extra overhead and minimal change to the system. The answer is yes by replacing USQ with lattice vector quantization (LVQ). To better capture scene-specific characteristics, we optimize the lattice basis for each scene, improving LVQ's adaptability and R-D efficiency. This scene-adaptive LVQ (SALVQ) strikes a balance between the R-D efficiency of vector quantization and the low complexity of USQ. SALVQ can be seamlessly integrated into existing 3DGS compression architectures, enhancing their R-D performance with minimal modifications and computational overhead. Moreover, by scaling the lattice basis vectors, SALVQ can dynamically adjust lattice density, enabling a single model to accommodate multiple bit rate targets. This flexibility eliminates the need to train separate models for different compression levels, significantly reducing training time and memory consumption.

### 8. InfraMind: A Novel Exploration-based GUI Agentic Framework for Mission-critical Industrial Management

**主要机构**: Nanyang Technological University
**作者数量**: 1人

**摘要**:
Mission-critical industrial infrastructure, such as data centers, increasingly depends on complex management software. Its operations, however, pose significant challenges due to the escalating system complexity, multivendor integration, and a shortage of expert operators. While Robotic Process Automation (RPA) offers partial automation through handcrafted scripts, it suffers from limited flexibility and high maintenance costs. Recent advances in Large Language Model (LLM)-based graphical user interface (GUI) agents have enabled more flexible automation, yet these general-purpose agents face five critical challenges when applied to industrial management, including unfamiliar element understanding, precision and efficiency, state localization, deployment constraints, and safety requirements. To address these issues, we propose InfraMind, a novel exploration-based GUI agentic framework specifically tailored for industrial management systems. InfraMind integrates five innovative modules to systematically resolve different challenges in industrial management: (1) systematic search-based exploration with virtual machine snapshots for autonomous understanding of complex GUIs; (2) memory-driven planning to ensure high-precision and efficient task execution; (3) advanced state identification for robust localization in hierarchical interfaces; (4) structured knowledge distillation for efficient deployment with lightweight models; and (5) comprehensive, multi-layered safety mechanisms to safeguard sensitive operations. Extensive experiments on both open-source and commercial DCIM platforms demonstrate that our approach consistently outperforms existing frameworks in terms of task success rate and operational efficiency, providing a rigorous and scalable solution for industrial management automation. CCS Concepts: • Computing methodologies → Artificial intelligence; • Software and its engineering → Software creation and management; • Human-centered computing → Human computer interaction (HCI).

### 9. MemGS: Memory-Efficient Gaussian Splatting for Real-Time SLAM

**主要机构**: 
**作者数量**: 7人

**摘要**:
Recent advancements in 3D Gaussian Splatting (3DGS) have made a significant impact on rendering and reconstruction techniques. Current research predominantly focuses on improving rendering performance and reconstruction quality using high-performance desktop GPUs, largely overlooking applications for embedded platforms like micro air vehicles (MAVs). These devices, with their limited computational resources and memory, often face a trade-off between system performance and reconstruction quality. In this paper, we improve existing methods in terms of GPU memory usage while enhancing rendering quality. Specifically, to address redundant 3D Gaussian primitives in SLAM, we propose merging them in voxel space based on geometric similarity. This reduces GPU memory usage without impacting system runtime performance. Furthermore, rendering quality is improved by initializing 3D Gaussian primitives via Patch-Grid (PG) point sampling, enabling more accurate modeling of the entire scene. Quantitative and qualitative evaluations on publicly available datasets demonstrate the effectiveness of our improvements.

### 10. MOCHA: Multi-modal Objects-aware Cross-arcHitecture Alignment

**主要机构**: Samsung R&D Institute UK
**作者数量**: 4人

**摘要**:
We introduce MOCHA (Multi-modal Objects-aware Cross-arcHitecture Alignment), a knowledge distillation approach that transfers region-level multimodal semantics from a large vision-language teacher (e.g., LLaVa) into a lightweight vision-only object detector student (e.g., YOLO). A translation module maps student features into a joint space, where the training of the student and translator is guided by a dualobjective loss that enforces both local alignment and global relational consistency. Unlike prior approaches focused on dense or global alignment, MOCHA operates at the object level, enabling efficient transfer of semantics without modifying the teacher or requiring textual input at inference. We validate our method across four personalized detection benchmarks under few-shot regimes. Results show consistent gains over baselines, with a +10.1 average score improvement. Despite its compact architecture, MOCHA reaches performance on par with larger multimodal models, proving its suitability for real-world deployment.

### 11. 

**主要机构**: 
**作者数量**: 0人

**摘要**:


### 12. 

**主要机构**: 
**作者数量**: 0人

**摘要**:
We introduce SAIL-VL2, an open-suite vision-language foundation model (LVM) for comprehensive multimodal understanding and reasoning. As the successor to SAIL-VL, SAIL-VL2 achieves state-of-the-art performance at the 2B and 8B parameter scales across diverse image and video benchmarks, demonstrating strong capabilities from fine-grained perception to complex reasoning. Its effectiveness is driven by three core innovations. First, a large-scale data curation pipeline with scoring and filtering strategies enhances both quality and distribution across captioning, OCR, QA, and video data, improving training efficiency. Second, a progressive training framework begins with a powerful pre-trained vision encoder (SAIL-ViT), advances through multimodal pre-training, and culminates in a thinking-fusion SFT-RL hybrid paradigm that systematically strengthens model capabilities. Third, architectural advances extend beyond dense LLMs to efficient sparse Mixture-of-Experts (MoE) designs. With these contributions, SAIL-VL2 demonstrates competitive performance across 106 datasets and achieves state-of-the-art results on challenging reasoning benchmarks such as MMMU and Math-Vista. Furthermore, on the OpenCompass leaderboard, SAIL-VL2-2B ranks first among officially released open-source models under the 4B parameter scale, while serving as an efficient and extensible foundation for the open-source multimodal community. a. Performance comparison of SAIL-VL2 basic models (no-thinking) across general multimodal understanding benchmarks. b. Performance comparison of SAIL-VL2-Thinking models on mathematical reasoning benchmarks.

### 13. Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency

**主要机构**: KTH Royal Institute of Technology
**作者数量**: 5人

**摘要**:
Recently, Test-Time Scaling (TTS) has gained increasing attention for improving LLM reasoning performance at test time without retraining the model. A notable TTS technique is Self-Consistency (SC), which generates multiple reasoning chains in parallel and selects the final answer via majority voting. While effective, the order-of-magnitude computational overhead limits its broad deployment. Prior attempts to accelerate SC mainly rely on modelbased confidence scores or heuristics with limited empirical support. For the first time, we theoretically and empirically analyze the inefficiencies of SC and reveal actionable opportunities for improvement. Building on these insights, we propose Slim-SC, a step-wise pruning strategy that identifies and removes redundant chains using inter-chain similarity at the thought level. Experiments on three STEM reasoning datasets and two recent LLM architectures show that Slim-SC reduces inference latency and KVC usage by up to 45% and 26%, respectively, with R1-Distill, while maintaining or improving accuracy, thus offering a simple yet efficient TTS alternative for SC.

### 14. SpecDiff : Accelerating Diffusion Model Inference with Self-Speculation

**主要机构**: Shanghai Jiao Tong University
**作者数量**: 4人

**摘要**:
Feature caching has recently emerged as a promising method for diffusion model acceleration. It effectively alleviates the inefficiency problem caused by high computational requirements by caching similar features in the inference process of the diffusion model. In this paper, we analyze existing feature caching methods from the perspective of information utilization, and point out that relying solely on historical information will lead to constrained accuracy and speed performance. And we propose a novel paradigm that introduces future information via self-speculation based on the information similarity at the same time step across different iteration times. Based on this paradigm, we present SpecDiff, a training-free multi-level feature caching strategy including a cached feature selection algorithm and a multi-level feature classification algorithm. (1) Feature selection algorithm based on self-speculative information. SpecDiff determines a dynamic importance score for each token based on selfspeculative information and historical information, and performs cached feature selection through the importance score. (2) Multi-level feature classification algorithm based on feature importance scores. SpecDiff classifies tokens by leveraging the differences in feature importance scores and introduces a multi-level feature calculation strategy. Extensive experiments show that SpecDiff achieves average 2.80×, 2.74×, and 3.17× speedup with negligible quality loss in Stable Diffusion 3, 3.5, and FLUX compared to RFlow on NVIDIA A800-80GB GPU. By merging speculative and historical information, SpecDiff overcomes the speedup-accuracy tradeoff bottleneck, pushing the Pareto frontier of speedup and accuracy in the efficient diffusion model inference.

### 15. Where Do Tokens Go? Understanding Pruning Behaviors in STEP at High Resolutions

**主要机构**: Université Côte d'Azur, CNRS, CEA, I3S, Université Paris-Saclay, Sophia Antipolis
**作者数量**: 3人

**摘要**:
Vision Transformers (ViTs) achieve state-of-the-art performance in semantic segmentation but are hindered by high computational and memory costs. To address this, we propose STEP (SuperToken and Early-Pruning), a hybrid tokenreduction framework that combines dynamic patch merging and token pruning to enhance efficiency without significantly compromising accuracy. At the core of STEP is dCTS, a lightweight CNN-based policy network that enables flexible merging into superpatches. Encoder blocks integrate also early-exits to remove high-confident supertokens, lowering computational load. We evaluate our method on high-resolution semantic segmentation benchmarks, including images up to 1024×1024, and show that when dCTS is applied alone, the token count can be reduced by a factor of 2.5 compared to the standard 16 × 16 pixel patching scheme. This yields a 2.6× reduction in computational cost and a 3.4× increase in throughput when using ViT-Large as the backbone. Applying the full STEP framework further improves efficiency, reaching up to a 4× reduction in computational complexity and a 1.7× gain in inference speed, with a maximum accuracy drop of no more than 2.0%. With the proposed STEP configurations, up to 40% of tokens can be confidently predicted and halted before reaching the final encoder layer.
