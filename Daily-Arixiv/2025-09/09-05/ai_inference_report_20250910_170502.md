# AI推理加速技术论文分析报告
生成时间: 2025-09-10 17:05:02
分析论文数量: 28篇

## 论文技术简报

### 1. 3DOF+Quantization: 3DGS quantization for large scenes with limited Degrees of Freedom

Orange Innovation发布了3DOF+Quantization论文，使用基于球坐标的新量化方案，解决了大场景中3DGS在有限自由度（3DoF+）下坐标量化导致的投影误差问题，在Garden场景中展示了良好的率失真性能。

### 2. 

请提供论文的标题、主要机构和摘要信息，以便生成技术简报。

### 3. Advanced Brain Tumor Segmentation Using EMCAD: Efficient Multi-scale Convolutional Attention Decoding

Texas Tech University发布了关于脑肿瘤分割的论文，使用EMCAD（高效多尺度卷积注意力解码器），解决了现有解码机制在脑肿瘤分割中计算成本高的问题，达成了优化性能与计算效率，在BraTs2020数据集上最佳Dice分数0.31、平均0.285±0.015且验证集性能稳定无过拟合的效果

### 4. Application of discrete Ricci curvature in pruning randomly wired neural networks: A case study with chest x-ray classification of COVID-19

哥廷根大学发布了关于离散Ricci曲率在随机连接神经网络剪枝中应用的论文，使用Forman-Ricci曲率（FRC）等边缘中心网络度量，解决了随机连接神经网络剪枝中减少复杂度同时保持性能的问题，达成了显著计算优势且性能与Ollivier-Ricci曲率（ORC）相当的效果

### 5. Ban&Pick: Achieving Free Performance Gains and Inference Speedup via Smarter Routing in MoE-LLMs

中科院自动化所与南京理工大学发布了Ban&Pick论文，使用后训练即插即用的智能路由策略（Ban&Pick），解决了MoE-LLMs中高影响力专家未充分利用及固定激活专家数冗余的问题，达成了无需重训练或架构改变即可实现性能提升和推理加速的效果

### 6. Beyond the Pre-Service Horizon: Infusing In-Service Behavior for Improved Financial Risk Forecasting

腾讯与中国人民大学发布了融合服务中行为数据改进金融风险预测的相关论文，使用MGKD（多粒度知识蒸馏）框架（含教师-学生知识蒸馏、粗/细/自多粒度蒸馏及重加权策略），解决了传统金融风险管理中服务前风险评估与服务中行为数据分离导致的预测不足问题，在腾讯移动支付大规模数据集上有效提升了服务前风险评估性能。

### 7. BioLite U-Net: Edge-Deployable Semantic Segmentation for In Situ Bioprinting Monitoring

ÚRAM - SFI Research Centre for Medical Devices发布了BioLite U-Net论文，使用基于深度可分离卷积的轻量化语义分割架构及手动标注生物打印数据集，解决了生物打印原位监测中有限数据和资源受限硬件下的实时语义分割挑战，达成了mIoU 92.85%、Dice 96.17%的高分割精度，模型体积比MobileNetV2-DeepLabV3+小1300×以上且边缘设备每帧推理仅需335ms的近实时效果。

### 8. BIR-Adapter: A Low-Complexity Diffusion Model Adapter for Blind Image Restoration

慕尼黑工业大学发布了BIR-Adapter论文，使用低复杂度扩散模型适配器技术，解决了盲图像恢复问题，达成了在多种退化情况下实现有效图像恢复的效果

### 9. BRANCHGRPO: STABLE AND EFFICIENT GRPO WITH STRUCTURED BRANCHING IN DIFFUSION MODELS

北京大学发布了BranchGRPO论文，使用结构化分支的GRPO技术（含分支采样、树基优势估计器及剪枝策略），解决了扩散模型中GRPO的高计算成本与训练不稳定性问题，达成图像对齐分数提升16%、训练时间减少55%及视频生成质量提升的效果。

### 10. COMPACT: COMMON-TOKEN OPTIMIZED MODEL PRUNING ACROSS CHANNELS AND TOKENS

Penn State University发布了COMPACT论文，使用联合剪枝稀有词汇与基于常见token加权激活的FFN通道技术，解决了现有剪枝方法破坏架构或导致精度骤降的局限，达成了保持标准架构、训练无关，在多模型上实现SOTA下游性能并显著减少参数、内存与延迟的效果。

### 11. Context-Aware Knowledge Distillation with Adaptive Weighting for Image Classification

浙江大学发布了Context-Aware Knowledge Distillation with Adaptive Weighting论文，使用Adaptive Knowledge Distillation (AKD)框架（含动态学习平衡因子α及MLP+Attention的Context-Aware Module），解决了传统知识蒸馏中固定平衡因子α次优的问题，在CIFAR-10上较固定权重KD基线实现更高准确率和更稳定收敛。

### 12. Delta Velocity Rectified Flow for Text-to-Image Editing

哈佛大学发布了Delta Velocity Rectified Flow (DVRF)论文，使用显式建模源与目标速度场差异并引入时间相关偏移项的无反转路径感知编辑框架，解决了蒸馏采样方法中过度平滑伪影问题，达成了更优的编辑质量、保真度和可控性，且无需架构修改高效适用于文本到图像编辑任务。

### 13. Dynamic Sensitivity Filter Pruning using Multi-Agent Reinforcement Learning For DCNN's

North South University Dhaka发布了Dynamic Sensitivity Filter Pruning using Multi-Agent Reinforcement Learning For DCNN's论文，使用Differential Sensitivity Fusion Pruning (DSFP)单阶段滤波器剪枝框架（融合梯度基灵敏度等多准则差异计算差分灵敏度分数，高效确定性仅需一次前向-反向传播），解决了DCNNs计算与内存开销大限制实际部署的问题，达成50%-70%剪枝率下超80% FLOPs减少、70%剪枝时保留98.23%基线精度且优于传统方法的效果

### 14. FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving

三星SDS发布了FineServe论文，使用KV Slab（精度感知自适应内存管理）和两级调度框架，解决了量化LLM服务中的内存碎片化与资源调度效率问题，达成了2.2×更高SLO达成率和1.8×更高令牌生成吞吐量的效果

### 15. From Long to Short: LLMs Excel at Trimming Own Reasoning Chains

悉尼大学与新加坡国立大学发布了《From Long to Short: LLMs Excel at Trimming Own Reasoning Chains》论文，使用EDIT（高效动态推理修剪）技术，解决了大型推理模型过度思考导致推理路径冗长复杂、难以平衡正确性与简洁性的问题，达成了显著提升推理效率、生成简洁且信息丰富的输出以改善可读性和用户体验的效果。

### 16. Hybrid Fourier Neural Operator-Plasma Fluid Model for Fast and Accurate Multiscale Simulations of High Power Microwave Breakdown

Dhirubhai Ambani University发布了混合Fourier Neural Operator-等离子体流体模型论文，使用结合微分方程等离子体流体求解器与FNO电磁求解器的混合建模技术，解决了高功率微波击穿多尺度模拟计算昂贵问题，达成与传统FDTD模拟结果一致且加速约60倍的效果。

### 17. INDEX-PRESERVING LIGHTWEIGHT TOKEN PRUNING FOR EFFICIENT DOCUMENT UNDERSTANDING IN VISION-LANGUAGE MODELS

Hana Institute of Technology Seoul发布了关于视觉语言模型文档理解的论文，使用索引保留的轻量级令牌剪枝框架（含二进制补丁级分类器去除非文本区域及最大池化优化步骤恢复文本区域），解决了视觉语言模型在文档理解中的高计算需求问题，达成了大幅降低计算成本同时保持相当准确率的效果。

### 18. LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba

研究团队发布了LocoMamba论文，使用基于Mamba的跨模态融合骨干的端到端深度强化学习框架，解决了四足机器人视觉驱动运动控制中的跨模态融合效率、计算效率及训练稳定性问题，相比SOTA在回报、碰撞次数、移动距离上有提升且相同计算预算下收敛更快。

### 19. MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs

Northwestern University发布了MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs论文，使用基于C的无运行时推理引擎及算子融合、内存布局优化技术，解决了微控制器上部署Mamba模型面临的内存有限等挑战，达成了减少83.0%峰值内存、保持分类精度并实现跨平台可移植的效果。

### 20. MEANFLOW-ACCELERATED MULTIMODAL VIDEO-TO-AUDIO SYNTHESIS VIA ONE-STEP GENERATION

武汉理工大学发布了MeanFlow加速的多模态视频转音频合成论文，使用MeanFlow加速模型（通过平均速度表征流场实现一步生成）及标量重缩放机制，解决了现有视频转音频合成中合成质量与推理效率的权衡问题，达成显著提升推理速度同时保持音频质量、语义对齐和时间同步的效果。

### 21. Mitigating Spurious Correlations Between Question and Answer via Chain-of-Thought Correctness Perception Distillation

北京航空航天大学发布了论文，使用Chain-of-Thought Correctness Perception Distillation (CoPeD)技术，解决了小型语言模型微调时因CoT数据噪声导致捕捉问题与答案间虚假关联、推理质量下降的问题，在分布内和分布外推理数据集上有效提升推理质量。

### 22. ProfilingAgent: Profiling-Guided Agentic Reasoning for Adaptive Model Optimization

Iowa State University发布了ProfilingAgent论文，使用基于大语言模型（LLMs）的多智能体系统进行分析引导的结构化剪枝与动态量化技术，解决了基础模型在资源受限平台部署的计算和内存瓶颈及现有压缩技术忽略架构与运行时异质性的问题，达成了剪枝在ImageNet-1K精度下降约1%且小数据集精度提升达+2%、量化内存节省达74%且推理加速1.74×（精度下降低于0.5%）的效果

### 23. Quantitative Currency Evaluation in Low-Resource Settings through Pattern Analysis to Assist Visually Impaired Users

乔治梅森大学发布了《Quantitative Currency Evaluation in Low-Resource Settings through Pattern Analysis to Assist Visually Impaired Users》论文，使用整合轻量级CNN面额分类、Unified Currency Damage Index损伤量化及特征模板匹配伪造检测的统一框架，解决了低资源环境下货币识别系统忽视可用性和真实性评估（尤其视觉障碍用户与离线验证场景）的问题，达成了支持实时设备端推理的准确、可解释且紧凑的包容性货币评估效果。

### 24. Quaternion Approximate Networks for Enhanced Image Classification and Oriented Object Detection

研究团队发布了Quaternion Approximate Networks (QUAN)论文，使用通过汉密尔顿积分解用实值运算近似四元数卷积、结合独立四元数批归一化(IQBN)与空间注意力机制的深度学习框架，解决了旋转等变图像分类与目标检测中的效率与几何特性保持问题，达成分类任务更高精度、更少参数、更快收敛，目标检测参数效率和旋转处理能力提升并确立四元数CNN的SOTA效果。

### 25. Sensitivity-Aware Post-Training Quantization for Deep Neural Networks

华南理工大学发布了Sensitivity-Aware Post-Training Quantization for Deep Neural Networks论文，使用基于参数敏感性分析的高效后训练量化（PTQ）方法，通过优先量化高敏感参数并利用低敏感参数补偿误差，引入行并行量化框架及全局共享逆Hessian矩阵更新机制，解决了现有后训练量化方法计算复杂度高、资源开销大的问题，达成在ResNet-50和YOLOv5s上量化速度较Optimal Brain Quantization提升20-200倍、平均精度损失低于0.3%的效果。

### 26. SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning

上海交通大学发布了SpecPrune-VLA论文，使用动作感知自推测剪枝技术（结合局部与全局信息的两级token剪枝及轻量级控制器），解决了现有VLA模型剪枝忽略全局信息导致成功率下降和加速有限的问题，达成了在LIBERO基准上平均1.46×（A800）和1.57×（3090）加速且成功率损失可忽略的效果

### 27. StripDet: Strip Attention-Based Lightweight 3D Object Detection from Point Cloud

中山大学发布了StripDet论文，使用Strip Attention Block (SAB)及硬件友好的分层骨干网络，解决了高精度3D点云目标检测模型计算和内存需求大、难以部署的问题，达成0.65M参数下实现79.97%汽车检测mAP，较PointPillars参数减少7倍并超越其性能的效果。

### 28. TrajAware: Graph Cross-Attention and Trajectory-Aware for Generalisable VANETs under Partial Observations

研究团队发布了TrajAware论文，使用整合动作空间剪枝、图交叉注意力和轨迹感知预测的RL框架，解决了VANETs动态拓扑、不完全观测及边缘设备资源有限下的路由挑战与现有RL方法泛化性差问题，实现近最短路径、高交付率和边缘适用效率，在全/部分观测场景下优于SOTA。

## 论文详细信息

### 1. 3DOF+Quantization: 3DGS quantization for large scenes with limited Degrees of Freedom

**主要机构**: Stéphane PATEUX Théo LADUNE Orange Innovation
**作者数量**: 1人

**摘要**:
3DGS [Kerbl et al., 2023] is a major breakthrough in 3D scene reconstruction. With a number of views of a given object or scene, the algorithm trains a model composed of 3D gaussians, which enables the production of novel views from arbitrary points of view. This freedom of movement is referred to as 6DoF for 6 degrees of freedom: a view is produced for any position (3 degrees), orientation of camera (3 other degrees). On large scenes, though, the input views are acquired from a limited zone in space, and the reconstruction is valuable for novel views from the same zone, even if the scene itself is almost unlimited in size. We refer to this particular case as 3DoF+, meaning that the 3 degrees of freedom of camera position are limited to small offsets around the central position. Considering the problem of coordinate quantization, the impact of position error on the projection error in pixels is studied. It is shown that the projection error is proportional to the squared inverse distance of the point being projected. Consequently, a new quantization scheme based on spherical coordinates is proposed. Rate-distortion performance of the proposed method are illustrated on the well-known Garden scene.

### 2. 

**主要机构**: 
**作者数量**: 0人

**摘要**:


### 3. Advanced Brain Tumor Segmentation Using EMCAD: Efficient Multi-scale Convolutional Attention Decoding

**主要机构**: Texas Tech University Lubbock, Department of Computer Science
**作者数量**: 4人

**摘要**:
Brain tumor segmentation is a critical preprocessing step in the medical image analysis pipeline that involves precise delineation of tumor regions from healthy brain tissue in medical imaging data, particularly MRI scans. An efficient and effective decoding mechanism is crucial in brain tumor segmentation especially in scenarios with limited computational resources. However these decoding mechanisms usually come with high computational costs. To address this concern EMCAD a new efficient multi-scale convolutional attention decoder designed was utilized to optimize both performance and computational efficiency for brain tumor segmentation on the BraTs2020 dataset consisting of MRI scans from 369 brain tumor patients. The preliminary result obtained by the model achieved a best Dice score of 0.31 and maintained a stable mean Dice score of 0.285 ± 0.015 throughout the training process which is moderate. The initial model maintained consistent performance across the validation set without showing signs of over-fitting.

### 4. Application of discrete Ricci curvature in pruning randomly wired neural networks: A case study with chest x-ray classification of COVID-19

**主要机构**: The Institute of Mathematical Sciences (IMSc), Institute of Computer Science and Campus Institute Data Science, University of Göttingen
**作者数量**: 5人

**摘要**:
Randomly Wired Neural Networks (RWNNs) serve as a valuable testbed for investigating the impact of network topology in deep learning by capturing how different connectivity patterns impact both learning efficiency and model performance. At the same time, they provide a natural framework for exploring edge-centric network measures as tools for pruning and optimization. In this study, we investigate three edge-centric network measures: Forman-Ricci curvature (FRC), Ollivier-Ricci curvature (ORC), and edge betweenness centrality (EBC), to compress RWNNs by selectively retaining important synapses (or edges) while pruning the rest. As a baseline, RWNNs are trained for COVID-19 chest x-ray image classification, aiming to reduce network complexity while preserving performance in terms of accuracy, specificity, and sensitivity. We extend prior work on pruning RWNN using ORC by incorporating two additional edge-centric measures, FRC and EBC, across three network generators: Erdös-Rényi (ER) model, Watts-Strogatz (WS) model, and Barabási-Albert (BA) model. We provide a comparative analysis of the pruning performance of the three measures in terms of compression ratio and theoretical speedup. A central focus of our study is to evaluate whether FRC, which is computationally more efficient than ORC, can achieve comparable pruning effectiveness. Along with performance evaluation, we further investigate the structural properties of the pruned networks through modularity and global efficiency, offering insights into the trade-off between modular segregation and network efficiency in compressed RWNNs. Our results provide initial evidence that FRC-based pruning can effectively simplify RWNNs, offering significant computational advantages while maintaining performance comparable to ORC.

### 5. Ban&Pick: Achieving Free Performance Gains and Inference Speedup via Smarter Routing in MoE-LLMs

**主要机构**: Nanjing University of Science and Technology, Chinese Academy of Sciences, Institute of Automation
**作者数量**: 5人

**摘要**:
Sparse Mixture-of-Experts (MoE) has become a key architecture for scaling large language models (LLMs) efficiently. Recent fine-grained MoE designs introduce hundreds of experts per layer, with multiple experts activated per token, enabling stronger specialization. However, during pre-training, routers are optimized mainly for stability and robustness: they converge prematurely and enforce balanced usage, limiting the full potential of model performance and efficiency. In this work, we uncover two overlooked issues: (i) a few highly influential experts are underutilized due to premature and balanced routing decisions; and (ii) enforcing a fixed number of active experts per token introduces substantial redundancy. Instead of retraining models or redesigning MoE architectures, we introduce Ban&Pick, a post-training, plug-and-play strategy for smarter MoE routing. Pick discovers and reinforces key experts-a small group with outsized impact on performance-leading to notable accuracy gains across domains. Ban complements this by dynamically pruning redundant experts based on layer and token sensitivity, delivering faster inference with minimal accuracy loss. Experiments on fine-grained MoE-LLMs (DeepSeek, Qwen3) across math, code, and general reasoning benchmarks demonstrate that Ban&Pick delivers free performance gains and inference acceleration without retraining or architectural changes.

### 6. Beyond the Pre-Service Horizon: Infusing In-Service Behavior for Improved Financial Risk Forecasting

**主要机构**: Institute of Intelligent Computing Technology, Renmin University of China Beijing, Tencent Weixin Group Shenzhen, Chinese Academy of Sciences, University of Chinese Academy of Sciences Beijing
**作者数量**: 15人

**摘要**:
Typical financial risk management involves distinct phases for pre-service risk assessment and in-service default detection, often modeled separately. This paper proposes a novel framework, Multi-Granularity Knowledge Distillation (abbreviated as MGKD), aimed at improving pre-service risk prediction through the integration of in-service user behavior data. MGKD follows the idea of knowledge distillation, where the teacher model, trained on historical in-service data, guides the student model, which is trained on pre-service data. By using soft labels derived from in-service data, the teacher model helps the student model improve its risk prediction prior to service activation. Meanwhile, a multi-granularity distillation strategy is introduced, including coarse-grained, fine-grained, and selfdistillation, to align the representations and predictions of the teacher and student models. This approach not only reinforces the representation of default cases but also enables the transfer of key behavioral patterns associated with defaulters from the teacher to the student model, thereby improving the overall performance of pre-service risk assessment. Moreover, we adopt a re-weighting strategy to mitigate the model's bias towards the minority class. Experimental results on large-scale real-world datasets from Tencent Mobile Payment demonstrate the effectiveness of our proposed approach in both offline and online scenarios.

### 7. BioLite U-Net: Edge-Deployable Semantic Segmentation for In Situ Bioprinting Monitoring

**主要机构**: College of Science and Engineering, ÚRAM -SFI Research Centre for Medical Devices, School of Computer Science, University of Galway
**作者数量**: 7人

**摘要**:
Bioprinting is a rapidly advancing field that offers a transformative approach to fabricating tissue and organ models through the precise deposition of cell-laden bioinks. Ensuring the fidelity and consistency of printed structures in real-time remains a core challenge, particularly under constraints imposed by limited imaging data and resourceconstrained embedded hardware. Semantic segmentation of the extrusion process, differentiating between nozzle, extruded bioink, and surrounding background, enables in situ monitoring critical to maintaining print quality and biological viability. In this work, we introduce a lightweight semantic segmentation framework tailored for real-time bioprinting applications. We present a novel, manually annotated dataset comprising 787 RGB images captured during the bioprinting process, labeled across three classes: nozzle, bioink, and background. To achieve fast and efficient inference suitable for integration with bioprinting systems, we propose a BioLite U-Net architecture that leverages depthwise separable convolutions to drastically reduce computational load without compromising accuracy. Our model is benchmarked against MobileNetV2 and MobileNetV3-based segmentation baselines using mean Intersection over Union (mIoU), Dice score, and pixel accuracy. All models were evaluated on a Raspberry Pi 4B to assess real-world feasibility. The proposed BioLite U-Net achieves an mIoU of 92.85% and a Dice score of 96.17%, while being over 1300× smaller than MobileNetV2-DeepLabV3+. On-device inference takes 335 ms per frame, demonstrating near real-time capability. Compared to MobileNet baselines, BioLite U-Net offers a superior tradeoff between segmentation accuracy, efficiency, and deployability, making it highly suitable for intelligent, closed-loop bioprinting systems.

### 8. BIR-Adapter: A Low-Complexity Diffusion Model Adapter for Blind Image Restoration

**主要机构**: Chair of Media Technology, † Chair of Communication Networks Munich Institute of Robotics and Machine Intelligence School of Computation, Information, and Technology, Technical University of Munich
**作者数量**: 4人

**摘要**:
Figure 1. Example outputs of BIR-Adapter under various degradations.

### 9. BRANCHGRPO: STABLE AND EFFICIENT GRPO WITH STRUCTURED BRANCHING IN DIFFUSION MODELS

**主要机构**: Beijing Normal University, Peking University
**作者数量**: 7人

**摘要**:
Recent progress in aligning image and video generative models with Group Relative Policy Optimization (GRPO) has improved human preference alignment, yet existing approaches still suffer from high computational cost due to sequential rollouts and large numbers of SDE sampling steps, as well as training instability caused by sparse rewards. In this paper, we present BranchGRPO, a method that restructures the rollout process into a branching tree, where shared prefixes amortize computation and pruning removes low-value paths and redundant depths. BranchGRPO introduces three contributions: (1) a branch sampling scheme that reduces rollout cost by reusing common segments; (2) a tree-based advantage estimator that converts sparse terminal rewards into dense, step-level signals; and (3) pruning strategies that accelerate convergence while preserving exploration. On HPDv2.1 image alignment, BranchGRPO improves alignment scores by up to 16% over strong baselines, while reducing per-iteration training time by nearly 55%. On WanX-1.3B video generation, it further achieves higher Video-Align scores with sharper and temporally consistent frames compared to DanceGRPO.

### 10. COMPACT: COMMON-TOKEN OPTIMIZED MODEL PRUNING ACROSS CHANNELS AND TOKENS

**主要机构**: Penn State University, Department of Computer Science & Engineering
**作者数量**: 2人

**摘要**:
Making LLMs more efficient in memory, latency, and serving cost is crucial for edge deployment, interactive applications, and sustainable inference at scale. Pruning is a key technique toward this goal. However, prior pruning methods are limited: width pruning often breaks the standard transformer layout or requires custom inference code, while depth pruning removes entire layers and can cause abrupt accuracy drops. In this work, we propose COMPACT, which jointly (i) prunes rare vocabulary to shrink embedding/unembedding and (ii) prunes FFN intermediate channels using common-token-weighted activations, aligning importance with the post-pruning token distribution. COMPACT enjoys merits of both depth and width pruning, such as: deployment-friendliness (keeps a standard transformer architecture), scale-adaptivity (trade off vocab vs. FFN pruning), training-free operation with competitive pruning time, and strong memory savings alongside throughput gains. Experiments across Qwen, LLaMA, and Gemma families (0.5B-70B) show state-of-the-art downstream task performance at similar or higher pruning ratios, with substantial reductions in parameters, GPU memory, and end-to-end latency.

### 11. Context-Aware Knowledge Distillation with Adaptive Weighting for Image Classification

**主要机构**: Zhejiang University HangZhou, School of Computing
**作者数量**: 1人

**摘要**:
Knowledge distillation (KD) is a widely used technique to transfer knowledge from a large teacher network to a smaller student model. Traditional KD uses a fixed balancing factor α as a hyperparameter to combine the hard-label cross-entropy loss with the soft-label distillation loss. However, a static α is suboptimal because the optimal trade-off between hard and soft supervision can vary during training. In this work, we propose an Adaptive Knowledge Distillation (AKD) framework. First we try to make α as learnable parameter that can be automatically learned and optimized during training. Then we introduce a formula to reflect the gap between the student and the teacher to compute α dynamically, guided by student-teacher discrepancies, and further introduce a Context-Aware Module (CAM) using MLP + Attention to adaptively reweight class-wise teacher outputs. Experiments on CIFAR-10 with ResNet-50 as teacher and ResNet-18 as student demonstrate that our approach achieves superior accuracy compared to fixed-weight KD baselines, and yields more stable convergence.

### 12. Delta Velocity Rectified Flow for Text-to-Image Editing

**主要机构**: Harvard University, Harvard AI and Robotics Lab, Computer Science Department
**作者数量**: 5人

**摘要**:
We propose Delta Velocity Rectified Flow (DVRF), a novel inversion-free, path-aware editing framework within rectified flow models for text-to-image editing. DVRF is a distillationbased method that explicitly models the discrepancy between the source and target velocity fields in order to mitigate over-smoothing artifacts rampant in prior distillation sampling approaches. We further introduce a time-dependent shift term to push noisy latents closer to the target trajectory, enhancing the alignment with the target distribution. We theoretically demonstrate that when this shift is disabled, DVRF reduces to Delta Denoising Score, thereby bridging score-based diffusion optimization and velocity-based rectified-flow optimization. Moreover, when the shift term follows a linear schedule under rectified-flow dynamics, DVRF generalizes the Inversion-free method FlowEdit and provides a principled theoretical interpretation for it. Experimental results indicate that DVRF achieves superior editing quality, fidelity, and controllability while requiring no architectural modifications, making it efficient and broadly applicable to text-to-image editing tasks. Code is available at https://github.com/gaspardbd/DeltaVelocityRectifiedFlow.

### 13. Dynamic Sensitivity Filter Pruning using Multi-Agent Reinforcement Learning For DCNN's

**主要机构**: Dept. of ECE North South University Dhaka, ECE North South University Dhaka, Dept.of ECE North South University Dhaka, rd Ahmed Faizul Haque Dhrubo Dept, Zaed Ikbal Syed Dept
**作者数量**: 6人

**摘要**:
Deep Convolutional Neural Networks (DCNNs) have achieved state-of-the-art performance across various computer vision tasks; however, their practical deployment is limited by computational and memory overhead. This paper introduces Differential Sensitivity Fusion Pruning (DSFP), a novel singleshot filter pruning framework that focuses on evaluating the stability and redundancy of filter importance scores across multiple criteria. DSFP computes a differential sensitivity score for each filter by fusing the discrepancies among gradient-based sensitivity, first-order Taylor expansion, and KL-divergence of activation distributions. An exponential scaling mechanism is applied to emphasize filters with inconsistent importance across metrics-identifying candidates that are structurally unstable or less critical to the model's performance. Unlike iterative or reinforcement learning-based pruning strategies, DSFP is efficient and deterministic, requiring only a single forward-backward pass for scoring and pruning. Extensive experiments across varying pruning rates (50%-70%) demonstrate that DSFP significantly reduces model complexity-achieving over 80% FLOPs reduction-while maintaining high accuracy. For instance, at 70% pruning, our approach retains up to 98.23% of baseline accuracy, surpassing traditional heuristics in both compression and generalization. The proposed method presents an effective solution for scalable and adaptive DCNN compression, paving the way for efficient deployment on edge and mobile platforms.

### 14. FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving

**主要机构**: Cloud Research Team, Samsung SDS
**作者数量**: 9人

**摘要**:
Recent advances in Post-Training Quantization (PTQ) techniques have significantly increased demand for serving quantized large language models (LLMs), enabling higher throughput and substantially reduced memory usage with minimal accuracy loss. Quantized models address memory constraints in LLMs and enhance GPU resource utilization through efficient GPU sharing. However, quantized models have smaller KV block sizes than non-quantized models, causing limited memory efficiency due to memory fragmentation. Also, distinct resource usage patterns between quantized and nonquantized models require efficient scheduling to maximize throughput. To address these challenges, we propose Fine-Serve, an inference serving framework for mixed-precision LLMs. FineServe's key contributions include: (1) KV Slab, a precision-aware adaptive memory management technique dynamically allocating KV cache based on model quantization characteristics, significantly reducing GPU memory fragmentation, and (2) a two-level scheduling framework comprising a global scheduler that places models to GPUs based on request rates, latency SLOs, and memory constraints and efficiency, and a local scheduler that adaptively adjusts batch sizes according to real-time request fluctuations. Experimental results demonstrate that FineServe achieves up to 2.2× higher SLO attainment and 1.8× higher token generation throughput compared to the state-of-the-art GPU sharing systems.

### 15. From Long to Short: LLMs Excel at Trimming Own Reasoning Chains

**主要机构**: University of Sydney, Independent Researcher, National University of Singapore, Singapore Management University
**作者数量**: 5人

**摘要**:
O1/R1-style large reasoning models (LRMs) signal a substantial leap forward over conventional instruction-following LLMs. By applying test-time scaling to generate extended reasoning paths, they establish many SOTAs across a wide range of complex reasoning tasks. However, recent studies show that LRMs are prone to suffer from overthinking-the tendency to overcomplicate simple problems, leading to excessive strategy switching and long, convoluted reasoning traces that hinder their interpretability. To mitigate this issue, we conduct a systematic investigation into the reasoning efficiency of a broad set of LRMs and uncover a common dilemma: the difficulty in balancing multiple generation objectives such as correctness and brevity. Based on this discovery, we propose a test-time scaling method, EDIT(Efficient Dynamic Inference Trimming), which efficiently guides LRMs to identify the shortest correct reasoning paths at test time. EDIT employs constraint-guided generation while jointly tracking length and answer distributions under varying constraints, allowing it to select responses that strike an optimal balance between conciseness and correctness. Extensive experiments across diverse models and datasets show that EDIT substantially enhance the reasoning efficiency, producing compact yet informative outputs that improve readability and user experience.

### 16. Hybrid Fourier Neural Operator-Plasma Fluid Model for Fast and Accurate Multiscale Simulations of High Power Microwave Breakdown

**主要机构**: Group in Computational Science, Dhirubhai Ambani University, DA-IICT, HPC
**作者数量**: 5人

**摘要**:
Modeling and simulation of High Power Microwave (HPM) breakdown, a multiscale phenomenon, is computationally expensive and requires solving Maxwell's equations (EM solver) coupled with a plasma continuity equation (plasma solver). In this work, we present a hybrid modeling approach that combines the accuracy of a differential equation-based plasma fluid solver with the computational efficiency of FNO (Fourier Neural Operator) based EM solver. Trained on data from an in-house FDTDbased plasma-fluid solver, the FNO replaces computationally expensive EM field updates, while the plasma solver governs the dynamic plasma response. The hybrid model is validated on microwave streamer formation, due to diffusion-ionization mechanism, in a 2D scenario for unseen incident electric fields corresponding to entirely new plasma streamer simulations not included in model training, showing excellent agreement with FDTD-based fluid simulations in terms of streamer shape, velocity, and temporal evolution. This hybrid FNO based strategy delivers significant acceleration of the order of 60X compared to traditional simulations for the specified problem size and offers an efficient alternative for computationally demanding multiscale and multiphysics simulations involved in HPM breakdown. Our work also demonstrate how such hybrid pipelines can be used to seamlessly to integrate existing C-based simulation codes with Python-based machine learning frameworks for simulations of plasma sceince and engineering problems.

### 17. INDEX-PRESERVING LIGHTWEIGHT TOKEN PRUNING FOR EFFICIENT DOCUMENT UNDERSTANDING IN VISION-LANGUAGE MODELS

**主要机构**: Hana Institute of Technology Seoul
**作者数量**: 3人

**摘要**:
Recent progress in vision-language models (VLMs) has led to impressive results in document understanding tasks, but their high computational demands remain a challenge. To mitigate the compute burdens, we propose a lightweight token pruning framework that filters out non-informative background regions from document images prior to VLM processing. A binary patch-level classifier removes non-text areas, and a maxpooling refinement step recovers fragmented text regions to enhance spatial coherence. Experiments on real-world document datasets demonstrate that our approach substantially lowers computational costs, while maintaining comparable accuracy.

### 18. LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba

**主要机构**: 
**作者数量**: 2人

**摘要**:
State-of-the-art LocoMamba: To the best of our knowledge, this is the first vision-driven cross-modal DRL framework for quadrupedal locomotion that utilizes the selective state-space model Mamba as the fusion backbone, enabling foresightful and efficient control. • Effective proprioception and depth encoders: A compact input representation is introduced in which an MLP embeds proprioceptive states and a CNN patchifies depth images into spatial tokens tailored for Mamba-based fusion. This design provides immediate state estimates and look-ahead while reducing sensitivity to appearance variation, thereby improving computational efficiency and training stability. • Efficient cross-modal Mamba fusion backbone: Encoded tokens from proprioception and depth are fused using stacked Mamba layers via selective state-space scanning, achieving near-linear time and memory scaling. The backbone supports long-horizon modeling, remains robust to token length and image resolution, and provides a regularizing inductive bias through input-gated, exponentially decaying dynamics. • Robust end-to-end RL training scheme: The policy is trained with PPO using Mamba-fused cross-modal features, complemented by terrain and appearance randomization and an obstacle-density curriculum. A compact, statecentric reward balances task-aligned progress, energy efficiency, and safety, enabling stable learning and consistent performance. • Comprehensive evaluation: Extensive experiments are conducted with static and moving obstacles and uneven terrain, and demonstrate consistent gains over state-of-the-art (SOTA) in terms of return, collision times, and distance moved, along with faster convergence under the same compute budget.

### 19. MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs

**主要机构**: Northwestern University
**作者数量**: 5人

**摘要**:
Deploying Mamba models on microcontrollers (MCUs) remains challenging due to limited memory, the lack of native operator support, and the absence of embeddedfriendly toolchains. We present, to our knowledge, the first deployment of a Mamba-based neural architecture on a resource-constrained MCU, a fully C-based runtime-free inference engine: MambaLite-Micro. Our pipeline maps a trained PyTorch Mamba model to on-device execution by (1) exporting model weights into a lightweight format, and (2) implementing a handcrafted Mamba layer and supporting operators in C with operator fusion and memory layout optimization. MambaLite-Micro eliminates large intermediate tensors, reducing 83.0% peak memory, while maintaining an average numerical error of only 1.7 × 10-5 relative to the PyTorch Mamba implementation. When evaluated on keyword spotting (KWS) and human activity recognition (HAR) tasks, MambaLite-Micro achieved 100% consistency with the PyTorch baselines, fully preserving classification accuracy. We further validated portability by deploying on both ESP32S3 and STM32H7 microcontrollers, demonstrating consistent operation across heterogeneous embedded platforms and paving the way for bringing advanced sequence models like Mamba to real-world resource-constrained applications.

### 20. MEANFLOW-ACCELERATED MULTIMODAL VIDEO-TO-AUDIO SYNTHESIS VIA ONE-STEP GENERATION

**主要机构**: Southwestern University of Finance and Economics, Wuhan University, School of Electronic Information, School of Computing and Artificial Intelligence, MiLM Plus, Xiaomi Inc
**作者数量**: 6人

**摘要**:
A key challenge in synthesizing audios from silent videos is the inherent trade-off between synthesis quality and inference efficiency in existing methods. For instance, flow matching based models rely on modeling instantaneous velocity, inherently require an iterative sampling process, leading to slow inference speeds. To address this efficiency bottleneck, we introduce a MeanFlow-accelerated model that characterizes flow fields using average velocity, enabling one-step generation and thereby significantly accelerating multimodal video-to-audio (VTA) synthesis while preserving audio quality, semantic alignment, and temporal synchronization. Furthermore, a scalar rescaling mechanism is employed to balance conditional and unconditional predictions when classifier-free guidance (CFG) is applied, effectively mitigating CFG-induced distortions in one step generation. Since the audio synthesis network is jointly trained with multimodal conditions, we further evaluate it on text-to-audio (TTA) synthesis task. Experimental results demonstrate that incorporating MeanFlow into the network significantly improves inference speed without compromising perceptual quality on both VTA and TTA synthesis tasks. Demos are provided in https://vta888.github.io/MF-MJT/

### 21. Mitigating Spurious Correlations Between Question and Answer via Chain-of-Thought Correctness Perception Distillation

**主要机构**: Beihang University, Institute of Artificial Intelligence (TeleAI), School of Computer
**作者数量**: 9人

**摘要**:
Large language models (LLMs) excel at reasoning tasks but are expensive to deploy. Thus small language models (SLMs) are fine-tuned on CoT data generated by LLMs to copy LLMs' abilities. However, these CoT data may include noisy rationales that either fail to substantiate the answers or contribute no additional information to support answer prediction, which leads SLMs to capture spurious correlations between questions and answers and compromise the quality of reasoning. In this work, we propose Chain-of-Thought Correctness Perception Distillation (CoPeD), which aims to improve the reasoning quality of the student model from the perspectives of task setting and data utilization. Firstly, we introduce a correctnessaware task setting that encourages the student model to predict answers based on correct rationales and revise them when they are incorrect. This setting improves the faithfulness of reasoning and allows the model to learn from its mistakes. Then, we propose a Correctness-Aware Weighted loss, which dynamically adjusts the contribution of each training instance based on the combined loss of the rationale and the answer. This strategy encourages the model to focus more on samples where the rationale offers stronger support for the correct answer. Experiments have shown that CoPeD is effective on both in-distribution (IND) and out-of-distribution (OOD) benchmark reasoning datasets 1 .

### 22. ProfilingAgent: Profiling-Guided Agentic Reasoning for Adaptive Model Optimization

**主要机构**: Iowa State University, Department of Computer Science
**作者数量**: 4人

**摘要**:
Foundation models face growing compute and memory bottlenecks, hindering deployment on resource-limited platforms without optimization. While model compression techniques like pruning and quantization are widely used, most existing approaches rely on uniform heuristics or rule-based strategies that ignore architectural and runtime heterogeneity. Profiling tools have been developed to expose such bottlenecks by capturing metrics like per-layer latency, memory usage, and compute cost; however, these insights are rarely integrated into automated compression pipelines. In this paper, we propose ProfilingAgent, a profilingguided agentic approach that leverages large language models (LLMs) to automate model compression through structured pruning and post-training dynamic quantization. Our modular pipeline consists of a multi-agent system that leverages both static (e.g., MACs, parameter counts) and dynamic (e.g., latency, memory usage) profiling signals to generate architecture-specific compression strategies. Unlike heuristic baselines for pruning and quantization, our LLM-guided agents reason over profiling traces to produce layer-wise decisions tailored to performance bottlenecks. Experiments conducted on benchmark datasets, such as ImageNet 1K, CIFAR-10 and CIFAR-100, utilizing ResNet-101, ViT-B/16, Swin-B, and DeiT-B/16, demonstrate that our approach to pruning maintains competitive or even improved accuracy (with accuracy drops of about 1% on ImageNet-1K, and gains of up to +2% for ViT-B/16 on smaller datasets without any post-pruning fine-tuning), while quantization achieves memory savings of up to 74% with accuracy drops below 0.5%. Furthermore, our quantization strategy consistently delivers inference speedups of up to 1.74×, effectively reducing latency while preserving performance. Comparative studies with GPT-4o and GPT-4-Turbo underscore the impact of LLM reasoning quality on iterative pruning efficacy. These results establish agentic systems as a scalable solution for profiling-guided model optimization.

### 23. Quantitative Currency Evaluation in Low-Resource Settings through Pattern Analysis to Assist Visually Impaired Users

**主要机构**: Department of Computer Science, George Mason University Fairfax
**作者数量**: 4人

**摘要**:
Currency recognition systems often overlook usability and authenticity assessment, especially in low-resource environments where visually impaired users and offline validation are common. While existing methods focus on denomination classification, they typically ignore physical degradation and forgery, limiting their applicability in real-world conditions. This paper presents a unified framework for currency evaluation that integrates three modules: denomination classification using lightweight CNN models, damage quantification through a novel Unified Currency Damage Index (UCDI), and counterfeit detection using feature-based template matching. The dataset consists of over 82,000 annotated images spanning clean, damaged, and counterfeit notes. Our Custom CNN model achieves high classification performance with low parameter count. The UCDI metric provides a continuous usability score based on binary mask loss, chromatic distortion, and structural feature loss. The counterfeit detection module demonstrates reliable identification of forged notes across varied imaging conditions. The framework supports real-time, on-device inference and addresses key deployment challenges in constrained environments. Results show that accurate, interpretable, and compact solutions can support inclusive currency evaluation in practical settings.

### 24. Quaternion Approximate Networks for Enhanced Image Classification and Oriented Object Detection

**主要机构**: 
**作者数量**: 2人

**摘要**:
This paper introduces Quaternion Approximate Networks (QUAN), a novel deep learning framework that leverages quaternion algebra for rotation equivariant image classification and object detection. Unlike conventional quaternion neural networks attempting to operate entirely in the quaternion domain, QUAN approximates quaternion convolution through Hamilton product decomposition using real-valued operations. This approach preserves geometric properties while enabling efficient implementation with custom CUDA kernels. We introduce Independent Quaternion Batch Normalization (IQBN) for training stability and extend quaternion operations to spatial attention mechanisms. QUAN is evaluated on image classification (CIFAR-10/100, ImageNet), object detection (COCO, DOTA), and robotic perception tasks. In classification tasks, QUAN achieves higher accuracy with fewer parameters and faster convergence compared to existing convolution and quaternion-based models. For objection detection, QUAN demonstrates improved parameter efficiency and rotation handling over standard Convolutional Neural Networks (CNNs) while establishing the SOTA for quaternion CNNs in this downstream task. These results highlight its potential for deployment in resource-constrained robotic systems requiring rotation-aware perception and application in other domains.

### 25. Sensitivity-Aware Post-Training Quantization for Deep Neural Networks

**主要机构**: South China University of Technology
**作者数量**: 6人

**摘要**:
Model quantization reduces neural network parameter precision to achieve compression, but often compromises accuracy. Existing post-training quantization (PTQ) methods employ iterative parameter updates to preserve accuracy under high compression ratios, incurring significant computational complexity and resource overhead, which limits applicability in resource-constrained edge computing and real-time inference scenarios. This paper proposes an efficient PTQ method guided by parameter sensitivity analysis. The approach prioritizes quantization of high-sensitivity parameters, leveraging unquantized low-sensitivity parameters to compensate for quantization errors, thereby mitigating accuracy degradation. Furthermore, by exploiting column-wise clustering of parameter sensitivity, the method introduces a row-parallel quantization framework with a globally shared inverse Hessian matrix update mechanism, reducing computational complexity by an order of magnitude. Experimental results on ResNet-50 and YOLOv5s demonstrate a 20-200-fold quantization speedup over the Optimal Brain Quantization baseline, with mean accuracy loss below 0.3%, confirming the method's efficacy in balancing efficiency and accuracy.

### 26. SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning

**主要机构**: Shanghai Jiao Tong University
**作者数量**: 5人

**摘要**:
Pruning is a typical acceleration method for compute-bound problems by effectively reducing the computation amount. Recently, pruning has been applied to the Vision-Language-Action (VLA) models and emerges as a promising acceleration method by evicting unimportant tokens. However, existing pruning methods only focus on the local information in current action generation and ignore the global information in previous actions, resulting in a reduction of more than 20% in the success rate and limited speedup in some scenarios. In this paper, we point out that the information across consecutive actions exhibits a high degree of similarity, and thus propose the novel insight that combines local information in the current action generation with global information from previous generation for token selection. Based on the insight, we further propose SpecPrune-VLA, a training-free pruning method including two-level token pruning with heuristic control. (1) Static token pruning at action level. We explore the token redundancy through global information in previous actions and the token through local information in the current action generation, statically reducing the number of visual tokens at the action level. (2) Dynamic token pruning at layer level. We exploit the relevance between the tokens and model layer, and dynamically prune tokens based on their layer-specific importance at the layer level. (3) Lightweight action-aware controller. We point out that the generated action can be categorized into coarse-grained and fine-grained based on the speed, with fine-grained actions being sensitive to the error brought by pruning. Therefore, we introduce a lightweight controller that can identify the current action granularity and adjust the pruning strategy accordingly. Extensive experiments show that, compared with the high-performing VLA model OpenVLA-OFT, SpecPrune-VLA achieves an average 1.46× speedup on NVIDIA A800 GPU and 1.57× speedup on NVIDIA GeForce RTX 3090 GPU with negligible loss on task success rate in the LIBERO simulation benchmark.

### 27. StripDet: Strip Attention-Based Lightweight 3D Object Detection from Point Cloud

**主要机构**: School of Integrated Circuits, Shenzhen Campus of Sun Yat-sen University
**作者数量**: 3人

**摘要**:
The deployment of high-accuracy 3D object detection models from point cloud remains a significant challenge due to their substantial computational and memory requirements. To address this, we introduce StripDet, a novel lightweight framework designed for on-device efficiency. First, we propose the novel Strip Attention Block (SAB), a highly efficient module designed to capture long-range spatial dependencies. By decomposing standard 2D convolutions into asymmetric strip convolutions, SAB efficiently extracts directional features while reducing computational complexity from quadratic to linear. Second, we design a hardware-friendly hierarchical backbone that integrates SAB with depthwise separable convolutions and a simple multiscale fusion strategy, achieving end-to-end efficiency. Extensive experiments on the KITTI dataset validate StripDet's superiority. With only 0.65M parameters, our model achieves a 79.97% mAP for car detection, surpassing the baseline PointPillars with a 7× parameter reduction. Furthermore, StripDet outperforms recent lightweight and knowledge distillation-based methods, achieving a superior accuracy-efficiency trade-off while establishing itself as a practical solution for real-world 3D detection on edge devices.

### 28. TrajAware: Graph Cross-Attention and Trajectory-Aware for Generalisable VANETs under Partial Observations

**主要机构**: 
**作者数量**: 3人

**摘要**:
Vehicular ad hoc networks (VANETs) are a crucial component of intelligent transportation systems; however, routing remains challenging due to dynamic topologies, incomplete observations, and the limited resources of edge devices. Existing reinforcement learning (RL) approaches often assume fixed graph structures and require retraining when network conditions change, making them unsuitable for deployment on constrained hardware. We present TrajAware, an RL-based framework designed for edge AI deployment in VANETs. TrajAware integrates three components: (i) action space pruning, which reduces redundant neighbour options while preserving two-hop reachability, alleviating the curse of dimensionality; (ii) graph cross-attention, which maps pruned neighbours to the global graph context, producing features that generalise across diverse network sizes; and (iii) trajectory-aware prediction, which uses historical routes and junction information to estimate real-time positions under partial observations. We evaluate TrajAware in the open-source SUMO simulator using real-world city maps with a leave-one-city-out setup. Results show that TrajAware achieves near-shortest paths and high delivery ratios while maintaining efficiency suitable for constrained edge devices, outperforming state-of-the-art baselines in both full and partial observation scenarios.
