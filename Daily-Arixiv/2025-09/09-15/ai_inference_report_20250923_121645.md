# AI推理加速技术论文分析报告
生成时间: 2025-09-23 12:16:45
分析论文数量: 24篇

## 论文技术简报

### 1. AsyMoE: Leveraging Modal Asymmetry for Enhanced Expert Specialization in Large Vision-Language Models

宾夕法尼亚大学、哥伦比亚大学发布了AsyMoE论文，使用含模态内专家、双曲跨模态专家及证据优先语言专家组的AsyMoE架构，解决了大视觉语言模型中MoE因模态不对称导致的语言专家上下文失焦与跨模态交互失衡问题，达成较传统MoE和模态特定MoE分别提升26.58%和15.45%准确率、激活参数较密集模型减少25.45%的效果。

### 2. A Novel Compression Framework for YOLOv8: Achieving Real-Time Aerial Object Detection on Edge Devices via Structured Pruning and Channel-Wise Distillation

Malek Ashtar University of Technology发布了新型YOLOv8压缩框架论文，使用结构化剪枝与通道级知识蒸馏（CWD）结合的压缩框架，解决了资源受限边缘设备上YOLOv8实时空中目标检测的高效部署问题，达成YOLOv8m参数减少73.51%、推理速度从26 FPS提升至45 FPS（结合TensorRT后达68 FPS）且AP50仅降2.7%的效果

### 3. CIARD: Cyclic Iterative Adversarial Robustness Distillation

北京大学、南京理工大学发布了CIARD论文，使用多教师框架与对比推损对齐及持续对抗重训练技术，解决了现有对抗鲁棒性蒸馏方法中干净样本性能下降的问题，达成了平均对抗防御率提升3.53%、干净样本准确率提升5.87%的效果。

### 4. COTT-ADNET: LIGHTWEIGHT REAL-TIME COTTON BOLL AND FLOWER DETECTION UNDER FIELD CONDITIONS

佐治亚大学发布了Cott-ADNet论文，使用基于YOLOv11n并引入NeLU增强的全局注意力机制和扩张感受野SPPF模块的轻量化检测器，解决复杂田间条件下棉花铃和花的准确识别问题，达成93.3% mAP50、7.5 GFLOPs的高精度轻量化实时检测效果

### 5. Drone Detection Using a Low-Power Neuromorphic Virtual Tripwire

瑞典国防研究局（FOI）发布了低功耗神经形态虚拟绊网无人机检测论文，使用脉冲神经网络与神经形态相机并部署于神经形态芯片的全神经形态系统，解决了小型无人机对军事人员及民用基础设施的早期自动检测问题，达成了比边缘GPU方案节能数个数量级、可电池运行一年以上且便于在无电力基础设施区域部署的效果。

### 6. Effective Gaussian Management for High-fidelity Object Reconstruction

清华大学发布了“Effective Gaussian Management for High-fidelity Object Reconstruction”论文，使用动态激活球谐函数/法线的致密化策略与基于梯度自适应调整SH阶数及任务解耦剪枝的轻量级高斯表示技术，解决了高斯溅射方法中属性分配不加区分导致的梯度冲突及表示效率问题，达成了在重建质量和效率上优于现有方法且参数显著减少的效果。

### 7. Energy-Efficient Quantized Federated Learning for Resource-constrained IoT devices

巴黎萨克雷大学CentraleSupélec发布了Energy-Efficient Quantized Federated Learning for Resource-constrained IoT devices论文，使用融合有限块长传输、模型量化、误差感知聚合机制并优化上行传输功率的联邦学习框架，解决了资源受限IoT设备在联邦学习中面临的能量有限、通信不可靠及无限块长传输不切实际的挑战，达成了相比标准联邦学习模型能耗降低达75%且保持稳健模型精度的效果。

### 8. Enhancing Physical Consistency in Lightweight World Models

研究团队发布了《Enhancing Physical Consistency in Lightweight World Models》论文，使用物理一致的轻量级世界模型（PIWM）技术，解决了轻量级世界模型的物理一致性问题，达成加权总分提升60.6%，且最小130M Soft Mask模型相比400M最大基线模型加权总分高7.4%、推理速度快28%的效果。

### 9. GhostNetV3-Small: A Tailored Architecture and Comparative Study of Distillation Strategies for Tiny Images

ETIT-KIT Karlsruhe发布了GhostNetV3-Small论文，通过提出针对低分辨率图像的GhostNetV3-Small架构，解决了资源受限边缘设备上小尺度图像分类的高效推理问题，达成了在CIFAR-10上准确率93.94%且显著优于原始GhostNetV3的效果。

### 10. HERO: Rethinking Visual Token Early Dropping in High-Resolution Large Vision-Language Models

复旦大学发布了HERO: Rethinking Visual Token Early Dropping in High-Resolution Large Vision-Language Models论文，使用HERO框架（内容自适应token预算分配与功能感知token选择），解决了高分辨率大视觉语言模型(HR-LVLMs)因视觉token数量激增导致的计算和内存开销大的问题，达成了在多种基准和模型规模上实现优异效率-精度权衡且无需训练的效果。

### 11. iCD: A Implicit Clustering Distillation Mathod for Structural Information Mining

内蒙古工业大学发布了iCD: A Implicit Clustering Distillation Method for Structural Information Mining论文，使用iCD隐式聚类蒸馏方法（利用Gram矩阵在解耦的局部logit表示上挖掘可解释的结构知识），解决了Logit知识蒸馏中决策过程可解释性有限且无需真实标签或特征空间对齐的问题，达成了在基准数据集上跨不同师生架构有效、细粒度分类任务较基线提升+5.08%的效果。

### 12. 

请提供论文的标题、主要机构和摘要信息以生成技术简报。

### 13. LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations

MongoDB Research Sydney发布了LEAF论文，使用基于教师对齐表示的文本嵌入模型知识蒸馏技术，解决了现有嵌入模型训练需判断、难负样本及高数据和基础设施要求的问题，达成了同尺寸下在MTEB v2（英文）排行榜上SOTA（第一）的效果

### 14. MSDNet: Efficient 4D Radar Super-Resolution via Multi-Stage Distillation

研究团队发布了MSDNet论文，使用多阶段蒸馏框架（含重建引导特征蒸馏RGFD、扩散引导特征蒸馏DGFD及噪声适配器），解决了4D雷达超分辨率中现有方法难以平衡精度与效率的问题，实现了高保真重建和低延迟推理并提升下游任务性能。

### 15. Neural Diffeomorphic-Neural Operator for Residual Stress-Induced Deformation Prediction

南京航空航天大学发布了神经微分同胚-神经算子（NDNO）相关论文，使用该框架通过微分同胚映射将复杂三维几何映射到公共参考域并训练神经算子，解决了传统数值方法在多样几何结构下残余应力诱导变形预测计算成本高、跨几何域应用受限的问题，达成了高效准确预测多样几何构件变形、实现跨几何域快速适应的效果。

### 16. REASONING MODELS CAN BE ACCURATELY PRUNED VIA CHAIN-OF-THOUGHT RECONSTRUCTION A PREPRINT

LinkedIn发布了推理模型剪枝相关论文，使用Reasoning-Aware Compression (RAC)技术（剪枝时联合重建输入与模型在线思维链轨迹），解决了推理语言模型剪枝时性能损失大甚至变慢的问题，达成了显著提升剪枝后模型性能并集成现有剪枝工作流（如SparseGPT）的效果。

### 17. Recurrent Cross-View Object Geo-Localization

清华大学、浙江大学发布了Recurrent Cross-View Object Geo-Localization论文，使用ReCOT（循环跨视角目标地理定位Transformer）技术，其通过将跨视角目标地理定位重新表述为循环定位任务，引入可学习令牌迭代优化预测位置，并结合SAM知识蒸馏与RFEM模块，解决了现有方法易受特征噪声影响且缺乏误差校正机制的问题，达成了在标准CVOGL基准上实现SOTA性能并减少60%参数的效果。

### 18. ResidualViT for Efficient Temporally Dense Video Encoding

CIIRC CTU发布了ResidualViT for Efficient Temporally Dense Video Encoding论文，使用ResidualViT技术，解决了时间密集型视频编码的效率问题，达成了高效编码的效果。

### 19. SAGA: Selective Adaptive Gating for Efficient and Expressive Linear Attention

北京交通大学发布了SAGA论文，使用选择性自适应门控（SAGA）技术，解决了线性注意力中均匀压缩KV信息导致的特征冗余、方向对齐丢失及低秩约束问题，达成1280×1280分辨率下吞吐量提升1.76倍、GPU峰值内存减少2.69倍（对比PVT-T）且ImageNet数据集top-1准确率提升达4.4%的效果。

### 20. SynergAI: Edge-to-Cloud Synergy for Architecture-Driven High-Performance Orchestration for AI Inference

雅典国立技术大学发布了SynergAI论文，使用架构驱动的边缘-云协同、整合离线与在线决策策略的动态工作负载分配技术，解决了AI推理服务中云部署网络拥堵/高能耗/隐私问题及边缘计算资源有限的矛盾，达成了平均减少2.4倍QoS违规的效果。

### 21. The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning

研究团队发布了《The Better You Learn, The Smarter You Prune》论文，使用可微视觉token剪枝框架LightVLA（通过动态查询评估token重要性及Gumbel softmax实现可微选择，无额外参数），解决了VLA模型在资源受限平台部署时因大量视觉token导致的计算瓶颈问题，达成了FLOPs减少59.1%、延迟降低38.2%且任务成功率提升2.6%的效果

### 22. The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations

南京大学发布了《The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations》论文，通过利用目标LLM的隐藏表示，将token生成过程建模为马尔可夫链并定义价值函数，解决了现有LLM问题难度估计方法计算成本高或通用性差的问题，实现了难度估计性能优于现有基线，并提升了推理效率。

### 23. TinyServe: Query-Aware Cache Selection for Efficient LLM Serving

哥伦比亚大学、耶鲁大学发布了TinyServe论文，使用查询感知页面选择机制（利用边界框元数据估计注意力相关性实现选择性KV加载），解决了LLM服务中KV缓存访问的高内存和延迟开销问题，达成了3.4倍加速和超2倍内存节省（准确率下降可忽略）的效果。

### 24. VI-SAFE: A SPATIAL-TEMPORAL FRAMEWORK FOR EFFICIENT VIOLENCE DETECTION IN PUBLIC SURVEILLANCE

多伦多大学发布了Vi-SAFE论文，使用集成优化YOLOv8（含Ghost-NetV3骨干、EMA注意力机制及剪枝）与Temporal Segment Network (TSN)的时空框架，解决了公共监控中暴力检测的小规模目标、复杂环境及实时时序分析挑战，达成了在RWF-2000数据集上准确率达0.88（超过TSN单独使用的0.77）且在准确率和效率上优于现有方法的效果。

## 论文详细信息

### 1. AsyMoE: Leveraging Modal Asymmetry for Enhanced Expert Specialization in Large Vision-Language Models

**主要机构**: University of Pennsylvania, Columbia University, University of Michigan, University of Chinese Academy of Sciences, Tsinghua University, Zhejiang University, Alibaba Cloud 3 Nanchang Research Institute, University of Science and Technology, Research Institute of China Telecom Corporate Ltd, Shanghai Jiao Tong University, South China Normal University
**作者数量**: 12人

**摘要**:
Large Vision-Language Models (LVLMs) have demonstrated impressive performance on multimodal tasks through scaled architectures and extensive training. However, existing Mixture of Experts (MoE) approaches face challenges due to the asymmetry between visual and linguistic processing. Visual information is spatially complete, while language requires maintaining sequential context. As a result, MoE models struggle to balance modality-specific features and cross-modal interactions. Through systematic analysis, we observe that language experts in deeper layers progressively lose contextual grounding and rely more on parametric knowledge rather than utilizing the provided visual and linguistic information. To address this, we propose AsyMoE, a novel architecture that models this asymmetry using three specialized expert groups. We design intra-modality experts for modality-specific processing, hyperbolic inter-modality experts for hierarchical crossmodal interactions, and evidence-priority language experts to suppress parametric biases and maintain contextual grounding. Extensive experiments demonstrate that AsyMoE achieves 26.58% and 15.45% accuracy improvements over vanilla MoE and modality-specific MoE respectively, with 25.45% fewer activated parameters than dense models.

### 2. A Novel Compression Framework for YOLOv8: Achieving Real-Time Aerial Object Detection on Edge Devices via Structured Pruning and Channel-Wise Distillation

**主要机构**: Malek Ashtar University of Technology, Faculty of Electrical & Computer Engineering
**作者数量**: 3人

**摘要**:
Efficient deployment of deep learning models for aerial object detection on resource-constrained devices requires significant compression without compromising performance. In this study, we propose a novel three-stage compression pipeline for the YOLOv8 object detection model, integrating sparsityaware training, structured channel pruning, and Channel-Wise Knowledge Distillation (CWD). First, sparsity-aware training introduces dynamic sparsity during model optimization, effectively balancing parameter reduction and detection accuracy. Second, we apply structured channel pruning by leveraging batch normalization scaling factors to eliminate redundant channels, significantly reducing model size and computational complexity. Finally, to mitigate the accuracy drop caused by pruning, we employ CWD to transfer knowledge from the original model, using an adjustable temperature and loss weighting scheme tailored for small and medium object detection. Extensive experiments on the VisDrone dataset demonstrate the effectiveness of our approach across multiple YOLOv8 variants. For YOLOv8m, our method reduces model parameters from 25.85M to 6.85M (a 73.51% reduction), FLOPs from 49.6G to 13.3G, and MACs from 101G to 34.5G, while reducing AP50 by only 2.7%. The resulting compressed model achieves 47.9 AP50 and boosts inference speed from 26 FPS (YOLOv8m baseline) to 45 FPS, enabling real-time deployment on edge devices. We further apply TensorRT as a lightweight optimization step. While this introduces a minor drop in AP50 (from 47.9 to 47.6), it significantly improves inference speed from 45 to 68 FPS, demonstrating the practicality of our approach for high-throughput, resource-constrained scenarios. To our knowledge, this is the first study to integrate CWD-based knowledge distillation with channel pruning on YOLOv8 for aerial object detection, effectively bridging the gap between state-of-the-art detection performance and realworld deployment needs.

### 3. CIARD: Cyclic Iterative Adversarial Robustness Distillation

**主要机构**: Peking University, Nanjing University of Science and Technology, Nanjing University of Industry Technology, HKUST(GZ)
**作者数量**: 7人

**摘要**:
Adversarial robustness distillation (ARD) aims to transfer both performance and robustness from teacher model to lightweight student model, enabling resilient performance on resource-constrained scenarios. Though existing ARD approaches enhance student model's robustness, the inevitable by-product leads to the degraded performance on clean examples. We summarize the causes of this problem inherent in existing methods with dual-teacher framework as: 1 ⃝ The divergent optimization objectives of dualteacher models, i.e., the clean and robust teachers, impede effective knowledge transfer to the student model, and 2 ⃝ The iteratively generated adversarial examples during training lead to performance deterioration of the robust teacher model. To address these challenges, we propose a novel Cyclic Iterative ARD (CIARD) method with two key innovations: 1 ⃝ A multi-teacher framework with contrastive push-loss alignment to resolve conflicts in dualteacher optimization objectives, and 2 ⃝ Continuous adversarial retraining to maintain dynamic teacher robustness against performance degradation from the varying adversarial examples. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate that CIARD achieves remarkable performance with an average 3.53% improvement in adversarial defense rates across various attack scenarios and a 5.87% increase in clean sample accuracy, establishing a new benchmark for balancing model robustness and generalization. Our code is available at https://github.com/eminentgu/CIARD.

### 4. COTT-ADNET: LIGHTWEIGHT REAL-TIME COTTON BOLL AND FLOWER DETECTION UNDER FIELD CONDITIONS

**主要机构**: Institute of Plant Breeding, Genetics and Genomics, School of Software Technology, University of Georgia, Zhejiang University, Department of Crop and Soil Sciences, Wake Forest University, Tifton Campus, Department of Computer Science
**作者数量**: 6人

**摘要**:
Cotton is one of the most important natural fiber crops worldwide, yet harvesting remains limited by labor-intensive manual picking, low efficiency, and yield losses from missing the optimal harvest window. Accurate recognition of cotton bolls and their maturity is therefore essential for automation, yield estimation, and breeding research. We propose Cott-ADNet, a lightweight real-time detector tailored to cotton boll and flower recognition under complex field conditions. Building on YOLOv11n, Cott-ADNet enhances spatial representation and robustness through improved convolutional designs, while introducing two new modules: a NeLUenhanced Global Attention Mechanism to better capture weak and low-contrast features, and a Dilated Receptive Field SPPF to expand receptive fields for more effective multi-scale context modeling at low computational cost. We curate a labeled dataset of 4,966 images, and release an external validation set of 1,216 field images to support future research. Experiments show that Cott-ADNet achieves 91.5% Precision, 89.8% Recall, 93.3% mAP50, 71.3% mAP, and 90.6% F1-Score with only 7.5 GFLOPs, maintaining stable performance under multi-scale and rotational variations. These results demonstrate Cott-ADNet as an accurate and efficient solution for in-field deployment, and thus provide a reliable basis for automated cotton harvesting and high-throughput phenotypic analysis. Code and dataset is available at https://github.com/SweefongWong/Cott-ADNet.

### 5. Drone Detection Using a Low-Power Neuromorphic Virtual Tripwire

**主要机构**: Swedish Defence Research Agency (FOI)
**作者数量**: 6人

**摘要**:
Small drones are an increasing threat to both military personnel and civilian infrastructure, making early and automated detection crucial. In this work we develop a system that uses spiking neural networks and neuromorphic cameras (event cameras) to detect drones. The detection model is deployed on a neuromorphic chip making this a fully neuromorphic system. Multiple detection units can be deployed to create a virtual tripwire which detects when and where drones enter a restricted zone. We show that our neuromorphic solution is several orders of magnitude more energy efficient than a reference solution deployed on an edge GPU, allowing the system to run for over a year on battery power. We investigate how synthetically generated data can be used for training, and show that our model most likely relies on the shape of the drone rather than the temporal characteristics of its propellers. The small size and low power consumption allows easy deployment in contested areas or locations that lack power infrastructure.

### 6. Effective Gaussian Management for High-fidelity Object Reconstruction

**主要机构**: Tsinghua University, School of Communications and Information Engineering, Nanjing University of Posts and Telecommunications, Department of Computer and Information Science, College of Automation, University of Macau, BNRist and School of Software
**作者数量**: 7人

**摘要**:
This paper proposes an effective Gaussian management approach for high-fidelity object reconstruction. Departing from recent Gaussian Splatting (GS) methods that employ indiscriminate attribute assignment, our approach introduces a novel densification strategy that dynamically activates spherical harmonics (SHs) or normals under the supervision of a surface reconstruction module, which effectively mitigates the gradient conflicts caused by dual supervision and achieves superior reconstruction results. To further improve representation efficiency, we develop a lightweight Gaussian representation that adaptively adjusts the SH orders of each Gaussian based on gradient magnitudes and performs task-decoupled pruning to remove Gaussian with minimal impact on a reconstruction task without sacrificing others, which balances the representational capacity with parameter quantity. Notably, our management approach is model-agnostic and can be seamlessly integrated into other frameworks, enhancing performance while reducing model size. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art approaches in both reconstruction quality and efficiency, achieving superior performance with significantly fewer parameters.

### 7. Energy-Efficient Quantized Federated Learning for Resource-constrained IoT devices

**主要机构**: College of Computing, University of Paris-Saclay, CentraleSupélec, Mohammed VI Polytechnic University
**作者数量**: 4人

**摘要**:
Federated Learning (FL) has emerged as a promising paradigm for enabling collaborative machine learning while preserving data privacy, making it particularly suitable for Internet of Things (IoT) environments. However, resource-constrained IoT devices face significant challenges due to limited energy, unreliable communication channels, and the impracticality of assuming infinite blocklength transmission. This paper proposes a federated learning framework for IoT networks that integrates finite blocklength transmission, model quantization, and an erroraware aggregation mechanism to enhance energy efficiency and communication reliability. The framework also optimizes uplink transmission power to balance energy savings and model performance. Simulation results demonstrate that the proposed approach significantly reduces energy consumption by up to 75% compared to a standard FL model, while maintaining robust model accuracy, making it a viable solution for FL in real-world IoT scenarios with constrained resources. This work paves the way for efficient and reliable FL implementations in practical IoT deployments.

### 8. Enhancing Physical Consistency in Lightweight World Models

**主要机构**: 
**作者数量**: 11人

**摘要**:
https://physics-wm.github.io/ the baseline by 60.6% in weighted overall score. Moreover, even when compared with the largest baseline model (400M), the smallest PIWM (130M Soft Mask) achieves a 7.4% higher weighted overall score with a 28% faster inference speed.

### 9. GhostNetV3-Small: A Tailored Architecture and Comparative Study of Distillation Strategies for Tiny Images

**主要机构**: IIIT at ETIT-KIT Karlsruhe, ETIT-KIT Karlsruhe
**作者数量**: 2人

**摘要**:
Deep neural networks have achieved remarkable success across a range of tasks, however their computational demands often make them unsuitable for deployment on resourceconstrained edge devices. This paper explores strategies for compressing and adapting models to enable efficient inference in such environments. We focus on GhostNetV3, a state-of-the-art architecture for mobile applications, and propose GhostNetV3-Small, a modified variant designed to perform better on lowresolution inputs such as those in the CIFAR-10 dataset. In addition to architectural adaptation, we provide a comparative evaluation of knowledge distillation techniques, including traditional knowledge distillation, teacher assistants, and teacher ensembles. Experimental results show that GhostNetV3-Small significantly outperforms the original GhostNetV3 on CIFAR-10, achieving an accuracy of 93.94%. Contrary to expectations, all examined distillation strategies led to reduced accuracy compared to baseline training. These findings indicate that architectural adaptation can be more impactful than distillation in smallscale image classification tasks, highlighting the need for further research on effective model design and advanced distillation techniques for low-resolution domains. 1

### 10. HERO: Rethinking Visual Token Early Dropping in High-Resolution Large Vision-Language Models

**主要机构**: Fudan University xu, Institute of Big Data, College of Computer Science and Artificial Intelligence
**作者数量**: 7人

**摘要**:
By cropping high-resolution images into local tiles and encoding them independently, High-Resolution Large Vision-Language Models (HR-LVLMs) have demonstrated remarkable fine-grained visual understanding capabilities. However, this divide-and-conquer paradigm significantly increases the number of visual tokens, resulting in substantial computational and memory overhead. To better understand and address this challenge, we empirically investigate visual token utilization in HR-LVLMs and uncover three key findings: (1) the local tiles have varying importance, jointly determined by visual saliency and task relevance; (2) the CLS token in CLIPbased vision encoders exhibits a two-stage attention pattern across layers, with each stage attending to different types of visual tokens; (3) the visual tokens emphasized at different stages encode information at varying levels of granularity, playing complementary roles within LVLMs. Building on these insights, we propose HERO, a High-resolution visual tokEn eaRly drOpping framework that integrates contentadaptive token budget allocation with function-aware token selection. By accurately estimating tile-level importance and selectively retaining visual tokens with complementary roles, HERO achieves superior efficiency-accuracy trade-offs across diverse benchmarks and model scales, all in a trainingfree manner. This study provides both empirical insights and practical solutions toward efficient inference in HR-LVLMs.

### 11. iCD: A Implicit Clustering Distillation Mathod for Structural Information Mining

**主要机构**: Inner Mongolia University of Technology
**作者数量**: 9人

**摘要**:
Logit Knowledge Distillation has gained substantial research interest in recent years due to its simplicity and lack of requirement for intermediate feature alignment; however, it suffers from limited interpretability in its decisionmaking process. To address this, we propose implicit Clustering Distillation (iCD): a simple and effective method that mines and transfers interpretable structural knowledge from logits, without requiring ground-truth labels or feature-space alignment. iCD leverages Gram matrices over decoupled local logit representations to enable student models to learn latent semantic structural patterns. Extensive experiments on benchmark datasets demonstrate the effectiveness of iCD across diverse teacher-student architectures, with particularly strong performance in finegrained classification tasks-achieving a peak improvement of +5.08% over the baseline. The code is available at: https://github.com/maomaochongaa/iCD.

### 12. 

**主要机构**: 
**作者数量**: 0人

**摘要**:


### 13. LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations

**主要机构**: MongoDB Research Sydney
**作者数量**: 4人

**摘要**:
SOTA, achieving no.1 on the public MTEB v2 (English) leaderboard for its size. LEAF is applicable to black-box models and in contrast to other embedding model training frameworks, it does not require judgments nor hard negatives, and training can be conducted using small batch sizes. Thus, dataset and training infrastructure requirements for our framework are modest. We make our models publicly available under a permissive Apache 2.0 license.

### 14. MSDNet: Efficient 4D Radar Super-Resolution via Multi-Stage Distillation

**主要机构**: 
**作者数量**: 6人

**摘要**:
4D radar super-resolution, which aims to reconstruct sparse and noisy point clouds into dense and geometrically consistent representations, is a foundational problem in autonomous perception. However, existing methods often suffer from high training cost or rely on complex diffusion-based sampling, resulting in high inference latency and poor generalization, making it difficult to balance accuracy and efficiency. To address these limitations, we propose MSDNet, a multi-stage distillation framework that efficiently transfers dense LiDAR priors to 4D radar features to achieve both high reconstruction quality and computational efficiency. The first stage performs reconstruction-guided feature distillation (RGFD), aligning and densifying the student's features through feature reconstruction. In the second stage, we propose diffusion-guided feature distillation (DGFD), which treats the stage-one distilled features as a noisy version of the teacher's representations and refines them via a lightweight diffusion network. Furthermore, we introduce a noise adapter that adaptively aligns the noise level of the feature with a predefined diffusion timestep, enabling a more precise denoising. Extensive experiments on the VoD and in-house datasets demonstrate that MSDNet achieves both high-fidelity reconstruction and low-latency inference in the task of 4D radar point cloud super-resolution, and consistently improves performance on downstream tasks. The code will be publicly available upon publication.

### 15. Neural Diffeomorphic-Neural Operator for Residual Stress-Induced Deformation Prediction

**主要机构**: Nanjing University of Aeronautics and Astronautics, College of Mechanical and Electrical Engineering
**作者数量**: 5人

**摘要**:
Accurate prediction of machining deformation in structural components is essential for ensuring dimensional precision and reliability. Such deformation often originates from residual stress fields, whose distribution and influence vary significantly with geometric complexity. Conventional numerical methods for modeling the coupling between residual stresses and deformation are computationally expensive, particularly when diverse geometries are considered. Neural operators have recently emerged as a powerful paradigm for efficiently solving partial differential equations, offering notable advantages in accelerating residual stress-deformation analysis. However, their direct application across changing geometric domains faces theoretical and practical limitations. To address this challenge, a novel framework based on diffeomorphic embedding neural operators named neural diffeomorphic-neural operator (NDNO) is introduced. Complex three-dimensional geometries are explicitly mapped to a common reference domain through a diffeomorphic neural network constrained by smoothness and invertibility. The neural operator is then trained on this reference domain, enabling efficient learning of deformation fields induced by residual stresses. Once trained, both the diffeomorphic neural network and the neural operator demonstrate efficient prediction capabilities, allowing rapid adaptation to varying geometries. The proposed method thus provides an effective and computationally efficient solution for deformation prediction in structural components subject to varying geometries. The proposed method is validated to predict both main-direction and multi-direction deformation fields, achieving high accuracy and efficiency across parts with diverse geometries including component types, dimensions and features.

### 16. REASONING MODELS CAN BE ACCURATELY PRUNED VIA CHAIN-OF-THOUGHT RECONSTRUCTION A PREPRINT

**主要机构**: Zhipeng Wang LinkedIn, LinkedIn, Qingquan Song LinkedIn, Massachusetts Institute of Technology, Shao Tang LinkedIn
**作者数量**: 11人

**摘要**:
Reasoning language models such as DeepSeek-R1 produce long chain-of-thought traces during inference time which make them costly to deploy at scale. We show that using compression techniques such as neural network pruning produces greater performance loss than in typical language modeling tasks, and in some cases can make the model slower since they cause the model to produce more thinking tokens but with worse performance. We show that this is partly due to the fact that standard LLM pruning methods often focus on input reconstruction, whereas reasoning is a decode-dominated task. We introduce a simple, drop-in fix: during pruning we jointly reconstruct activations from the input and the model's on-policy chain-of-thought traces. This "Reasoning-Aware Compression" (RAC) integrates seamlessly into existing pruning workflows such as SparseGPT, and boosts their performance significantly. Code reproducing the results in the paper can be found at: https: //github.com/RyanLucas3/RAC

### 17. Recurrent Cross-View Object Geo-Localization

**主要机构**: State Key Laboratory of Intelligent Green Vehicle and Mobility, Tsinghua University, Zhejiang University, College of Information Science and Electronic Engineering
**作者数量**: 8人

**摘要**:
Cross-view object geo-localization (CVOGL) aims to determine the location of a specific object in high-resolution satellite imagery given a query image with a point prompt. Existing approaches treat CVOGL as a one-shot detection task, directly regressing object locations from cross-view information aggregation, but they are vulnerable to feature noise and lack mechanisms for error correction. In this paper, we propose ReCOT, a Recurrent Cross-view Object geolocalization Transformer, which reformulates CVOGL as a recurrent localization task. ReCOT introduces a set of learnable tokens that encode task-specific intent from the query image and prompt embeddings, and iteratively attend to the reference features to refine the predicted location. To enhance this recurrent process, we incorporate two complementary modules: (1) a SAM-based knowledge distillation strategy that transfers segmentation priors from the Segment Anything Model (SAM) to provide clearer semantic guidance without additional inference cost, and (2) a Reference Feature Enhancement Module (RFEM) that introduces a hierarchical attention to emphasize object-relevant regions in the reference features. Extensive experiments on standard CVOGL benchmarks demonstrate that ReCOT achieves state-of-theart (SOTA) performance while reducing parameters by 60% compared to previous SOTA approaches.

### 18. ResidualViT for Efficient Temporally Dense Video Encoding

**主要机构**: CIIRC CTU
**作者数量**: 6人

**摘要**:


### 19. SAGA: Selective Adaptive Gating for Efficient and Expressive Linear Attention

**主要机构**: Beijing Jiaotong University No, Haidian District, Shangyuan Village, Institute of Information Science
**作者数量**: 4人

**摘要**:
While Transformer architecture excel at modeling longrange dependencies-contributing to its widespread adoption in vision tasks-the quadratic complexity of softmax-based attention mechanisms imposes a major bottleneck, particularly when processing highresolution images. Linear attention presents a promising alternative by reformulating the attention computation from (QK)V to Q(KV), thereby reducing the complexity from O(N 2) to O(N) while preserving the global receptive field. However, most existing methods compress historical key-value (KV) information uniformly, which can lead to feature redundancy and the loss of directional alignment with the query (Q). This uniform compression results in low-rank KV feature maps, contributing to a performance gap compared to softmax attention. To mitigate this limitation, we propose Selective Adaptive GAting for Efficient and Expressive Linear Attention (SAGA) , which introduces input-adaptive learnable gates to selectively modulate information aggregation into the KV feature map. These gates enhance semantic diversity and alleviate the low-rank constraint inherent in conventional linear attention. Additionally, we propose an efficient Hadamard-product decomposition method for gate computation, which introduces no additional memory overhead. Experiments demonstrate that SAGA achieves a 1.76× improvement in throughput and a 2.69× reduction in peak GPU memory compared to PVT-T at a resolution of 1280 × 1280. Moreover, it improves top-1 accuracy by up to 4.4% on the Ima-geNet dataset, demonstrating both computational efficiency and model effectiveness.

### 20. SynergAI: Edge-to-Cloud Synergy for Architecture-Driven High-Performance Orchestration for AI Inference

**主要机构**: National Technical University of Athens
**作者数量**: 11人

**摘要**:
The rapid evolution of Artificial Intelligence (AI) and Machine Learning (ML) has significantly heightened computational demands, particularly for inference-serving workloads. While traditional cloud-based deployments offer scalability, they face challenges such as network congestion, high energy consumption, and privacy concerns. In contrast, edge computing provides low-latency and sustainable alternatives but is constrained by limited computational resources. In this work, we introduce SynergAI, a novel framework designed for performance-and architecture-aware inference serving across heterogeneous edge-to-cloud infrastructures. Built upon a comprehensive performance characterization of modern inference engines, SynergAI integrates a combination of offline and online decision-making policies to deliver intelligent, lightweight, and architectureaware scheduling. By dynamically allocating workloads across diverse hardware architectures, it effectively minimizes Quality of Service (QoS) violations. We implement SynergAI within a Kubernetes-based ecosystem and evaluate its efficiency. Our results demonstrate that architecture-driven inference serving enables optimized and architecture-aware deployments on emerging hardware platforms, achieving an average reduction of 2.4× in QoS violations compared to a State-of-the-Art (SotA) solution. CCS Concepts: • Computer systems organization → Heterogeneous (hybrid) systems; Embedded systems; • Software and its engineering → Scheduling; • Computing methodologies → Machine learning.

### 21. The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning

**主要机构**: 
**作者数量**: 10人

**摘要**:
We present LightVLA, a simple yet effective differentiable token pruning framework for vision-language-action (VLA) models. While VLA models have shown impressive capability in executing real-world robotic tasks, their deployment on resource-constrained platforms is often bottlenecked by the heavy attention-based computation over large sets of visual tokens. LightVLA addresses this challenge through adaptive, performance-driven pruning of visual tokens: It generates dynamic queries to evaluate visual token importance, and adopts Gumbel softmax to enable differentiable token selection. Through fine-tuning, LightVLA learns to preserve the most informative visual tokens while pruning tokens which do not contribute to task execution, thereby improving efficiency and performance simultaneously. Notably, LightVLA requires no heuristic "magic numbers" and introduces no additional trainable parameters, making it compatible with modern inference frameworks. Experimental results demonstrate that LightVLA outperforms different VLA models and existing token pruning methods across diverse tasks on the LIBERO benchmark, achieving higher success rates with substantially reduced computational overhead. Specifically, LightVLA reduces FLOPs and latency by 59.1% and 38.2% respectively, with a 2.6% improvement in task success rate. Meanwhile, we also investigate the learnable query-based token pruning method LightVLA * with additional trainable parameters, which also achieves satisfactory performance. Our work reveals that as VLA pursues optimal performance, LightVLA spontaneously learns to prune tokens from a performance-driven perspective. To the best of our knowledge, LightVLA is the first work to apply adaptive visual token pruning to VLA tasks with the collateral goals of efficiency and performance, marking a significant step toward more efficient, powerful and practical real-time robotic systems. Project site: https://liauto-research.github.io/LightVLA.

### 22. The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations

**主要机构**: Nanjing University, State Key Laboratory for Novel Software Technology, Shanghai Artificial Intelligence Laboratory
**作者数量**: 6人

**摘要**:
Estimating the difficulty of input questions as perceived by large language models (LLMs) is essential for accurate performance evaluation and adaptive inference. Existing methods typically rely on repeated response sampling, auxiliary models, or fine-tuning the target model itself, which may incur substantial computational costs or compromise generality. In this paper, we propose a novel approach for difficulty estimation that leverages only the hidden representations produced by the target LLM. We model the token-level generation process as a Markov chain and define a value function to estimate the expected output quality given any hidden state. This allows for efficient and accurate difficulty estimation based solely on the initial hidden state, without generating any output tokens. Extensive experiments across both textual and multimodal tasks demonstrate that our method consistently outperforms existing baselines in difficulty estimation. Moreover, we apply our difficulty estimates to guide adaptive reasoning strategies, including Self-Consistency, Best-of-N, and Self-Refine, achieving higher inference efficiency with fewer generated tokens.

### 23. TinyServe: Query-Aware Cache Selection for Efficient LLM Serving

**主要机构**: Columbia University New York, Yale University New Haven
**作者数量**: 2人

**摘要**:
Serving large language models (LLMs) efficiently remains challenging due to the high memory and latency overhead of key-value (KV) cache access during autoregressive decoding. We present Tiny-Serve, a lightweight and extensible serving system for deploying tiny LLMs (e.g., TinyLLaMA, GPT2-345M) with support for structured KV sparsity, plugin-based token selection, and hardwareefficient attention kernels. Unlike prior simulation frameworks, TinyServe executes real-time decoding with configurable sparsity strategies and fine-grained instrumentation. To reduce decoding cost, we introduce a query-aware page selection mechanism that leverages bounding-box metadata to estimate attention relevance between the query and KV cache blocks. This enables selective KV loading with minimal overhead and no model modifications. Our fused CUDA kernel integrates page scoring, sparse memory access, and masked attention in a single pass. Experiments show that TinyServe achieves up to 3.4× speedup and over 2× memory savings with negligible accuracy drop. Additional analysis of cache reuse, page hit rate, and multi-GPU scaling confirms its practicality as an efficient system-level design for LLM training and inference research on resource-constrained hardware. CCS Concepts • Computer systems organization → Parallel architectures.

### 24. VI-SAFE: A SPATIAL-TEMPORAL FRAMEWORK FOR EFFICIENT VIOLENCE DETECTION IN PUBLIC SURVEILLANCE

**主要机构**: Faculty of Applied Science and Engineering, University of Toronto, College of Computer Science, University of China, Sichuan University, School of Information Network Security, People's Public Security, City University of Hong Kong, Department of Architecture and Civil Engineering
**作者数量**: 7人

**摘要**:
The automatic detection of violent behaviors, such as physical altercations in public areas, is critical for public safety. This study addresses challenges in violence detection, including small-scale targets, complex environments, and real-time temporal analysis. We propose Vi-SAFE, a spatial-temporal framework that integrates an enhanced YOLOv8 with a Temporal Segment Network (TSN) for video surveillance. The YOLOv8 model is optimized with Ghost-NetV3 as a lightweight backbone, an exponential moving average (EMA) attention mechanism, and pruning to reduce computational cost while maintaining accuracy. YOLOv8 and TSN are trained separately on pedestrian and violence datasets, where YOLOv8 extracts human regions and TSN performs binary classification of violent behavior. Experiments on the RWF-2000 dataset show that Vi-SAFE achieves an accuracy of 0.88, surpassing TSN alone (0.77) and outperforming existing methods in both accuracy and efficiency, demonstrating its effectiveness for public safety surveillance.Code is available at https://anonymous. 4open.science/r/Vi-SAFE-3B42/README.md.
