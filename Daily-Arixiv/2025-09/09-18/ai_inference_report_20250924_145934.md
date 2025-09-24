# AI推理加速技术论文分析报告
生成时间: 2025-09-24 14:59:34
分析论文数量: 17篇

## 论文技术简报

### 1. BIRQ: BI-LEVEL SELF-LABELING RANDOM QUANTIZATION FOR SELF-SUPERVISED SPEECH RECOGNITION

IBM Research与Cornell University发布了BIRQ论文，使用双层自标记随机量化技术，解决了自监督语音识别中伪标签既信息丰富又高效的生成挑战，达成了在保持低复杂度和计算效率的同时持续优于BEST-RQ、在LibriSpeech等多个数据集上展示一致增益的效果。

### 2. ChronoForge-RL: Chronological Forging through Reinforcement Learning for Enhanced Video Understanding

发布了ChronoForge-RL论文，使用结合Temporal Apex Distillation (TAD)和KeyFrame-aware Group Relative Policy Optimization (KF-GRPO)的框架及可微分关键帧选择机制，解决了视频理解中处理密集帧的计算不可行性及语义显著帧识别困难问题，达成在VideoMME达69.1%、LVBench达52.7%，超过基线方法且7B参数模型性能可比72B参数模型、性能-参数比提升10倍的效果

### 3. Chunk Based Speech Pre-training with High Resolution Finite Scalar Quantization

Samsung Research America发布了Chunk Based Speech Pre-training with High Resolution Finite Scalar Quantization论文，使用基于块的自监督学习(Chunk SSL)结合高分辨率有限标量量化(FSQ)及组掩码预测损失技术，解决了自监督学习在流式应用中处理部分语音时的妥协问题，达成了流式和离线语音转文本任务（语音识别、语音翻译）的竞争力结果。

### 4. DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

香港科技大学发布了DiEP论文，使用自适应非均匀专家剪枝策略（Differentiable Expert Pruning），解决了现有MoE剪枝方法因各层专家冗余差异导致的性能下降问题，达成了在Mixtral 8×7B上仅保留一半专家即可维持约92%原始性能、在MMLU数据集上较其他剪枝方法性能提升7.1%的效果。

### 5. DistillMatch: Leveraging Knowledge Distillation from Vision Foundation Model for Multimodal Image Matching

武汉大学发布了DistillMatch论文，使用来自视觉基础模型（如DINOv2/DINOv3）的知识蒸馏、模态类别信息注入及V2I-GAN数据增强技术，解决了多模态图像匹配中模态外观差异大、标注数据稀缺导致的性能差和适应性不足问题，达成了在公共数据集上优于现有算法的效果

### 6. Enhancing WSI-Based Survival Analysis with Report-Auxiliary Self-Distillation

南方医科大学与IMT Mines Ales发布了Enhancing WSI-Based Survival Analysis with Report-Auxiliary Self-Distillation论文，使用Report-auxiliary self-distillation (Rasa)框架（结合大型语言模型提取WSI相关文本描述、自蒸馏管道过滤冗余特征及风险感知mix-up策略增强数据），解决了传统基于WSI的生存分析面临特征嘈杂和数据有限导致难以有效捕捉关键预后特征的问题，达成了相比最先进方法更优的有效性。

### 7. LowDiff: Efficient Diffusion Sampling with Low-Resolution Condition

Rice University发布了LowDiff论文，使用级联方法和统一模型逐步从低分辨率生成并优化至目标分辨率的高效扩散框架，解决了扩散模型采样速度慢的问题，达成了超过50%的吞吐量提升，同时保持相当或更好生成质量的效果。

### 8. MEC-Quant: Maximum Entropy Coding for Extremely Low Bit Quantization-Aware Training

研究团队发布了MEC-Quant论文，使用最大熵编码量化（MEC-Quant）技术，解决了极低比特量化感知训练中量化引入的表示偏差问题，首次将QAT极限推至x比特激活且精度媲美甚至超越全精度模型，树立QAT新标杆。

### 9. MoE-CE: Enhancing Generalization for Deep Learning based Channel Estimation via a Mixture-of-Experts Framework

研究团队发布了MoE-CE论文，使用混合专家（MoE）框架（通过多个专家子网络和学习路由器动态选择相关专家），解决了传统深度学习基信道估计在多样条件下（如SNR、RB数量、信道轮廓）泛化能力差（尤其多任务和零样本场景）的问题，达成了在多任务和零样本评估中持续优于传统DL方法、实现显著性能提升并保持效率的效果。

### 10. RadarGaussianDet3D: An Efficient and Effective Gaussian-based 3D Detector with 4D Automotive Radars

天津大学发布了RadarGaussianDet3D论文，使用Point Gaussian Encoder (PGE)与Box Gaussian Loss (BGL)，通过高斯基元和3D Gaussian Splatting技术，解决了现有4D雷达3D检测器特征图稀疏、边界框优化次优及嵌入式设备推理速度不足的问题，达成了SOTA检测精度并显著提升推理速度，满足车载嵌入式设备实时部署需求。

### 11. Region-Aware Deformable Convolutions

Tarbiat Modares University发布了Region-Aware Deformable Convolutions论文，使用Region-Aware Deformable Convolution (RAD-Conv)——通过每个核元素四个边界偏移量创建灵活矩形区域以动态调整大小和形状的卷积算子，解决了传统可变形卷积固定四边形采样区域的限制，达成了精确控制感受野宽高、捕获局部细节和长程依赖（即使1x1小核）并结合注意力适应性与卷积效率构建更具表达性和高效视觉模型的效果。

### 12. SAMPO: Scale-wise Autoregression with Motion PrOmpt for generative world models

西安交通大学与亚马逊发布了SAMPO论文，使用尺度自回归与运动提示（SAMPO）混合框架（结合时间因果解码与双向空间注意力、不对称多尺度分词器及轨迹感知运动提示模块），解决了现有自回归世界模型视觉一致性预测中的空间结构破坏、解码效率低及运动建模不足问题，达成了动作条件视频预测和模型控制中生成质量提升、推理速度快4.4倍及零样本泛化增强的效果。

### 13. SIGHTSOUND-R1: CROSS-MODAL REASONING DISTILLATION FROM VISION TO AUDIO LANGUAGE MODELS

哥伦比亚大学发布了SIGHTSOUND-R1论文，使用跨模态推理蒸馏框架（通过测试时扩展生成音频思维链、音频接地验证及SFT+GRPO蒸馏管道），解决了大型音频语言模型因缺乏大规模音频思维链数据导致复杂音景推理能力不足的问题，达成了提升LALM在域内AVQA测试集及未见听觉场景/问题的推理性能并超过预训练和仅标签蒸馏基线的效果。

### 14. TISDISS: A TRAINING-TIME AND INFERENCE-TIME SCALABLE FRAMEWORK FOR DISCRIMINATIVE SOURCE SEPARATION

中央音乐学院发布了TISDISS论文，使用整合earlysplit多损失监督、共享参数设计与动态推理重复的训练及推理时可扩展统一框架，解决了源分离依赖大型网络导致训练和部署成本高的问题，达成了在标准语音分离基准上实现SOTA性能且减少参数数量、提升低延迟应用性能的效果。

### 15. Training-Free Pyramid Token Pruning for Efficient Large Vision-Language Models via Region, Token, and Instruction-Guided Importance

复旦大学发布了题为《Training-Free Pyramid Token Pruning for Efficient Large Vision-Language Models via Region, Token, and Instruction-Guided Importance》的论文，使用整合区域、token级视觉显著性与指令引导重要性的无训练Pyramid Token Pruning (PTP)技术，解决了大型视觉语言模型处理高分辨率图像时因视觉token过多导致的计算开销大、推理延迟高的问题，达成了显著减少计算开销和推理延迟且性能损失最小的效果。

### 16. ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding

北京大学、华为诺亚方舟实验室发布了ViSpec论文，使用视觉感知的推测解码（ViSpec）框架（含轻量级视觉适配器模块压缩图像令牌并整合到草稿模型注意力机制、提取全局特征增强多模态连贯性及专门训练策略），解决了现有方法在视觉语言模型推测解码中加速有限（<1.5×）的问题，达成了VLM推测解码的显著加速效果。

### 17. Walk and Read Less: Improving the Efficiency of Vision-and-Language Navigation via Tuning-Free Multimodal Token Pruning

Boston University发布了《Walk and Read Less: Improving the Efficiency of Vision-and-Language Navigation via Tuning-Free Multimodal Token Pruning》论文，使用Navigation-Aware Pruning (NAP)技术，解决了现有标记剪枝方法在视觉-语言导航（VLN）中因无法识别无信息标记导致信息丢失、增加计算成本的问题，达成了在保持更高成功率的同时节省超50% FLOPS的效果。

## 论文详细信息

### 1. BIRQ: BI-LEVEL SELF-LABELING RANDOM QUANTIZATION FOR SELF-SUPERVISED SPEECH RECOGNITION

**主要机构**: University of Rochester † IBM Research ‡ Cornell University
**作者数量**: 5人

**摘要**:
Speech is a rich signal, and labeled audio-text pairs are costly to obtain, making self-supervised learning (SSL) essential for scalable representation learning. A core challenge in speech SSL is generating pseudo-labels that are both informative and efficient: strong labels, such as those used in HuBERT [1], improve downstream performance but rely on external encoders and multi-stage pipelines, while efficient methods like BEST-RQ [2] achieve simplicity at the cost of weaker labels. We propose BiRQ, a bilevel SSL framework that combines the efficiency of BEST-RQ with the refinement benefits of HuBERT-style label enhancement. The key idea is to reuse part of the model itself as a pseudo-label generator: intermediate representations are discretized by a random-projection quantizer to produce enhanced labels, while anchoring labels derived directly from the raw input stabilize training and prevent collapse. Training is formulated as an efficient first-order bilevel optimization problem, solved end-to-end with differentiable Gumbel-softmax selection. This design eliminates the need for external label encoders, reduces memory cost, and enables iterative label refinement in an end-to-end fashion. BiRQ consistently improves over BEST-RQ while maintaining low complexity and computational efficiency. We validate our method on various datasets, including 960hour LibriSpeech, 150-hour AMI meetings and 5,000-hour YODAS, demonstrating consistent gains over BEST-RQ.

### 2. ChronoForge-RL: Chronological Forging through Reinforcement Learning for Enhanced Video Understanding

**主要机构**: 
**作者数量**: 1人

**摘要**:
Current state-of-the-art video understanding methods typically struggle with two critical challenges: (1) the computational infeasibility of processing every frame in dense video content and (2) the difficulty in identifying semantically significant frames through naive uniform sampling strategies. In this paper, we propose a novel video understanding framework, called ChronoForge-RL, which combines Temporal Apex Distillation (TAD) and KeyFrame-aware Group Relative Policy Optimization (KF-GRPO) to tackle these issues. Concretely, we introduce a differentiable keyframe selection mechanism that systematically identifies semantic inflection points through a three-stage process to enhance computational efficiency while preserving temporal information. Then, two particular modules are proposed to enable effective temporal reasoning: Firstly, TAD leverages variation scoring, inflection detection, and prioritized distillation to select the most informative frames. Secondly, we introduce KF-GRPO which implements a contrastive learning paradigm with a saliency-enhanced reward mechanism that explicitly incentivizes models to leverage both frame content and temporal relationships. Finally, our proposed ChronoForge-RL achieves 69.1% on VideoMME and 52.7% on LVBench compared to baseline methods, clearly surpassing previous approaches while enabling our 7B parameter model to achieve performance comparable to 72B parameter alternatives, a 10× improvement in performance-to-parameter ratio.

### 3. Chunk Based Speech Pre-training with High Resolution Finite Scalar Quantization

**主要机构**: Samsung Research America
**作者数量**: 2人

**摘要**:
Low latency speech human-machine communication is becoming increasingly necessary as speech technology advances quickly in the last decade. One of the primary factors behind the advancement of speech technology is self-supervised learning. Most self-supervised learning algorithms are designed with full utterance assumption and compromises have to made if partial utterances are presented, which are common in the streaming applications. In this work, we propose a chunk based selfsupervised learning (Chunk SSL) algorithm as an unified solution for both streaming and offline speech pre-training. Chunk SSL is optimized with the masked prediction loss and an acoustic encoder is encouraged to restore indices of those masked speech frames with help from unmasked frames in the same chunk and preceding chunks. A copy and append data augmentation approach is proposed to conduct efficient chunk based pretraining. Chunk SSL utilizes a finite scalar quantization (FSQ) module to discretize input speech features and our study shows a high resolution FSQ codebook, i.e., a codebook with vocabulary size up to a few millions, is beneficial to transfer knowledge from the pre-training task to the downstream tasks. A group masked prediction loss is employed during pre-training to alleviate the high memory and computation cost introduced by the large codebook. The proposed approach is examined in two speech to text tasks, i.e., speech recognition and speech translation. Experimental results on the Librispeech and Must-C datasets show that the proposed method could achieve very competitive results for speech to text tasks at both streaming and offline modes.

### 4. DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

**主要机构**: HKUST Hong Kong
**作者数量**: 5人

**摘要**:
Despite the significant breakthrough of Mixture-of-Experts (MoE), the increasing scale of these MoE models presents huge memory and storage challenges. Existing MoE pruning methods, which involve reducing parameter size with a uniform sparsity across all layers, often lead to suboptimal outcomes and performance degradation due to varying expert redundancy in different MoE layers. To address this, we propose a non-uniform pruning strategy, dubbed Differentiable Expert Pruning (DiEP), which adaptively adjusts pruning rates at the layer level while jointly learning inter-layer importance, effectively capturing the varying redundancy across different MoE layers. By transforming the global discrete search space into a continuous one, our method handles exponentially growing non-uniform expert combinations, enabling adaptive gradient-based pruning. Extensive experiments on five advanced MoE models demonstrate the efficacy of our method across various NLP tasks. Notably, DiEP retains around 92% of original performance on Mixtral 8×7B with only half the experts, outperforming other pruning methods by up to 7.1% on the challenging MMLU dataset.

### 5. DistillMatch: Leveraging Knowledge Distillation from Vision Foundation Model for Multimodal Image Matching

**主要机构**: Electronic Information School, Wuhan University
**作者数量**: 6人

**摘要**:
Multimodal image matching seeks pixel-level correspondences between images of different modalities, crucial for cross-modal perception, fusion and analysis. However, the significant appearance differences between modalities make this task challenging. Due to the scarcity of high-quality annotated datasets, existing deep learning methods that extract modality-common features for matching perform poorly and lack adaptability to diverse scenarios. Vision Foundation Model (VFM), trained on large-scale data, yields generalizable and robust feature representations adapted to data and tasks of various modalities, including multimodal matching. Thus, we propose DistillMatch, a multimodal image matching method using knowledge distillation from VFM. Distill-Match employs knowledge distillation to build a lightweight student model that extracts high-level semantic features from VFM (including DINOv2 and DINOv3) to assist matching across modalities. To retain modality-specific information, it extracts and injects modality category information into the other modality's features, which enhances the model's understanding of cross-modal correlations. Furthermore, we design V2I-GAN to boost the model's generalization by translating visible to pseudo-infrared images for data augmentation. Experiments show that DistillMatch outperforms existing algorithms on public datasets.

### 6. Enhancing WSI-Based Survival Analysis with Report-Auxiliary Self-Distillation

**主要机构**: IMT Mines Ales, Southern Medical University, School of Artificial Intelligence and Data Science, University of Science and Technology of China, Xiamen University, Univ Montpellier, Nanfang Hospital, EuroMov Digital Health in Motion, School of Informatics
**作者数量**: 7人

**摘要**:
Survival analysis based on Whole Slide Images (WSIs) is crucial for evaluating cancer prognosis, as they offer detailed microscopic information essential for predicting patient outcomes. However, traditional WSI-based survival analysis usually faces noisy features and limited data accessibility, hindering their ability to capture critical prognostic features effectively. Although pathology reports provide rich patientspecific information that could assist analysis, their potential to enhance WSI-based survival analysis remains largely unexplored. To this end, this paper proposes a novel Report-auxiliary self-distillation (Rasa) framework for WSI-based survival analysis. First, advanced large language models (LLMs) are utilized to extract fine-grained, WSI-relevant textual descriptions from original noisy pathology reports via a carefully designed task prompt. Next, a self-distillation-based pipeline is designed to filter out irrelevant or redundant WSI features for the student model under the guidance of the teacher model's textual knowledge. Finally, a risk-aware mix-up strategy is incorporated during the training of the student model to enhance both the quantity and diversity of the training data. Extensive experiments carried out on our collected data (CRC) and public data (TCGA-BRCA) demonstrate the superior effectiveness of Rasa against state-of-the-art methods. Our code is available at https://github.com/zhengwang9/Rasa.

### 7. LowDiff: Efficient Diffusion Sampling with Low-Resolution Condition

**主要机构**: California Institute for Creative Technologies, Colorado School of Mines, Rice University, University of Southern, Independent Researcher
**作者数量**: 6人

**摘要**:
Diffusion models have achieved remarkable success in image generation but their practical application is often hindered by the slow sampling speed. Prior efforts of improving efficiency primarily focus on compressing models or reducing the total number of denoising steps, largely neglecting the possibility to leverage multiple input resolutions in the generation process. In this work, we propose LowDiff, a novel and efficient diffusion framework based on a cascaded approach by generating increasingly higher resolution outputs. Besides, LowDiff employs a unified model to progressively refine images from low resolution to the desired resolution. With the proposed architecture design and generation techniques, we achieve comparable or even superior performance with much fewer highresolution sampling steps. LowDiff is applicable to diffusion models in both pixel space and latent space. Extensive experiments on both conditional and unconditional generation tasks across CIFAR-10, FFHQ and ImageNet demonstrate the effectiveness and generality of our method. Results show over 50% throughput improvement across all datasets and settings while maintaining comparable or better quality.

### 8. MEC-Quant: Maximum Entropy Coding for Extremely Low Bit Quantization-Aware Training

**主要机构**: 
**作者数量**: 3人

**摘要**:
Quantization-Aware Training (QAT) has driven much attention to produce efficient neural networks. Current QAT still obtains inferior performances compared with the Full Precision (FP) counterpart. In this work, we argue that quantization inevitably introduce biases into the learned representation, especially under the extremely low-bit setting. To cope with this issue, we propose Maximum Entropy Coding Quantization (MEC-Quant), a more principled objective that explicitly optimizes on the structure of the representation, so that the learned representation is less biased and thus generalizes better to unseen in-distribution samples. To make the objective end-toend trainable, we propose to leverage the minimal coding length in lossy data coding as a computationally tractable surrogate for the entropy, and further derive a scalable reformulation of the objective based on Mixture Of Experts (MOE) that not only allows fast computation but also handles the longtailed distribution for weights or activation values. Extensive experiments on various tasks on computer vision tasks prove its superiority. With MEC-Qaunt, the limit of QAT is pushed to the x-bit activation for the first time and the accuracy of MEC-Quant is comparable to or even surpass the FP counterpart. Without bells and whistles, MEC-Qaunt establishes a new state of the art for QAT. Our code is available at this https URL and has been integrated into MQBench (this https URL)

### 9. MoE-CE: Enhancing Generalization for Deep Learning based Channel Estimation via a Mixture-of-Experts Framework

**主要机构**: 
**作者数量**: 3人

**摘要**:
Reliable channel estimation (CE) is fundamental for robust communication in dynamic wireless environments, where models must generalize across varying conditions such as signal-to-noise ratios (SNRs), the number of resource blocks (RBs), and channel profiles. Traditional deep learning (DL)-based methods struggle to generalize effectively across such diverse settings, particularly under multitask and zero-shot scenarios. In this work, we propose MoE-CE, a flexible mixture-ofexperts (MoE) framework designed to enhance the generalization capability of DL-based CE methods. MoE-CE provides an appropriate inductive bias by leveraging multiple expert subnetworks, each specialized in distinct channel characteristics, and a learned router that dynamically selects the most relevant experts per input. This architecture enhances model capacity and adaptability without a proportional rise in computational cost while being agnostic to the choice of the backbone model and the learning algorithm. Through extensive experiments on synthetic datasets generated under diverse SNRs, RB numbers, and channel profiles, including multitask and zero-shot evaluations, we demonstrate that MoE-CE consistently outperforms conventional DL approaches, achieving significant performance gains while maintaining efficiency.

### 10. RadarGaussianDet3D: An Efficient and Effective Gaussian-based 3D Detector with 4D Automotive Radars

**主要机构**: 
**作者数量**: 4人

**摘要**:
4D automotive radars have gained increasing attention for autonomous driving due to their low cost, robustness, and inherent velocity measurement capability. However, existing 4D radar-based 3D detectors rely heavily on pillar encoders for BEV feature extraction, where each point contributes to only a single BEV grid, resulting in sparse feature maps and degraded representation quality. In addition, they also optimize bounding box attributes independently, leading to suboptimal detection accuracy. Moreover, their inference speed, while sufficient for high-end GPUs, may fail to meet the realtime requirement on vehicle-mounted embedded devices. To overcome these limitations, an efficient and effective Gaussianbased 3D detector, namely RadarGaussianDet3D is introduced, leveraging Gaussian primitives and distributions as intermediate representations for radar points and bounding boxes. In RadarGaussianDet3D, a novel Point Gaussian Encoder (PGE) is designed to transform each point into a Gaussian primitive after feature aggregation and employs the 3D Gaussian Splatting (3DGS) technique for BEV rasterization, yielding denser feature maps. PGE exhibits exceptionally low latency, owing to the optimized algorithm for point feature aggregation and fast rendering of 3DGS. In addition, a new Box Gaussian Loss (BGL) is proposed, which converts bounding boxes into 3D Gaussian distributions and measures their distance to enable more comprehensive and consistent optimization. Extensive experiments on TJ4DRadSet and View-of-Delft demonstrate that RadarGaussianDet3D achieves state-of-the-art detection accuracy while delivering substantially faster inference, highlighting its potential for real-time deployment in autonomous driving.

### 11. Region-Aware Deformable Convolutions

**主要机构**: Tarbiat Modares University
**作者数量**: 2人

**摘要**:
We introduce Region-Aware Deformable Convolution (RAD-Conv), a new convolutional operator that enhances neural networks' ability to adapt to complex image structures. Unlike traditional deformable convolutions, which are limited to fixed quadrilateral sampling areas, RAD-Conv uses four boundary offsets per kernel element to create flexible, rectangular regions that dynamically adjust their size and shape to match image content. This approach allows precise control over the receptive field's width and height, enabling the capture of both local details and long-range dependencies, even with small 1x1 kernels. By decoupling the receptive field's shape from the kernel's structure, RAD-Conv combines the adaptability of attention mechanisms with the efficiency of standard convolutions. This innovative design offers a practical solution for building more expressive and efficient vision models, bridging the gap between rigid convolutional architectures and computationally costly attention-based methods.

### 12. SAMPO: Scale-wise Autoregression with Motion PrOmpt for generative world models

**主要机构**: Xi'an Jiaotong University, Amazon.com, Institute of Artificial Intelligence and Robotics, National Engineering Research Center for Visual Information and Applications, National Key Laboratory of Human-Machine Hybrid Augmented Intelligence
**作者数量**: 10人

**摘要**:
World models allow agents to simulate the consequences of actions in imagined environments for planning, control, and long-horizon decision-making. However, existing autoregressive world models struggle with visually coherent predictions due to disrupted spatial structure, inefficient decoding, and inadequate motion modeling. In response, we propose Scale-wise Autoregression with Motion PrOmpt (SAMPO), a hybrid framework that combines visual autoregressive modeling for intra-frame generation with causal modeling for next-frame generation. Specifically, SAMPO integrates temporal causal decoding with bidirectional spatial attention, which preserves spatial locality and supports parallel decoding within each scale. This design significantly enhances both temporal consistency and rollout efficiency. To further improve dynamic scene understanding, we devise an asymmetric multi-scale tokenizer that preserves spatial details in observed frames and extracts compact dynamic representations for future frames, optimizing both memory usage and model performance. Additionally, we introduce a trajectory-aware motion prompt module that injects spatiotemporal cues about object and robot trajectories, focusing attention on dynamic regions and improving temporal consistency and physical realism. Extensive experiments show that SAMPO achieves competitive performance in action-conditioned video prediction and model-based control, improving generation quality with 4.4× faster inference. We also evaluate SAMPO's zero-shot generalization and scaling behavior, demonstrating its ability to generalize to unseen tasks and benefit from larger model sizes.

### 13. SIGHTSOUND-R1: CROSS-MODAL REASONING DISTILLATION FROM VISION TO AUDIO LANGUAGE MODELS

**主要机构**: Columbia University
**作者数量**: 4人

**摘要**:
While large audio-language models (LALMs) have demonstrated state-of-the-art audio understanding, their reasoning capability in complex soundscapes still falls behind large vision-language models (LVLMs). Compared to the visual domain, one bottleneck is the lack of large-scale chain-of-thought audio data to teach LALM stepwise reasoning. To circumvent this data and modality gap, we present SightSound-R1, a cross-modal distillation framework that transfers advanced reasoning from a stronger LVLM teacher to a weaker LALM student on the same audiovisual question answering (AVQA) dataset. SightSound-R1 consists of three core steps: (i) test-time scaling to generate audio-focused chains of thought (CoT) from an LVLM teacher, (ii) audio-grounded validation to filter hallucinations, and (iii) a distillation pipeline with supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) for the LALM student. Results show that SightSound-R1 improves LALM reasoning performance both in the in-domain AVQA test set as well as in unseen auditory scenes and questions, outperforming both pretrained and label-only distilled baselines. Thus, we conclude that vision reasoning can be effectively transferred to audio models and scaled with abundant audiovisual data.

### 14. TISDISS: A TRAINING-TIME AND INFERENCE-TIME SCALABLE FRAMEWORK FOR DISCRIMINATIVE SOURCE SEPARATION

**主要机构**: Central Conservatory of Music, Department of Music AI and Music IT
**作者数量**: 7人

**摘要**:
Source separation is a fundamental task in speech, music, and audio processing, and it also provides cleaner and larger data for training generative models. However, improving separation performance in practice often depends on increasingly large networks, inflating training and deployment costs. Motivated by recent advances in inference-time scaling for generative modeling, we propose Training-Time and Inference-Time Scalable Discriminative Source Separation (TISDiSS), a unified framework that integrates earlysplit multi-loss supervision, shared-parameter design, and dynamic inference repetitions. TISDiSS enables flexible speed-performance trade-offs by adjusting inference depth without retraining additional models. We further provide systematic analyses of architectural and training choices and show that training with more inference repetitions improves shallow-inference performance, benefiting low-latency applications. Experiments on standard speech separation benchmarks demonstrate state-of-the-art performance with a reduced parameter count, establishing TISDiSS as a scalable and practical framework for adaptive source separation. Code is available at https://github.com/WingSingFung/TISDiSS.

### 15. Training-Free Pyramid Token Pruning for Efficient Large Vision-Language Models via Region, Token, and Instruction-Guided Importance

**主要机构**: Institute of Big Data Fudan University College of Computer Science and Artificial Intelligence, Fudan University, College of Computer Science and Artificial Intelligence, Institute of Big Data
**作者数量**: 7人

**摘要**:
Large Vision-Language Models (LVLMs) have significantly advanced multimodal understanding but still struggle with efficiently processing high-resolution images. Recent approaches partition high-resolution images into multiple sub-images, dramatically increasing the number of visual tokens and causing exponential computational overhead during inference. To address these limitations, we propose a training-free token pruning strategy, Pyramid Token Pruning (PTP), that integrates bottom-up visual saliency at both region and token levels with top-down instruction-guided importance. Inspired by human visual attention mechanisms, PTP selectively retains more tokens from visually salient regions and further leverages textual instructions to pinpoint tokens most relevant to specific multimodal tasks. Extensive experiments across 13 diverse benchmarks demonstrate that our method substantially reduces computational overhead and inference latency with minimal performance loss.

### 16. ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding

**主要机构**: Peking University, School of Intelligence Science and Technology, Huawei Noah's Ark Lab
**作者数量**: 5人

**摘要**:
Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), yet its application to vision-language models (VLMs) remains underexplored, with existing methods achieving only modest speedups (< 1.5×). This gap is increasingly significant as multimodal capabilities become central to large-scale models. We hypothesize that large VLMs can effectively filter redundant image information layer by layer without compromising textual comprehension, whereas smaller draft models struggle to do so. To address this, we introduce Vision-Aware Speculative Decoding (ViSpec), a novel framework tailored for VLMs. ViSpec employs a lightweight vision adaptor module to compress image tokens into a compact representation, which is seamlessly integrated into the draft model's attention mechanism while preserving original image positional information. Additionally, we extract a global feature vector for each input image and augment all subsequent text tokens with this feature to enhance multimodal coherence. To overcome the scarcity of multimodal datasets with long assistant responses, we curate a specialized training dataset by repurposing existing datasets and generating extended outputs using the target VLM with modified prompts. Our training strategy mitigates the risk of the draft model exploiting direct access to the target model's hidden states, which could otherwise lead to shortcut learning when training solely on target model outputs. Extensive experiments validate ViSpec, achieving, to our knowledge, the first substantial speedup in VLM speculative decoding. Code is available at https://github.com/KangJialiang/ViSpec.

### 17. Walk and Read Less: Improving the Efficiency of Vision-and-Language Navigation via Tuning-Free Multimodal Token Pruning

**主要机构**: Boston University
**作者数量**: 4人

**摘要**:
Large models achieve strong performance on Vision-and-Language Navigation (VLN) tasks, but are costly to run in resource-limited environments. Token pruning offers appealing tradeoffs for efficiency with minimal performance loss by reducing model input size, but prior work overlooks VLN-specific challenges. For example, information loss from pruning can effectively increase computational cost due to longer walks. Thus, the inability to identify uninformative tokens undermines the supposed efficiency gains from pruning. To address this, we propose Navigation-Aware Pruning (NAP), which uses navigation-specific traits to simplify the pruning process by pre-filtering tokens into foreground and background. For example, image views are filtered based on whether the agent can navigate in that direction. We also extract navigation-relevant instructions using a Large Language Model. After filtering, we focus pruning on background tokens, minimizing information loss. To further help avoid increases in navigation length, we discourage backtracking by removing low-importance navigation nodes. Experiments on standard VLN benchmarks show NAP significantly outperforms prior work, preserving higher success rates while saving more than 50% FLOPS 1 .
