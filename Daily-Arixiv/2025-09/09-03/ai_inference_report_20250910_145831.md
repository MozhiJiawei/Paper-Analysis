# AI推理加速技术论文分析报告
生成时间: 2025-09-10 14:58:31
分析论文数量: 11篇

## 论文技术简报

### 1. ContraGS: Codebook-Condensed and Trainable Gaussian Splatting for Fast, Memory-Efficient Reconstruction

多伦多大学发布了ContraGS论文，使用码本压缩和可训练的高斯溅射技术，解决了高斯溅射重建中的速度与内存效率问题，达成了快速且内存高效的重建效果。

### 2. Data-Augmented Quantization-Aware Knowledge Distillation

Oakland University发布了Data-Augmented Quantization-Aware Knowledge Distillation论文，使用基于最大化Contextual Mutual Information并确保预测接近真实标签的新指标评估数据增强策略，解决了低精度模型量化感知知识蒸馏中数据增强策略选择问题，达成了显著提升多种模型架构和数据集上现有QAT和KD方法性能的效果。

### 3. DIFFUSION GENERATIVE MODELS MEET COMPRESSED SENSING, WITH APPLICATIONS TO IMAGE DATA AND FINANCIAL TIME SERIES

研究团队发布了《DIFFUSION GENERATIVE MODELS MEET COMPRESSED SENSING, WITH APPLICATIONS TO IMAGE DATA AND FINANCIAL TIME SERIES》论文，使用整合压缩感知与扩散生成模型的降维技术（压缩数据到潜空间、在潜空间训练扩散模型、对潜空间生成样本应用压缩感知算法），解决了扩散模型训练和推理效率问题，达成了在数据稀疏假设下更快收敛并获得潜空间维度最优值的效果。

### 4. E-ARMOR: Edge case Assessment and Review of Multilingual Optical Character Recognition

Sprinklr发布了E-ARMOR论文，使用Sprinklr-Edge-OCR（针对资源受限环境边缘部署优化的新型OCR系统）技术，解决了多语言、嘈杂、多样化真实世界图像中OCR的边缘部署挑战，达成了F1分数0.46、处理速度35×更快（0.17秒/图）且成本仅为LVLMs的0.01×（0.006 USD/千图）的效果

### 5. EGTM: Event-guided Efficient Turbulence Mitigation

西安电子科技大学发布了EGTM论文，使用基于“event-lucky insight”的事件相机引导框架（EGTM）提取像素级无湍流引导，解决了现有湍流缓解方法计算与存储效率低的瓶颈，达成了高效湍流缓解并构建首个真实世界事件驱动TM数据集的效果。

### 6. FROM EDITOR TO DENSE GEOMETRY ESTIMATOR

Zero-shot Depth Estimation Input Image机构发布了《FROM EDITOR TO DENSE GEOMETRY ESTIMATOR》论文，使用从编辑器到密集几何估计器的转换技术，解决了零样本深度估计中的密集几何估计问题，达成了提升零样本场景下深度估计性能的效果

### 7. Multilevel Analysis of Cryptocurrency News using RAG Approach with Fine-Tuned Mistral Large Language Model

Ivan Franko National University of Lviv发布了加密货币新闻多级分析论文，使用微调Mistral 7B大语言模型结合检索增强生成（RAG）、PEFT/LoRA的4-bit量化及知识图谱表示技术，解决了大语言模型幻觉问题，达成了加密货币新闻信息丰富的定性和定量分析效果。

### 8. QUANTV2X: A FULLY QUANTIZED MULTI-AGENT SYSTEM FOR COOPERATIVE PERCEPTION

Purdue University发布了QuantV2X论文，使用首个全量化多智能体系统及统一端到端量化策略，解决了V2X协同感知中计算和传输成本高、效率低、延迟大及难以实时部署的问题，达成了系统级延迟降低3.2倍、mAP30提升9.5且精度与全精度系统相当的效果。

### 9. STA-Net: A Decoupled Shape and Texture Attention Network for Lightweight Plant Disease Classification

研究团队发布了STA-Net论文，使用结合DeepMAD架构搜索和Shape-Texture Attention Module（STAM，含DCNv4形状分支与Gabor滤波器组纹理分支）的技术，解决了轻量级植物病害分类模型在边缘设备部署时现有注意力机制难以捕捉病变形状和纹理细微特征的问题，达成了在CCMT数据集上以401K参数、51.1M FLOPs实现89.00%准确率和88.96% F1分数且STAM显著优于基线模型的效果。

### 10. Towards Efficient General Feature Prediction in Masked Skeleton Modeling

中国科学技术大学等机构发布了Towards Efficient General Feature Prediction in Masked Skeleton Modeling论文，使用General Feature Prediction (GFP)框架（以高层次特征预测替代低层次重构，结合协作学习、动态生成多样化监督信号及约束优化），解决了现有蒙面骨架建模中计算冗余和语义表示有限的问题，达成了6.2×更快训练及下游任务中SOTA性能的效果

### 11. TriLiteNet Lightweight Model for Multi-Task Visual Perception

University of Information Technology发布了TriLiteNet Lightweight Model for Multi-Task Visual Perception论文，使用TriLiteNet轻量级多任务视觉感知模型，解决了Advanced Driver Assistance Systems (ADAS)的实时执行需求，达成了在BDD100k数据集上实现车辆检测、可行驶区域及车道线分割多任务的竞争性能，基础版仅2.35M参数、7.72 GFLOPs，tiny版0.14M参数，且嵌入式设备上低延迟低功耗的效果

## 论文详细信息

### 1. ContraGS: Codebook-Condensed and Trainable Gaussian Splatting for Fast, Memory-Efficient Reconstruction

**主要机构**: University of Toronto
**作者数量**: 10人

**摘要**:


### 2. Data-Augmented Quantization-Aware Knowledge Distillation

**主要机构**: Oakland University
**作者数量**: 2人

**摘要**:
Quantization-aware training (QAT) and Knowledge Distillation (KD) are combined to achieve competitive performance in creating low-bit deep learning models. Existing KD and QAT works focus on improving the accuracy of quantized models from the network output perspective by designing better KD loss functions or optimizing QAT's forward and backward propagation. However, limited attention has been given to understanding the impact of input transformations, such as data augmentation (DA). The relationship between quantization-aware KD and DA remains unexplored. In this paper, we address the question: how to select a good DA in quantization-aware KD, especially for the models with low precisions? We propose a novel metric which evaluates DAs according to their capacity to maximize the Contextual Mutual Information-the information not directly related to an image's label-while also ensuring the predictions for each class are close to the ground truth labels on average. The proposed method automatically ranks and selects DAs, requiring minimal training overhead, and it is compatible with any KD or QAT algorithm. Extensive evaluations demonstrate that selecting DA strategies using our metric significantly improves state-of-the-art QAT and KD works across various model architectures and datasets.

### 3. DIFFUSION GENERATIVE MODELS MEET COMPRESSED SENSING, WITH APPLICATIONS TO IMAGE DATA AND FINANCIAL TIME SERIES

**主要机构**: 
**作者数量**: 4人

**摘要**:
This paper develops dimension reduction techniques for accelerating diffusion model inference in the context of synthetic data generation. The idea is to integrate compressed sensing into diffusion models: (i) compress the data into a latent space, (ii) train a diffusion model in the latent space, and (iii) apply a compressed sensing algorithm to the samples generated in the latent space, facilitating the efficiency of both model training and inference. Under suitable sparsity assumptions on data, the proposed algorithm is proved to enjoy faster convergence by combining diffusion model inference with sparse recovery. As a byproduct, we obtain an optimal value for the latent space dimension. We also conduct numerical experiments on a range of datasets, including image data (handwritten digits, medical images, and climate data) and financial time series for stress testing.

### 4. E-ARMOR: Edge case Assessment and Review of Multilingual Optical Character Recognition

**主要机构**: AI Team Sprinklr Gurgaon, Intern Sprinklr Gurgaon, Ratnesh Jamidar AI Team Sprinklr Gurgaon
**作者数量**: 9人

**摘要**:
Optical Character Recognition (OCR) in multilingual, noisy, and diverse real-world images remains a significant challenge for optical character recognition systems. With the rise of Large Vision-Language Models (LVLMs), there is growing interest in their ability to generalize and reason beyond fixed OCR pipelines. In this work, we introduce Sprinklr-Edge-OCR, a novel OCR system built specifically optimized for edge deployment in resource-constrained environments. We present a large-scale comparative evaluation of five state-of-the-art LVLMs (InternVL, Qwen, GOT OCR, LLaMA, MiniCPM) and two traditional OCR systems (Sprinklr-Edge-OCR, SuryaOCR) on a proprietary, doubly hand annotated dataset of multilingual (54 languages) images. Our benchmark covers a broad range of metrics including accuracy, semantic consistency, language coverage, computational efficiency (latency, memory, GPU usage), and deployment cost. To better reflect real-world applicability, we also conducted edge case deployment analysis, evaluating model performance on CPU only environments. Among the results, Qwen achieved the highest precision (0.54), while Sprinklr-Edge-OCR delivered the best overall F1 score (0.46) and outperformed others in efficiency, processing images 35× faster (0.17 seconds per image on average) and at less than 0.01× of the cost (0.006 USD per 1,000 images) compared to LVLM. Our findings demonstrate that the most optimal OCR systems for edge deployment are the traditional ones even in the era of LLMs due to their low compute requirements, low latency, and very high affordability.

### 5. EGTM: Event-guided Efficient Turbulence Mitigation

**主要机构**: School of Integrated Circuits, Xidian University
**作者数量**: 8人

**摘要**:
Turbulence mitigation (TM) aims to remove the stochastic distortions and blurs introduced by atmospheric turbulence into frame cameras. Existing state-of-the-art deep-learning TM methods extract turbulence cues from multiple degraded frames to find the so-called "lucky", not distorted patch, for "lucky fusion". However, it requires high-capacity network to learn from coarse-grained turbulence dynamics between synchronous frames with limited frame-rate, thus fall short in computational and storage efficiency. Event cameras, with microsecond-level temporal resolution, have the potential to fundamentally address this bottleneck with efficient sparse and asynchronous imaging mechanism. In light of this, we (i) present the fundamental "event-lucky insight" to reveal the correlation between turbulence distortions and inverse spatiotemporal distribution of event streams. Then, build upon this insight, we (ii) propose a novel EGTM framework that extracts pixel-level reliable turbulence-free guidance from the explicit but noisy turbulent events for temporal lucky fusion. Moreover, we (iii) build the first turbulence data acquisition system to contribute the first real-world event-driven TM dataset. This demonstrating the great efficiency merit of introducing event modality into TM task. Demo code and data have been uploaded in supplementary material and will be released once accepted.

### 6. FROM EDITOR TO DENSE GEOMETRY ESTIMATOR

**主要机构**: Zero-shot Depth Estimation Input Image
**作者数量**: 13人

**摘要**:


### 7. Multilevel Analysis of Cryptocurrency News using RAG Approach with Fine-Tuned Mistral Large Language Model

**主要机构**: Ivan Franko National University of Lviv
**作者数量**: 1人

**摘要**:
In the paper, we consider multilevel multitask analysis of cryptocurrency news using a fine-tuned Mistral 7B large language model with retrieval-augmented generation (RAG). On the first level of analytics, the fine-tuned model generates graph and text summaries with sentiment scores as well as JSON representations of summaries. Higher levels perform hierarchical stacking that consolidates sets of graph-based and text-based summaries as well as summaries of summaries into comprehensive reports. The combination of graph and text summaries provides complementary views of cryptocurrency news. The model is fine-tuned with 4-bit quantization using the PEFT/LoRA approach. The representation of cryptocurrency news as knowledge graph can essentially eliminate problems with large language model hallucinations. The obtained results demonstrate that the use of fine-tuned Mistral 7B LLM models for multilevel cryptocurrency news analysis can conduct informative qualitative and quantitative analytics, providing important insights.

### 8. QUANTV2X: A FULLY QUANTIZED MULTI-AGENT SYSTEM FOR COOPERATIVE PERCEPTION

**主要机构**: Purdue University
**作者数量**: 15人

**摘要**:
Cooperative perception through Vehicle-to-Everything (V2X) communication offers significant potential for enhancing vehicle perception by mitigating occlusions and expanding the field of view. However, past research has predominantly focused on improving accuracy metrics without addressing the crucial system-level considerations of efficiency, latency, and real-world deployability. Noticeably, most existing systems rely on full-precision models, which incur high computational and transmission costs, making them impractical for real-time operation in resource-constrained environments. In this paper, we introduce QuantV2X, the first fully quantized multi-agent system designed specifically for efficient and scalable deployment of multi-modal, multi-agent V2X cooperative perception. QuantV2X introduces a unified end-to-end quantization strategy across both neural network models and transmitted message representations that simultaneously reduces computational load and transmission bandwidth. Remarkably, despite operating under low-bit constraints, QuantV2X achieves accuracy comparable to full-precision systems. More importantly, when evaluated under deployment-oriented metrics, QuantV2X reduces system-level latency by 3.2× and achieves a +9.5 improvement in mAP30 over full-precision baselines. Furthermore, QuantV2X scales more effectively, enabling larger and more capable models to fit within strict memory budgets. These results highlight the viability of a fully quantized multi-agent intermediate fusion system for real-world deployment. The system will be publicly released to promote research in this field: https://github.com/ucla-mobility/QuantV2X.

### 9. STA-Net: A Decoupled Shape and Texture Attention Network for Lightweight Plant Disease Classification

**主要机构**: 
**作者数量**: 1人

**摘要**:
Responding to rising global food security needs, precision agriculture and deep learning-based plant disease diagnosis have become crucial. Yet, deploying high-precision models on edge devices is challenging. Most lightweight networks use attention mechanisms designed for generic object recognition, which poorly capture subtle pathological features like irregular lesion shapes and complex textures. To overcome this, we propose a twofold solution: first, using a training-free neural architecture search method (DeepMAD) to create an efficient network backbone for edge devices; second, introducing the Shape-Texture Attention Module (STAM). STAM splits attention into two branches-one using deformable convolutions (DCNv4) for shape awareness and the other using a Gabor filter bank for texture awareness. On the public CCMT plant disease dataset, our STA-Net model (with 401K parameters and 51.1M FLOPs) reached 89.00% accuracy and an F1 score of 88.96%. Ablation studies confirm STAM significantly improves performance over baseline and standard attention models. Integrating domain knowledge via decoupled attention thus presents a promising path for edge-deployed precision agriculture AI. The source code is available at https://github.com/RzMY/STA-Net.

### 10. Towards Efficient General Feature Prediction in Masked Skeleton Modeling

**主要机构**: Zhejiang Gongshang University, Jilin University, University of Science and Technology of China, Hefei University of Technology
**作者数量**: 6人

**摘要**:
Recent advances in the masked autoencoder (MAE) paradigm have significantly propelled self-supervised skeleton-based action recognition. However, most existing approaches limit reconstruction targets to raw joint coordinates or their simple variants, resulting in computational redundancy and limited semantic representation. To address this, we propose a novel General Feature Prediction framework (GFP) for efficient mask skeleton modeling. Our key innovation is replacing conventional low-level reconstruction with high-level feature prediction that spans from local motion patterns to global semantic representations. Specifically, we introduce a collaborative learning framework where a lightweight target generation network dynamically produces diversified supervision signals across spatial-temporal hierarchies, avoiding reliance on pre-computed offline features. The framework incorporates constrained optimization to ensure feature diversity while preventing model collapse. Experiments on NTU RGB+D 60, NTU RGB+D 120 and PKU-MMD demonstrate the benefits of our approach: Computational efficiency (with 6.2× faster training than standard masked skeleton modeling methods) and superior representation quality, achieving state-of-the-art performance in various downstream tasks.

### 11. TriLiteNet Lightweight Model for Multi-Task Visual Perception

**主要机构**: Laboratory of Multimedia Communications, Faculty of Computer Engineering, University of Information Technology
**作者数量**: 2人

**摘要**:
Efficient perception models are essential for Advanced Driver Assistance Systems (ADAS), as these applications require rapid processing and response to ensure safety and effectiveness in real-world environments. To address the real-time execution needs of such perception models, this study introduces the TriLiteNet model. This model can simultaneously manage multiple tasks related to panoramic driving perception. TriLiteNet is designed to optimize performance while maintaining low computational costs. Experimental results on the BDD100k dataset demonstrate that the model achieves competitive performance across three key tasks: vehicle detection, drivable area segmentation, and lane line segmentation. Specifically, the TriLiteNet base demonstrated a recall of 85.6% for vehicle detection, a mean Intersection over Union (mIoU) of 92.4% for drivable area segmentation, and an Acc of 82.3% for lane line segmentation with only 2.35M parameters and a computational cost of 7.72 GFLOPs. Our proposed model includes a tiny configuration with just 0.14M parameters, which provides a multi-task solution with minimal computational demand. Evaluated for latency and power consumption on embedded devices, TriLiteNet in both configurations shows low latency and reasonable power during inference. By balancing performance, computational efficiency, and scalability, TriLiteNet offers a practical and deployable solution for real-world autonomous driving applications. Code is available at https://github.com/chequanghuy/TriLiteNet.
