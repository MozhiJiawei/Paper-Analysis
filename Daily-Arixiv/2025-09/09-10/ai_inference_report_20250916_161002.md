# AI推理加速技术论文分析报告
生成时间: 2025-09-16 16:10:02
分析论文数量: 11篇

## 论文技术简报

### 1. ADAPTIVE KNOWLEDGE DISTILLATION USING A DEVICE-AWARE TEACHER FOR LOW-COMPLEXITY ACOUSTIC SCENE CLASSIFICATION

Seoul National University of Science and Technology发布了低复杂度设备鲁棒性声学场景分类论文，使用基于设备感知教师集成（含Device-Aware Feature Alignment损失）的自适应知识蒸馏及设备特定微调技术，解决了低复杂度声学场景分类中严格复杂度约束及对已见/未见设备鲁棒泛化的问题，达成开发集准确率57.93%、显著优于官方基线（尤其在未见设备上）的效果

### 2. Adaptive Pareto-Optimal Token Merging for Edge Transformer Models in Semantic Communication

卡尔加里大学发布了关于边缘Transformer模型语义通信的论文，使用自适应帕累托最优令牌合并框架（结合多目标优化与高斯过程贝叶斯优化构建帕累托前沿），解决了大尺度Transformer模型在资源受限6G网络中的高计算需求与部署难题，达成了显著减少浮点运算及传输资源使用、在多SNR条件下保持高精度并实现基于信道质量的延迟与语义保真度动态平衡的效果。

### 3. Benchmarking Energy Efficiency of Large Language Models Using vLLM

University of Applied Sciences Eindhoven发布了Benchmarking Energy Efficiency of Large Language Models Using vLLM论文，使用vLLM技术，解决了大型语言模型能效基准测试问题，达成了建立能效评估基准的效果。

### 4. BUTTERFLYQUANT: ULTRA-LOW-BIT LLM QUANTI-ZATION THROUGH LEARNABLE ORTHOGONAL BUT-TERFLY TRANSFORMS

相关机构发布了BUTTERFLYQUANT论文，使用可学习的正交蝴蝶变换及均匀性正则化技术，解决了极端2位LLM量化中因激活值异常值导致的性能骤降问题，在LLaMA-2-7B 2位量化下困惑度从22.1降至15.4。

### 5. CCF: A Context Compression Framework for Efficient Long-Sequence Language Modeling

厦门大学发布了CCF论文，使用上下文压缩框架（通过学习分层潜在表示整合分段语义聚合与键值记忆编码，并结合增量分段解码与稀疏水库采样的训练优化策略），解决了长序列语言模型扩展时的计算和内存负担问题，达成了在高压缩比下实现有竞争力的困惑度并显著提升吞吐量和内存效率的效果。

### 6. DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech

University of Alabama at Birmingham和FPT Software AI Center发布了DiFlow-TTS论文，使用纯离散流匹配与因子化语音属性建模技术，解决了现有流匹配方法未充分利用离散表示优势及推理慢、重复伪影问题，达成了低延迟零样本语音合成，推理速度较最新基线提升25.8倍并提升自然度等关键指标的效果

### 7. In-Loop Filtering Using Learned Look-Up Tables for Video Coding

研究团队发布了《In-Loop Filtering Using Learned Look-Up Tables for Video Coding》论文，使用基于学习查找表（LUT）的环内滤波框架LUT-ILF++（含多类LUT协作、跨分量索引及LUT压缩技术），解决了基于神经网络的环内滤波计算复杂度高、对专用硬件要求高的问题，达成了在通用测试序列上平均0.82%-4.11%码率降低且时间复杂度和存储成本更低的效果

### 8. 

University of British Columbia发布了相关论文，使用3D Attention U-Net（带残差块）及基于BraTS 2021数据集的迁移学习技术，解决了撒哈拉以南非洲高质量标注MRI数据稀缺导致难以部署高级胶质瘤分割模型的问题，达成了ET、NETC、SNFH的Dice分数分别达0.76、0.80、0.85且模型紧凑（约90MB）、消费级硬件上亚分钟推理的效果。

### 9. SQAP-VLA: A SYNERGISTIC QUANTIZATION-AWARE PRUNING FRAMEWORK FOR HIGH-PERFORMANCE VISION-LANGUAGE-ACTION MODELS

南京大学和亚利桑那大学发布了SQAP-VLA论文，使用协同量化感知剪枝框架（通过协同设计量化与令牌剪枝克服不兼容性），解决了VLA模型计算和内存成本高及现有方法不兼容的部署难题，达成了1.93倍推理加速和平均成功率提升4.5%的效果。

### 10. Unified Start, Personalized End: Progressive Pruning for Efficient 3D Medical Image Segmentation

西北工业大学发布了PSP-Seg论文，使用渐进式剪枝框架（结合块级剪枝与功能解耦损失），解决了3D医学图像分割资源与时间消耗大、现有模型静态适应性差的问题，达成了轻量级变体PSP-Seg-S性能与nnU-Net相当，同时减少GPU内存42-45%、训练时间29-48%及参数83-87%的效果。

### 11. VoxelFormer: Parameter-Efficient Multi-Subject Visual Decoding from fMRI

研究团队发布了VoxelFormer论文，使用整合Token Merging Transformer (ToMer)与query-driven Q-Former的轻量级Transformer架构，解决了fMRI视觉解码中依赖特定受试者训练导致可扩展性和部署受限的问题，达成了参数显著少于现有方法且在训练受试者上实现竞争性检索性能的效果。

## 论文详细信息

### 1. ADAPTIVE KNOWLEDGE DISTILLATION USING A DEVICE-AWARE TEACHER FOR LOW-COMPLEXITY ACOUSTIC SCENE CLASSIFICATION

**主要机构**: Seoul National University of Science and Technology
**作者数量**: 2人

**摘要**:
In this technical report, we describe our submission for Task 1, Low-Complexity Device-Robust Acoustic Scene Classification, of the DCASE 2025 Challenge. Our work tackles the dual challenges of strict complexity constraints and robust generalization to both seen and unseen devices, while also leveraging the new rule allowing the use of device labels at test time. Our proposed system is based on a knowledge distillation framework where an efficient CP-MobileNet student learns from a compact, specialized two-teacher ensemble. This ensemble combines a baseline PaSST teacher, trained with standard cross-entropy, and a 'generalization expert' teacher. This expert is trained using our novel Device-Aware Feature Alignment (DAFA) loss, adapted from prior work, which explicitly structures the feature space for device robustness. To capitalize on the availability of test-time device labels, the distilled student model then undergoes a final device-specific fine-tuning stage. Our proposed system achieves a final accuracy of 57.93% on the development set, demonstrating a significant improvement over the official baseline, particularly on unseen devices.

### 2. Adaptive Pareto-Optimal Token Merging for Edge Transformer Models in Semantic Communication

**主要机构**: Department of Electrical and Software Engineering, Centre for Wireless Communications, University of Calgary, KU 6G Research Centre, University of Oulu, Khalifa University, College of Computing and Mathematical Sciences
**作者数量**: 4人

**摘要**:
Large-scale transformer models have emerged as a powerful tool for semantic communication systems, enabling edge devices to extract rich representations for robust inference across noisy wireless channels. However, their substantial computational demands remain a major barrier to practical deployment in resource-constrained 6G networks. In this paper, we present a training-free framework for adaptive token merging in pretrained vision transformers to jointly reduce inference time and transmission resource usage. We formulate the selection of per-layer merging proportions as a multi-objective optimization problem to balance accuracy and computational cost. We employ Gaussian process-based Bayesian optimization to construct a Pareto frontier of optimal configurations, enabling flexible runtime adaptation to dynamic application requirements and channel conditions. Extensive experiments demonstrate that our method consistently outperforms other baselines and achieves significant reductions in floating-point operations while maintaining competitive accuracy across a wide range of signal-tonoise ratio (SNR) conditions. Additional results highlight the effectiveness of adaptive policies that adjust merging aggressiveness in response to channel quality, providing a practical mechanism to trade off latency and semantic fidelity on demand. These findings establish a scalable and efficient approach for deploying transformer-based semantic communication in future edge intelligence systems.

### 3. Benchmarking Energy Efficiency of Large Language Models Using vLLM

**主要机构**: University of Applied Sciences Eindhoven, Allumni Master Applied IT Fontys
**作者数量**: 1人

**摘要**:


### 4. BUTTERFLYQUANT: ULTRA-LOW-BIT LLM QUANTI-ZATION THROUGH LEARNABLE ORTHOGONAL BUT-TERFLY TRANSFORMS

**主要机构**: 
**作者数量**: 4人

**摘要**:
Large language models require massive memory footprints, severely limiting deployment on consumer hardware. Quantization reduces memory through lower numerical precision, but extreme 2-bit quantization suffers from catastrophic performance loss due to outliers in activations. Rotation-based methods such as QuIP and QuaRot apply orthogonal transforms to eliminate outliers before quantization, using computational invariance: y = Wx = (WQ T)(Qx) for orthogonal Q. However, these methods use fixed transforms-Hadamard matrices achieving optimal worst-case coherence µ = 1/ √ n-that cannot adapt to specific weight distributions. We identify that different transformer layers exhibit distinct outlier patterns, motivating layer-adaptive rotations rather than one-size-fits-all approaches. We propose ButterflyQuant, which replaces Hadamard rotations with learnable butterfly transforms parameterized by continuous Givens rotation angles. Unlike Hadamard's discrete {+1,-1} entries that are non-differentiable and prohibit gradient-based learning, butterfly transforms' continuous parameterization enables smooth optimization while guaranteeing orthogonality by construction. This orthogonal constraint ensures theoretical guarantees in outlier suppression while achieving O(n log n) computational complexity with only n log n 2 learnable parameters. We further introduce a uniformity regularization on posttransformation activations to promote smoother distributions amenable to quantization. Learning requires only 128 calibration samples and converges in minutes on a single GPU-a negligible one-time cost. On LLaMA-2-7B with 2-bit quantization, ButterflyQuant achieves 15.4 perplexity versus 22.1 for QuaRot.

### 5. CCF: A Context Compression Framework for Efficient Long-Sequence Language Modeling

**主要机构**: Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University
**作者数量**: 7人

**摘要**:
Scaling language models to longer contexts is essential for capturing rich dependencies across extended discourse. However, naïve context extension imposes significant computational and memory burdens, often resulting in inefficiencies during both training and inference. In this work, we propose CCF, a novel context compression framework designed to enable efficient long-context modeling by learning hierarchical latent representations that preserve global semantics while aggressively reducing input redundancy. CCF integrates segment-wise semantic aggregation with key-value memory encoding, forming compact representations that support accurate reconstruction and long-range understanding. To further enhance scalability, we introduce a trainingefficient optimization strategy that couples incremental segment decoding with sparse reservoir sampling, substantially reducing memory overhead without degrading performance. Empirical results on multiple long-context language modeling benchmarks demonstrate that CCF achieves competitive perplexity under high compression ratios, and significantly improves throughput and memory efficiency compared to existing approaches. These findings highlight the potential of structured compression for scalable and effective longcontext language modeling. * Equal contribution. The ranking is decided by a coin flip. AAAI 2026 underreview.

### 6. DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech

**主要机构**: FPT Software AI Center, University of Alabama at Birmingham
**作者数量**: 5人

**摘要**:
Zero-shot Text-to-Speech (TTS) aims to synthesize highquality speech that mimics the voice of an unseen speaker using only a short reference sample, requiring not only speaker adaptation but also accurate modeling of prosodic attributes. Recent approaches based on language models, diffusion, and flow matching have shown promising results in zero-shot TTS, but still suffer from slow inference and repetition artifacts. Discrete codec representations have been widely adopted for speech synthesis, and recent works have begun to explore diffusion models in purely discrete settings, suggesting the potential of discrete generative modeling for speech synthesis. However, existing flow-matching methods typically embed these discrete tokens into a continuous space and apply continuous flow matching, which may not fully leverage the advantages of discrete representations. To address these challenges, we introduce DiFlow-TTS, which, to the best of our knowledge, is the first model to explore purely Discrete Flow Matching for speech synthesis. DiFlow-TTS explicitly models factorized speech attributes within a compact and unified architecture. It leverages in-context learning by conditioning on textual content, along with prosodic and acoustic attributes extracted from a reference speech, enabling effective attribute cloning in a zero-shot setting. In addition, the model employs a factorized flow prediction mechanism with distinct heads for prosody and acoustic details, allowing it to learn aspect-specific distributions. Experimental results demonstrate that DiFlow-TTS achieves promising performance in several key metrics, including naturalness, prosody, preservation of speaker style, and energy control. It also maintains a compact model size and achieves low-latency inference, generating speech up to 25.8 times faster than the latest existing baselines. Code and audio samples are available on our demo page 1 .

### 7. In-Loop Filtering Using Learned Look-Up Tables for Video Coding

**主要机构**: 
**作者数量**: 6人

**摘要**:
In-loop filtering (ILF) is a key technology in video coding standards to reduce artifacts and enhance visual quality. Recently, neural network-based ILF schemes have achieved remarkable coding gains, emerging as a powerful candidate for next-generation video coding standards. However, the use of deep neural networks (DNN) brings significant computational and time complexity or high demands for dedicated hardware, making it challenging for general use. To address this limitation, we study a practical ILF solution by adopting look-up tables (LUTs). After training a DNN with a restricted reference range for ILF, all possible inputs are traversed, and the output values of the DNN are cached into LUTs. During the coding process, the filtering process is performed by simply retrieving the filtered pixel through locating the input pixels and interpolating between the cached values, instead of relying on heavy inference computations. In this paper, we propose a universal LUT-based ILF framework, termed LUT-ILF++. First, we introduce the cooperation of multiple kinds of filtering LUTs and propose a series of customized indexing mechanisms to enable better filtering reference perception with limited storage consumption. Second, we propose the cross-component indexing mechanism to enable the filtering of different color components jointly. Third, in order to make our solution practical for coding uses, we propose the LUT compaction scheme to enable the LUT pruning, achieving a lower storage cost of the entire solution. The proposed framework is implemented in the Versatile Video Coding reference software. Experimental results show that the proposed framework achieves on average 0.82%/2.97%/1.63% and 0.85%/4.11%/2.06% bitrate reduction for common test sequences, under the all-intra and random-access configurations, respectively. Compared to DNN-based solutions, our proposed solution has much lower time complexity and storage cost.

### 8. 

**主要机构**: University of British Columbia, Department of Clinical & Radiation Oncology, College of Science, University of Cape Town, CES Laboratory, London School of Hygiene and Tropical Medicine, Medical Artificial Intelligence Laboratory (MAI Lab), University of Sfax, Department of Medicine, McGill University, Majmaah University, National School of Engineers of Sfax, Department of Anatomy, Faculty of Medicine of Sfax, Department of Computer Science and Information, Higher Institute of Medical Technologies of Tunis, AL-Majmaah, University ElManar, Laboratory of Biophysics and Medical Technologies, Mulungushi University, Department of Neurology and Neurosurgery, Department of Electrical and Computer Engineering, Montreal Neurological Institute, Bayero University, Aminu Kano Teaching Hospital, Multimodal Imaging of Neurodegenerative Diseases (MiND), Department of Computer Science and IT, Department of Radiology, Lawson Health Research Institute, School of Computing and Information Systems, Federal University Lokoja college of health sciences
**作者数量**: 17人

**摘要**:
Gliomas are the most prevalent type of primary brain tumors, and their accurate segmentation from MRI is critical for diagnosis, treatment planning, and longitudinal monitoring. However, the scarcity of high-quality annotated imaging data in Sub-Saharan Africa (SSA) poses a significant challenge for deploying advanced segmentation models in clinical workflows. This study introduces a robust and computationally efficient deep learning framework tailored for resource-constrained settings. We leveraged a 3D Attention U-Net architecture augmented with residual blocks and enhanced through transfer learning from pretrained weights on the BraTS 2021 dataset. Our model was evaluated on 95 MRI cases from the BraTS-Africa dataset, a benchmark for glioma segmentation in SSA MRI data. Despite the limited data quality and quantity, our approach achieved Dice scores of 0.76 (Enhancing Tumor-ET), 0.80 (Necrotic and Non-Enhancing Tumor Core-NETC), and 0.85 (Surrounding Non-Functional Hemisphere-SNFH). These results demonstrate the generalizability of the proposed model and its potential to support clinical decision making in low-resource settings. The compact architecture (∼ 90 MB) and sub-minute per-volume inference time on consumer-grade hardware, further underscores its practicality for deployment in SSA health systems. This work contributes toward closing the gap in equitable AI for global health by empowering underserved regions with high performing and accessible medical imaging solutions.

### 9. SQAP-VLA: A SYNERGISTIC QUANTIZATION-AWARE PRUNING FRAMEWORK FOR HIGH-PERFORMANCE VISION-LANGUAGE-ACTION MODELS

**主要机构**: Nanjing University, School of Electronic Science and Engineering, University of Arizona
**作者数量**: 6人

**摘要**:
Vision-Language-Action (VLA) models exhibit unprecedented capabilities for embodied intelligence. However, their extensive computational and memory costs hinder their practical deployment. Existing VLA compression and acceleration approaches conduct quantization or token pruning in an ad-hoc manner but fail to enable both for a holistic efficiency improvement due to an observed incompatibility. This work introduces SQAP-VLA, the first structured, training-free VLA inference acceleration framework that simultaneously enables state-of-the-art quantization and token pruning. We overcome the incompatibility by co-designing the quantization and token pruning pipeline, where we propose new quantization-aware token pruning criteria that work on an aggressively quantized model while improving the quantizer design to enhance pruning effectiveness. When applied to standard VLA models, SQAP-VLA yields significant gains in computational efficiency and inference speed while successfully preserving core model performance, achieving a ×1.93 speedup and up to a 4.5% average success rate enhancement compared to the original model.

### 10. Unified Start, Personalized End: Progressive Pruning for Efficient 3D Medical Image Segmentation

**主要机构**: National Engineering Laboratory for Integrated Aero-Space-Ground-Ocean Big Data Application Technology, School of Computer Science and Engineering, Northwestern Polytechnical University
**作者数量**: 4人

**摘要**:
3D medical image segmentation often faces heavy resource and time consumption, limiting its scalability and rapid deployment in clinical environments. Existing efficient segmentation models are typically static and manually designed prior to training, which restricts their adaptability across diverse tasks and makes it difficult to balance performance with resource efficiency. In this paper, we propose PSP-Seg, a progressive pruning framework that enables dynamic and efficient 3D segmentation. PSP-Seg begins with a redundant model and iteratively prunes redundant modules through a combination of block-wise pruning and a functional decoupling loss. We evaluate PSP-Seg on five public datasets, benchmarking it against seven state-of-the-art models and six efficient segmentation models. Results demonstrate that the lightweight variant, PSP-Seg-S, achieves performance on par with nnU-Net while reducing GPU memory usage by 42-45%, training time by 29-48%, and parameter number by 83-87% across all datasets. These findings underscore PSP-Seg's potential as a cost-effective yet high-performing alternative for widespread clinical application. Code and weights will be available once accepted.

### 11. VoxelFormer: Parameter-Efficient Multi-Subject Visual Decoding from fMRI

**主要机构**: 
**作者数量**: 7人

**摘要**:
Recent advances in fMRI-based visual decoding have enabled compelling reconstructions of perceived images. However, most approaches rely on subject-specific training, limiting scalability and practical deployment. We introduce VoxelFormer, a lightweight transformer architecture that enables multi-subject training for visual decoding from fMRI. VoxelFormer integrates a Token Merging Transformer (ToMer) for efficient voxel compression and a query-driven Q-Former that produces fixed-size neural representations aligned with the CLIP image embedding space. Evaluated on the 7T Natural Scenes Dataset, VoxelFormer achieves competitive retrieval performance on subjects included during training with significantly fewer parameters than existing methods. These results highlight token merging and query-based transformers as promising strategies for parameter-efficient neural decoding. The source code is available at https://github.com/ kushagrayadv/voxel-former.
