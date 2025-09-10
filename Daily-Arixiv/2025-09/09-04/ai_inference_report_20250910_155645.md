# AI推理加速技术论文分析报告
生成时间: 2025-09-10 15:56:45
分析论文数量: 13篇

## 论文技术简报

### 1. Accuracy-Constrained CNN Pruning for Efficient and Reliable EEG-Based Seizure Detection

Vellore Institute of Technology (VIT-AP)发布了相关论文，使用带结构化剪枝和轻度早停的轻量级一维CNN模型，解决了基于EEG的癫痫检测中深度学习模型的尺寸和计算需求问题，达成了减少50%权重/内存的同时保持预测能力并提升精度至92.87%、宏F1分数至0.8707的效果。

### 2. AFD-SLU: ADAPTIVE FEATURE DISTILLATION FOR SPOKEN LANGUAGE UNDERSTANDING

北京大学发布了AFD-SLU论文，使用自适应特征蒸馏框架（含动态适配器与残差投影神经网络RPNN及动态蒸馏系数DDC），解决了口语语言理解中标注数据稀缺及部署大语言模型的计算负担问题，达成了最先进性能（意图准确率95.67%、槽位F1分数92.02%、总体准确率85.50%）

### 3. Elucidating the Design Space of Decay in Linear Attention

研究团队发布了《Elucidating the Design Space of Decay in Linear Attention》论文，通过系统阐述线性注意力衰减机制设计空间（涵盖参数化策略、参数共享、衰减粒度及与相对位置编码兼容性四个维度），解决了线性注意力中衰减机制设计的关键问题，揭示了参数化策略需细致考虑、向量衰减通常优于标量、RoPE对多数线性注意力无显著益处等核心见解。

### 4. Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet

Clemson University发布了增强3D点云分类的论文，使用改进数据集ModelNet-R和轻量级图神经网络Point-SkipNet，解决了常用数据集ModelNet40的质量问题（如标签不一致、含2D数据等）及现有模型计算开销大的问题，达成了提升分类性能且Point-SkipNet在ModelNet-R上实现SOTA精度并降低参数的效果。

### 5. Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference

北京人工智能研究院发布了《Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference》论文，使用针对预填充-解码解耦推理的定向剪枝方法（含构建剪枝和蒸馏集独立迭代移除两阶段块及token-aware缓存剪枝机制），解决了LLM部署中高计算和内存成本问题，达成默认设置下推理速度提升20.56%、数据传输带宽消耗减少4.95倍的效果。

### 6. Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution

杭州电子科技大学发布了光场超分辨率论文，使用混合Mamba-Transformer框架（LFMT）及Subspace Simple Scanning策略、双阶段建模，解决了Mamba方法在复杂光场数据特征提取低效冗余及状态空间保留空间-角度和视差信息的局限，显著优于现有最先进方法，在真实和合成光场数据集上性能大幅提升且保持低计算复杂度。

### 7. Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation

卡内基梅隆大学发布了相关论文，使用难度感知推理框架（无需架构修改，通过在难度比例推理链数据集上后训练并结合SFT与DPO），解决了Chain-of-thought推理对简单问题产生不必要冗长输出的问题，达成了减少推理长度并保持或提升性能，使模型能“按比例思考”（简单问题少推理、复杂问题深推理）的效果。

### 8. 

研究团队发布了PLaMo 2系列日语大语言模型论文，使用混合Samba架构、高效剪枝及SFT/DPO后训练流程，解决了日语数据稀缺与计算效率问题，达成8B模型性能媲美100B模型且日语基准测试SOTA的效果。

### 9. Quantized Large Language Models in Biomedical Natural Language Processing: Evaluation and Recommendation

明尼苏达大学发布了Quantized Large Language Models in Biomedical Natural Language Processing: Evaluation and Recommendation论文，使用量化技术，解决了大型语言模型因尺寸和计算需求大在数据隐私高、资源有限的医疗环境中难以部署的问题，达成了GPU内存需求减少75%、保留性能并实现70B参数模型在40GB消费级GPU部署的效果。

### 10. A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling SCALING UP, SPEEDING UP: A BENCHMARK OF SPEC-ULATIVE DECODING FOR EFFICIENT LLM TEST-TIME SCALING

华为诺亚方舟实验室与香港城市大学发布了投机解码基准测试论文，使用投机解码基准测试技术，解决了大型语言模型测试时的高效扩展问题，为LLM测试时高效扩展提供了评估基准。

### 11. SpikingBrain Technical Report: Spiking Brain-inspired Large Models

中国科学院、北京航空航天大学发布了Spiking Brain-inspired Large Models技术报告，使用线性/混合线性注意力与自适应脉冲神经元的模型架构、基于转换的高效训练流水线及针对MetaX硬件的定制系统框架，解决了主流Transformer-based大语言模型的效率瓶颈（训练计算随序列长度二次增长、推理内存线性增长）及非NVIDIA平台的稳定高效训练部署挑战，实现高效长上下文训练与推理。

### 12. STADI: Fine-Grained Step-Patch Diffusion Parallelism for Heterogeneous GPUs

中山大学发布了STADI论文，使用细粒度时空混合并行调度（计算感知步骤分配与弹性补丁并行）技术，解决了异构多GPU环境下扩散模型推理的负载不平衡与资源利用率低问题，达成了相比现有补丁并行框架端到端推理延迟降低45%、资源利用率显著提升的效果。

### 13. VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation

德克萨斯大学奥斯汀分校发布了VCMamba论文，使用融合卷积神经网络（CNNs）与多方向Mamba状态空间模型（SSMs）的混合视觉骨干技术，解决了CNN难以捕捉全局上下文与ViTs/Mamba难以有效捕捉细粒度局部特征的矛盾，达成了在ImageNet-1K分类上达82.6%Top-1准确率（超越PlainMamba-L3 0.3%且参数少37%）、在ADE20K分割上获47.1 mIoU（超越EfficientFormer-L7 2.0 mIoU且参数少62%）的效果

## 论文详细信息

### 1. Accuracy-Constrained CNN Pruning for Efficient and Reliable EEG-Based Seizure Detection

**主要机构**: School of Computer Science and Engineering, Vellore Institute of Technology (VIT-AP)
**作者数量**: 1人

**摘要**:
Deep learning models, especially convolutional neural networks (CNN's) have shown considerable promise when it comes to biomedical signals, such as EEGbased seizure detection. However, these models come with challenges, primarily due to their size and compute requirements in environments where real-time detection or fewer resources are present. In this study, we presented a lightweight one-dimensional CNN model with structured pruning to improve efficiency and reliability. The overall model was trained with mild early stopping to try and address possible overfitting, achieving a reasonable accuracy of 92.78% and a macro-F1 score of 0.8686. Structured pruning for CNN of the baseline involved removing 50% of the convolutional kernels based on the importance of the kernel from the model predictions. Surprisingly, after pruning the weight/memory by 50%, our new network was still able to predict, maintain the predictive capabilities of the model, and we even modestly increased our precision to 92.87% and improved the macro-F1 score metric to 0.8707. Overall, we have presented a convincing case that structured pruning removes redundancy, improves generalization and in combination with mild early stopping, achieves a promising way forward to improve seizure detection efficiencies and reliability, which is a clear motivation for resource-limited settings. [1, 2].

### 2. AFD-SLU: ADAPTIVE FEATURE DISTILLATION FOR SPOKEN LANGUAGE UNDERSTANDING

**主要机构**: Peking University, Academy of Military Sciences, National Institute of Defense Technology Innovation, School of Advanced Manufacturing and Robotics
**作者数量**: 4人

**摘要**:
Spoken Language Understanding (SLU) is a core component of conversational systems, enabling machines to interpret user utterances. Despite its importance, developing effective SLU systems remains challenging due to the scarcity of labeled training data and the computational burden of deploying Large Language Models (LLMs) in real-world applications. To further alleviate these issues, we propose an Adaptive Feature Distillation framework that transfers rich semantic representations from a General Text Embeddings (GTE)-based teacher model to a lightweight student model. Our method introduces a dynamic adapter equipped with a Residual Projection Neural Network (RPNN) to align heterogeneous feature spaces, and a Dynamic Distillation Coefficient (DDC) that adaptively modulates the distillation strength based on real-time feedback from intent and slot prediction performance. Experiments on the Chinese profile-based ProSLU benchmark demonstrate that AFD-SLU achieves state-of-the-art results, with 95.67% intent accuracy, 92.02% slot F1 score, and 85.50% overall accuracy.

### 3. Elucidating the Design Space of Decay in Linear Attention

**主要机构**: 
**作者数量**: 4人

**摘要**:
This paper presents a comprehensive investigation into the decay mechanisms inherent in linear complexity sequence models. We systematically delineate the design space of decay mechanisms across four pivotal dimensions: parameterization strategy, which refers to the computational methodology for decay; parameter sharing, which involves the utilization of supplementary parameters for decay computation; decay granularity, comparing scalar versus vector-based decay; and compatibility with relative positional encoding methods, such as Rotary Position Embedding (RoPE). Through an extensive series of experiments conducted on diverse language modeling tasks, we uncovered several critical insights. Firstly, the design of the parameterization strategy for decay requires meticulous consideration. Our findings indicate that effective configurations are typically confined to a specific range of parameters. Secondly, parameter sharing cannot be used arbitrarily, as it may cause decay values to be too large or too small, thereby significantly impacting performance. Thirdly, under identical parameterization strategies, scalar decay generally underperforms compared to its vector-based counterpart. However, in certain scenarios with alternative parameterization strategies, scalar decay may unexpectedly surpass vector decay in efficacy. Lastly, our analysis reveals that RoPE, a commonly employed relative positional encoding method, typically fails to provide tangible benefits to the majority of linear attention mechanisms.

### 4. Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet

**主要机构**: Islamic Azad University-Lahijan Branch, Sirjan University of Technology Sirjan, Clemson University Clemson
**作者数量**: 7人

**摘要**:
The classification of 3D point clouds is crucial for applications such as autonomous driving, robotics, and augmented reality. However, the commonly used ModelNet40 dataset suffers from limitations such as inconsistent labeling, 2D data, size mismatches, and inadequate class differentiation, which hinder model performance. This paper introduces ModelNet-R, a meticulously refined version of ModelNet40 designed to address these issues and serve as a more reliable benchmark. Additionally, this paper proposes Point-SkipNet, a lightweight graph-based neural network that leverages efficient sampling, neighborhood grouping, and skip connections to achieve high classification accuracy with reduced computational overhead. Extensive experiments demonstrate that models trained in ModelNet-R exhibit significant performance improvements. Notably, Point-SkipNet achieves state-of-the-art accuracy on ModelNet-R with a substantially lower parameter count compared to contemporary models. This research highlights the crucial role of dataset quality in optimizing model efficiency for 3D point cloud classification. For more details, see the code at: https://github.com/m-saeid/ ModeNetR_PointSkipNet.

### 5. Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference

**主要机构**: Beijing Academy of Artificial Intelligence
**作者数量**: 4人

**摘要**:
Large Language Models (LLMs) demonstrate exceptional capabilities across various tasks, but their deployment is constrained by high computational and memory costs. Model pruning provides an effective means to alleviate these demands. However, existing methods often ignore the characteristics of prefill-decode (PD) disaggregation in practice. In this paper, we propose a novel pruning method for PD disaggregation inference, enabling more precise and efficient block and KV Cache pruning. Our approach constructs pruning and distillation sets to perform iterative block removal independently for the prefill and decode stages, obtaining better pruning solutions. Moreover, we introduce a token-aware cache pruning mechanism that retains all KV Cache in the prefill stage but selectively reuses entries for the first and last token sequences in selected layers during decode, reducing communication costs with minimal overhead. Extensive experiments demonstrate that our approach consistently achieves strong performance in both PD disaggregation and PD unified settings without disaggregation. Under the default settings, our method achieves a 20.56% inference speedup and a 4.95× reduction in data transmission bandwidth consumption.

### 6. Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution

**主要机构**: Xiamen University of Technology, Hangzhou Dianzi University, School of Information Science and Engineering, Huaqiao University, School of Engineering, Department of Computer Science, School of Mechanical Engineering and Automation, School of Optoelectronic and Communication Engineering, City University of Hong Kong, Artificial Intelligence Institute
**作者数量**: 10人

**摘要**:
Recently, Mamba-based methods, with its advantage in long-range information modeling and linear complexity, have shown great potential in optimizing both computational cost and performance of light field image super-resolution (LFSR). However, current multi-directional scanning strategies lead to inefficient and redundant feature extraction when applied to complex LF data. To overcome this challenge, we propose a Subspace Simple Scanning (Sub-SS) strategy, based on which we design the Subspace Simple Mamba Block (SSMB) to achieve more efficient and precise feature extraction. Furthermore, we propose a dualstage modeling strategy to address the limitation of state space in preserving spatial-angular and disparity information, thereby enabling a more comprehensive exploration of non-local spatialangular correlations. Specifically, in stage I, we introduce the Spatial-Angular Residual Subspace Mamba Block (SA-RSMB) for shallow spatial-angular feature extraction; in stage II, we use a dual-branch parallel structure combining the Epipolar Plane Mamba Block (EPMB) and Epipolar Plane Transformer Block (EPTB) for deep epipolar feature refinement. Building upon meticulously designed modules and strategies, we introduce a hybrid Mamba-Transformer framework, termed LFMT. LFMT integrates the strengths of Mamba and Transformer models for LFSR, enabling comprehensive information exploration across spatial, angular, and epipolar-plane domains. Experimental results demonstrate that LFMT significantly outperforms current state-of-the-art methods in LFSR, achieving substantial improvements in performance while maintaining low computational complexity on both real-word and synthetic LF datasets. The code is available at https://github.com/hsliu01/LFMT.

### 7. Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation

**主要机构**: Carnegie Mellon University
**作者数量**: 5人

**摘要**:
Chain-of-thought reasoning, while powerful, can produce unnecessarily verbose output for simpler problems. We present a framework for difficulty-aware reasoning that teaches models to dynamically adjust reasoning depth based on problem complexity. Remarkably, we show that models can be endowed with such dynamic inference pathways without any architectural modifications; we simply post-train on data that is carefully curated to include chain-of-thought traces that are proportional in length to problem difficulty. Our analysis reveals that post-training via supervised fine-tuning (SFT) primarily captures patterns like reasoning length and format, while direct preference optimization (DPO) preserves reasoning accuracy, with their combination reducing length and maintaining or improving performance. Both quantitative metrics and qualitative assessments confirm that models can learn to "think proportionally"-reasoning minimally on simple problems while maintaining depth for complex ones.

### 8. 

**主要机构**: 
**作者数量**: 0人

**摘要**:
In this report, we introduce PLaMo 2, a series of Japanese-focused large language models featuring a hybrid Samba-based architecture that transitions to full attention via continual pre-training to support 32K token contexts. Training leverages extensive synthetic corpora to overcome data scarcity, while computational efficiency is achieved through weight reuse and structured pruning. This efficient pruning methodology produces an 8B model that achieves performance comparable to our previous 100B model. Post-training further refines the models using a pipeline of supervised fine-tuning (SFT) and direct preference optimization (DPO), enhanced by synthetic Japanese instruction data and model merging techniques. Optimized for inference using vLLM and quantization with minimal accuracy loss, the PLaMo 2 models achieve state-of-the-art results on Japanese benchmarks, outperforming similarly-sized open models in instruction-following, language fluency, and Japanese-specific knowledge.

### 9. Quantized Large Language Models in Biomedical Natural Language Processing: Evaluation and Recommendation

**主要机构**: University of Minnesota, Division of Computational Health Sciences, Department of Surgery, Department of Electrical and Computer Engineering, School of Nursing
**作者数量**: 9人

**摘要**:
Large language models have demonstrated remarkable capabilities in biomedical natural language processing, yet their rapid growth in size and computational requirements present a major barrier to adoption in healthcare settings where data privacy precludes cloud deployment and resources are limited. In this study, we systematically evaluated the impact of quantization on 12 state-of-the-art large language models, including both general-purpose and biomedical-specific models, across eight benchmark datasets covering four key tasks: named entity recognition, relation extraction, multi-label classification, and question answering. We show that quantization substantially reduces GPU memory requirements-by up to 75%-while preserving model performance across diverse tasks, enabling the deployment of 70B-parameter models on 40GB consumer-grade GPUs. In addition, domain-specific knowledge and responsiveness to advanced prompting methods are largely maintained. These findings provide significant practical and guiding value, highlighting quantization as a practical and effective strategy for enabling the secure, local deployment of large yet high-capacity language models in biomedical contexts, bridging the gap between technical advances in AI and real-world clinical translation.

### 10. A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling SCALING UP, SPEEDING UP: A BENCHMARK OF SPEC-ULATIVE DECODING FOR EFFICIENT LLM TEST-TIME SCALING

**主要机构**: City University of Hong Kong, Huawei Noah's Ark Lab
**作者数量**: 11人

**摘要**:


### 11. SpikingBrain Technical Report: Spiking Brain-inspired Large Models

**主要机构**: Chinese Academy of Sciences, Beihang University, The Hong Kong Polytechnic University, Institute of Automation
**作者数量**: 18人

**摘要**:
Mainstream Transformer-based large language models (LLMs) face significant efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly. These constraints limit their ability to process long sequences effectively. In addition, building large models on non-NVIDIA computing platforms poses major challenges in achieving stable and efficient training and deployment. To address these issues, we introduce SpikingBrain, a new family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX 1 GPU cluster and focuses on three core aspects: i) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; ii) Algorithmic Optimizations: an efficient, conversion-based training pipeline compatible with existing LLMs, along with a dedicated spike coding framework; iii) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to the MetaX hardware.

### 12. STADI: Fine-Grained Step-Patch Diffusion Parallelism for Heterogeneous GPUs

**主要机构**: Sun Yat-sen University
**作者数量**: 5人

**摘要**:
The escalating adoption of diffusion models for applications such as image generation demands efficient parallel inference techniques to manage their substantial computational cost. However, existing diffusion parallelism inference schemes often underutilize resources in heterogeneous multi-GPU environments, where varying hardware capabilities or background tasks cause workload imbalance. This paper introduces Spatio-Temporal Adaptive Diffusion Inference (STADI), a novel framework to accelerate diffusion model inference in such settings. At its core is a hybrid scheduler that orchestrates fine-grained parallelism across both temporal and spatial dimensions. Temporally, STADI introduces a novel computation-aware step allocator applied after warmup phases, using a least-common-multipleminimizing quantization technique to reduce denoising steps on slower GPUs and execution synchronization. To further minimize GPU idle periods, STADI executes an elastic patch parallelism mechanism that allocates variably sized image patches to GPUs according to their computational capability, ensuring balanced workload distribution through a complementary spatial mechanism. Extensive experiments on both load-imbalanced and heterogeneous multi-GPU clusters validate STADI's efficacy, demonstrating improved load balancing and mitigation of performance bottlenecks. Compared to patch parallelism, a state-of-the-art diffusion inference framework, our method significantly reduces end-to-end inference latency by up to 45% and significantly improves resource utilization on heterogeneous GPUs.

### 13. VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation

**主要机构**: The University of Texas at Austin
**作者数量**: 3人

**摘要**:
Recent advances in Vision Transformers (ViTs) and State Space Models (SSMs) have challenged the dominance of Convolutional Neural Networks (CNNs) in computer vision. ViTs excel at capturing global context, and SSMs like Mamba offer linear complexity for long sequences, yet they do not capture fine-grained local features as effectively as CNNs. Conversely, CNNs possess strong inductive biases for local features but lack the global reasoning capabilities of transformers and Mamba. To bridge this gap, we introduce VCMamba, a novel vision backbone that integrates the strengths of CNNs and multi-directional Mamba SSMs. VCMamba employs a convolutional stem and a hierarchical structure with convolutional blocks in its early stages to extract rich local features. These convolutional blocks are then processed by later stages incorporating multi-directional Mamba blocks designed to efficiently model long-range dependencies and global context. This hybrid design allows for superior feature representation while maintaining linear complexity with respect to image resolution. We demonstrate VCMamba's effectiveness through extensive experiments on ImageNet-1K classification and ADE20K semantic segmentation. Our VCMamba-B achieves 82.6% top-1 accuracy on ImageNet-1K, surpassing PlainMamba-L3 by 0.3% with 37% fewer parameters, and outperforming Vision GNN-B by 0.3% with 64% fewer parameters. Furthermore, VCMamba-B obtains 47.1 mIoU on ADE20K, exceeding EfficientFormer-L7 by 2.0 mIoU while utilizing 62% fewer parameters. Code is available at https://github.com/Wertyuui345/VCMamba.
