# AI推理加速技术论文分析报告
生成时间: 2025-10-15 14:19:18
分析论文数量: 15篇

## 论文技术简报

### 1. Can Less Precise Be More Reliable? A Systematic Evaluation of Quantization's Impact on CLIP Beyond Accuracy

CEA与Université Paris-Saclay发布了关于量化对CLIP模型可靠性影响的系统评估论文，使用量化技术及特定量化感知训练方法，解决了量化对CLIP模型可靠性（超越准确性）影响未被充分探索的问题，达成了量化能改善校准、提升OOD检测，特定QAT方法可同时提升零样本准确性、校准和OOD鲁棒性并挑战效率-性能权衡的效果。

### 2. Dual-Path Phishing Detection: Integrating Transformer-Based NLP with Structural URL Analysis

University of Doha for Science and Technology发布了双路径钓鱼检测框架论文，使用整合Transformer-based NLP（如DistilBERT）与结构化URL分析（字符级TF-IDF+传统分类器）的技术，解决了传统方法孤立分析邮件内容或URL的不足，达成了显著提高检测准确率、DistilBERT平衡准确性与效率且随机森林在恶意URL识别上表现突出的效果。

### 3. Dynamic Reasoning Chains through Depth-Specialized Mixture-of-Experts in Transformer Architectures

University of Petroleum and Energy Studies (UPES) Dehradun发布了Dynamic Reasoning Chains through Depth-Specialized Mixture-of-Experts in Transformer Architectures论文，使用深度专业化混合专家的动态推理链（DS-MoE）技术，解决了Transformer对所有输入采用相同处理深度导致的效率低下和推理质量受限问题，达成了计算节省16%、推理速度提升35%及复杂多步推理准确率提高2.8%的效果

### 4. Fast-SEnSeI: Lightweight Sensor-Independent Cloud Masking for On-board Multispectral Sensors

Zaitra s.r.o. Brno发布了Fast-SEnSeI论文，使用轻量级、传感器独立的编码器模块（结合改进光谱描述符、轻量级架构及CPU-FPGA混合部署），解决了现有云分割模型与特定传感器耦合、依赖地面处理的问题，在Sentinel-2和Landsat 8数据集上达成了跨不同输入配置的准确云分割效果。

### 5. FERD: Fairness-Enhanced Data-Free Robustness Distillation

HKUST(GZ)、南洋理工大学发布了FERD论文，使用FERD框架（通过鲁棒性引导的类别重加权及生成公平感知示例与统一目标对抗示例），解决了现有数据无鲁棒性蒸馏中不同类别间鲁棒性公平性差的问题，达成了在CIFAR-10上使用MobileNet-V2时FGSM和AutoAttack下最差类别鲁棒性分别提升15.1%和6.4%的效果

### 6. Hierarchical Resolution Transformers: A Wavelet-Inspired Architecture for Multi-Scale Language Understanding

UPES Dehradun发布了Hierarchical Resolution Transformers论文，使用小波启发的多分辨率Hierarchical Resolution Transformer (HRT)架构（含多分辨率注意力机制与指数序列缩减实现O(n log n)复杂度），解决了标准Transformer因扁平序列处理导致的二次计算成本、弱组合泛化及话语级建模不足问题，达成在GLUE、SuperGLUE、Long Range Arena等基准上平均提升3.8%-6.1%，内存减少42%，推理延迟降低37%的效果。

### 7. Interactive Recommendation Agent with Active User Commands

Alibaba Group发布了“Interactive Recommendation Agent with Active User Commands”论文，使用IRF范式及RecBot双代理架构（Parser Agent与Planner Agent）和模拟增强知识蒸馏技术，解决了传统推荐系统被动反馈无法捕捉用户细微意图导致偏好建模不准的问题，达成了用户满意度和业务成果显著提升的效果。

### 8. MASSIVE ACTIVATIONS ARE THE KEY TO LOCAL DE-TAIL SYNTHESIS IN DIFFUSION TRANSFORMERS

上海交通大学、香港大学发布了相关论文，使用Detail Guidance (DG)技术，解决了扩散Transformer中局部细节合成的问题，达成了增强细粒度视觉细节、生成高质量输出并与Classifier-Free Guidance (CFG)无缝集成进一步优化细节的效果

### 9. Punching Above Precision: Small Quantized Model Distillation with Learnable Regularizer

Opt-AI发布了Punching Above Precision论文，使用Game of Regularizer (GoR)可学习正则化技术，解决了现有QAT-KD方法在低比特量化下难以平衡任务特定与蒸馏损失的问题，达成了小量化模型性能提升、多任务上优于SOTA且边缘设备推理更快并保持全精度准确性（最优条件下超越全精度模型）的效果。

### 10. QUANTIZED VISUAL GEOMETRY GROUNDED TRANSFORMER

ETH Zürich和中国科学院计算技术研究所发布了QUANTIZED VISUAL GEOMETRY GROUNDED TRANSFORMER论文，使用量化视觉几何接地Transformer技术，解决了视觉Transformer中计算效率与几何信息有效融合不足的问题，达成了提升计算效率并增强几何感知能力，实现视觉任务性能提升的效果。

### 11. RAM-NAS: Resource-aware Multiobjective Neural Architecture Search Method for Robot Vision Tasks

研究团队发布了RAM-NAS论文，使用子网互蒸馏、延迟代理预测器及多目标进化搜索技术，解决了传统NAS超网训练不足及机器人硬件资源感知缺失问题，达成了ImageNet准确率76.7%-81.4%、降低机器人边缘硬件推理延迟且检测分割任务推理时间低于MobileNetv3的效果。

### 12. Real-Time Object Detection Meets DINOv3

Intellindust AI Lab发布了Real-Time Object Detection Meets DINOv3论文，使用DINOv3技术，解决了实时目标检测中精度与速度难以兼顾的问题，达成了在保持实时性的同时提升检测精度的效果。

### 13. Under review as a conference paper at ICLR 2026 REJUVENATING CROSS-ENTROPY LOSS IN KNOWL-EDGE DISTILLATION FOR RECOMMENDER SYSTEMS

华东师范大学发布了《Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems》论文，使用Rejuvenated Cross-Entropy for Knowledge Distillation (RCE-KD)方法（将教师top项分为学生高排名与非高排名子集，对非高排名子集采用师生协作采样策略近似闭包假设并自适应结合两子集损失），解决了推荐系统知识蒸馏中交叉熵损失闭包假设与蒸馏教师top项排序目标的矛盾，有效提升了知识蒸馏性能。

### 14. Seedream 4.0: Toward Next-generation Multimodal Image Generation

火山引擎发布了Seedream 4.0论文，使用高效扩散Transformer与强大VAE（减少图像tokens）、微调VLM模型及多模态联合训练技术，解决了传统T2I系统功能分散的问题，达成1K-4K高分辨率图像快速生成（2K图像生成仅需1.4秒）且在T2I和多模态图像编辑上达到SOTA的效果。

### 15. SINGER: A CLEARER VOICE DISTILLS VISION TRANSFORMERS FURTHER

MOBILTECH CO与庆熙大学发布了SINGER论文，使用零空间引导扰动的SiNGER框架及LoRA适配器，解决了视觉Transformer知识蒸馏中伪影抑制与有效信号保留的权衡问题，达成学生模型性能提升并在多个下游任务实现SOTA，生成更清晰可解释表示的效果

## 论文详细信息

### 1. Can Less Precise Be More Reliable? A Systematic Evaluation of Quantization's Impact on CLIP Beyond Accuracy

**主要机构**: CEA, Computer Vision Center, Université Paris-Saclay
**作者数量**: 5人

**摘要**:
The powerful zero-shot generalization capabilities of visionlanguage models (VLMs) like CLIP have enabled new paradigms for safety-related tasks such as out-of-distribution (OOD) detection. However, additional aspects crucial for the computationally efficient and reliable deployment of CLIP are still overlooked. In particular, the impact of quantization on CLIP's performance beyond accuracy remains underexplored. This work presents a large-scale evaluation of quantization on CLIP models, assessing not only in-distribution accuracy but a comprehensive suite of reliability metrics and revealing counterintuitive results driven by pre-training source. We demonstrate that quantization consistently improves calibration for typically underconfident pre-trained models, while often degrading it for overconfident variants. Intriguingly, this degradation in calibration does not preclude gains in other reliability metrics; we find that OOD detection can still improve for these same poorly calibrated models. Furthermore, we identify specific quantization-aware training (QAT) methods that yield simultaneous gains in zero-shot accuracy, calibration, and OOD robustness, challenging the view of a strict efficiency-performance trade-off. These findings offer critical insights for navigating the multi-objective problem of deploying efficient, reliable, and robust VLMs by utilizing quantization beyond its conventional role.

### 2. Dual-Path Phishing Detection: Integrating Transformer-Based NLP with Structural URL Analysis

**主要机构**: University of Doha for Science and Technology, College of Computing and IT
**作者数量**: 5人

**摘要**:
Phishing emails pose a persistent and increasingly sophisticated threat, undermining email security through deceptive tactics designed to exploit both semantic and structural vulnerabilities. Traditional detection methods, often based on isolated analysis of email content or embedded URLs, fail to comprehensively address these evolving attacks. In this paper, we propose a dual-path phishing detection framework that integrates transformer-based natural language processing (NLP) with classical machine learning to jointly analyze email text and embedded URLs. Our approach leverages the complementary strengths of semantic analysis using fine-tuned transformer architectures (e.g., DistilBERT) and structural link analysis via character-level TF-IDF vectorization paired with classical classifiers (e.g., Random Forest). Empirical evaluation on representative email and URL datasets demonstrates that this combined approach significantly improves detection accuracy. Specifically, the DistilBERT model achieves a near-optimal balance between accuracy and computational efficiency for textual phishing detection, while Random Forest notably outperforms other classical classifiers in identifying malicious URLs. The modular design allows flexibility for standalone deployment or ensemble integration, facilitating realworld adoption. Collectively, our results highlight the efficacy and practical value of this dual-path approach, establishing a scalable, accurate, and interpretable solution capable of enhancing email security against contemporary phishing threats.

### 3. Dynamic Reasoning Chains through Depth-Specialized Mixture-of-Experts in Transformer Architectures

**主要机构**: School of Computer Science, University of Petroleum and Energy Studies (UPES) Dehradun
**作者数量**: 6人

**摘要**:
Contemporary transformer architectures apply identical processing depth to all inputs, creating inefficiencies and limiting reasoning quality. Simple factual queries are subjected to the same multi-layered computation as complex logical problems, wasting resources while constraining deep inference. To overcome this, we came up with a concept of Dynamic Reasoning Chains through Depth-Specialised Mixture-of-Experts (DS-MoE), a modular framework that extends the Mixture-of-Experts paradigm from width-based to depth-specialised computation. DS-MoE introduces expert modules optimised for distinct reasoning depths-shallow pattern recognition, compositional reasoning, logical inference, memory integration, and meta-cognitive supervision. A learned routing network dynamically assembles custom reasoning chains, activating only the necessary experts to match input complexity. The dataset on which we trained and evaluated DS-MoE is on The Pile, an 800GB corpus covering diverse domains such as scientific papers, legal texts, programming code, and web content, enabling systematic assessment across reasoning depths. Experimental results demonstrate that DS-MoE achieves up to 16 per cent computational savings and 35 per cent faster inference compared to uniform-depth transformers, while delivering 2.8 per cent higher accuracy on complex multi-step reasoning benchmarks. Furthermore, routing decisions yield interpretable reasoning chains, enhancing transparency and scalability. These findings establish DS-MoE as a significant advancement in adaptive neural architectures, demonstrating that depth-specialised modular processing can simultaneously improve efficiency, reasoning quality, and interpretability in large-scale language models.

### 4. Fast-SEnSeI: Lightweight Sensor-Independent Cloud Masking for On-board Multispectral Sensors

**主要机构**: Zaitra s.r.o. Brno
**作者数量**: 1人

**摘要**:
Cloud segmentation is a critical preprocessing step for many Earth observation tasks, yet most models are tightly coupled to specific sensor configurations and rely on groundbased processing. In this work, we propose Fast-SEnSeI, a lightweight, sensor-independent encoder module that enables flexible, on-board cloud segmentation across multispectral sensors with varying band configurations. Building upon SEnSeI-v2, Fast-SEnSeI integrates an improved spectral descriptor, lightweight architecture, and robust padding-band handling. It accepts arbitrary combinations of spectral bands and their wavelengths, producing fixed-size feature maps that feed into a compact, quantized segmentation model based on a modified U-Net. The module runs efficiently on embedded CPUs using Apache TVM, while the segmentation model is deployed on FPGA, forming a CPU-FPGA hybrid pipeline suitable for spacequalified hardware. Evaluations on Sentinel-2 and Landsat 8 datasets demonstrate accurate segmentation across diverse input configurations.

### 5. FERD: Fairness-Enhanced Data-Free Robustness Distillation

**主要机构**: STCA, HKUST(GZ), Nanyang Technological University, Nanjing University of Science and Technology
**作者数量**: 7人

**摘要**:
Data-Free Robustness Distillation (DFRD) aims to transfer the robustness from the teacher to the student without accessing the training data. While existing methods focus on overall robustness, they overlook the robust fairness issues, leading to severe disparity of robustness across different categories. In this paper, we find two key problems: (1) student model distilled with equal class proportion data behaves significantly different across distinct categories; and (2) the robustness of student model is not stable across different attacks target. To bridge these gaps, we present the first Fairness-Enhanced data-free Robustness Distillation (FERD) framework to adjust the proportion and distribution of adversarial examples. For the proportion, FERD adopts a robustnessguided class reweighting strategy to synthesize more samples for the less robust categories, thereby improving robustness of them. For the distribution, FERD generates complementary data samples for advanced robustness distillation. It generates Fairness-Aware Examples (FAEs) by enforcing a uniformity constraint on feature-level predictions, which suppress the dominance of class-specific non-robust features, providing a more balanced representation across all categories. Then, FERD constructs Uniform-Target Adversarial Examples (UTAEs) from FAEs by applying a uniform target class constraint to avoid biased attack directions, which distribute the attack targets across all categories and prevents overfitting to specific vulnerable categories. Extensive experiments on three public datasets show that FERD achieves state-of-the-art worst-class robustness under all adversarial attack (e.g., the worst-class robustness under FGSM and Au-toAttack are improved by 15.1% and 6.4% using MobileNet-V2 on CIFAR-10), demonstrating superior performance in both robustness and fairness aspects.

### 6. Hierarchical Resolution Transformers: A Wavelet-Inspired Architecture for Multi-Scale Language Understanding

**主要机构**: School of Computer Science, University of Petroleum and Energy Studies (UPES) Dehradun
**作者数量**: 6人

**摘要**:
Transformer architectures have achieved state-ofthe-art performance across natural language tasks, yet they fundamentally misrepresent the hierarchical nature of human language by processing text as flat token sequences. This results in quadratic computational cost, weak computational cost, weak compositional generalization, and inadequate discourselevel modeling. We propose Hierarchical Resolution Transformer (HRT), a novel wavelet-inspired neural architecture that processes language simultaneously across multiple resolutions, from characters to discourse-level units. HRT constructs a multiresolution attention, enabling bottom-up composition and topdown contextualization. By employing exponential sequence reduction across scales, HRT achieves O(n log n) complexity, offering significant efficiency improvements over standard transformers. We evaluated HRT on a diverse suite of benchmarks, including GLUE, SuperGLUE, Long Range Arena, and WikiText-103, and results demonstrated that HRT outperforms standard transformer baselines by an average of +3.8% on GLUE, +4.5% on SuperGLUE, and +6.1% on Long Range Arena, while reducing memory usage by 42% and inference latency by 37% compared to BERT and GPT style models of similar parameter count. Ablation studies confirm the effectiveness of cross-resolution attention and scale-specialized modules, showing that each contributes independently to both efficiency and accuracy. Our findings establish HRT as the first architecture to align computational structure with the hierarchical organization of human language, demonstrating that multi-scale, waveletinspired processing yields both theoretical efficiency gains and practical improvements in language understanding.

### 7. Interactive Recommendation Agent with Active User Commands

**主要机构**: Alibaba Group, Renmin University of China, University of Chinese Academy of Sciences, Gaoling School of Artificial Intelligence
**作者数量**: 15人

**摘要**:
Traditional recommender systems rely on passive feedback mechanisms that limit users to simple choices such as like and dislike. However, these coarse-grained signals fail to capture users' nuanced behavior motivations and intentions. In turn, current systems cannot also distinguish which specific item attributes drive user satisfaction or dissatisfaction, resulting in inaccurate preference modeling. These fundamental limitations create a persistent gap between user intentions and system interpretations, ultimately undermining user satisfaction and harming system effectiveness. To address these limitations, we introduce the Interactive Recommendation Feed (IRF), a pioneering paradigm that enables natural language commands within mainstream recommendation feeds. Unlike traditional systems that confine users to passive implicit behavioral influence, IRF empowers active explicit control over recommendation policies through real-time linguistic commands. To support this paradigm, we develop RecBot, a dual-agent architecture where a Parser Agent transforms linguistic expressions into structured preferences and a Planner Agent dynamically orchestrates adaptive tool chains for on-the-fly policy adjustment. To enable practical deployment, we employ simulation-augmented knowledge distillation to achieve efficient performance while maintaining strong reasoning capabilities. Through extensive offline and longterm online experiments, RecBot shows significant improvements in both user satisfaction and business outcomes. CCS Concepts • Information systems → Recommender systems.

### 8. MASSIVE ACTIVATIONS ARE THE KEY TO LOCAL DE-TAIL SYNTHESIS IN DIFFUSION TRANSFORMERS

**主要机构**: Shanghai Jiao Tong University, Monash University, The University of Hong Kong
**作者数量**: 8人

**摘要**:
Visual results of our Detail Guidance (DG). Left: DG explicitly enhances fine-grained visual details, yielding high-quality outputs. Right: DG integrates seamlessly with Classifier-Free Guidance (CFG), allowing for further refinement of details.

### 9. Punching Above Precision: Small Quantized Model Distillation with Learnable Regularizer

**主要机构**: Opt-AI
**作者数量**: 6人

**摘要**:
Quantization-aware training (QAT) combined with knowledge distillation (KD) is a promising strategy for compressing Artificial Intelligence (AI) models for deployment on resource-constrained hardware. However, existing QAT-KD methods often struggle to balance task-specific (TS) and distillation losses due to heterogeneous gradient magnitudes, especially under low-bit quantization. We propose Game of Regularizer (GoR), a novel learnable regularization method that adaptively balances TS and KD objectives using only two trainable parameters for dynamic loss weighting. GoR reduces conflict between supervision signals, improves convergence, and boosts the performance of small quantized models (SQMs). Experiments on image classification, object detection (OD), and large language model (LLM) compression show that GoR consistently outperforms state-of-the-art QAT-KD methods. On low-power edge devices, it delivers faster inference while maintaining full-precision accuracy. We also introduce QAT-EKD-GoR, an ensemble distillation framework that uses multiple heterogeneous teacher models. Under optimal conditions, the proposed EKD-GoR can outperform full-precision models, providing a robust solution for real-world deployment.

### 10. QUANTIZED VISUAL GEOMETRY GROUNDED TRANSFORMER

**主要机构**: ETH Zürich, Institute of Computing Technology, Chinese Academy of Sciences, University of Chinese Academy of Sciences
**作者数量**: 11人

**摘要**:


### 11. RAM-NAS: Resource-aware Multiobjective Neural Architecture Search Method for Robot Vision Tasks

**主要机构**: 
**作者数量**: 5人

**摘要**:
Neural architecture search (NAS) has shown great promise in automatically designing lightweight models. However, conventional approaches are insufficient in training the supernet and pay little attention to actual robot hardware resources. To meet such challenges, we propose RAM-NAS, a resource-aware multi-objective NAS method that focuses on improving the supernet pretrain and resource-awareness on robot hardware devices. We introduce the concept of subnets mutual distillation, which refers to mutually distilling all subnets sampled by the sandwich rule. Additionally, we utilize the Decoupled Knowledge Distillation (DKD) loss to enhance logits distillation performance. To expedite the search process with consideration for hardware resources, we used data from three types of robotic edge hardware to train Latency Surrogate predictors. These predictors facilitated the estimation of hardware inference latency during the search phase, enabling a unified multi-objective evolutionary search to balance model accuracy and latency trade-offs. Our discovered model family, RAM-NAS models, can achieve top-1 accuracy ranging from 76.7% to 81.4% on ImageNet. In addition, the resource-aware multi-objective NAS we employ significantly reduces the model's inference latency on edge hardware for robots. We conducted experiments on downstream tasks to verify the scalability of our methods. The inference time for detection and segmentation is reduced on all three hardware types compared to MobileNetv3-based methods. Our work fills the gap in NAS for robot hardware resource-aware.

### 12. Real-Time Object Detection Meets DINOv3

**主要机构**: Intellindust AI Lab
**作者数量**: 5人

**摘要**:


### 13. Under review as a conference paper at ICLR 2026 REJUVENATING CROSS-ENTROPY LOSS IN KNOWL-EDGE DISTILLATION FOR RECOMMENDER SYSTEMS

**主要机构**: Department of Computer Science East, China Normal University Shanghai
**作者数量**: 4人

**摘要**:
This paper analyzes Cross-Entropy (CE) loss in knowledge distillation (KD) for recommender systems. KD for recommender systems targets at distilling rankings, especially among items most likely to be preferred, and can only be computed on a small subset of items. Considering these features, we reveal the connection between CE loss and NDCG in the field of KD. We prove that when performing KD on an item subset, minimizing CE loss maximizes the lower bound of NDCG, only if an assumption of closure is satisfied. It requires that the item subset consists of the student's top items. However, this contradicts our goal of distilling rankings of the teacher's top items. We empirically demonstrate the vast gap between these two kinds of top items. To bridge the gap between our goal and theoretical support, we propose Rejuvenated Cross-Entropy for Knowledge Distillation (RCE-KD). It splits the top items given by the teacher into two subsets based on whether or not it is ranked highly by the student. For the subset that defies the condition, a sampling strategy is devised to use teacher-student collaboration to approximate our assumption of closure. We also combine the losses on the two subsets adaptively. Extensive experiments demonstrate the effectiveness of our method. Our code is available at https://anonymous.4open.science/r/RCE-KD.

### 14. Seedream 4.0: Toward Next-generation Multimodal Image Generation

**主要机构**: 
**作者数量**: 0人

**摘要**:
We introduce Seedream 4.0, an efficient and high-performance multimodal image generation system that unifies text-to-image (T2I) synthesis, image editing, and multi-image composition within a single framework. We develop a highly efficient diffusion transformer with a powerful VAE which also can reduce the number of image tokens considerably. This allows for efficient training of our model, and enables it to fast generate native high-resolution images (e.g., 1K-4K). Seedream 4.0 is pretrained on billions of text-image pairs spanning diverse taxonomies and knowledgecentric concepts. Comprehensive data collection across hundreds of vertical scenarios, coupled with optimized strategies, ensures stable and large-scale training, with strong generalization. By incorporating a carefully fine-tuned VLM model, we perform multi-modal post-training for training both T2I and image editing tasks jointly. For inference acceleration, we integrate adversarial distillation, distribution matching, and quantization, as well as speculative decoding. It achieves an inference time of up to 1.4 seconds for generating a 2K image (without a LLM/VLM as PE model). Comprehensive evaluations reveal that Seedream 4.0 can achieve state-of-the-art results on both T2I and multimodal image editing. In particular, it demonstrates exceptional multimodal capabilities in complex tasks, including precise image editing and in-context reasoning, and also allows for multi-image reference, and can generate multiple output images. This extends traditional T2I systems into an more interactive and multidimensional creative tool, pushing the boundary of generative AI for both creativity and professional applications. Seedream 4.0 is now accessible on Volcano Engine α .

### 15. SINGER: A CLEARER VOICE DISTILLS VISION TRANSFORMERS FURTHER

**主要机构**: MOBILTECH CO, Kyung Hee University, LTD, Department of Software Convergence
**作者数量**: 5人

**摘要**:
Vision Transformers are widely adopted as the backbone of vision foundation models, but they are known to produce high-norm artifacts that degrade representation quality. When knowledge distillation transfers these features to students, high-norm artifacts dominate the objective, so students overfit to artifacts and underweight informative signals, diminishing the gains from larger models. Prior work attempted to remove artifacts but encountered an inherent trade-off between artifact suppression and preserving informative signals from teachers. To address this, we introduce Singular Nullspace-Guided Energy Reallocation (SiNGER), a novel distillation framework that suppresses artifacts while preserving informative signals. The key idea is principled teacher feature refinement: during refinement, we leverage the nullspace-guided perturbation to preserve information while suppressing artifacts. Then, the refined teacher's features are distilled to a student. We implement this perturbation efficiently with a LoRA-based adapter that requires minimal structural modification. Extensive experiments show that SiNGER consistently improves student models, achieving state-of-the-art performance in multiple downstream tasks and producing clearer and more interpretable representations.
