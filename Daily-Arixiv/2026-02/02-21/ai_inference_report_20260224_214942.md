# AI推理加速技术论文分析报告
生成时间: 2026-02-24 21:49:42
分析论文数量: 6篇

## 论文技术简报

### 1. Bayesian Lottery Ticket Hypothesis

卡尔斯鲁厄理工学院（KIT）发布了Bayesian Lottery Ticket Hypothesis论文，使用将彩票假说实验迁移至贝叶斯神经网络场景并采用幅度优先、标准差其次的剪枝策略及移植方法，解决了贝叶斯神经网络计算资源需求过高的问题，验证了贝叶斯场景下彩票假说成立，得到了精度相当甚至更优的稀疏子网络，同时揭示了模型对掩码结构和权重初始化的差异化依赖程度

### 2. 

Advanced Micro Devices, Inc发布了DUET-VLM论文，使用双阶段协同视觉令牌压缩框架，先通过视觉编码器冗余感知压缩生成信息保留令牌再结合分层文本引导令牌丢弃，解决了视觉语言模型因密集视觉令牌化计算成本高昂且现有效率方法常以精度换取速度的问题，达成了在大幅缩减视觉令牌的同时仍维持甚至反超基线精度的效果，在LLaVA-1.5-7B上以67%令牌缩减保留99%以上基线精度，在Video-LLaVA-7B中53.1%令牌缩减下精度超过基线。

### 3. Federated Reasoning Distillation Framework with Model Learnability-Aware Data Allocation

北京建筑大学、燕山大学发布了LaDa联邦推理蒸馏框架论文，使用模型可学习性感知数据分配和领域自适应对比蒸馏技术，解决了联邦大、小语言模型协作中的双向可学习性差距和推理迁移无法适配本地领域的问题，达成了O(1/√T)的理论收敛率，相比最优基线实现了最高13.8%的准确率提升

### 4. Joint Post-Training Quantization of Vision Transformers with Learned Prompt-Guided Data Generation

vivo与牛津大学发布了《Joint Post-Training Quantization of Vision Transformers with Learned Prompt-Guided Data Generation》论文，使用端到端全局联合量化结合学习多模态提示引导Stable Diffusion Turbo的无数据校准技术，解决了现有视觉Transformer训练后量化无法全局优化层间依赖、低比特量化精度损失严重以及依赖标注校准数据的问题，达成了ImageNet上SOTA的W4A4与W3A3精度，首次实现极低比特设置下维持ViT等模型高精度，无数据校准性能媲美真实ImageNet数据校准的效果

### 5. Rank-Aware Spectral Bounds on Attention Logits for Stable Low-Precision Training

相关研究团队发布了《Rank-Aware Spectral Bounds on Attention Logits for Stable Low-Precision Training》论文，使用秩感知注意力对数谱界推导及几何感知缩放结合隐式幂迭代技术，解决低精度Transformer训练中的注意力得分溢出问题，达成相比无秩感知边界紧8-28倍的浓度约束，消除延迟缩放失效场景下的溢出同时保持相当的MMLU下游任务精度

### 6. UFO: Unlocking Ultra-Efficient Quantized Private Inference with Protocol and Algorithm Co-Optimization

北京区块链和边缘计算研究院发布了UFO论文，使用协议与量化算法协同优化的量化2PC推理框架，结合Winograd卷积、图级协议优化、混合精度量化感知训练与比特重加权算法，解决了卷积神经网络私密推理通信开销高、量化训练精度劣化的问题，达成了相比现有主流框架通信量最高降低11.7倍且精度最多提升1.29%的效果

## 论文详细信息

### 1. Bayesian Lottery Ticket Hypothesis

**主要机构**: Scientific Computing Center (SCC), Karlsruhe Institute of Technology (KIT)
**作者数量**: 6人

**摘要**:
Bayesian neural networks (BNNs) are a useful tool for uncertainty quantification, but require substantially more computational resources than conventional neural networks. For non-Bayesian networks, the Lottery Ticket Hypothesis (LTH) posits the existence of sparse subnetworks that can train to the same or even surpassing accuracy as the original dense network. Such sparse networks can lower the demand for computational resources at inference, and during training. The existence of the LTH and corresponding sparse subnetworks in BNNs could motivate the development of sparse training algorithms and provide valuable insights into the underlying training process. Towards this end, we translate the LTH experiments to a Bayesian setting using common computer vision models. We investigate the defining characteristics of Bayesian lottery tickets, and extend our study towards a transplantation method connecting BNNs with deterministic Lottery Tickets. We generally find that the LTH holds in BNNs, and winning tickets of matching and surpassing accuracy are present independent of model size, with degradation at very high sparsities. However, the pruning strategy should rely primarily on magnitude, secondly on standard deviation. Furthermore, our results demonstrate that models rely on mask structure and weight initialization to varying degrees.

### 2. 

**主要机构**: Advanced Micro Devices, Inc
**作者数量**: 5人

**摘要**:
Vision-language models (VLMs) have achieved remarkable multimodal understanding and reasoning capabilities, yet remain computationally expensive due to dense visual tokenization. Existing efficiency approaches either merge redundant visual tokens or drop them progressively in language backbone, often trading accuracy for speed. In this work, we propose DUET-VLM, a versatile plug-and-play dual compression framework that consists of (a) vision-only redundancy aware compression of vision encoder's output into information-preserving tokens, followed by (b) layerwise, salient text-guided dropping of visual tokens within the language backbone to progressively prune less informative tokens. This coordinated token management enables aggressive compression while retaining critical semantics. On LLaVA-1.5-7B, our approach maintains over 99% of baseline accuracy with 67% fewer tokens ↓, and still retains >97% even at 89% ↓ reduction. With this dual-stage compression during training, it achieves 99.7% accuracy at 67% ↓ and 97.6% at 89% ↓, surpassing prior SoTA visual token reduction methods across multiple benchmarks. When integrated into Video-LLaVA-7B, it even surpasses the baseline-achieving >100% ↑ accuracy with a substantial 53.1% ↓ token reduction and retaining 97.6% accuracy under an extreme 93.4% ↓ setting. These results highlight end-to-end training with DUET-VLM, enabling robust adaptation to reduced visual (image/video) input without sacrificing accuracy, producing compact yet semantically rich representations within the same computational budget. Our code is available at https://github.com/AMD-AGI/DUET-VLM.

### 3. Federated Reasoning Distillation Framework with Model Learnability-Aware Data Allocation

**主要机构**: Beijing University of Civil, Yanshan University, Heilongjiang University, Beihang University, Renmin University of China, Engineering and Architecture, Shandong University
**作者数量**: 11人

**摘要**:
Data allocation plays a critical role in federated large language model (LLM) and small language models (SLMs) reasoning collaboration. Nevertheless, existing data allocation methods fail to address an under-explored challenge in collaboration: bidirectional model learnability gap, where client-side SLMs cannot identify high-reward samples matching their learnability constraints for effective knowledge transfer from LLMs, while LLMs struggle to select samples contributing novel knowledge beyond their existing data. Furthermore, these collaboration frameworks face another key challenge: domain-agnostic reasoning transfer, where existing reasoning transfer methods fail to flexibly adapt to the local domain data, preventing SLMs from effectively acquiring step-by-step reasoning abilities within from general LLM. To address these challenges, we propose LaDa, a federated reasoning distillation framework with model learnability-aware data allocation. It introduces a model learnability-aware data filter that adaptively allocates high-reward samples based on the learnability gap between each SLM and LLM pair, effectively facilitating bidirectional knowledge transfer. We further design a domain adaptive reasoning distillation method that aligns joint probabilities of reasoning paths on filtered high-reward samples through contrastive distillation learning between SLM and LLM, enabling SLM to capture underlying reasoning patterns under local data distribution. LaDa operates as a plug-in module for existing collaboration frameworks, adapting knowledge transfer based on model learnability gaps. We provide theoretical convergence guarantees with O(1/ √ T) rate for classic collaboration frameworks enhanced with our methods and demonstrate up to 13.8% accuracy improvements over state-of-the-art baselines through extensive experiments across four LLM-SLM collaborative scenarios on two widely-used datasets. Our code is available at https://github.com/GUoGUoWi/LaDa.

### 4. Joint Post-Training Quantization of Vision Transformers with Learned Prompt-Guided Data Generation

**主要机构**: vivo Tech Research GmbH, Torr Vision Group Universtiy of Oxford
**作者数量**: 3人

**摘要**:
We present a framework for end-to-end joint quantization of Vision Transformers trained on ImageNet for the purpose of image classification. Unlike prior post-training or block-wise reconstruction methods, we jointly optimize over the entire set of all layers and inter-block dependencies without any labeled data, scaling effectively with the number of samples and completing in just one hour on a single GPU for ViT-small. We achieve state-of-the-art W4A4 and W3A3 accuracies on ImageNet and, to the best of our knowledge, the first PTQ results that maintain strong accuracy on ViT, DeiT, and Swin-T models under extremely lowbit settings (W1.58A8), demonstrating the potential for efficient edge deployment. Furthermore, we introduce a datafree calibration strategy that synthesizes diverse, label-free samples using Stable Diffusion Turbo guided by learned multi-mode prompts. By encouraging diversity in both the learned prompt embeddings and the generated image features, our data-free approach achieves performance on par with real-data ImageNet calibration and surpasses simple text-prompt baselines such as "a <adjective> photo of <adjective> <cls>".

### 5. Rank-Aware Spectral Bounds on Attention Logits for Stable Low-Precision Training

**主要机构**: 
**作者数量**: 1人

**摘要**:
Attention scores in transformers are bilinear forms S ij = x ⊤ i M x j / √ d h whose maximum magnitude governs overflow risk in low-precision training. We derive a rank-aware concentration inequality: when the interaction matrix M = W Q W K⊤ has rank r ≪ d, tail probabilities for max i,j |S ij | decay as exp(-d 2 α 2 /(γr)) rather than exp(-dα 2), where γ > 1 is a typicality parameter. For transformer attention where r = d h , this yields 8-28× tighter concentration than rankagnostic bounds in modern architectures. We apply this result to FP8 training, deriving geometryaware scale factors that provide principled overflow guarantees without observing activations. The method computes per-layer scales from the spectral norm ∥W Q W K⊤ ∥ 2 via implicit power iteration, includes a grouped query attention formulation that avoids key expansion, and remains compatible with fused attention kernels. Across GPT-2 XL to Llama-2-70B, geometry-aware scaling eliminates overflows in transient scenarios where delayed scaling fails, while achieving comparable downstream MMLU accuracy.

### 6. UFO: Unlocking Ultra-Efficient Quantized Private Inference with Protocol and Algorithm Co-Optimization

**主要机构**: Beijing Academy of Blockchain and Edge Computing, School of Integrated Circuits, Institute for Artificial Intelligence, Peking University, School of Software and Microelectronics
**作者数量**: 10人

**摘要**:
Private convolutional neural network (CNN) inference based on secure two-party computation (2PC) suffers from high communication and latency overhead, especially from convolution layers. In this paper, we propose UFO, a quantized 2PC inference framework that jointly optimizes the 2PC protocols and quantization algorithm. UFO features a novel 2PC protocol that systematically combines the efficient Winograd convolution algorithm with quantization to improve inference efficiency. However, we observe that naively combining quantization and Winograd convolution faces the following challenges: 1) From the inference perspective, Winograd transformations introduce extensive additions and require frequent bit width conversions to avoid inference overflow, leading to non-negligible communication overhead; 2) From the training perspective, Winograd transformations introduce weight outliers that make quantization-aware training (QAT) difficult, resulting in inferior model accuracy. To address these challenges, we co-optimize both protocol and algorithm. 1) At the protocol level, we propose a series of graph-level optimizations for 2PC inference to minimize the communication. 2) At the algorithm level, we develop a mixed-precision QAT algorithm based on layer sensitivity to optimize model accuracy given communication constraints. To accommodate the outliers, we further introduce a 2PC-friendly bit re-weighting algorithm to increase the representation range without explicitly increasing bit widths. With extensive experiments, UFO demonstrates 11.7×, 3.6×, and 6.3× communication reduction with 1.29%, 1.16%, and 1.29% higher accuracy compared to state-of-the-art frameworks SiRNN, COINN, and CoPriv, respectively.
