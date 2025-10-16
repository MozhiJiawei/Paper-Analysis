# AI推理加速技术论文分析报告
生成时间: 2025-10-16 13:40:53
分析论文数量: 12篇

## 论文技术简报

### 1. Accelerating Diffusion LLM Inference via Local Determinism Propagation ACCELERATING DIFFUSION LLM INFERENCE VIA LOCAL DETERMINISM PROPAGATION

Klear Team发布了加速扩散LLM推理的论文，使用基于局部确定性传播和渐进空间一致性衰减的LocalLeap训练无关自适应并行解码策略，解决了扩散大语言模型（dLLMs）推理时的质量-速度权衡问题（延迟解码导致的效率低下），达成了6.94倍吞吐量提升、解码步骤减少至14.2%且质量影响可忽略的效果。

### 2. Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods

上海交通大学、香港科技大学（广州）发布了《Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods》论文，使用VTC-Bench评估框架（含数据过滤机制），解决了现有基准评估视觉令牌压缩方法时的任务不匹配及噪声问题，达成了对视觉令牌压缩方法更公平、更准确评估的效果。

### 3. Artificial Hippocampus Networks for Efficient Long-Context Modeling

相关机构发布了Artificial Hippocampus Networks for Efficient Long-Context Modeling论文，使用人工海马体网络技术，解决了长上下文建模效率问题，达成了高效长上下文处理效果。

### 4. AUDIOMARATHON: A COMPREHENSIVE BENCHMARK FOR LONG-CONTEXT AUDIO UNDERSTANDING AND EFFICIENCY IN AUDIO LLMS

清华大学发布了AUDIOMARATHON论文，使用长上下文音频基准（含90-300秒长输入、全领域覆盖及复杂推理）技术，解决了现有音频基准无法评估长音频下Large Audio Language Models理解与效率的问题，达成了揭示现有模型随音频长度增长性能下降、推动更先进音频理解模型研发的效果。

### 5. DISTILLING LIGHTWEIGHT LANGUAGE MODELS FOR C/C++ VULNERABILITIES

北京理工大学发布了FineSec论文，使用知识蒸馏技术将大型语言模型知识转移到轻量级模型，解决C/C++代码漏洞检测问题，达成高效精确识别复杂漏洞和逻辑缺陷且优于基础模型及更大LLM的效果。

### 6. EFFICIENT DISCRIMINATIVE JOINT ENCODERS FOR LARGE SCALE VISION-LANGUAGE RERANKING

Ben-Gurion University of the Negev发布了EFFICIENT DISCRIMINATIVE JOINT ENCODERS FOR LARGE SCALE VISION-LANGUAGE RERANKING论文，使用EDJE（Efficient Discriminative Joint Encoder）——通过离线预计算视觉令牌并经轻量级注意力适配器压缩，在线运行紧凑联合编码器处理少量视觉令牌与文本的技术，解决了现有视觉语言联合编码器（如BLIP）因视觉特征提取开销大而难以大规模部署的问题，达成了每秒处理50k图像-文本对、每张图像仅需49kB存储，且在Flickr（零样本）和COCO（微调）检索上匹配现有技术性能的效果。

### 7. Enhanced Self-Distillation Framework for Efficient Spiking Neural Network Training

浙江大学发布了《Enhanced Self-Distillation Framework for Efficient Spiking Neural Network Training》论文，使用增强的自蒸馏框架（通过将中间层firing rates投影到轻量级ANN分支并解耦教师信号为可靠部分），解决了SNN传统训练方法性能不足且计算与内存开销随时间维度线性增长的问题，达成了在CIFAR-10/100、CIFAR10-DVS及ImageNet上减少训练复杂度并实现高性能SNN训练的效果。

### 8. HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving

研究团队发布了HyPlan论文，使用混合学习辅助规划方法（结合多智能体行为预测、深度强化学习PPO及带启发式置信度垂直剪枝的近似在线POMDP规划），解决了自动驾驶汽车在部分可观测交通环境中的无碰撞导航问题，达成了比相关基线更安全且比其他在线POMDP规划器执行速度显著更快的效果。

### 9. OBS-DIFF: ACCURATE PRUNING FOR DIFFUSION MODELS IN ONE-SHOT

西湖大学发布了OBS-DIFF论文，使用一次性精确剪枝技术，解决了扩散模型剪枝精度不足的问题，达成了扩散模型高效准确剪枝的效果。

### 10. SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation

清华大学和上海人工智能实验室发布了SDAR论文，提出协同扩散-自回归范式，通过将自回归模型高效转换为块级扩散模型，解决了序列生成中自回归训练效率与扩散并行推理能力难以兼顾的问题，实现了保持AR级性能的同时支持并行生成加速，提升了推理和领域适应性。

### 11. Sharpness-Aware Data Generation for Zero-shot Quantization

研究团队发布了《Sharpness-Aware Data Generation for Zero-shot Quantization》论文，使用锐度感知的合成数据生成技术（通过最大化生成样本与邻居的梯度匹配以最小化量化模型锐度），解决了零样本量化中合成数据未考虑模型锐度导致泛化能力不足的问题，在低比特量化场景下性能优于现有技术。

### 12. TALENT: Table VQA via Augmented Language-Enhanced Natural-text Transcription

Johns Hopkins University和Purdue University发布了TALENT论文，使用利用表格双表征（OCR文本和自然语言叙述）的轻量级框架并结合小VLM与LLM进行推理的技术，解决了现有大VLM计算成本高且细节缺失、轻量级方案误差大的问题，达成了使小VLM-LLM组合在显著降低计算成本的同时性能匹配或超越大VLM的效果。

## 论文详细信息

### 1. Accelerating Diffusion LLM Inference via Local Determinism Propagation ACCELERATING DIFFUSION LLM INFERENCE VIA LOCAL DETERMINISM PROPAGATION

**主要机构**: Klear Team
**作者数量**: 6人

**摘要**:
Diffusion large language models (dLLMs) represent a significant advancement in text generation, offering parallel token decoding capabilities. However, existing open-source implementations suffer from quality-speed trade-offs that impede their practical deployment. Conservative sampling strategies typically decode only the most confident token per step to ensure quality (i.e., greedy decoding), at the cost of inference efficiency due to repeated redundant refinement iterations-a phenomenon we term delayed decoding. Through systematic analysis of dLLM decoding dynamics, we characterize this delayed decoding behavior and propose a training-free adaptive parallel decoding strategy, named LocalLeap, to address these inefficiencies. LocalLeap is built on two fundamental empirical principles: local determinism propagation centered on high-confidence anchors and progressive spatial consistency decay. By applying these principles, LocalLeap identifies anchors and performs localized relaxed parallel decoding within bounded neighborhoods, achieving substantial inference step reduction through early commitment of alreadydetermined tokens without compromising output quality. Comprehensive evaluation on various benchmarks demonstrates that LocalLeap achieves 6.94× throughput improvements and reduces decoding steps to just 14.2% of the original requirement, achieving these gains with negligible performance impact. The source codes are available at: https://github.com/friedrichor/LocalLeap.

### 2. Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods

**主要机构**: Hong Kong University of Science and Technology (Guangzhou), Shanghai Jiao Tong University, INSAIT, Shanghai AI Laboratory, Sofia University
**作者数量**: 14人

**摘要**:
Recent endeavors to accelerate inference in Multimodal Large Language Models (MLLMs) have primarily focused on visual token compression. The effectiveness of these methods is typically assessed by measuring the accuracy drop on established benchmarks, comparing model performance before and after compression. However, these benchmarks are originally designed to assess the perception and reasoning capabilities of MLLMs, rather than to evaluate compression techniques. As a result, directly applying them to visual token compression introduces a task mismatch. Strikingly, our investigation reveals that simple image downsampling consistently outperforms many advanced compression methods across multiple widely used benchmarks. Through extensive experiments, we make the following observations: (i) Current benchmarks are noisy for the visual token compression task. (ii) Down-sampling is able to serve as a data filter to evaluate the difficulty of samples in the visual token compression task. Motivated by these findings, we introduce VTC-Bench, an evaluation framework that incorporates a data filtering mechanism to denoise existing benchmarks, thereby enabling fairer and more accurate assessment of visual token compression methods. All data and code are available at https://github.com/Chenfei-Liao/VTC-Bench.

### 3. Artificial Hippocampus Networks for Efficient Long-Context Modeling

**主要机构**: 
**作者数量**: 6人

**摘要**:


### 4. AUDIOMARATHON: A COMPREHENSIVE BENCHMARK FOR LONG-CONTEXT AUDIO UNDERSTANDING AND EFFICIENCY IN AUDIO LLMS

**主要机构**: University of Chinese Academy of Sciences, Tsinghua University, Shanghai AI Laboratory, Shanghai Jiao Tong University
**作者数量**: 16人

**摘要**:
Processing long-form audio is a major challenge for Large Audio Language models (LALMs). These models struggle with the quadratic cost of attention (O(N 2)) and with modeling long-range temporal dependencies. Existing audio benchmarks are built mostly from short clips and do not evaluate models in realistic long context settings. To address this gap, we introduce AUDIOMARATHON, a benchmark designed to evaluate both understanding and inference efficiency on long-form audio. AUDIOMARATHON provides a diverse set of tasks built upon three pillars: long-context audio inputs with durations ranging from 90.0 to 300.0 seconds, which correspond to encoded sequences of 2,250 to 7,500 audio tokens, respectively, full domain coverage across speech, sound, and music, and complex reasoning that requires multi-hop inference. We evaluate state-of-the-art LALMs and observe clear performance drops as audio length grows. We also study acceleration techniques and analyze the trade-offs of token pruning and KV cache eviction. The results show large gaps across current LALMs and highlight the need for better temporal reasoning and memory-efficient architectures. We believe AUDIOMARATHON will drive the audio and multimodal research community to develop more advanced audio understanding models capable of solving complex audio tasks.

### 5. DISTILLING LIGHTWEIGHT LANGUAGE MODELS FOR C/C++ VULNERABILITIES

**主要机构**: Beijing Institute of Technology Beijing, School of Computer Science, School of Computer Science and Technology, School of Cyberspace Science and Technology, University of Auckland Auckland
**作者数量**: 8人

**摘要**:
The increasing complexity of modern software systems exacerbates the prevalence of security vulnerabilities, posing risks of severe breaches and substantial economic loss. Consequently, robust code vulnerability detection is essential for software security. While Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing, their potential for automated code vulnerability detection remains underexplored. This paper presents FineSec, a novel framework that harnesses LLMs through knowledge distillation to enable efficient and precise vulnerability identification in C/C++ codebases. FineSec utilizes knowledge distillation to transfer expertise from large teacher models to compact student models, achieving high accuracy with minimal computational cost. By integrating data preparation, training, evaluation, and continuous learning into a unified, single-task workflow, FineSec offers a streamlined approach. Extensive evaluations on C/C++ codebases demonstrate its superiority over both base models and larger LLMs in identifying complex vulnerabilities and logical flaws, establishing FineSec as a practical and scalable solution for real-world software security. To facilitate reproducibility, the datasets, source code, and experimental results are made publicly available at: https://github.com/yangxiaoxuan123/ FineSec_detect.

### 6. EFFICIENT DISCRIMINATIVE JOINT ENCODERS FOR LARGE SCALE VISION-LANGUAGE RERANKING

**主要机构**: Ben-Gurion University of the Negev, INSIGHT Lab
**作者数量**: 3人

**摘要**:
Multimodal retrieval still leans on embedding-based models like CLIP for fast vector search over pre-computed image embeddings. Yet, unlike text retrieval where joint-encoder rerankers are standard, comparable vision-language rerankers are largely absent. We find that seminal joint encoders such as BLIP are severely bottlenecked by an expensive visual feature-extraction stage, preventing practical deployment at scale. Motivated by this bottleneck, we introduce EDJE , an Efficient Discriminative Joint Encoder that precomputes vision tokens offline and compresses them via a lightweight attention-based adapter, so online inference runs only a compact joint encoder over a small set of visual tokens plus the text. EDJE preserves strong retrieval performance while drastically reducing storage and online compute, enabling high-throughput inference. Specifically, EDJE processes 50k image-text pairs/second while requiring 49kB of disk storage per image, matching prior art on Flickr (zero-shot) and COCO (fine-tuned) retrieval. The implementation and checkpoints will be made publicly available shortly.

### 7. Enhanced Self-Distillation Framework for Efficient Spiking Neural Network Training

**主要机构**: Zhejiang University, ZJU-UIUC Institute
**作者数量**: 5人

**摘要**:
Spiking Neural Networks (SNNs) exhibit exceptional energy efficiency on neuromorphic hardware due to their sparse activation patterns. However, conventional training methods based on surrogate gradients and Backpropagation Through Time (BPTT) not only lag behind Artificial Neural Networks (ANNs) in performance, but also incur significant computational and memory overheads that grow linearly with the temporal dimension. To enable high-performance SNN training under limited computational resources, we propose an enhanced self-distillation framework, jointly optimized with rate-based backpropagation. Specifically, the firing rates of intermediate SNN layers are projected onto lightweight ANN branches, and high-quality knowledge generated by the model itself is used to optimize substructures through the ANN pathways. Unlike traditional self-distillation paradigms, we observe that low-quality self-generated knowledge may hinder convergence. To address this, we decouple the teacher signal into reliable and unreliable components, ensuring that only reliable knowledge is used to guide the optimization of the model. Extensive experiments on CIFAR-10, CIFAR-100, CIFAR10-DVS, and Im-ageNet demonstrate that our method reduces training complexity while achieving high-performance SNN training. Our code is available at https://github.com/Intelli-Chip-Lab/enhanced-self-distillation-framework-for-snn.

### 8. HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving

**主要机构**: 
**作者数量**: 3人

**摘要**:
We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners.

### 9. OBS-DIFF: ACCURATE PRUNING FOR DIFFUSION MODELS IN ONE-SHOT

**主要机构**: Westlake University
**作者数量**: 5人

**摘要**:
Prompt: A portrait of a human growing colorful flowers from her hair. Hyperrealistic oil painting. Intricate details.

### 10. SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation

**主要机构**: University of Maryland, Shanghai AI Laboratory, Tsinghua University
**作者数量**: 11人

**摘要**:
We propose SDAR, a Synergistic Diffusion-AutoRegression paradigm that establishes a new language modeling framework combining the training efficiency of autoregression with the parallel inference capability of diffusion. Instead of costly end-to-end diffusion training, SDAR performs a lightweight paradigm conversion that transforms a well-trained autoregressive (AR) model into a blockwise diffusion model through brief, data-efficient adaptation. During inference, SDAR models generate sequences autoregressively across blocks for global coherence while decoding all tokens within each block in parallel via a discrete diffusion process. Through extensive controlled experiments, we demonstrate that AR models remain substantially more compute-efficient than masked diffusion models, providing a strong foundation for adaptation. Building on this insight, SDAR achieves efficient AR-to-diffusion conversion with minimal cost, preserving AR-level performance while enabling parallel generation. Scaling studies across both dense and Mixture-of-Experts architectures further confirm that SDAR scales without compromise-larger models exhibit increasing robustness to block size and decoding thresholds, yielding greater parallel speedups without loss of accuracy. Beyond efficiency, SDAR also exhibits enhanced reasoning and domain adaptability. Our 30B MoE model surpasses its AR counterpart on challenging scientific reasoning benchmarks such as GPQA and ChemBench, benefiting from local bidirectional context and reduced causal constraints. When combined with test-time scaling strategies such as majority voting and pass@k, SDAR achieves substantial additional gains, indicating strong potential for reinforcement learning optimization. Together, these results establish SDAR as a new and practical language modeling paradigm that unifies the complementary strengths of autoregression and diffusion, enabling scalable, high-throughput inference while preserving the accuracy and reasoning competence of state-of-the-art AR models.

### 11. Sharpness-Aware Data Generation for Zero-shot Quantization

**主要机构**: 
**作者数量**: 5人

**摘要**:
Zero-shot quantization aims to learn a quantized model from a pre-trained full-precision model with no access to original real training data. The common idea in zero-shot quantization approaches is to generate synthetic data for quantizing the full-precision model. While it is wellknown that deep neural networks with low sharpness have better generalization ability, none of the previous zero-shot quantization works considers the sharpness of the quantized model as a criterion for generating training data. This paper introduces a novel methodology that takes into account quantized model sharpness in synthetic data generation to enhance generalization. Specifically, we first demonstrate that sharpness minimization can be attained by maximizing gradient matching between the reconstruction loss gradients computed on synthetic and real validation data, under certain assumptions. We then circumvent the problem of the gradient matching without real validation set by approximating it with the gradient matching between each generated sample and its neighbors. Experimental evaluations on CIFAR-100 and ImageNet datasets demonstrate the superiority of the proposed method over the state-of-the-art techniques in low-bit quantization settings.

### 12. TALENT: Table VQA via Augmented Language-Enhanced Natural-text Transcription

**主要机构**: Johns Hopkins University, Purdue University, University at Albany, Independent Researcher
**作者数量**: 5人

**摘要**:
Table Visual Question Answering (Table VQA) is typically addressed by large vision-language models (VLMs). While such models can answer directly from images, they often miss fine-grained details unless scaled to very large sizes, which are computationally prohibitive, especially for mobile deployment. A lighter alternative is to have a small VLM perform OCR and then use a large language model (LLM) to reason over structured outputs such as Markdown tables. However, these representations are not naturally optimized for LLMs and still introduce substantial errors. We propose TALENT (Table VQA via Augmented Language-Enhanced Natural-text Transcription), a lightweight framework that leverages dual representations of tables. TALENT prompts a small VLM to produce both OCR text and natural language narration, then combines them with the question for reasoning by an LLM. This reframes Table VQA as an LLM-centric multimodal reasoning task, where the VLM serves as a perception-narration module rather than a monolithic solver. Additionally, we construct ReTabVQA, a more challenging Table VQA dataset requiring multi-step quantitative reasoning over table images. Experiments show that TALENT enables a small VLM-LLM combination to match or surpass a single large VLM at significantly lower computational cost on both public datasets and ReTabVQA.
