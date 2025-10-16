# AI推理加速技术论文分析报告
生成时间: 2025-10-16 10:55:32
分析论文数量: 18篇

## 论文技术简报

### 1. Action Deviation-Aware Inference for Low-Latency Wireless Robots

延世大学、阿德莱德大学发布了Action Deviation-Aware Inference for Low-Latency Wireless Robots论文，使用动作偏差感知混合推理技术，解决了分布式机器学习中行为克隆策略无法并行验证多个草稿导致的低延迟协作推理难题，达成了减少上行传输和服务器操作40%、端到端延迟降低33.32%，任务成功率达目标模型单独推理97.03%的效果

### 2. Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression

研究团队发布了Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression论文，使用动态专家聚类与结构化压缩框架，解决了MoE大语言模型的负载不平衡、参数冗余及通信开销三难问题，达成总参数减少约80%、吞吐量提升10%-20%、专家负载方差降低超3倍且匹配标准MoE模型质量的效果。

### 3. CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration

浙江大学与华为诺亚方舟实验室发布了CHORD论文，使用通道级混合精度量化与设备-云协作的CHORD框架（含多粒度参数敏感性分析及超网络模块），解决了现有量化方法忽略设备用户兴趣导致推荐精度下降及设备端微调计算负担重的问题，达成了个性化与资源自适应部署、加速推理并消除重训练周期、最小化通信开销的效果。

### 4. CHUNKLLM: A LIGHTWEIGHT PLUGGABLE FRAME-WORK FOR ACCELERATING LLMS INFERENCE

北京邮电大学发布了CHUNKLLM论文，使用QK Adapter、Chunk Adapter及注意力蒸馏方法的轻量级可插拔框架，解决了Transformer大模型自注意力二次复杂度导致的计算效率低及现有方法语义不完整或效率差的问题，达成了长文本处理最高提速4.48倍并在长上下文基准保持98.64%性能的效果

### 5. DIFFUSPEC: UNLOCKING DIFFUSION LANGUAGE MODELS FOR SPECULATIVE DECODING

南方科技大学与OPPO研究院发布了DIFFUSPEC论文，使用基于扩散语言模型的DiffuSpec框架（含因果一致性路径搜索和自适应草稿长度控制器），解决了大语言模型自回归解码延迟问题，达成最高3倍墙钟速度提升

### 6. Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention

香港科技大学、上海交通大学发布了相关论文，使用HoloV视觉token剪枝框架（通过从整体角度自适应分配不同空间裁剪的剪枝预算以保留全局视觉上下文），解决了MLLMs中因依赖大量视觉token导致的计算开销大及现有注意力优先剪枝方法在高剪枝率下因保留语义相似token而性能下降的问题，达成了在多种任务、MLLM架构和剪枝率下优于SOTA的效果，如LLaVA1.5剪枝88.9%视觉token仍保留95.8%原始性能。

### 7. ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

华为技术有限公司发布了ElasticMoE论文，使用ElasticMoE弹性扩展框架（结合HMM零拷贝重映射与虚拟内存专家重分配），解决了MoE模型弹性扩展粒度粗、延迟高及需重启的问题，达成了9倍扩展延迟降低、2倍吞吐量提升及SLO达标率显著改善的效果

### 8. Flip Distribution Alignment VAE for Multi-Phase MRI Synthesis

中南大学发布了Flip Distribution Alignment VAE for Multi-Phase MRI Synthesis论文，使用Flip Distribution Alignment Variational Autoencoder (FDA-VAE)（一种轻量级特征解耦VAE模型，通过将输入和目标图像编码为关于标准正态分布对称的两个潜分布及Y形双向训练策略），解决了多期增强MRI合成中分离共享和独立特征时现有方法参数效率低、缺乏可解释训练策略的问题，达成了显著减少模型参数和推理时间、同时有效提升合成质量的效果。

### 9. HALO: Memory-Centric Heterogeneous Accelerator with 2.5D Integration for Low-Batch LLM Inference

普渡大学发布了HALO论文，使用2.5D集成的异构加速器（结合HBM基Compute-in-DRAM与片上模拟Compute-in-Memory），解决了低批量LLM推理中预填充与解码阶段的计算和内存需求差异问题，达成相比AttAcc 18倍几何平均加速、相比全CiD的CENT 2.5倍加速的效果

### 10. Hyperparameters are all you need: Using five-step inference for an original diffusion model to generate images comparable to the latest distillation model

University of Nottingham发布了Hyperparameters are all you need论文，使用基于扩散ODE/SDE截断误差分析的无需训练五步推理技术，解决了扩散模型生成高质量图像需高步数和额外训练的问题，达成8步内生成512x512/1024x1024图像且FID性能可比最新蒸馏模型的效果

### 11. Input-Aware Sparse Attention for Real-Time Co-Speech Video Generation

卡内基梅隆大学发布了Input-Aware Sparse Attention for Real-Time Co-Speech Video Generation论文，使用Input-Aware Sparse Attention技术，解决了实时伴随语音视频生成的效率问题，达成了实时生成性能。

### 12. PocketSR: The Super-Resolution Expert in Your Pocket Mobiles

清华大学和香港科技大学发布了PocketSR论文，使用LiteED（高效VAE替代方案）和U-Net在线退火剪枝（结合多层特征蒸馏损失）技术，解决了现有RealSR方法计算成本高、延迟大不适合边缘部署的问题，达成了146M参数模型处理4K图像仅需0.8秒且性能媲美SOTA的效果。

### 13. Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval

Tencent发布了Retrv-R1多模态检索框架，使用带细节检查机制的信息压缩模块及检索定制合成CoT数据集激活+课程奖励RL的新训练范式，解决了直接应用R1方法到多模态检索的高计算成本和RL训练不稳定性问题，达成了SOTA性能、高效率和强泛化能力。

### 14. SAFE AND EFFICIENT IN-CONTEXT LEARNING VIA RISK CONTROL

阿姆斯特丹大学与约翰·霍普金斯大学发布了《SAFE AND EFFICIENT IN-CONTEXT LEARNING VIA RISK CONTROL》论文，使用无分布风险控制（DFRC）结合动态提前退出预测技术，解决了大型语言模型上下文学习中受错误或恶意示例影响的安全隐患，达成了有效控制有害示例风险并在有益示例上提升计算效率的效果。

### 15. SELFJUDGE: FASTER SPECULATIVE DECODING VIA SELF-SUPERVISED JUDGE VERIFICATION

相关研究机构发布了SELFJUDGE: FASTER SPECULATIVE DECODING VIA SELF-SUPERVISED JUDGE VERIFICATION论文，使用SelfJudge技术（通过目标模型自监督训练判断验证器，评估token替换响应语义保留以实现跨多样NLP任务自动验证器训练），解决了现有判断解码依赖人类标注或可验证真值导致的泛化性受限问题，达成了优于判断解码基线的推理-准确性权衡，为更快的LLM推理提供广泛适用解决方案。

### 16. SPEECHCT-CLIP: DISTILLING TEXT-IMAGE KNOWLEDGE TO SPEECH FOR VOICE-NATIVE MULTIMODAL CT ANALYSIS

慕尼黑工业大学发布了SPEECHCT-CLIP论文，使用从文本-图像CLIP模型蒸馏知识到语音的对比模型，解决了医疗AI依赖书面文本、缺乏对口语报告支持的问题，达成零样本分类F1从0.623提升至0.705、恢复88%性能差距且无需文本实现强检索的效果。

### 17. To Compress or Not? Pushing the Frontier of Lossless GenAI Model Weights Compression with Exponent Concentration

Stevens Institute of Technology与Lambda, Inc发布了GenAI模型权重无损压缩论文，使用指数集中现象及Exponent-Concentrated FP8 (ECF8)框架，解决了大模型低精度无损压缩部署问题，达成26.9%内存节省、177.1%吞吐量提升及无损计算效果。

### 18. WAVE-GMS: LIGHTWEIGHT MULTI-SCALE GENERATIVE MODEL FOR MEDICAL IMAGE SEGMENTATION

拉合尔管理科学大学发布了WAVE-GMS论文，使用轻量级多尺度生成模型技术，解决了医疗图像分割中在有限内存GPU上大批次训练且无需预训练基础模型的问题，达成了在四个公开数据集上实现SOTA分割性能、跨域泛化性优异且仅需约2.6M参数的效果。

## 论文详细信息

### 1. Action Deviation-Aware Inference for Low-Latency Wireless Robots

**主要机构**: School of Electrical and Electronic Engineering, Yonsei University, University of Adelaide, School of Electrical and Mechanical Engineering, ISTD Pillar, University of Waterloo, Department of Mechanical and Mechatronics Engineering, Singapore University of Technology and Design
**作者数量**: 6人

**摘要**:
To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML, connecting distributed computational resources in edge and cloud over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: an on-device draft model locally generates drafts and a remote server-based target model verifies and corrects them, resulting lower latency. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications like robot manipulation and autonomous driving, cannot parallelize verification and correction for multiple drafts as each action depends on observation which needs to be updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference, wherein the draft model estimates an action's need for verification and correction by the target model and selectively skips communication and computation for server operations. Action deviation shows a strong correlation with action's rejection probability by the target model, enabling selective skipping. We derive the path deviation threshold that balances the transmission rate and the inference performance, and we empirically show that action deviation-aware hybrid inference reduces uplink transmission and server operation by 40%, while lowering end-to-end latency by 33.32% relative to hybrid inference without skipping and achieving task success rate up to 97.03% of that of target model only inference.

### 2. Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression

**主要机构**: 
**作者数量**: 5人

**摘要**:
Mixture-of-Experts (MoE) Large Language Models (LLMs) face a trilemma of load imbalance, parameter redundancy, and communication overhead. We introduce a unified framework based on dynamic expert clustering and structured compression to address these issues cohesively. Our method employs an online clustering procedure that periodically regroups experts using a fused metric of parameter and activation similarity, which stabilizes expert utilization. To our knowledge, this is one of the first frameworks to leverage the semantic embedding capability of the router to dynamically reconfigure the model's architecture during training for substantial efficiency gains. Within each cluster, we decompose expert weights into a shared base matrix and extremely low-rank residual adapters, achieving up to fivefold parameter reduction per group while preserving specialization. This structure enables a two-stage hierarchical routing strategy: tokens are first assigned to a cluster, then to specific experts within it, drastically reducing the routing search space and the volume of all-to-all communication. Furthermore, a heterogeneous precision scheme, which stores shared bases in FP16 and residual factors in INT4, coupled with dynamic offloading of inactive clusters, reduces peak memory consumption to levels comparable to dense models. Evaluated on GLUE and WikiText-103, our framework matches the quality of standard MoE models while reducing total parameters by approximately 80%, improving throughput by 10% to 20%, and lowering expert load variance by a factor of over three. Our work demonstrates that structural reorganization is a principled path toward scalable, efficient, and memory-effective MoE LLMs.

### 3. CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration

**主要机构**: Zhejiang University Hangzhou, Shanghai Institute for Advanced Study of Zhejiang University Shanghai, Huawei Noah's Ark Lab Shenzhen, Shanghai Jiao Tong University
**作者数量**: 14人

**摘要**:
With the advancement of mobile device capabilities, deploying reranking models directly on devices has become feasible, enabling real-time contextual recommendations. When migrating models from cloud to devices, resource heterogeneity inevitably necessitates model compression. Recent quantization methods show promise for efficient deployment, yet they overlook device-specific user interests, resulting in compromised recommendation accuracy. While on-device finetuning captures personalized user preference, it imposes additional computational burden through local retraining. To address these challenges, we propose a framework for Customizing Hybrid-precision On-device model for sequential Recommendation with Device-cloud collaboration (CHORD), leveraging channel-wise mixed-precision quantization to simultaneously achieve personalization and resource-adaptive deployment. CHORD distributes randomly initialized models across heterogeneous devices and identifies user-specific critical parameters through auxiliary hypernetwork modules on the cloud. Our parameter sensitivity analysis operates across multiple granularities (layer, filter, and element levels), enabling precise mapping from user profiles to quantization strategy. Through on-device mixedprecision quantization, CHORD delivers dynamic model adaptation and accelerated inference without backpropagation, eliminating costly retraining cycles. We minimize communication overhead by encoding quantization strategies using only 2 bits per channel

### 4. CHUNKLLM: A LIGHTWEIGHT PLUGGABLE FRAME-WORK FOR ACCELERATING LLMS INFERENCE

**主要机构**: Beijing University of Posts, School of Artificial Intelligence
**作者数量**: 6人

**摘要**:
Transformer-based large models excel in natural language processing and computer vision, but face severe computational inefficiencies due to the self-attention's quadratic complexity with input tokens. Recently, researchers have proposed a series of methods based on block selection and compression to alleviate this problem, but they either have issues with semantic incompleteness or poor traininginference efficiency. To comprehensively address these challenges, we propose ChunkLLM, a lightweight and pluggable training framework. Specifically, we introduce two components: QK Adapter (Q-Adapter and K-Adapter) and Chunk Adapter. The former is attached to each Transformer layer, serving dual purposes of feature compression and chunk attention acquisition. The latter operates at the bottommost layer of the model, functioning to detect chunk boundaries by leveraging contextual semantic information. During the training phase, the parameters of the backbone remain frozen, with only the QK Adapter and Chunk Adapter undergoing training. Notably, we design an attention distillation method for training the QK Adapter, which enhances the recall rate of key chunks. During the inference phase, chunk selection is triggered exclusively when the current token is detected as a chunk boundary, thereby accelerating model inference. Experimental evaluations are conducted on a diverse set of long-text and short-text benchmark datasets spanning multiple tasks. ChunkLLM not only attains comparable performance on short-text benchmarks but also maintains 98.64% of the performance on long-context benchmarks while preserving a 48.58% key-value cache retention rate. Particularly, ChunkLLM attains a maximum speedup of 4.48× in comparison to the vanilla Transformer in the processing of 120K long texts.

### 5. DIFFUSPEC: UNLOCKING DIFFUSION LANGUAGE MODELS FOR SPECULATIVE DECODING

**主要机构**: SIGS, Southern University of Science and Technology, OPPO Research Institute, Tsinghua University
**作者数量**: 7人

**摘要**:
As large language models (LLMs) scale up, accuracy improves, but the autoregressive (AR) nature of decoding increases latency since each token requires a serial forward pass. Speculative decoding addresses this by employing a fast drafter to propose multi-token drafts, which are then verified in parallel by the target model. However, many deployments still rely on AR drafters, where sequential passes limit wall-clock gains. We revisit the drafting stage and present DiffuSpec, a training-free drop-in framework that uses a pretrained diffusion language model (DLM) to produce multi-token drafts in a single forward pass, while remaining compatible with standard AR verifiers. Because DLM drafts are generated under bidirectional conditioning, parallel per-position candidates form a token lattice in which the locally highest-probability token at each position need not form a causal left-to-right path. Moreover, DLM drafting requires pre-specifying a draft length, inducing a speed-quality trade-off. To address these challenges, we introduce two practical components: (i) a causal-consistency path search (CPS) over this lattice that extracts a left-to-right path aligned with AR verification; and (ii) an adaptive draft-length (ADL) controller that adjusts next proposal size based on recent acceptance feedback and realized generated length. Across benchmarks, DiffuSpec yields up to 3× wall-clock speedup, establishing diffusion-based drafting as a robust alternative to autoregressive drafters for speculative decoding.

### 6. Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention

**主要机构**: The Hong Kong University of Science and Technology, Shanghai Jiao Tong University
**作者数量**: 9人

**摘要**:
Despite their powerful capabilities, Multimodal Large Language Models (MLLMs) suffer from considerable computational overhead due to their reliance on massive visual tokens. Recent studies have explored token pruning to alleviate this problem, which typically uses text-vision cross-attention or [CLS] attention to assess and discard redundant visual tokens. In this work, we identify a critical limitation of such attention-first pruning approaches, i.e., they tend to preserve semantically similar tokens, resulting in pronounced performance drops under high pruning ratios. To this end, we propose HoloV, a simple yet effective, plug-and-play visual token pruning framework for efficient inference. Distinct from previous attention-first schemes, HoloV rethinks token retention from a holistic perspective. By adaptively distributing the pruning budget across different spatial crops, HoloV ensures that the retained tokens capture the global visual context rather than isolated salient features. This strategy minimizes representational collapse and maintains task-relevant information even under aggressive pruning. Experimental results demonstrate that our HoloV achieves superior performance across various tasks, MLLM architectures, and pruning ratios compared to SOTA methods. For instance, LLaVA1.5 equipped with HoloV preserves 95.8% of the original performance after pruning 88.9% of visual tokens, achieving superior efficiency-accuracy trade-offs.

### 7. ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

**主要机构**: Huawei Technologies China, Huawei Technologies, Qintao Zhang Huawei Technologies China
**作者数量**: 17人

**摘要**:
Mixture-of-Experts (MoE) models promise efficient scaling of large language models (LLMs) by activating only a small subset of experts per token, but their parallelized inference pipelines make elastic serving challenging. Existing strategies fall short: horizontal scaling provisions entire replicas of the current configuration, often tens to hundreds of accelerators, leading to coarse granularity, long provisioning delays, and costly overprovisioning; vertical scaling offers finer adjustments but typically requires instance restarts, incurring downtime. These limitations make current approaches illsuited for the bursty, short-lived traffic patterns common in cloud deployments. We present ElasticMoE, an elastic scaling framework for MoE LLMs that achieves fine-grained, low latency, and zerodowntime scaling. ElasticMoE decouples inference execution from memory operations, enabling scaling steps to proceed concurrently with serving. An HBM Management Module (HMM) reuses weights and KV caches via zero-copy remapping, while high-bandwidth peer-to-peer transfers bring newly added accelerators online without interrupting service. A virtual memory-based expert redistribution mechanism migrates MoE experts without costly buffer reallocations, reducing peak memory usage during expert parallelism reconfiguration. Our evaluation on Ascend NPUs with three popular MoE LLMs shows that ElasticMoE achieves up to ≈9X lower scaleup latency, up to ≈2X better throughput during scaling, and results in significant improvement in SLO attainment compared to baselines. By enabling fine-grained, concurrent scaling with minimal disruption, ElasticMoE advances the practicality of deploying massive MoE LLMs in dynamic cloud environments. * Equal Contribution. † Corresponding authors (<yong.zhang3, zhenan.fan1>@huawei.com).

### 8. Flip Distribution Alignment VAE for Multi-Phase MRI Synthesis

**主要机构**: The Second Xiangya Hospital, Big Data Institute, Central South University, Department of Radiology, School of Computer Science and Engineering
**作者数量**: 6人

**摘要**:
Separating shared and independent features is crucial for multi-phase contrast-enhanced (CE) MRI synthesis. However, existing methods use deep autoencoder generators with low parameter efficiency and lack interpretable training strategies. In this paper, we propose Flip Distribution Alignment Variational Autoencoder (FDA-VAE), a lightweight feature-decoupled VAE model for multi-phase CE MRI synthesis. Our method encodes input and target images into two latent distributions that are symmetric concerning a standard normal distribution, effectively separating shared and independent features. The Y-shaped bidirectional training strategy further enhances the interpretability of feature separation. Experimental results show that compared to existing deep autoencoder-based end-to-end synthesis methods, FDA-VAE significantly reduces model parameters and inference time while effectively improving synthesis quality. The source code is publicly available at https://github.com/QianMuXiao/FDA-VAE.

### 9. HALO: Memory-Centric Heterogeneous Accelerator with 2.5D Integration for Low-Batch LLM Inference

**主要机构**: Purdue University West Lafayette, Elmore Family School of Electrical and Computer Engineering
**作者数量**: 2人

**摘要**:
The rapid adoption of Large Language Models (LLMs) has driven a growing demand for efficient inference, particularly in latency-sensitive applications such as chatbots and personalized assistants. Unlike traditional deep neural networks, LLM inference proceeds in two distinct phases: the prefill phase, which processes the full input sequence in parallel, and the decode phase, which generates tokens sequentially. These phases exhibit highly diverse compute and memory requirements, which makes accelerator design particularly challenging. Prior works have primarily been optimized for high-batch inference or evaluated only short input context lengths, leaving the low-batch and longcontext regime, which is critical for interactive applications, largely underexplored. In this work, we propose HALO, a heterogeneous memorycentric accelerator specifically designed to address the unique challenges of prefill and decode phases in low-batch LLM inference. HALO integrates HBM based Compute-in-DRAM (CiD) with an on-chip analog Compute-in-Memory (CiM), co-packaged using 2.5D integration. To further improve the hardware utilization, we introduce a phase-aware mapping strategy that adapts to the distinct demands of the prefill and decode phases. Computebound operations in the prefill phase are mapped to CiM to exploit its high throughput matrix multiplication capability, while memory-bound operations in the decode phase are executed on CiD to benefit from reduced data movement within DRAM. Additionally, we present an analysis of the performance tradeoffs of LLMs under two architectural extremes: a fully CiD and a fully on-chip analog CiM design to highlight the need for a heterogeneous design. We evaluate HALO on LLaMA-2 7B and Qwen3 8B models. Our experimental results show that LLMs mapped to HALO achieve up to 18× geometric mean speedup over AttAcc, an attention-optimized mapping and 2.5× over CENT, a fully CiD based mapping.

### 10. Hyperparameters are all you need: Using five-step inference for an original diffusion model to generate images comparable to the latest distillation model

**主要机构**: University of Nottingham
**作者数量**: 1人

**摘要**:
The diffusion probability model is a state-of-the-art generative model that generates an image by applying a neural network iteratively. Moreover, this generation process is regarded as an algorithm solving a diffusion ordinary differential equation (ODE) or stochastic differential equation (SDE). Based on the analysis of the truncation error of the diffusion ODE and SDE, our study proposes a training-free algorithm that generates high-quality 512 x 512 and 1024 x 1024 images in eight steps, with flexible guidance scales. To the best of my knowledge, our algorithm is the first one that samples a 1024 x 1024 resolution image in 8 steps with an FID performance comparable to that of the latest distillation model, but without additional training. Meanwhile, our algorithm can also generate a 512 x 512 image in 8 steps, and its FID performance is better than the inference result using state-of-the-art ODE solver DPM++ 2m in 20 steps. The result of our algorithm in generating high-quality 512 x 512 images and 1024 x 1024 images with five-step and sixstep inference is also comparable to the latest distillation model. Moreover, unlike most distillation algorithms, which achieve state-of-the-art FID performance by fixing the sampling guidance scale, and which sometimes cannot improve their performance by adding inference steps, our algorithm uses a flexible guidance scale on classifier-free guidance sampling. The increase in inference steps enhances its FID performance. Additionally, the algorithm can be considered a plug-in component compatible with most ODE solvers and latent diffusion models. Extensive experiments are performed using the COCO 2014, COCO 2017, and the LAION dataset. Specifically, we validate our eight-step image generation algorithm using the COCO 2014, COCO 2017, and LAION validation datasets with a 5.5 guidance scale and a 7.5 guidance scale, respectively. Furthermore, the FID performance of the image synthesis in 512 x 512 resolution with a 5.5 guidance scale is 15.7, 22.35, and 17.52, meaning it is comparable with the state-of-the-art ODE solver DPM++ in 20 steps, whose best FID performance is 17.3, 23.75, and 17.33, respectively. Further, it also outperforms the state-of-the-art AMED-plugin solver, whose FID performance is 19.07, 25.50, and 18.06. We also apply the algorithm in five-step inference without additional training, for which the best FID performance of our algorithm in COCO 2014, COCO 2017, and LAION is 19.18, 23.24, and 19.61, respectively, which is comparable to the performance of the state-of-the-art AMED Pulgin solver in eight steps, SDXLturbo in four steps, and the state-of-the-art diffusion distillation model Flash Diffusion in five steps. Then, we validate our algorithm in synthesizing 1024 * 1024 images, whose FID performance in COCO 2014, COCO 2017, and LAION using eight-step inference is 17.84, 24.42, and 19.25, respectively. Thus, it outperforms the SDXL-lightning in eight steps, Flash DiffusionXL in eight steps, and DMD2 in four steps. Moreover, the FID performance of the six-step inference of our algorithm in the 1024 x 1024 image synthesis is 23, which only has a limited distance to the state-of-the-art distillation model mentioned above. We also use information theory to explain the advantage of our algorithm and why it achieves a strong FID performance.

### 11. Input-Aware Sparse Attention for Real-Time Co-Speech Video Generation

**主要机构**: Carnegie Mellon University
**作者数量**: 1人

**摘要**:


### 12. PocketSR: The Super-Resolution Expert in Your Pocket Mobiles

**主要机构**: Tsinghua University Joy Future Academy HKUST, Equal Contribution
**作者数量**: 12人

**摘要**:
Real-world image super-resolution (RealSR) aims to enhance the visual quality of in-the-wild images, such as those captured by mobile phones. While existing methods leveraging large generative models demonstrate impressive results, the high computational cost and latency make them impractical for edge deployment. In this paper, we introduce PocketSR, an ultra-lightweight, single-step model that brings generative modeling capabilities to RealSR while maintaining high fidelity. To achieve this, we design LiteED, a highly efficient alternative to the original computationally intensive VAE in SD, reducing parameters by 97.5% while preserving high-quality encoding and decoding. Additionally, we propose online annealing pruning for the U-Net, which progressively shifts generative priors from heavy modules to lightweight counterparts, ensuring effective knowledge transfer and further optimizing efficiency. To mitigate the loss of prior knowledge during pruning, we incorporate a multi-layer feature distillation loss. Through an in-depth analysis of each design component, we provide valuable insights for future research. PocketSR, with a model size of 146M parameters, processes 4K images in just 0.8 seconds, achieving a remarkable speedup over previous methods. Notably, it delivers performance on par with state-of-the-art single-step and even multi-step RealSR models, making it a highly practical solution for edge-device applications.

### 13. Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval

**主要机构**: Tencent, City University of Hong, Zhejiang University
**作者数量**: 5人

**摘要**:
The success of DeepSeek-R1 demonstrates the immense potential of using reinforcement learning (RL) to enhance LLMs' reasoning capabilities. This paper introduces Retrv-R1, the first R1-style MLLM specifically designed for multimodal universal retrieval, achieving higher performance by employing step-by-step reasoning to produce more accurate retrieval results. We find that directly applying the methods of DeepSeek-R1 to retrieval tasks is not feasible, mainly due to (1) the high computational cost caused by the large token consumption required for multiple candidates with reasoning processes, and (2) the instability and suboptimal results when directly applying RL to train for retrieval tasks. To address these issues, Retrv-R1 introduces an information compression module with a details inspection mechanism, which enhances computational efficiency by reducing the number of tokens while ensuring that critical information for challenging candidates is preserved. Furthermore, a new training paradigm is proposed, including an activation stage using a retrieval-tailored synthetic CoT dataset for more effective optimization, followed by RL with a novel curriculum reward to improve both performance and efficiency. Incorporating these novel designs, Retrv-R1 achieves SOTA performance, high efficiency, and strong generalization ability, as demonstrated by experiments across multiple benchmarks and tasks. Project page.

### 14. SAFE AND EFFICIENT IN-CONTEXT LEARNING VIA RISK CONTROL

**主要机构**: Department of Computer Science, UvA-Bosch Delta Lab University of Amsterdam, Johns Hopkins University
**作者数量**: 7人

**摘要**:
Large language models (LLMs) demonstrate a remarkable ability to learn new tasks from a few in-context examples. However, this flexibility introduces safety concerns: LLMs can be influenced by incorrect or malicious demonstrations-for example, if an adversary tampers with or injects harmful examples without a human supervisor noticing. This motivates principled designs in which the system itself includes built-in mechanisms to guard against such attacks. We propose a novel approach to limit the degree to which harmful demonstrations can degrade model performance. First, we define a baseline "safe" behavior for the modelthe model's performance given no in-context demonstrations (zero-shot). Next, we apply distribution-free risk control (DFRC) to control the extent to which incontext samples can decay performance below zero-shot. We achieve this by leveraging dynamic early exit prediction, ignoring later attention heads that attend the most to the unsafe inputs. Finally, we propose modifications to DFRC that allow it to both control risk for harmful inputs and leverage performance and efficiency gains on helpful inputs. We present both theoretical and empirical results showing that our approach can effectively control risk for harmful in-context demonstrations while simultaneously achieving substantial computational efficiency gains with helpful demonstrations.

### 15. SELFJUDGE: FASTER SPECULATIVE DECODING VIA SELF-SUPERVISED JUDGE VERIFICATION

**主要机构**: 
**作者数量**: 11人

**摘要**:
Speculative decoding accelerates LLM inference by verifying candidate tokens from a draft model against a larger target model. Recent "judge" decoding boosts this process by relaxing verification criteria by accepting draft tokens that may exhibit minor discrepancies from target model output, but existing methods are restricted by their reliance on human annotations or tasks with verifiable ground truths, limiting generalizability across diverse NLP tasks. We propose SelfJudge, which trains judge verifiers via self-supervision of the target model. Our method measures semantic preservation by assessing whether token-substituted responses preserve the meaning of original responses, enabling automatic verifier training across diverse NLP tasks. Our experiments show SelfJudge achieves superior inference-accuracy trade-offs than judge decoding baselines, offering a broadly applicable solution for faster LLM inference.

### 16. SPEECHCT-CLIP: DISTILLING TEXT-IMAGE KNOWLEDGE TO SPEECH FOR VOICE-NATIVE MULTIMODAL CT ANALYSIS

**主要机构**: Technical University of Munich, Pattern Recognition Lab, Computer Aided Medical Procedures, Friedrich-Alexander-Universität Erlangen-Nürnberg
**作者数量**: 9人

**摘要**:
Spoken communication plays a central role in clinical workflows. In radiology, for example, most reports are created through dictation. Yet, nearly all medical AI systems rely exclusively on written text. In this work, we address this gap by exploring the feasibility of learning visual-language representations directly from spoken radiology reports. Specifically, we synthesize a large-scale dataset (Speech-RATE) of spoken radiology reports and train SpeechCT-CLIP, a contrastive model that aligns speech and 3D CT volumes in a shared representation space. While naïve speech-based models underperform compared to text-trained counterparts, we show that knowledge distillation from a pretrained text-image CLIP model effectively transfers semantic alignment capabilities from text to speech, substantially narrowing this gap. Experiments demonstrate improved zero-shot classification F1 from 0.623 to 0.705, recovering 88% of the performance difference, and strong retrieval results without requiring text at inference. These findings highlight speech as a practical alternative to text in multimodal pretraining and open the door to voice-driven diagnostic support tools in clinical practice.

### 17. To Compress or Not? Pushing the Frontier of Lossless GenAI Model Weights Compression with Exponent Concentration

**主要机构**: Stevens Institute of Technology, Dept. of Computer Science, Lambda, Inc, Rice University, Dept. of Electrical and Computer Engineering
**作者数量**: 6人

**摘要**:
The scaling of Generative AI (GenAI) models into the hundreds of billions of parameters makes low-precision computation indispensable for efficient deployment. We argue that the fundamental solution lies in developing low-precision floating-point formats, which inherently provide numerical stability, memory savings, and hardware efficiency without dequantization overhead. In this paper, we present a theoretical and empirical study of an exponent concentration phenomenon in GenAI weights: exponents consistently exhibit low entropy across architectures and modalities. We show that this arises naturally from α-stable distributions induced by stochastic gradient descent, and we prove tight bounds on the entropy of exponents. Our analysis establishes a theoretical compression limit near FP4.67, which motivates the design of a practical FP8 format. Building on these insights, we propose Exponent-Concentrated FP8 (ECF8), a lossless compression framework with entropy-aware encoding and GPU-optimized decoding. Experiments on LLMs and DiTs up to 671B parameters demonstrate up to 26.9% memory savings and 177.1% throughput acceleration, with perfectly lossless computations, i.e., no deviation in model outputs. Our results establish exponent concentration as a statistical law of trained models and open a principled path for lossless low-precision floating-point design in the FP8 era.

### 18. WAVE-GMS: LIGHTWEIGHT MULTI-SCALE GENERATIVE MODEL FOR MEDICAL IMAGE SEGMENTATION

**主要机构**: Lahore University of Management Sciences, School of Science and Engineering
**作者数量**: 3人

**摘要**:
For equitable deployment of AI tools in hospitals and healthcare facilities, we need Deep Segmentation Networks that offer high performance and can be trained on cost-effective GPUs with limited memory and large batch sizes. In this work, we propose Wave-GMS, a lightweight and efficient multi-scale generative model for medical image segmentation. Wave-GMS has a substantially smaller number of trainable parameters, does not require loading memory-intensive pretrained vision foundation models, and supports training with large batch sizes on GPUs with limited memory. We conducted extensive experiments on four publicly available datasets (BUS, BUSI, Kvasir-Instrument, and HAM10000), demonstrating that Wave-GMS achieves state-of-the-art segmentation performance with superior cross-domain generalizability, while requiring only ∼2.6M trainable parameters. Code is available at https://github.com/ ATPLab-LUMS/Wave-GMS.
