# AI推理加速技术论文分析报告
生成时间: 2025-10-15 15:43:59
分析论文数量: 39篇

## 论文技术简报

### 1. ACTION-AWARE DYNAMIC PRUNING FOR EFFICIENT VISION-LANGUAGE-ACTION MANIPULATION

悉尼大学发布了Action-aware Dynamic Pruning (ADP)框架论文，使用结合文本驱动令牌选择与动作感知轨迹门控的多模态剪枝技术，解决了Vision-Language-Action (VLA)模型在机器人操作中因忽略操作阶段冗余差异导致的推理效率与感知精度平衡问题，达成了显著减少FLOPs和动作推理延迟（如OpenVLA-OFT上1.35倍加速）并提升成功率（如OpenVLA上25.8%提升）的效果。

### 2. Adaptive Dual-Mode Distillation with Incentive Schemes for Scalable, Heterogeneous Federated Learning on Non-IID Data

University of Gujrat发布了《Adaptive Dual-Mode Distillation with Incentive Schemes for Scalable, Heterogeneous Federated Learning on Non-IID Data》论文，使用自适应双模式蒸馏与激励机制（DL-SH、DL-MH、I-DL-MH），解决了联邦学习中统计异构性、模型异构性及客户端参与激励问题，达成DL-SH较标准FL提升153%全局模型准确率、I-DL-MH在非IID条件下提升225%准确率并降低通信成本的效果

### 3. BRAIN PATHOGRAPH LEARNING

Macquarie University与Dalian University of Technology发布了《BRAIN PATHOGRAPH LEARNING》论文，使用病理模式过滤与病理特征蒸馏技术，解决了现有脑图学习方法难以选择性学习疾病相关知识导致的参数及计算成本高、临床应用受限问题，达成了在脑疾病检测任务中模型性能与计算效率的双重提升效果。

### 4. BRIDGING DRAFT POLICY MISALIGNMENT: GROUP TREE OPTIMIZATION FOR SPECULATIVE DECODING

复旦大学与新加坡国立大学发布了BRIDGING DRAFT POLICY MISALIGNMENT: GROUP TREE OPTIMIZATION FOR SPECULATIVE DECODING论文，使用Group Tree Optimization (GTO)技术，解决了推测解码中草稿策略与树解码策略的错位问题，达成了接受长度增加7.4%并实现额外7.7%推理加速的效果。

### 5. COSPADI: COMPRESSING LLMS VIA CALIBRATION-GUIDED SPARSE DICTIONARY LEARNING

研究团队发布了COSPADI论文，使用校准引导的稀疏字典学习（CoSpaDi），通过结构化稀疏因子分解（密集字典与列稀疏系数矩阵）替代低秩分解，解决了现有低秩方法结构约束刚性导致的LLM压缩后精度下降问题，在20-50%压缩比下，多个Llama和Qwen模型的精度与困惑度优于SOTA数据感知低秩方法。

### 6. CubistMerge: Spatial-Preserving Token Merging For Diverse ViT Backbones

英属哥伦比亚大学发布了CubistMerge论文，使用结合2D缩减策略、空间感知合并算法和max-magnitude-per-dimension表示的空间保留token合并技术，解决了现有token缩减方法无法保留空间架构ViT骨干网依赖的空间结构的问题，达成了在SAM-H上1.25倍加速且mIOU仅降0.7%、DeiT-B微调1个epoch 1.15倍加速且ImageNet准确率无下降的效果。

### 7. Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning

University of Waterloo发布了Dual-Head Reasoning Distillation (DHRD)论文，使用为decoder-only语言模型添加训练推理时池化分类头和仅训练时推理头的双头部推理蒸馏技术，解决了Chain-of-Thought提示提高分类准确率但导致吞吐量下降的权衡问题，达成在七个SuperGLUE任务上相对增益0.65-5.47%（尤其蕴含/因果任务）且推理吞吐量与池化分类器匹配、较CoT解码快96-142× QPS的效果。

### 8. DYNAMIC EXPERTS SEARCH: ENHANCING REASONING IN MIXTURE-OF-EXPERTS LLMS AT TEST TIME

南洋理工大学、浙江大学发布了Dynamic Experts Search论文，使用动态专家搜索技术（DES），解决了现有Test-Time Scaling方法忽略MoE模型架构、依赖输出级采样的问题，在数学、代码和知识等推理基准上可靠优于TTS基线，提升准确性和稳定性且无额外成本。

### 9. ELASTIC MOE: UNLOCKING THE INFERENCE-TIME SCALABILITY OF MIXTURE-OF-EXPERTS

百度发布了ELASTIC MOE论文，使用Elastic Mixture-of-Experts (EMoE)训练框架（通过训练专家在不同组合中协作并优化路由器选择），解决了MoE模型推理时增加激活专家数量导致性能快速下降的问题，达成了将有效性能扩展范围扩展至训练时k的2-3倍并提升模型峰值性能的效果

### 10. Enriching Knowledge Distillation with Intra-Class Contrastive Learning

东南大学发布了Enriching Knowledge Distillation with Intra-Class Contrastive Learning论文，使用在教师训练中加入类内对比损失并整合margin loss的技术，解决了现有知识蒸馏中教师模型未考虑同一类内多样表示导致软标签类内信息不足及类内损失带来的训练不稳定和收敛慢问题，达成了丰富软标签类内信息、提升训练稳定性和收敛速度的效果。

### 11. FASTGRPO: ACCELERATING POLICY OPTIMIZATION VIA CONCURRENCY-AWARE SPECULATIVE DECODING AND ONLINE DRAFT LEARNING

兰州大学、香港大学发布了FASTGRPO论文，使用并发感知的推测解码框架和在线草稿学习机制，解决了GRPO训练中生成阶段慢及高并发下推测解码加速有限、分布偏移导致性能下降的问题，达成了端到端加速2.35x至2.72x的效果

### 12. From Long to Lean: Performance-aware and Adaptive Chain-of-Thought Compression via Multi-round Refinement

Harbin Institute of Technology发布了From Long to Lean: Performance-aware and Adaptive Chain-of-Thought Compression via Multi-round Refinement论文，使用Multiround Adaptive Chain-of-Thought Compression (MACC)框架，该框架利用token elasticity现象通过多轮优化渐进压缩CoT，解决了Chain-of-Thought推理因冗长导致的推理延迟问题，达成了平均准确率提升5.6%、CoT长度减少47个token并显著降低延迟的效果。

### 13. HEAPR: HESSIAN-BASED EFFICIENT ATOMIC EXPERT PRUNING IN OUTPUT SPACE

浙江大学发布了HEAPr论文，使用将专家分解为原子专家并利用基于Hessian的二阶信息进行原子级剪枝的技术，解决了MoE模型内存需求高及专家级剪枝精度损失大的问题，达成了20%-25%压缩率下近无损压缩且FLOPs降低近20%的效果

### 14. HierLight-YOLO: A Hierarchical and Lightweight Object Detection Network for UAV Photography

深圳大学发布了HierLight-YOLO论文，使用Hierarchical Extended Path Aggregation Network (HEPAN)、Inverted Residual Depthwise Convolution Block (IRDCB)、Lightweight Downsample (LDown)模块及小目标检测头，解决了无人机摄影中小目标检测高误检率及资源受限平台实时性问题，达成nano-scale模型2.2M参数（比YOLOv8-N少26.7%）下AP 0.5达35.8%、medium-scale模型AP 0.5 50.2%（超越YOLOv8-M 5.6%）的效果。

### 15. IIET: Efficient Numerical Transformer via Implicit Iterative Euler Method

美团与东北大学发布了IIET论文，使用隐式迭代欧拉方法简化高阶数值方法的Iterative Implicit Euler Transformer (IIET)及Iteration Influence-Aware Distillation (IIAD)技术，解决了高阶数值方法Transformer的性能-效率权衡及传统优化技术损害性能问题，达成了平均精度较vanilla Transformer提升2.65%、较PCformer提升0.8%，其高效变体E-IIET减少55%推理开销同时保留99.4%精度的效果。

### 16. IN THEIR OWN WORDS: REASONING TRACES TAI-LORED FOR SMALL MODELS MAKE THEM BETTER REASONERS

延世大学发布了《IN THEIR OWN WORDS: REASONING TRACES TAILORED FOR SMALL MODELS MAKE THEM BETTER REASONERS》论文，使用Reverse Speculative Decoding (RSD)技术，解决了大模型推理轨迹与小模型分布错位导致的推理能力转移失败问题，使Qwen3-0.6B在主要推理基准上平均性能提升4.9%（直接蒸馏则下降20.5%）。

### 17. Joint graph entropy knowledge distillation for point cloud classification and robustness against corruptions

武汉科技大学发布了Joint graph entropy knowledge distillation (JGEKD)相关论文，使用基于联合图熵的知识蒸馏策略（含联合图熵损失函数、孪生结构及自/教师知识蒸馏框架），解决了3D点云分类中类相关性被破坏及对数据损坏鲁棒性不足的问题，达成了在ScanObject、ModelNet40等多个数据集上的竞争分类性能与抗损坏鲁棒性提升。

### 18. KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cache

多伦多大学发布了KV-Efficient VLA论文，使用RNN门控分块KV缓存机制，解决了VLA模型长程推理中注意力二次成本和KV内存无限增长的效率问题，达成了1.21×推理加速和36%KV内存减少且对任务成功影响小的效果

### 19. Large AI Model-Enabled Generative Semantic Communications for Image Transmission

相关机构发布了《Large AI Model-Enabled Generative Semantic Communications for Image Transmission》论文，使用通过分割图像为关键与非关键区域细化语义粒度（关键区域用图像语义编码器处理、非关键区域用图像到文本建模压缩）及轻量级部署策略（模型量化与低秩适应微调）的技术，解决了现有方法忽略图像区域重要性差异导致视觉关键内容重建质量低及大模型存储计算需求大的问题，达成了在语义保真度和视觉质量上优于传统方法的效果。

### 20. LFA-Net: A Lightweight Network with LiteFusion Attention for Retinal Vessel Segmentation

贝尔法斯特女王大学发布了LFA-Net论文，使用新设计的LiteFusion-Attention模块（结合残差学习、Vision Mamba启发的动态特性及调制注意力），解决了现有模型小血管分割难和计算成本高的挑战，达成轻量化（0.11百万参数、0.42MB内存、4.46 GFLOPs）并在DRIVE等数据集上取得高Dice分数（83.28%-87.44%）的效果。

### 21. LIGHTWEIGHT ERROR MITIGATION STRATE-GIES FOR POST-TRAINING N:M ACTIVATION SPARSITY IN LLMS

研究团队发布了LLM训练后N:M激活稀疏化轻量级误差缓解策略论文，使用轻量级误差缓解的训练后N:M激活剪枝技术，解决了LLM推理中激活剪枝研究不足的问题，达成了同等稀疏度下相比权重剪枝更好保留生成能力、16:32模式性能接近非结构化稀疏且8:16模式为更优选择的效果。

### 22. LONGLIVE: REAL-TIME INTERACTIVE LONG VIDEO GENERATION

NVlabs发布了LONGLIVE论文，使用因果帧级自回归框架并整合KV-recache机制、流式长调优及短窗口注意力+帧级注意力池技术，解决了长视频生成的效率与质量挑战及交互式流式提示输入下的视觉一致性与语义连贯问题，达成了在单个H100上20.7 FPS实时生成、支持240秒视频且VBench性能强劲的效果

### 23. MS-YOLO: Infrared Object Detection for Edge Deployment via MobileNetV4 and SlideLoss

Kummer Institute Center for Artificial Intelligence and Autonomous Systems发布了MS-YOLO论文，使用MobileNetV4骨干与SlideLoss损失函数，解决了红外目标检测中的类别不平衡、热噪声及计算限制问题，达成减少1.5%计算开销、提升精度并实现6.7 GFLOPs高效边缘部署的效果。

### 24. OJAKV: CONTEXT-AWARE ONLINE LOW-RANK KV CACHE COMPRESSION WITH OJA'S RULE

IBM Research发布了OjaKV论文，使用混合存储策略（保留首尾关键token全秩）与Oja算法在线子空间适应技术，解决了大语言模型KV缓存内存瓶颈（现有静态离线子空间在数据分布变化时表现差的问题），达成了在高压缩比下保持甚至提升零样本准确率、尤其长上下文复杂推理任务性能提升的效果

### 25. Progressive Weight Loading: Accelerating Initial Inference and Gradually Boosting Performance on Resource-Constrained Environments

Intel发布了Progressive Weight Loading (PWL)技术论文，使用渐进式权重加载技术，解决了资源受限环境中深度学习模型初始推理延迟高和模型加载时间长的问题，达成了保持竞争蒸馏性能、逐步提升准确率至与完整教师模型相当且不牺牲初始推理速度的效果。

### 26. Pushing Toward the Simplex Vertices: A Simple Remedy for Code Collapse in Smoothed Vector Quantization

中部大学发布了关于平滑向量量化中代码崩溃的论文，使用通过最小化单纯形顶点与K近邻平滑量化器距离的正则化技术，解决了现有方法难以同时保证平滑量化器接近onehot向量和防止代码崩溃的问题，达成了更可靠的码本利用和性能提升效果。

### 27. RAPID 3 : TRI-LEVEL REINFORCED ACCELERATION POLICIES FOR DIFFUSION TRANSFORMER

The University of Texas at Austin与Zhejiang University发布了RAPID³: Tri-Level Reinforced Acceleration Policies for Diffusion Transformer论文，使用三层次强化加速策略（含Step-Skip、Cache-Reuse、Sparse-Attention轻量级策略头及GRPO在线训练、对抗判别器奖励增强），解决了Diffusion Transformers采样慢且现有加速方法存在局限的问题，达成了近3倍采样加速且生成质量有竞争力的效果。

### 28. REFINE-CONTROL: A SEMI-SUPERVISED DISTILLATION METHOD FOR CONDITIONAL IMAGE GENERATION

东南大学与联想研究院发布了REFINE-CONTROL论文，使用半监督蒸馏框架（含三级知识融合损失及标记与未标记数据利用），解决了条件图像生成模型高资源需求及标注数据稀缺阻碍边缘部署的问题（含成本与隐私隐患），达成了显著降低计算成本和延迟同时保持高保真生成能力与可控性的效果。

### 29. RESIDUAL VECTOR QUANTIZATION FOR COMMUNICATION-EFFICIENT MULTI-AGENT PERCEPTION

卡内基梅隆大学发布了通信高效的多智能体感知论文，使用基于多阶段残差向量量化（RVQ）的学习型特征编解码器ReVQom技术，解决了通信带宽限制多智能体协作感知可扩展性的问题，达成了6-30 bpp下273×-1365×压缩比、18 bpp时匹配或超过原始特征协作感知性能的效果

### 30. Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts RETRIEVAL-OF-THOUGHT: EFFICIENT REASONING VIA REUSING THOUGHTS

加州大学发布了Retrieval-of-Thought（RoT）论文，使用重用先前推理作为可组合“思想”步骤构建思想图并动态生成问题模板的技术，解决了大型推理模型推理轨迹长导致延迟和成本过高的问题，达成了在保持准确率的同时减少输出tokens达40%、推理延迟82%及成本59%的效果。

### 31. SCALE-WISE VAR IS SECRETLY DISCRETE DIFFUSION

约翰霍普金斯大学发布了SCALE-WISE VAR IS SECRETLY DISCRETE DIFFUSION论文，使用发现配备马尔可夫注意力掩码的VAR等价于离散扩散（SRDD）并建立AR转换器与扩散模型桥梁的技术，解决了VAR的架构效率问题，达成了更快收敛、更低推理成本、改进零样本重建及提升多数据集效率与生成质量的效果

### 32. Self-Speculative Biased Decoding for Faster Live Translation

发布了《Self-Speculative Biased Decoding for Faster Live Translation》论文，使用Self-Speculative Biased Decoding技术，解决了实时翻译等流式应用中输入扩展时需持续更新输出且计算成本高、延迟大的问题，达成了相比传统自回归重翻译高达1.7倍加速且不损失质量、同时减少80%闪烁的效果

### 33. SGNNBench: A Holistic Evaluation of Spiking Graph Neural Network on Large-scale Graph

中南大学、中山大学发布了SGNNBench论文，提出脉冲图神经网络(SGNNs)评估基准，解决SGNNs缺乏系统评估以探索设计原则的问题，全面评估9个先进SGNNs在18个数据集上的有效性、能效与架构设计，揭示能效瓶颈并促进通用SGNN范式发展。

### 34. SIMULSENSE: SENSE-DRIVEN INTERPRETING FOR EFFICIENT SIMULTANEOUS SPEECH TRANSLATION

Nara Institute of Science and Technology发布了SIMULSENSE论文，使用感知语义单元触发写决策的框架，解决了同步语音翻译系统依赖专门交错训练数据和昂贵LLM推理导致的决策效率低问题，达成了更优的质量-延迟权衡及实时效率提升，决策速度比基线快9.6倍。

### 35. SlimDiff: Training-Free, Activation-Guided Hands-free Slimming of Diffusion Models

Purdue University发布了SlimDiff论文，使用激活引导的无训练结构压缩框架，解决了扩散模型计算成本高且现有效率技术依赖微调/重训练的问题，达成了高达35%加速、约1亿参数减少、生成质量与未压缩模型相当，且仅需约500校准样本（较现有方法少70倍）的效果。

### 36. Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models

ETH Zürich发布了Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models论文，使用结构化稀疏转移矩阵参数化方法（PD-SSM，将转移矩阵表示为列独热矩阵与复数值对角矩阵乘积），解决了状态空间模型（SSMs）转移矩阵表达性与计算成本的矛盾，达成高效计算与最优状态追踪表达性，性能显著优于现有SSM变体并在时间序列分类上表现良好

### 37. Towards Adapting Federated & Quantum Machine Learning for Network Intrusion Detection: A Survey

研究团队发布了《Towards Adapting Federated & Quantum Machine Learning for Network Intrusion Detection: A Survey》论文，使用联邦学习与量子机器学习融合技术（含量子联邦学习），解决了网络入侵检测中敏感流量数据无法集中化的隐私保护及复杂模式识别效率问题，达成了保护数据隐私并承诺复杂网络流量模式识别指数级加速的效果。

### 38. Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting

斯坦福大学发布了Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting论文，使用2D Gaussian Splatting（2DGS）作为压缩图像表示的视觉基底并改编CLIP（冻结RGB transformer骨干+轻量级输入茎与perceiver resampler），解决了现代视觉语言管道中密集RGB图像传输耗能成本高及patch token化序列长度爆炸的问题，达成输入压缩3-20倍同时保持有意义零样本ImageNet-1K性能，且2DGS管道拟合速度提升90×、GPU利用率达∼97%的效果。

### 39. γ-Quant: Towards Learnable Quantization for Low-bit Pattern Recognition

University of Siegen和University of Mannheim发布了γ-Quant论文，使用任务特定的非线性可学习量化技术，解决了低带宽和能量受限场景下低比特深度传感器数据的模式识别效率与性能问题，达成了4比特原始数据性能与12比特原始数据相当的效果

## 论文详细信息

### 1. ACTION-AWARE DYNAMIC PRUNING FOR EFFICIENT VISION-LANGUAGE-ACTION MANIPULATION

**主要机构**: The University of Sydney, School of Computer Science
**作者数量**: 6人

**摘要**:
Robotic manipulation with Vision-Language-Action models requires efficient inference over long-horizon multi-modal context, where attention to dense visual tokens dominates computational cost. Existing methods optimize inference speed by reducing visual redundancy within VLA models, but they overlook the varying redundancy across robotic manipulation stages. We observe that the visual token redundancy is higher in coarse manipulation phase than in fine-grained operations, and is strongly correlated with the action dynamic. Motivated by this observation, we propose Action-aware Dynamic Pruning (ADP), a multi-modal pruning framework that integrates text-driven token selection with action-aware trajectory gating. Our method introduces a gating mechanism that conditions the pruning signal on recent action trajectories, using past motion windows to adaptively adjust token retention ratios in accordance with dynamics, thereby balancing computational efficiency and perceptual precision across different manipulation stages. Extensive experiments on the LIBERO suites and diverse real-world scenarios demonstrate that our method significantly reduces FLOPs and action inference latency (e.g. 1.35× speed up on OpenVLA-OFT) while maintaining competitive success rates (e.g. 25.8% improvements with OpenVLA) compared to baselines, thereby providing a simple plug-in path to efficient robot policies that advances the efficiency and performance frontier of robotic manipulation. Our project website is: ADP.com.

### 2. Adaptive Dual-Mode Distillation with Incentive Schemes for Scalable, Heterogeneous Federated Learning on Non-IID Data

**主要机构**: Department of Computer Science, University of Gujrat
**作者数量**: 1人

**摘要**:
It has been increasingly difficult to effectively use the vast amounts of valuable, real-time data generated by smart devices for machine learning model training due to privacy concerns. Federated Learning (FL) has emerged as a promising decentralized learning (DL) approach that enables the use of distributed data without compromising user privacy. However, FL poses several key challenges. First, it is frequently assumed that every client can train the same machine learning models, however, not all clients are able to meet this assumption because of differences in their business needs and computational resources. Second, statistical heterogeneity (a.k.a. non-IID data) poses a major challenge in FL, which can lead to lower global model performance. Third, while addressing these challenges, there is a need for a cost-effective incentive mechanism to encourage clients to participate in FL training. In response to these challenges, we propose several methodologies: DL-SH, which facilitates efficient, privacy-preserving, and communication-efficient learning in the context of statistical heterogeneity; DL-MH, designed to manage fully heterogeneous models while tackling statistical disparities; and I-DL-MH, an incentivebased extension of DL-MH that promotes client engagement in federated learning training by providing incentives within this complex federated learning framework. Comprehensive experiments were carried out to assess the performance and scalability of the proposed approaches across a range of complex experimental settings. This involved utilizing various model architectures, such as ResNet18, DenseNet, and ResNet8, in diverse data distributions, including IID and several non-IID scenarios, as well as multiple datasets, including CIFAR10, CIFAR100, CINIC10, FMNIST, and MNIST. Empirical analysis with various SOTA approaches shows promising results. Experimental results demonstrate that the proposed approaches significantly enhance accuracy and decrease communication costs while effectively addressing statistical heterogeneity and model heterogeneity in comparison to existing state-of-the-art approaches and baselines. The experimental outcomes demonstrate significant performance gains, with DL-SH improving global model accuracy by 153% over standard FL, and I-DL-MH achieving a 225% improvement under non-IID conditions.

### 3. BRAIN PATHOGRAPH LEARNING

**主要机构**: Macquarie University, Dalian University of Technology, Federation University Australia, RMIT University
**作者数量**: 7人

**摘要**:
Brain graph learning has demonstrated significant achievements in the fields of neuroscience and artificial intelligence. However, existing methods struggle to selectively learn disease-related knowledge, leading to heavy parameters and computational costs. This challenge diminishes their efficiency, as well as limits their practicality for real-world clinical applications. To this end, we propose a lightweight Brain PathoGraph Learning (BrainPoG) model that enables efficient brain graph learning by pathological pattern filtering and pathological feature distillation. Specifically, BrainPoG first contains a filter to extract the pathological pattern formulated by highly disease-relevant subgraphs, achieving graph pruning and lesion localization. A PathoGraph is therefore constructed by dropping less disease-relevant subgraphs from the whole brain graph. Afterwards, a pathological feature distillation module is designed to reduce disease-irrelevant noise features and enhance pathological features of each node in the PathoGraph. Brain-PoG can exclusively learn informative disease-related knowledge while avoiding less relevant information, achieving efficient brain graph learning. Extensive experiments on four benchmark datasets demonstrate that BrainPoG exhibits superiority in both model performance and computational efficiency across various brain disease detection tasks.

### 4. BRIDGING DRAFT POLICY MISALIGNMENT: GROUP TREE OPTIMIZATION FOR SPECULATIVE DECODING

**主要机构**: Singapore Management University, Fudan University, National University of Singapore
**作者数量**: 4人

**摘要**:
Speculative decoding accelerates large language model (LLM) inference by letting a lightweight draft model propose multiple tokens that the target model verifies in parallel. Yet existing training objectives optimize only a single greedy draft path, while decoding follows a tree policy that re-ranks and verifies multiple branches. This draft policy misalignment limits achievable speedups. We introduce Group Tree Optimization (GTO), which aligns training with the decoding-time tree policy through two components: (i) Draft Tree Reward, a sampling-free objective equal to the expected acceptance length of the draft tree under the target model, directly measuring decoding performance; (ii) Group-based Draft Policy Training, a stable optimization scheme that contrasts trees from the current and a frozen reference draft model, forming debiased group-standardized advantages and applying a PPO-style surrogate along the longest accepted sequence for robust updates. We further prove that increasing our Draft Tree Reward provably improves acceptance length and speedup. Across dialogue (MT-Bench), code (HumanEval), and math (GSM8K), and multiple LLMs (e.g., LLaMA-3.1-8B, LLaMA-3.3-70B, Vicuna-1.3-13B, DeepSeek-R1-Distill-LLaMA-8B), GTO increases acceptance length by 7.4% and yields an additional 7.7% speedup over prior state-of-the-art EAGLE-3. By bridging draft policy misalignment, GTO offers a practical, general solution for efficient LLM inference.

### 5. COSPADI: COMPRESSING LLMS VIA CALIBRATION-GUIDED SPARSE DICTIONARY LEARNING

**主要机构**: 
**作者数量**: 6人

**摘要**:
Post-training compression of large language models (LLMs) largely relies on lowrank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose CoSpaDi (Compression via Sparse Dictionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, CoSpaDi leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse-dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. We evaluate CoSpaDi across multiple Llama and Qwen models under per-layer and per-group settings at 20-50% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment.

### 6. CubistMerge: Spatial-Preserving Token Merging For Diverse ViT Backbones

**主要机构**: University of British Columbia Mieszko Lis University of British Columbia
**作者数量**: 1人

**摘要**:
Many modern ViT backbones adopt spatial architectural designs, such as window attention, decomposed relative positional embeddings in SAM, and RoPE in DINOv3. Such architectures impose new challenges on token reduction, as the vast majority of existing methods fail to preserve the spatial structure these architectures depend on. In this paper, we introduce a simple yet effective token merging method that maintains spatial integrity, enabling seamless compatibility with spatial architectures. We reconcile two seemingly conflicting requirements: (i) exploiting the uneven information distribution across the spatial layout while (ii) preserving the spatial structure post-merging. Our approach employs (i) a 2D reduction strategy to enforce structured token layouts, (ii) a spatial-aware merging algorithm that maintains relative token positions, and (iii) a novel max-magnitude-per-dimension token representation that preserves salient features. Our method demonstrates strong performance both off-the-shelf and with fine-tuning, achieving stateof-the-art results on spatial and non-spatial architectures across various vision tasks. Specifically, we achieve 1.25× speedup on SAM-H with only 0.7% mIOU drop evaluated on COCO off-the-shelf, and 1.15× speedup on DeiT-B with no top-1 accuracy drop on ImageNet within just one epoch of fine-tuning.

### 7. Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning

**主要机构**: University of Waterloo
**作者数量**: 9人

**摘要**:
Chain-of-Thought (CoT) prompting often improves classification accuracy but it introduces a significant throughput penalty with rationale generation (Wei et al., 2022; Cheng and Van Durme, 2024). To resolve this trade-off, we introduce Dual-Head Reasoning Distillation (DHRD), a simple training method for decoder-only language models (LMs) that adds (i) a pooled classification head used during training and inference and (ii) a reasoning head supervised by teacher rationales used only in training. We train with a loss function that is a weighted sum of label cross-entropy and token-level LM loss over input-plus-rationale sequences. On seven SuperGLUE tasks, DHRD yields relative gains of 0.65-5.47% over pooled baselines, with notably larger gains on entailment/causal tasks. Since we disable the reasoning head at test time, inference throughput matches pooled classifiers and exceeds CoT decoding on the same backbones by 96-142× in QPS.

### 8. DYNAMIC EXPERTS SEARCH: ENHANCING REASONING IN MIXTURE-OF-EXPERTS LLMS AT TEST TIME

**主要机构**: Nanyang Technological University, Zhejiang University
**作者数量**: 4人

**摘要**:
Test-Time Scaling (TTS) enhances the reasoning ability of large language models (LLMs) by allocating additional computation during inference. However, existing approaches primarily rely on output-level sampling while overlooking the role of model architecture. In mainstream Mixture-of-Experts (MoE) LLMs, we observe that varying the number of activated experts yields complementary solution sets with stable accuracy, revealing a new and underexplored source of diversity. Motivated by this observation, we propose Dynamic Experts Search (DES), a TTS strategy that elevates expert activation into a controllable dimension of the search space. DES integrates two key components: (1) Dynamic MoE, which enables direct control of expert counts during inference to generate diverse reasoning trajectories without additional cost; and (2) Expert Configuration Inheritance, which preserves consistent expert counts within a reasoning path while varying them across runs, thereby balancing stability and diversity throughout the search. Extensive experiments across MoE architectures, verifiers and reasoning benchmarks (i.e., math, code and knowledge) demonstrate that DES reliably outperforms TTS baselines, enhancing accuracy and stability without additional cost. These results highlight DES as a practical and scalable form of architecture-aware TTS, illustrating how structural flexibility in modern LLMs can advance reasoning.

### 9. ELASTIC MOE: UNLOCKING THE INFERENCE-TIME SCALABILITY OF MIXTURE-OF-EXPERTS

**主要机构**: Institute of Information Engineering, Chinese Academy of Sciences, Baidu Inc
**作者数量**: 11人

**摘要**:
Mixture-of-Experts (MoE) models typically fix the number of activated experts k at both training and inference. Intuitively, activating more experts at inference k ′ (where k ′ > k) means engaging a larger set of model parameters for the computation and thus is expected to improve performance. However, contrary to this intuition, we find the scaling range to be so narrow that performance begins to degrade rapidly after only a slight increase in the number of experts. Further investigation reveals that this degradation stems from a lack of learned collaboration among experts. To address this, we introduce Elastic Mixture-of-Experts (EMoE), a novel training framework that enables MoE models to scale the number of activated experts at inference without incurring additional training overhead. By simultaneously training experts to collaborate in diverse combinations and encouraging the router for high-quality selections, EMoE ensures robust performance across computational budgets at inference. We conduct extensive experiments on various MoE settings. Our results show that EMoE significantly expands the effective performance-scaling range, extending it to as much as 2-3× the training-time k, while also pushing the model's peak performance to a higher level.

### 10. Enriching Knowledge Distillation with Intra-Class Contrastive Learning

**主要机构**: School of Computer Science and Engineering, Southeast University
**作者数量**: 4人

**摘要**:
Since the advent of knowledge distillation, much research has focused on how the soft labels generated by the teacher model can be utilized effectively. Allen-Zhu & Li (2020) points out that the implicit knowledge within soft labels originates from the multi-view structure present in the data. Feature variations within samples of the same class allow the student model to generalize better by learning diverse representations. However, in existing distillation methods, teacher models predominantly adhere to ground-truth labels as targets, without considering the diverse representations within the same class. Therefore, we propose incorporating an intra-class contrastive loss during teacher training to enrich the intra-class information contained in soft labels. In practice, we find that intra-class loss causes instability in training and slows convergence. To mitigate these issues, margin loss is integrated into intra-class contrastive learning to improve the training stability and convergence speed. Simultaneously, we theoretically analyze the impact of this loss on the intra-class distances and inter-class distances. It has been proved that the intra-class contrastive loss can enrich the intra-class diversity. Experimental results demonstrate the effectiveness of the proposed method.

### 11. FASTGRPO: ACCELERATING POLICY OPTIMIZATION VIA CONCURRENCY-AWARE SPECULATIVE DECODING AND ONLINE DRAFT LEARNING

**主要机构**: Lanzhou University, The University of Hong
**作者数量**: 4人

**摘要**:
Group relative policy optimization (GRPO) has demonstrated significant potential in improving the reasoning capabilities of large language models (LLMs) via reinforcement learning. However, its practical deployment is impeded by an excessively slow training process, primarily attributed to the computationally intensive autoregressive generation of multiple responses per query, which makes the generation phase the primary performance bottleneck. Although speculative decoding presents a promising direction for acceleration, its direct application in GRPO achieves limited speedup under high-concurrency training conditions. To overcome this limitation, we propose a concurrency-aware speculative decoding framework that dynamically adjusts the drafting and verification strategy according to real-time concurrency levels, thereby maximizing the acceleration of the generation process. Furthermore, to address performance degradation arising from distributional drift between the evolving target model and the fixed draft model during training, we introduce an online draft learning mechanism that enables the draft model to continuously adapt using feedback signals from the target model. Experimental results across multiple mathematical reasoning datasets and models demonstrate that the proposed method achieves end-to-end speedups of 2.35x to 2.72x, significantly surpassing baseline approaches in efficiency. The code is available at https://github.com/yedaotian9/GRPO speculative.

### 12. From Long to Lean: Performance-aware and Adaptive Chain-of-Thought Compression via Multi-round Refinement

**主要机构**: Pengcheng Laboratory, Harbin Institute of Technology
**作者数量**: 7人

**摘要**:
Chain-of-Thought (CoT) reasoning improves performance on complex tasks but introduces significant inference latency due to its verbosity. In this work, we propose Multiround Adaptive Chain-of-Thought Compression (MACC), a framework that leverages the token elasticity phenomenon-where overly small token budgets may paradoxically increase output length-to progressively compress CoTs via multiround refinement. This adaptive strategy allows MACC to dynamically determine the optimal compression depth for each input. Our method achieves an average accuracy improvement of 5.6% over state-of-the-art baselines, while also reducing CoT length by an average of 47 tokens and significantly lowering latency. Furthermore, we show that test-time performance-accuracy and token length-can be reliably predicted using interpretable features like perplexity and compression rate on training set. Evaluated across different models, our method enables efficient model selection and forecasting without repeated fine-tuning, demonstrating that CoT compression is both effective and predictable. Our code will be released in https://github.com/Leon221220/ MACC.

### 13. HEAPR: HESSIAN-BASED EFFICIENT ATOMIC EXPERT PRUNING IN OUTPUT SPACE

**主要机构**: FABU Inc, School of Software Technology, Zhejiang University, DiDi Global Inc. 4 Geely Automobile Research Institute
**作者数量**: 6人

**摘要**:
Mixture-of-Experts (MoE) architectures in large language models (LLMs) deliver exceptional performance and reduced inference costs compared to dense LLMs. However, their large parameter counts result in prohibitive memory requirements, limiting practical deployment. While existing pruning methods primarily focus on expert-level pruning, this coarse granularity often leads to substantial accuracy degradation. In this work, we introduce HEAPr, a novel pruning algorithm that decomposes experts into smaller, indivisible atomic experts, enabling more precise and flexible atomic expert pruning. To measure the importance of each atomic expert, we leverage second-order information based on principles similar to Optimal Brain Surgeon (OBS) theory. To address the computational and storage challenges posed by second-order information, HEAPr exploits the inherent properties of atomic experts to transform the second-order information from expert parameters into that of atomic expert parameters, and further simplifies it to the second-order information of atomic expert outputs. This approach reduces the space complexity from O(d 4), where d is the model's dimensionality, to O(d 2). HEAPr requires only two forward passes and one backward pass on a small calibration set to compute the importance of atomic experts. Extensive experiments on MoE models, including DeepSeek MoE and Qwen MoE family, demonstrate that HEAPr outperforms existing expert-level pruning methods across a wide range of compression ratios and benchmarks. Specifically, HEAPr achieves nearly lossless compression at compression ratios of 20% ∼ 25% in most models, while also reducing FLOPs nearly by 20%. The code can be found at https://github.com/LLIKKE/HEAPr.

### 14. HierLight-YOLO: A Hierarchical and Lightweight Object Detection Network for UAV Photography

**主要机构**: Shenzhen University, School of Mathematical Sciences
**作者数量**: 3人

**摘要**:
The real-time detection of small objects in complex scenes, such as the unmanned aerial vehicle (UAV) photography captured by drones, has dual challenges of detecting small targets (<32×32 pixels) and maintaining real-time efficiency on resource-constrained platforms. While YOLO-series detectors have achieved remarkable success in real-time large object detection, they suffer from significantly higher false negative rates for drone-based detection where small objects dominate, compared to large object scenarios. This paper proposes HierLight-YOLO, a hierarchical feature fusion and lightweight model that enhances the real-time detection of small objects, based on the YOLOv8 architecture. We propose the Hierarchical Extended Path Aggregation Network (HEPAN), a multi-scale feature fusion method through hierarchical cross-level connections, enhancing the small object detection accuracy. HierLight-YOLO includes two innovative lightweight modules: Inverted Residual Depthwise Convolution Block (IRDCB) and Lightweight Downsample (LDown) module, which significantly reduce the model's parameters and computational complexity without sacrificing detection capabilities. Small object detection head is designed to further enhance spatial resolution and feature fusion to tackle the tiny object (4×4 pixels) detection. Comparison experiments and ablation studies on the VisDrone2019 benchmark demonstrate state-of-the-art performance of HierLight-YOLO: the nano-scale HierLight-YOLO-N maintains strong detection capability (35.8% AP 0.5) at just 2.2M parameters (26.7% fewer than YOLOv8-N), proving its suitability for edge deployment; the small-scale HierLight-YOLO-S achieves the state-of-the-art accuracy 44.9% AP 0.5 among S-scale models with only 7.8M parameters (29.7% fewer than YOLOv8-S); the medium-scale variant HierLight-YOLO-M outperforms other M-scale models by 50.2% AP 0.5 (surpassing YOLOv8-M by 5.6%), with competitive 17.9M parameters. Ablation studies validate each component's contribution, with HEPAN improving AP 0.5 by 0.8% over conventional feature pyramids and IRDCB reducing parameters by 22.1% without accuracy loss.

### 15. IIET: Efficient Numerical Transformer via Implicit Iterative Euler Method

**主要机构**: School of Computer Science and Engineering, Northeastern University, Meituan Inc, Tsinghua University, NLP Lab
**作者数量**: 9人

**摘要**:
High-order numerical methods enhance Transformer performance in tasks like NLP and CV, but introduce a performance-efficiency trade-off due to increased computational overhead. Our analysis reveals that conventional efficiency techniques, such as distillation, can be detrimental to the performance of these models, exemplified by PCformer. To explore more optimizable ODE-based Transformer architectures, we propose the Iterative Implicit Euler Transformer (IIET), which simplifies highorder methods using an iterative implicit Euler approach. This simplification not only leads to superior performance but also facilitates model compression compared to PCformer. To enhance inference efficiency, we introduce Iteration Influence-Aware Distillation (IIAD). Through a flexible threshold, IIAD allows users to effectively balance the performanceefficiency trade-off. On lm-evaluation-harness, IIET boosts average accuracy by 2.65% over vanilla Transformers and 0.8% over PCformer. Its efficient variant, E-IIET, significantly cuts inference overhead by 55% while retaining 99.4% of the original task accuracy. Moreover, the most efficient IIET variant achieves an average performance gain exceeding 1.6% over vanilla Transformer with comparable speed.

### 16. IN THEIR OWN WORDS: REASONING TRACES TAI-LORED FOR SMALL MODELS MAKE THEM BETTER REASONERS

**主要机构**: Yonsei University
**作者数量**: 3人

**摘要**:
Transferring reasoning capabilities from larger language models to smaller ones through supervised fine-tuning often fails counterintuitively, with performance degrading despite access to high-quality teacher demonstrations. We identify that this failure stems from distributional misalignment: reasoning traces from larger models contain tokens that are low probability under the student's distribution, exceeding the internal representation capacity of smaller architectures and creating learning barriers rather than helpful guidance. We propose Reverse Speculative Decoding (RSD), a mechanism for generating student-friendly reasoning traces in which the teacher model proposes candidate tokens but the student model determines acceptance based on its own probability distributions, filtering low probability tokens. When applied to Qwen3-0.6B, direct distillation of s1K-1.1 reasoning trace data degrades average performance across major reasoning benchmarks by 20.5%, while the same model trained on RSDgenerated reasoning traces achieves meaningful improvements of 4.9%. Our analysis reveals that low probability tokens constitute the critical bottleneck in reasoning ability transfer. However, cross-model experiments demonstrate that RSD traces are model-specific rather than universally applicable, indicating that distributional alignment must be tailored for each student architecture's unique internal representation. Code, datasets, and models are available at https://github.com/jaeh8nkim/equigranular.

### 17. Joint graph entropy knowledge distillation for point cloud classification and robustness against corruptions

**主要机构**: School of Computer Science and Technology, School of Information Science and Engineering, Wuhan University of Science and Technology
**作者数量**: 4人

**摘要**:
Classification tasks in 3D point clouds often assume that class events are independent and identically distributed (IID), although this assumption destroys the correlation between classes. This study proposes a classification strategy, Joint Graph Entropy Knowledge Distillation (JGEKD), suitable for non-independent and identically distributed 3D point cloud data, which achieves knowledge transfer of class correlations through knowledge distillation by constructing a loss function based on joint graph entropy. First, we employ joint graphs to capture addthe hidden relationships between classes and implement knowledge distillation to train our model by calculating the entropy of addadd graph. Subsequently, to handle 3D point clouds invariant to spatial transformations, we construct Siamese structures and develop two frameworks, self-knowledge distillation and teacher-knowledge distillation, to facilitate information transfer between different transformation forms of the same data. In addition, we use the above framework to achieve knowledge transfer between point clouds and their corrupted forms, and increase the robustness against corruption of model. Extensive experiments on ScanObject, ModelNet40, ScanntV2_cls and ModelNet-C demonstrate that the proposed strategy can achieve competitive results.

### 18. KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cache

**主要机构**: University of Toronto
**作者数量**: 2人

**摘要**:
Vision-Language-Action (VLA) models promise unified robotic perception and control, yet their scalability is constrained by the quadratic cost of attention and the unbounded growth of key-value (KV) memory during long-horizon inference. While recent methods improve generalization through scaling backbone architectures, they often neglect the inference inefficiencies critical to real-time deployment. In this work, We present KV-Efficient VLA, a model-agnostic memory compression framework that addresses these limitations by introducing a lightweight, training-friendly mechanism to selectively retain high-utility context. Our method partitions the KV cache into fixed-size chunks and employs a recurrent gating module to summarize and filter historical context according to learned utility scores. This design preserves recent fine-grained detail while aggressively pruning stale, low-relevance memory, all while maintaining causality. Theoretically, KV-Efficient VLA yields up to 1.21× inference speedup and 36% KV memory reduction, with minimal impact on task success. Our method integrates seamlessly into existing autoregressive and hybrid VLA stacks, enabling scalable inference without modifying training pipelines or downstream control logic.

### 19. Large AI Model-Enabled Generative Semantic Communications for Image Transmission

**主要机构**: 
**作者数量**: 3人

**摘要**:
The rapid development of generative artificial intelligence (AI) has introduced significant opportunities for enhancing the efficiency and accuracy of image transmission within semantic communication systems. Despite these advancements, existing methodologies often neglect the difference in importance of different regions of the image, potentially compromising the reconstruction quality of visually critical content. To address this issue, we introduce an innovative generative semantic communication system that refines semantic granularity by segmenting images into key and non-key regions. Key regions, which contain essential visual information, are processed using an image oriented semantic encoder, while non-key regions are efficiently compressed through an image-to-text modeling approach. Additionally, to mitigate the substantial storage and computational demands posed by large AI models, the proposed system employs a lightweight deployment strategy incorporating model quantization and low-rank adaptation fine-tuning techniques, significantly boosting resource utilization without sacrificing performance. Simulation results demonstrate that the proposed system outperforms traditional methods in terms of both semantic fidelity and visual quality, thereby affirming its effectiveness for image transmission tasks.

### 20. LFA-Net: A Lightweight Network with LiteFusion Attention for Retinal Vessel Segmentation

**主要机构**: School of EEECS Queen's University Belfast
**作者数量**: 6人

**摘要**:
Lightweight retinal vessel segmentation is important for the early diagnosis of vision-threatening and systemic diseases, especially in a real-world clinical environment with limited computational resources. Although segmentation methods based on deep learning are improving, existing models are still facing challenges of small vessel segmentation and high computational costs. To address these challenges, we proposed a new vascular segmentation network, LFA-Net, which incorporates a newly designed attention module, LiteFusion-Attention. This attention module incorporates residual learning connections, Vision Mamba-inspired dynamics, and modulation-based attention, enabling the model to capture local and global context efficiently and in a lightweight manner. LFA-Net offers high performance with 0.11 million parameters, 0.42 MB memory size, and 4.46 GFLOPs, which make it ideal for resource-constrained environments. We validated our proposed model on DRIVE, STARE, and CHASE DB with outstanding performance in terms of dice scores of 83.28, 87.44, and 84.50% and jaccard indices of 72.85, 79.31, and 74.70%, respectively. The code of LFA-Net is available online https://github.com/Mehwish4593/LFA-Net.

### 21. LIGHTWEIGHT ERROR MITIGATION STRATE-GIES FOR POST-TRAINING N:M ACTIVATION SPARSITY IN LLMS

**主要机构**: 
**作者数量**: 10人

**摘要**:
The demand for efficient large language model (LLM) inference has intensified the focus on sparsification techniques. While semi-structured (N:M) pruning is well-established for weights, its application to activation pruning remains underexplored despite its potential for dynamic, input-adaptive compression and reductions in I/O overhead. This work presents a comprehensive analysis of methods for post-training N:M activation pruning in LLMs. Across multiple LLMs, we demonstrate that pruning activations enables superior preservation of generative capabilities compared to weight pruning at equivalent sparsity levels. We evaluate lightweight, plug-and-play error mitigation techniques and pruning criteria, establishing strong hardware-friendly baselines that require minimal calibration. Furthermore, we explore sparsity patterns beyond NVIDIA's standard 2:4, showing that the 16:32 pattern achieves performance nearly on par with unstructured sparsity. However, considering the trade-off between flexibility and hardware implementation complexity, we focus on the 8:16 pattern as a superior candidate. Our findings provide both effective practical methods for activation pruning and a motivation for future hardware to support more flexible sparsity patterns. Our code is available here.

### 22. LONGLIVE: REAL-TIME INTERACTIVE LONG VIDEO GENERATION

**主要机构**: 
**作者数量**: 15人

**摘要**:
We present LONGLIVE, a frame-level autoregressive (AR) framework for realtime and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference, but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LONGLIVE adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with new prompts for smooth, adherent switches; streaming long tuning to enable long video training and to align training and inference (train-long-test-long); and short window attention paired with a frame-level attention sink, shorten as frame sink, preserving long-range consistency while enabling faster generation. With these key designs, LONGLIVE fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LONGLIVE sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short and long videos. LONGLIVE supports up to 240-second videos on a single H100 GPU. LONGLIVE further supports INT8-quantized inference with only marginal quality loss. Code, Model, and Demo Page are available at https://github.com/NVlabs/LongLive.

### 23. MS-YOLO: Infrared Object Detection for Edge Deployment via MobileNetV4 and SlideLoss

**主要机构**: Kummer Institute Center for Artificial Intelligence and Autonomous Systems, Dept. of Mathematics & Statistics, School of Electrical & Computer Engineering, Dept. of Computer Science, Missouri University of Science and Technology, University of Oklahoma
**作者数量**: 6人

**摘要**:
Infrared imaging has emerged as a robust solution for urban object detection under low-light and adverse weather conditions, offering significant advantages over traditional visible-light cameras. However, challenges such as class imbalance, thermal noise, and computational constraints can significantly hinder model performance in practical settings. To address these issues, we evaluate multiple YOLO variants on the FLIR ADAS V2 dataset, ultimately selecting YOLOv8 as our baseline due to its balanced accuracy and efficiency. Building on this foundation, we present MS-YOLO (MobileNetv4 and SlideLoss based on YOLO), which replaces YOLOv8's CSPDarknet backbone with the more efficient MobileNetV4, reducing computational overhead by 1.5% while sustaining high accuracy. In addition, we introduce SlideLoss, a novel loss function that dynamically emphasizes under-represented and occluded samples, boosting precision without sacrificing recall. Experiments on the FLIR ADAS V2 benchmark show that MS-YOLO attains competitive mAP and superior precision while operating at only 6.7 GFLOPs. These results demonstrate that MS-YOLO effectively addresses the dual challenge of maintaining high detection quality while minimizing computational costs, making it well-suited for real-time edge deployment in urban environments.

### 24. OJAKV: CONTEXT-AWARE ONLINE LOW-RANK KV CACHE COMPRESSION WITH OJA'S RULE

**主要机构**: Department of Computer Science, IBM Research Yorktown Heights, Rensselaer Polytechnic Institute Troy
**作者数量**: 8人

**摘要**:
The expanding long-context capabilities of large language models are constrained by a significant memory bottleneck: the key-value (KV) cache required for autoregressive generation. This bottleneck is substantial; for instance, a Llama-3.1-8B model processing a 32K-token prompt at a batch size of 4 requires approximately 16 GB for its KV cache, a size exceeding the model's weights. While KV-cache compression via low-rank projection is a promising direction, existing methods' rely on a static, offline-learned subspace that performs poorly under data distribution shifts. To overcome these limitations, we introduce OjaKV, a novel framework that integrates a strategic hybrid storage policy with online subspace adaptation. First, OjaKV recognizes that not all tokens are equally important for compression; it preserves the crucial first and most recent tokens in full-rank, maintaining high-fidelity anchors for attention. Second, for the vast majority of intermediate tokens, it applies low-rank compression by incrementally adapting the projection basis using Oja's algorithm for online principal component analysis. This adaptation involves a comprehensive update during prompt prefilling and lightweight periodic updates during decoding, ensuring the subspace remains aligned with the evolving context. Crucially, our framework is fully compatible with modern attention modules like FlashAttention. Experiments demonstrate that OjaKV maintains or even improves zero-shot accuracy at high compression ratios. In particular, OjaKV achieves its strongest gains on very long-context benchmarks that require complex reasoning, highlighting the importance of online subspace adaptation in dynamically tracking context shifts. Furthermore, our approach is compatible with token-selection methods, enabling compounded memory savings. These results establish our hybrid framework as a practical, plug-and-play solution for memory-efficient long-context inference without requiring model fine-tuning.

### 25. Progressive Weight Loading: Accelerating Initial Inference and Gradually Boosting Performance on Resource-Constrained Environments

**主要机构**: Yonsei University, Intel
**作者数量**: 5人

**摘要**:
Deep learning models have become increasingly large and complex, resulting in higher memory consumption and computational demands. Consequently, model loading times and initial inference latency have increased, posing significant challenges in mobile and latencysensitive environments where frequent model loading and unloading are required, which directly impacts user experience. While Knowledge Distillation (KD) offers a solution by compressing large teacher models into smaller student ones, it often comes at the cost of reduced performance. To address this trade-off, we propose Progressive Weight Loading (PWL), a novel technique that enables fast initial inference by first deploying a lightweight student model, then incrementally replacing its layers with those of a pre-trained teacher model. To support seamless layer substitution, we introduce a training method that not only aligns intermediate feature representations between student and teacher layers, but also improves the overall output performance of the student model. Our experiments on VGG, ResNet, and ViT architectures demonstrate that models trained with PWL maintain competitive distillation performance and gradually improve accuracy as teacher layers are loaded-matching the final accuracy of the full teacher model without compromising initial inference speed. This makes PWL particularly suited for dynamic, resource-constrained deployments where both responsiveness and performance are critical.

### 26. Pushing Toward the Simplex Vertices: A Simple Remedy for Code Collapse in Smoothed Vector Quantization

**主要机构**: Chubu University, Academy of Emerging Sciences
**作者数量**: 1人

**摘要**:
Vector quantization, which discretizes a continuous vector space into a finite set of representative vectors (a codebook), has been widely adopted in modern machine learning. Despite its effectiveness, vector quantization poses a fundamental challenge: the non-differentiable quantization step blocks gradient backpropagation. Smoothed vector quantization addresses this issue by relaxing the hard assignment of a codebook vector into a weighted combination of codebook entries, represented as the matrix product of a simplex vector and the codebook. Effective smoothing requires two properties: (1) smoothed quantizers should remain close to a onehot vector, ensuring tight approximation, and (2) all codebook entries should be utilized, preventing code collapse. Existing methods typically address these desiderata separately. By contrast, the present study introduces a simple and intuitive regularization that promotes both simultaneously by minimizing the distance between each simplex vertex and its 𝐾-nearest smoothed quantizers. Experiments on representative benchmarks-including discrete image autoencoding and contrastive speech representation learning-demonstrate that the proposed method achieves more reliable codebook utilization and improves performance compared to prior approaches.

### 27. RAPID 3 : TRI-LEVEL REINFORCED ACCELERATION POLICIES FOR DIFFUSION TRANSFORMER

**主要机构**: The University of Texas at Austin, Zhejiang University, ZIP Lab, National University of Singapore
**作者数量**: 13人

**摘要**:
Diffusion Transformers (DiTs) excel at visual generation yet remain hampered by slow sampling. Existing training-free accelerators-step reduction, feature caching, and sparse attention-enhance inference speed but typically rely on a uniform heuristic or manually designed adaptive strategy for all images, leaving quality on the table. Alternatively, dynamic neural networks offer per-image adaptive acceleration, but their high fine-tuning costs limit broader applicability. To address these limitations, we introduce RAPID 3 : Tri-Level Reinforced Acceleration PolIcies for Diffusion Transformer framework that delivers image-wise acceleration with zero updates to the base generator. Specifically, three lightweight policy heads-Step-Skip, Cache-Reuse, and Sparse-Attention-observe the current denoising state and independently decide their corresponding speed-up at each timestep. All policy parameters are trained online via Group Relative Policy Optimization (GRPO) while the generator remains frozen. Meanwhile, an adversarially learned discriminator augments the reward signal, discouraging reward hacking by boosting returns only when generated samples stay close to the original model's distribution. Across state-of-the-art DiT backbones including Stable Diffusion 3 and FLUX, RAPID 3 achieves nearly 3× faster sampling with competitive generation quality.

### 28. REFINE-CONTROL: A SEMI-SUPERVISED DISTILLATION METHOD FOR CONDITIONAL IMAGE GENERATION

**主要机构**: School of Computer Science and Engineering, Southeast University, AI Lab, Lenovo Research
**作者数量**: 5人

**摘要**:
Conditional image generation models have achieved remarkable results by leveraging text-based control to generate customized images. However, the high resource demands of these models and the scarcity of well-annotated data have hindered their deployment on edge devices, leading to enormous costs and privacy concerns, especially when user data is sent to a third party. To overcome these challenges, we propose Refine-Control, a semi-supervised distillation framework. Specifically, we improve the performance of the student model by introducing a tri-level knowledge fusion loss to transfer different levels of knowledge. To enhance generalization and alleviate dataset scarcity, we introduce a semi-supervised distillation method utilizing both labeled and unlabeled data. Our experiments reveal that Refine-Control achieves significant reductions in computational cost and latency, while maintaining high-fidelity generation capabilities and controllability, as quantified by comparative metrics.

### 29. RESIDUAL VECTOR QUANTIZATION FOR COMMUNICATION-EFFICIENT MULTI-AGENT PERCEPTION

**主要机构**: Carnegie Mellon University
**作者数量**: 2人

**摘要**:
Multi-agent collaborative perception (CP) improves scene understanding by sharing information across connected agents such as autonomous vehicles, unmanned aerial vehicles, and robots. Communication bandwidth, however, constrains scalability. We present ReVQom, a learned feature codec that preserves spatial identity while compressing intermediate features. ReVQom is an end-to-end method that compresses feature dimensions via a simple bottleneck network followed by multi-stage residual vector quantization (RVQ). This allows only per-pixel code indices to be transmitted, reducing payloads from 8192 bits per pixel (bpp) of uncompressed 32-bit float features to 6-30 bpp per agent with minimal accuracy loss. On DAIR-V2X real-world CP dataset, ReVQom achieves 273× compression at 30 bpp to 1365× compression at 6 bpp. At 18 bpp (455×), ReVQom matches or outperforms raw-feature CP, and at 6-12 bpp it enables ultra-low-bandwidth operation with graceful degradation. ReVQom allows efficient and accurate multi-agent collaborative perception with a step toward practical V2X deployment.

### 30. Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts RETRIEVAL-OF-THOUGHT: EFFICIENT REASONING VIA REUSING THOUGHTS

**主要机构**: University of California, Argonne National Laboratory, University of Minnesota
**作者数量**: 7人

**摘要**:
Large reasoning models improve accuracy by producing long reasoning traces, but this inflates latency and cost, motivating inference-time efficiency. We propose Retrieval-of-Thought (RoT), which reuses prior reasoning as composable "thought" steps to guide new problems. RoT organizes steps into a thought graph with sequential and semantic edges to enable fast retrieval and flexible recombination. At inference, RoT retrieves query-relevant nodes and applies rewardguided traversal to assemble a problem-specific template that guides generation. This dynamic template reuse reduces redundant exploration and, therefore, reduces output tokens while preserving accuracy. We evaluate RoT on reasoning benchmarks with multiple models, measuring accuracy, token usage, latency, and memory overhead. Findings show small prompt growth but substantial efficiency gains, with RoT reducing output tokens by up to 40%, inference latency by 82%, and cost by 59% while maintaining accuracy. RoT establishes a scalable paradigm for efficient LRM reasoning via dynamic template construction through retrieval.

### 31. SCALE-WISE VAR IS SECRETLY DISCRETE DIFFUSION

**主要机构**: Inpainting OURS VAR Outpainting Super-Resolution Image Generation Zero-Shot Image Reconstruction, Johns Hopkins University
**作者数量**: 3人

**摘要**:
Autoregressive (AR) transformers have emerged as a powerful paradigm for visual generation, largely due to their scalability, computational efficiency and unified architecture with language and vision. Among them, next scale prediction Visual Autoregressive Generation (VAR) has recently demonstrated remarkable performance, even surpassing diffusion-based models. In this work, we revisit VAR and uncover a theoretical insight: when equipped with a Markovian attention mask, VAR is mathematically equivalent to a discrete diffusion. We term this reinterpretation as Scalable Visual Refinement with Discrete Diffusion (SRDD), establishing a principled bridge between AR transformers and diffusion models. Leveraging this new perspective, we show how one can directly import the advantages of diffusion-such as iterative refinement and reduce architectural inefficiencies into VAR, yielding faster convergence, lower inference cost, and improved zero-shot reconstruction. Across multiple datasets, we show that the diffusion-based perspective of VAR leads to consistent gains in efficiency and generation.

### 32. Self-Speculative Biased Decoding for Faster Live Translation

**主要机构**: 
**作者数量**: 4人

**摘要**:
Large Language Models (LLMs) have recently demonstrated impressive capabilities in various text generation tasks. However, it remains challenging to use them off-the-shelf in streaming applications (such as live translation), where the output must continually update as the input context expands, while still maintaining a reasonable computational cost to meet the latency requirement. In this work, we reexamine the re-translation approach to simultaneous translation and propose Self-Speculative Biased Decoding, a novel inference paradigm designed to avoid repeatedly generating output from scratch for a consistently growing input stream. We propose using the most recent output as a draft for the current growing input context. During the verification stage, the output will be biased towards the draft token for a higher draft acceptance rate. This strategy not only minimizes flickering that might distract users but also leads to higher speedups. Conventional decoding may take charge from the point of divergence after draft verification and continue until the end condition is met. Unlike existing speculative decoding strategies, our approach eliminates the need for draft computations, making it a model-agnostic and plug-and-play solution for accelerating latency-sensitive streaming applications. Experimental results on simultaneous text-to-text re-translation demonstrate that our approach achieves up to 1.7x speedup compared to conventional auto-regressive re-translation without compromising quality. Additionally, it significantly reduces flickering by 80% by incorporating the display-only mask-k technique.

### 33. SGNNBench: A Holistic Evaluation of Spiking Graph Neural Network on Large-scale Graph

**主要机构**: Central South University Changsha, Sun Yat-sen University Guangzhou
**作者数量**: 10人

**摘要**:
Graph Neural Networks (GNNs) are exemplary deep models designed for graph data. Message passing mechanism enables GNNs to effectively capture graph topology and push the performance boundaries across various graph tasks. However, the trend of developing such complex machinery for graph representation learning has become unsustainable on large-scale graphs. The computational and time overhead make it imperative to develop more energyefficient GNNs to cope with the explosive growth of real-world graphs. Spiking Graph Neural Networks (SGNNs), which integrate biologically plausible learning via unique spike-based neurons, have emerged as a promising energy-efficient alternative. Different layers communicate with sparse and binary spikes, which facilitates computation and storage of intermediate graph representations. Despite the proliferation of SGNNs proposed in recent years, there is no systematic benchmark to explore the basic design principles of these brain-inspired networks on the graph data. To bridge this gap, we present SGNNBench to quantify progress in the field of SGNNs. Specifically, SGNNBench conducts an in-depth investigation of SGNNs from multiple perspectives, including effectiveness, energy efficiency, and architectural design. We comprehensively evaluate 9 state-of-the-art SGNNs across 18 datasets. Regarding efficiency, we empirically compare these baselines w.r.t model size, memory usage, and theoretical energy consumption to reveal the often-overlooked energy bottlenecks of SGNNs. Besides, we elaborately investigate the design space of SGNNs to promote the development of a general SGNN paradigm. 1

### 34. SIMULSENSE: SENSE-DRIVEN INTERPRETING FOR EFFICIENT SIMULTANEOUS SPEECH TRANSLATION

**主要机构**: Nara Institute of Science and Technology
**作者数量**: 3人

**摘要**:
How to make human-interpreter-like read/write decisions for simultaneous speech translation (SimulST) systems? Current state-of-theart systems formulate SimulST as a multi-turn dialogue task, requiring specialized interleaved training data and relying on computationally expensive large language model (LLM) inference for decisionmaking. In this paper, we propose SimulSense, a novel framework for SimulST that mimics human interpreters by continuously reading input speech and triggering write decisions to produce translation when a new sense unit is perceived. Experiments against two stateof-the-art baseline systems demonstrate that our proposed method achieves a superior quality-latency tradeoff and substantially improved real-time efficiency, where its decision-making is up to 9.6× faster than the baselines.

### 35. SlimDiff: Training-Free, Activation-Guided Hands-free Slimming of Diffusion Models

**主要机构**: Purdue University
**作者数量**: 3人

**摘要**:
Diffusion models (DMs), lauded for their generative performance, are computationally prohibitive due to their billion-scale parameters and iterative denoising dynamics. Existing efficiency techniques, such as quantization, timestep reduction, or pruning, offer savings in compute, memory, or runtime but are strictly bottle-necked by reliance on fine-tuning or retraining to recover performance. In this work, we introduce SlimDiff, an automated activation-informed structural compression framework that reduces both attention and feedforward dimensionalities in DMs, while being entirely gradient-free. SlimDiff reframes DM compression as a spectral approximation task, where activation covariances across denoising timesteps define low-rank subspaces that guide dynamic pruning under a fixed compression budget. This activation-aware formulation mitigates error accumulation across timesteps by applying module-wise decompositions over functional weight groups: query-key interactions, value-output couplings, and feedforward projections-rather than isolated matrix factorizations, while adaptively allocating sparsity across modules to respect the non-uniform geometry of diffusion trajectories. SlimDiff achieves up to 35% acceleration and ∼ 100M parameter reduction over baselines, with generation quality on par with uncompressed models without any backpropagation. Crucially, our approach requires only about 500 calibration samples, over 70× fewer than prior methods. To our knowledge, this is the first closed-form, activation-guided structural compression of DMs that is entirely training-free, providing both theoretical clarity and practical efficiency.

### 36. Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models

**主要机构**: Department of Computer Science, ETH Zürich
**作者数量**: 4人

**摘要**:
Modern state-space models (SSMs) often utilize transition matrices which enable efficient computation but pose restrictions on the model's expressivity, as measured in terms of the ability to emulate finite-state automata (FSA). While unstructured transition matrices are optimal in terms of expressivity, they come at a prohibitively high compute and memory cost even for moderate state sizes. We propose a structured sparse parametrization of transition matrices in SSMs that enables FSA state tracking with optimal state size and depth, while keeping the computational cost of the recurrence comparable to that of diagonal SSMs. Our method, PD-SSM, parametrizes the transition matrix as the product of a column one-hot matrix (P) and a complex-valued diagonal matrix (D). Consequently, the computational cost of parallel scans scales linearly with the state size. Theoretically, the model is BIBO-stable and can emulate any N-state FSA with one layer of dimension N and a linear readout of size N × N , significantly improving on all current structured SSM guarantees. Experimentally, the model significantly outperforms a wide collection of modern SSM variants on various FSA state tracking tasks. On multiclass time-series classification, the performance is comparable to that of neural controlled differential equations, a paradigm explicitly built for time-series analysis. Finally, we integrate PD-SSM into a hybrid Transformer-SSM architecture and demonstrate that the model can effectively track the states of a complex FSA in which transitions are encoded as a set of variable-length English sentences. The code is available at https://github.com/IBM/expressive-sparse-state-space-model * Equal contribution. 39th Conference on Neural Information Processing Systems (NeurIPS 2025).

### 37. Towards Adapting Federated & Quantum Machine Learning for Network Intrusion Detection: A Survey

**主要机构**: 
**作者数量**: 3人

**摘要**:
This survey explores the integration of Federated Learning (FL) with Network Intrusion Detection Systems (NIDS), with particular emphasis on deep learning and quantum machine learning approaches. FL enables collaborative model training across distributed devices while preserving data privacy-a critical requirement in network security contexts where sensitive traffic data cannot be centralized. Our comprehensive analysis systematically examines the full spectrum of FL architectures, deployment strategies, communication protocols, and aggregation methods specifically tailored for intrusion detection. We provide an in-depth investigation of privacy-preserving techniques, model compression approaches, and attack-specific federated solutions for threats including DDoS, MITM, and botnet attacks. The survey further delivers a pioneering exploration of Quantum FL (QFL), discussing quantum feature encoding, quantum machine learning algorithms, and quantum-specific aggregation methods that promise exponential speedups for complex pattern recognition in network traffic. Through rigorous comparative analysis of classical and quantum approaches, identification of research gaps, and evaluation of real-world deployments, we outline a concrete roadmap for industrial adoption and future research directions. This work serves as an authoritative reference for researchers and practitioners seeking to enhance privacy, efficiency, and robustness of federated intrusion detection systems in increasingly complex network environments, while preparing for the quantum-enhanced cybersecurity landscape of tomorrow.

### 38. Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting

**主要机构**: Equal advising Department of Electrical Engineering, Stanford University
**作者数量**: 4人

**摘要**:
Modern vision-language pipelines are driven by RGB vision encoders trained on massive image-text corpora. While these pipelines have enabled impressive zero-shot capabilities and strong transfer across tasks, they still inherit two structural inefficiencies from the pixel domain: (i) transmitting dense RGB images from edge devices to the cloud is energy-intensive and costly, and (ii) patchbased tokenization explodes sequence length, stressing attention budgets and context limits. We explore 2D Gaussian Splatting (2DGS) as an alternative visual substrate for alignment: a compact, spatially adaptive representation that parameterizes images by a set of colored anisotropic Gaussians. We develop a scalable 2DGS pipeline with structured initialization, luminance-aware pruning, and batched CUDA kernels, achieving over 90× faster fitting and ∼ 97% GPU utilization compared to prior implementations. We further adapt contrastive language-image pretraining (CLIP) to 2DGS by reusing a frozen RGB-based transformer backbone with a lightweight splat-aware input stem and a perceiver resampler, training only ∼ 7% of the total parameters. On large DataComp subsets, GS encoders yield meaningful zero-shot ImageNet-1K performance while compressing inputs 3-20× relative to pixels. While accuracy currently trails RGB encoders, our results establish 2DGS as a viable multimodal substrate, pinpoint architectural bottlenecks, and open a path toward representations that are both semantically powerful and transmission-efficient for edge-cloud learning.

### 39. γ-Quant: Towards Learnable Quantization for Low-bit Pattern Recognition

**主要机构**: University of Siegen, University of Mannheim
**作者数量**: 7人

**摘要**:
Most pattern recognition models are developed on pre-processed data. In computer vision, for instance, RGB images processed through image signal processing (ISP) pipelines designed to cater to human perception are the most frequent input to image analysis networks. However, many modern vision tasks operate without a human in the loop, raising the question of whether such pre-processing is optimal for automated analysis. Similarly, human activity recognition (HAR) on body-worn sensor data commonly takes normalized floating-point data arising from a high-bit analog-to-digital converter (ADC) as an input, despite such an approach being highly inefficient in terms of data transmission, significantly affecting the battery life of wearable devices. In this work, we target low-bandwidth and energy-constrained settings where sensors are limited to low-bit-depth capture. We propose γ-Quant, i.e. the taskspecific learning of a non-linear quantization for pattern recognition. We exemplify our approach on raw-image object detection as well as HAR of wearable data, and demonstrate that raw data with a learnable quantization using as few as 4-bits can perform on par with the use of raw 12-bit data. All code to reproduce our experiments is publicly available via github.com/Mishalfatima/Gamma-Quant.
