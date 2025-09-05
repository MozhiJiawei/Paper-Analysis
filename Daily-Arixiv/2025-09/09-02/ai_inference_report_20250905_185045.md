# AI推理加速技术论文分析报告
生成时间: 2025-09-05 18:50:45
分析论文数量: 63篇

## 论文技术简报

### 1. A-FloPS: Accelerating Diffusion Sampling with Adaptive Flow Path Sampler

清华大学发布了A-FloPS论文，使用A-FloPS（自适应流路径采样器）通过将预训练扩散模型采样轨迹重参数化为流匹配形式并结合自适应速度分解，解决了扩散模型采样计算昂贵的问题，在样本质量和效率上优于现有无训练采样器，仅5次函数评估下实现更低FID和更清晰、连贯图像。

### 2. Acoustic Interference Suppression in Ultrasound images for Real-Time HIFU Monitoring Using an Image-Based Latent Diffusion Model

超声医学工程国家重点实验室发布了《Acoustic Interference Suppression in Ultrasound images for Real-Time HIFU Monitoring Using an Image-Based Latent Diffusion Model》论文，使用基于潜在扩散模型的HIFU-ILDiff深度学习方法（结合VQ-VAE编码与潜空间去干扰），解决了HIFU治疗中超声引导图像的声学干扰问题以提升实时监测能力，达成了显著优于Notch Filter（SSIM 0.796 vs 0.443、PSNR 23.780 vs 14.420）且实现15帧/秒实时处理的效果。

### 3. -A Multi-level Bias Elimination through a Decoding Approach with Knowledge Augmentation for Robust Constitutional Alignment of Language Models

南卡罗来纳大学人工智能研究所发布了稳健宪法对齐语言模型论文，使用AMBEDKAR框架（含宪法感知解码层与推测解码公平机制），解决了印度语境下种姓和宗教偏见及西方中心策略不足的问题，达成了相比基线绝对减少偏见高达26.41%的效果

### 4. AMMKD: Adaptive Multimodal Multi-teacher Distillation for Lightweight Vision-Language Models

University of Southern California等机构发布了AMMKD论文，使用自适应多模态多教师蒸馏（AMMKD）框架（整合多模态特征融合、多教师知识蒸馏与自适应动态加权），解决了大规模视觉语言预训练模型因模型大、计算复杂导致移动设备部署受限的问题，达成了在三个基准数据集上性能优越且显著降低模型复杂度的效果。

### 5. ANY-ORDER FLEXIBLE LENGTH MASKED DIFFUSION

Institute for Artificial Intelligence and Fundamental Interactions发布了ANY-ORDER FLEXIBLE LENGTH MASKED DIFFUSION论文，使用Flexible Masked Diffusion Models (FlexMDMs)技术，解决了Masked diffusion models不支持token插入、仅能固定长度生成的问题，达成了匹配MDMs困惑度、更好建模长度统计，迷宫规划任务成功率高≈60%，并将LLaDA-8B改造后数学(GSM8K 58%→67%)和代码填充(52%→65%)性能提升的效果

### 6. AN EFFICIENT GNNS-TO-KANS DISTILLATION VIA SELF-ATTENTION DYNAMIC SAMPLING WITH POTENTIAL FOR CONSUMER ELECTRONICS EDGE DEPLOYMENT A PREPRINT

天津大学、大连交通大学发布了相关论文，使用Self-Attention Dynamic Sampling Distillation (SA-DSD)框架及改进的FR-KAN+学生模型，解决了MLP难以捕捉GNN复杂邻域依赖导致边缘环境性能受限的问题，达成了性能提升3.05%-3.62%、参数减少16.96倍、推理时间减少55.75%的效果

### 7. An End-to-End Framework for Video Multi-Person Pose Estimation

中国科学技术大学发布了An End-to-End Framework for Video Multi-Person Pose Estimation论文，使用VEPE框架（含三个时空Transformer组件及实例一致性机制），解决了现有两阶段视频人体姿态估计方法分离时空维度、依赖单独检测器和复杂后处理导致的推理效率低等问题，达成了在Posetrack数据集上超过多数两阶段模型且推理效率提升300%的效果

### 8. 

清华大学发布了关于AppCopilot的论文，使用多模态多智能体通用设备端全栈闭环系统（集成多模态基础模型、思维链推理与分层任务规划等技术），解决了移动智能体的泛化、准确性、长程能力及资源受限设备效率四大核心问题，达成在所有四个维度均实现显著提升（更强泛化、更高精度屏幕操作、更可靠长程任务完成及更快更资源高效运行）的效果

### 9. A Continuous Encoding-Based Representation for Efficient Multi-Fidelity Multi-Objective Neural Architecture Search

A*STAR (IHPC)发布了高效多保真多目标神经架构搜索论文，使用连续编码表示及基于聚类的局部多保真填充采样策略，解决了神经架构搜索中多目标优化的高计算成本问题，在多个基准任务及实际应用中优于现有方法，降低计算复杂度并实现良好预测。

### 10. Bidirectional Sparse Attention for Faster Video Diffusion Training

研究团队发布了Bidirectional Sparse Attention for Faster Video Diffusion Training论文，使用双向稀疏注意力（BSA）框架技术，解决了视频扩散Transformer（DiT）模型因全注意力二次复杂度导致的高训练和推理成本问题，达成了加速训练、减少FLOPs达20倍、注意力训练速度提升17.79倍且保持甚至超越生成质量的效果。

### 11. Cache Management for Mixture-of-Experts LLMs -extended version Spyros Angelopoulos 1[0000-0001-9819-9158] , Loris Marchal 1[0000-0002-5519-9913] , Adrien Obrecht 2[0009-0007-6037-9787] , and

CNRS发布了关于混合专家大型语言模型缓存管理的论文，使用新分页问题模型及基于层的LRU扩展算法，解决了混合专家大型语言模型的专家缓存管理优化问题，在合成数据集和实际MoE使用轨迹上的模拟中性能优于标准LRU等经典分页策略。

### 12. Communication-Aware Knowledge Distillation for Federated LLM Fine-Tuning over Wireless Networks

伦敦国王学院发布了联邦LLM微调的通信感知知识蒸馏论文，使用自适应Top-k logit选择、自适应logits聚合及LoRA-adapted隐藏层投影融合的技术，解决了联邦LLM微调中高通信开销及带宽有限下logits高维传输问题，达成了性能优于基线方法且通信开销减少约50%的效果

### 13. CSFMAMBA: CROSS STATE FUSION MAMBA OPERATOR FOR MULTIMODAL REMOTE SENSING IMAGE CLASSIFICATION

上海交通大学发布了CSFMAMBA论文，使用跨状态融合Mamba算子，解决了多模态遥感图像分类中计算复杂度高及Mamba无法直接特征融合的问题，在MUUFL和Houston2018数据集上性能优于Transformer且降低网络训练负担。

### 14. DaMoC: Efficiently Selecting the Optimal Large Language Model for Fine-tuning Domain Tasks Based on Data and Model Compression

Ant Group发布了DaMoC论文，使用数据与模型压缩框架（含数据分类、关键token压缩、迭代重写及模型层重要性评估与稀疏合并），解决了快速选择最优大语言模型进行领域任务微调的挑战，达成了在四个数据集上节省约20倍训练时间的效果。

### 15. Democratizing Agentic AI with Fast Test-Time Scaling on the Edge

Microsoft Research Beijing发布了相关论文，使用FlashTTS（含Speculative Beam Extension、Asymmetric Multi-Model Memory Allocation、Dynamic Prefix-Aware Scheduling三个协同优化）技术，解决了边缘设备部署智能体AI时内存受限导致LLM推理能力不足、现有Test-Time Scaling方法开销过大的问题，达成了使≤7B边缘LLM在24GB消费级GPU上匹配大型云模型精度和延迟、goodput提升2.2倍、延迟降低38%-68%的效果。

### 16. Domain Adaptation-Based Crossmodal Knowledge Distillation for 3D Semantic Segmentation

北京大学发布了Domain Adaptation-Based Crossmodal Knowledge Distillation for 3D Semantic Segmentation论文，使用基于域适应的跨模态知识蒸馏方法（UDAKD和FSKD）及自校准卷积的域适应模块，解决了3D LiDAR数据标注成本高的问题，达成了性能超过现有最先进方法的效果

### 17. DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving

三星SDS发布了DSDE论文，使用基于KLD散度方差的预测信号和自适应推测长度上限的无训练框架，解决了推测解码在多样请求的大批量服务中依赖固定推测长度的次优问题，达成端到端延迟与领先基线相当且在多样工作负载下鲁棒性更优的效果。

### 18. DTRNet: Dynamic Token Routing Network to Reduce Quadratic Costs in Transformers

华为诺亚方舟实验室、AMD发布了DTRNet论文，使用动态令牌路由网络让令牌动态跳过二次成本的跨令牌混合并保留MLP模块（将大多数令牌注意力成本降至线性），解决了Transformer二次计算成本高的问题，达成了每层仅路由10%令牌通过注意力即可保持性能，同时减少FLOPs且长序列效率显著提升的效果。

### 19. Efficient Large Language Models with Zero-Shot Adjustable Acceleration

Sharif University of Technology Tehran发布了论文，使用Zero-Shot Adjustable Acceleration技术，解决了LLMs实际应用中计算效率与性能的平衡问题（尤其是微调后和推理时的加速优化），达成零样本广泛加速且相比基线高达11×提速的效果。

### 20. Efficient Pyramidal Analysis of Gigapixel Images on a Decentralized Modest Computer Cluster

EPFL和IP Paris发布了相关论文，使用PyramidAI技术（通过渐进式分析从低分辨率逐步聚焦感兴趣区域并优化精度-计算权衡），解决了gigapixel图像分析计算量大的问题，达成处理数据量减少2.65倍、单机能保持精度且12台普通计算机集群将分析时间从1小时以上降至几分钟的效果

### 21. Encoder-Only Image Registration

研究团队发布了Encoder-Only Image Registration论文，使用EOIR框架（分离特征学习与流估计，构建拉普拉斯特征金字塔处理大变形），解决了图像配准中计算复杂度高和大变形的挑战，达成了更好的精度-效率与精度-平滑度权衡。

### 22. Energy Efficient Exact and Approximate Systolic Array Architecture for Matrix Multiplication

Indian Institute of Technology发布了Energy Efficient Exact and Approximate Systolic Array Architecture for Matrix Multiplication论文，使用结合新型精确和近似处理单元（PEs）及能量高效正/负部分积单元（PPC/NPPC）的脉动阵列架构，解决了深度神经网络（DNNs）矩阵乘法的能量效率问题，达成8位精确PE节能22%、近似PE节能32%，并在DCT和边缘检测中实现高输出质量（PSNR分别为38.21 dB和30.45 dB）的效果。

### 23. Entropy-based Coarse and Compressed Semantic Speech Representation Learning

浙江大学发布了Entropy-based Coarse and Compressed Semantic Speech Representation Learning论文，使用基于熵的动态聚合框架（通过预训练语音语言模型、预测熵自适应确定聚合边界及交叉注意力模块融合段内信息，可灵活控制表示粒度和压缩比），解决了现有离散语音表示学习中令牌冗余、效率低及语义表示粒度过细的问题，达成了在ASR、语音到文本翻译、语音转换任务上，压缩表示性能与或优于密集令牌序列的效果。

### 24. Faster and Better: Reinforced Collaborative Distillation and Self-Learning for Infrared-Visible Image Fusion

北京理工大学发布了《Faster and Better: Reinforced Collaborative Distillation and Self-Learning for Infrared-Visible Image Fusion》论文，使用基于强化学习的协作蒸馏和自学习框架，解决了轻量级模型下实现高质量红外-可见光图像融合的挑战，达成了显著提升学生模型性能、融合效果优于现有技术的效果。

### 25. FastVGGT: Training-Free Acceleration of Visual Geometry Transformer

中国教育部发布了FastVGGT论文，使用训练无关加速技术，解决了视觉几何Transformer（VGGT）的计算效率问题，达成了无需训练即可加速VGGT的效果。

### 26. From TLinFormer to TConstFormer: The Leap to Constant-Time Transformer Attention Achieving O(1) Computation and O(1) KV Cache during Autoregressive Inference

研究团队发布了TConstFormer论文，使用创新的周期性状态更新机制，解决了Transformer自回归推理中KV Cache线性增长及计算复杂度O(N²d)的问题，达成了O(1)计算复杂度和O(1)KV Cache，在长文本推理任务上速度、内存效率和整体性能远超基线模型的效果

### 27. Gated Associative Memory: A Parallel O(N) Architecture for Efficient Sequence Modeling

Independent Researcher发布了Gated Associative Memory (GAM)网络论文，使用Gated Associative Memory (GAM)网络（通过因果卷积与并行关联记忆检索机制两条并行路径，并结合门控机制动态融合局部与全局信息），解决了Transformer自注意力机制在序列建模中因O(N²)复杂度导致的长序列处理瓶颈，达成了线性复杂度(O(N))下训练速度优于Transformer和Mamba且验证困惑度相当或更优的效果。

### 28. GraphKV: Breaking the Static Selection Paradigm with Graph-Based KV Cache Eviction

上海交通大学发布了GraphKV: Breaking the Static Selection Paradigm with Graph-Based KV Cache Eviction论文，使用基于图的KV缓存驱逐框架（GraphKV），将tokens建模为节点（带重要性分数）、边表示相似度关系，并通过衰减信号传播机制动态更新token重要性，解决了传统KV驱逐策略依赖静态启发式、无法捕捉推理过程中tokens间演化隐式依赖的问题，达成了能无缝插件式用于现有KV缓存驱逐方法（如SnapKV、PyramidKV）的效果。

### 29. Guidance and Control Neural Network Acceleration using Memristors

ESA Advanced Concepts Team与Delft University of Technology发布了Guidance and Control Neural Network Acceleration using Memristors论文，使用PCM和RRAM忆阻器进行星载内存计算AI加速技术，解决了小卫星/立方体卫星能量预算受限及辐射问题导致的星载AI应用受限问题，达成了忆阻器加速器能学习专家动作且退化后重新训练可恢复性能至标称水平的效果。

### 30. Guided Model-based LiDAR Super-Resolution for Resource-Efficient Automotive scene Segmentation

Athena Research Center发布了Guided Model-based LiDAR Super-Resolution for Resource-Efficient Automotive scene Segmentation论文，使用首个端到端集成LiDAR超分辨率与分割的框架，通过联合优化和轻量级模型解决低成本16通道LiDAR稀疏点云导致的分割性能低问题，达成分割性能媲美高分辨率64通道LiDAR数据的效果

### 31. ENHANCING IMAGE QUALITY AND ANOMALY DETECTION FOR SMALL AND DENSE INDUSTRIAL OBJECTS IN NUCLEAR RECYCLING

Mines Saint-Etienne发布了《增强核回收中微小密集工业物体的图像质量与异常检测》论文，使用基于监督深度学习的方法（含新数据集及轻量级全连接卷积网络模型），解决了工业环境中小、密集、重叠物体检测及嘈杂图像质量提升问题，达成了识别可靠检测系统并改善图像质量的效果。

### 32. Knowledge distillation as a pathway toward next-generation intelligent ecohydrological modeling systems

香港大学发布了Knowledge distillation as a pathway toward next-generation intelligent ecohydrological modeling systems论文，使用三阶段知识蒸馏框架（整合过程模型与机器学习，通过行为、结构、认知蒸馏），解决了传统过程模型结构刚性、计算成本高及机器学习缺乏可解释性和可迁移性的问题，达成了重现过程模型输出、提高预测准确性、支持情景决策的效果

### 33. KVComp: A High-Performance, LLM-Aware, Lossy Compression Framework for KV Cache

University of Houston和Temple University发布了KVComp论文，使用针对KV缓存数据特征设计的新型有损压缩技术及算法与系统架构协同设计，解决了LLMs长上下文推理中KV缓存巨大内存需求的问题，达成了平均内存减少率比现有方法高47%（最高83%）、模型精度几乎无损失且提升执行吞吐量的效果。

### 34. LatentEdit: Adaptive Latent Control for Consistent Semantic Editing

南方科技大学发布了LatentEdit论文，使用自适应潜在融合框架，解决了基于扩散的图像编辑在保持背景相似性的同时实现高质量编辑且不牺牲速度与内存效率的挑战，达成在PIE-Bench数据集上保真度与可编辑性的最佳平衡，8-15步内优于最先进方法，无反演变体提升实时部署效率的效果。

### 35. Learning to Shard: RL for Co-optimizing the Parallelism Degrees and Per-operator Sharding Dimensions in Distributed LLM Inference

Microsoft Azure与Yale University发布了Learn to Shard论文，使用基于RL的方法共同优化分布式LLM推理的并行度与算子分片维度，解决了静态启发式方法分别配置导致的性能问题，在1.6T参数MoE模型上实现吞吐量较元启发式基线提升3.5×、较Megatron启发式提升1.06×的效果。

### 36. LightVLM: Acceleraing Large Multimodal Models with Pyramid Token Merging and KV Cache Compression

天津大学发布了LightVLM论文，使用金字塔令牌合并与KV缓存压缩技术，解决了大型多模态模型（VLMs）的推理效率问题，达成保留35%图像令牌性能100%、3%时约98%，吞吐量提升2.02倍、预填充时间减少3.65倍、长文本推理时间减少3.21倍的效果。

### 37. LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

字节跳动与上海交通大学发布了LiquidGEMM论文，使用LiquidQuant和隐式细粒度流水线技术，解决了现有W4A8 GEMM内核因CUDA核反量化低效无法跟上Tensor核高吞吐量的问题，达成相比最先进W4A8内核提速2.90倍、端到端系统级提速4.94倍的效果

### 38. 

发布了LongCat-Flash论文，使用Zero-computation Experts和Shortcut-connected MoE技术的5600亿参数混合专家(MoE)语言模型，解决了大规模语言模型计算效率与高级智能体能力兼顾的问题，达成30天内完成超20万亿tokens训练、推理速度超100 TPS且成本0.70美元/百万输出tokens，在智能体任务上表现突出的效果。

### 39. LUT-Fuse: Towards Extremely Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up Tables

武汉大学电子信息学院、东南大学发布了LUT-Fuse论文，使用通过蒸馏到可学习查找表（LUT）的红外与可见光图像融合技术（结合低阶近似编码和高阶联合上下文场景编码的查找表结构），解决了红外与可见光图像融合研究中忽视实时融合设备适用性的问题，达成了比现有轻量级SOTA融合算法快十分之一以上、在低功耗移动设备等场景下实现高运行速度的效果。

### 40. MAMBA-CNN: A HYBRID ARCHITECTURE FOR EFFICIENT AND ACCURATE FACIAL BEAUTY PREDICTION

Scientific and Technical Research Centre for Arid Areas发布了MAMBA-CNN论文，使用集成轻量级Mamba启发的状态空间模型（SSM）门控机制的混合架构，解决了面部吸引力评估中CNN感受野有限与ViT计算成本高的权衡问题，达成了在SCUT-FBP5500基准上的新SOTA，Pearson Correlation达0.9187。

### 41. 

请提供论文的标题、主要机构和摘要信息，以便生成符合要求的技术简报。

### 42. MoPEQ: Mixture of Mixed Precision Quantized Experts

Argonne National Laboratory和Illinois Institute of Technology发布了MoPEQ论文，使用MoPEQ混合精度量化算法（通过Hessian迹近似分析专家敏感度分配最优位宽），解决了MoE架构大语言/视觉模型部署的计算与内存需求挑战，达成了在保持精度竞争力的同时显著改善内存占用的效果

### 43. OmniReason: A Temporal-Guided Vision-Language-Action Framework for Autonomous Driving

Li Auto Inc与香港科技大学发布了OmniReason论文，使用OmniReason框架（含OmniReason-Data数据集与OmniReason-Agent架构，集成稀疏时间记忆模块、解释生成器及时空知识蒸馏），解决了现有视觉语言模型在自动驾驶中忽视时间维度、静态场景理解的局限，达成了在开环规划任务和视觉问答基准上显著提升，并建立可解释、时间感知的自动驾驶能力的效果。

### 44. Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption

CNRS发布了Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption论文，使用SAFHIRE混合推理框架（线性层加密服务器执行、非线性客户端明文计算、随机打乱保护模型机密性及密文打包优化），解决了FHE推理中bootstrapping昂贵、非线性激活低效及模型机密性问题，达成比ORION推理延迟降低1.5×-10.5×、通信开销可控且精度相当的效果，确立了混合FHE推理的实用性。

### 45. Principled Approximation Methods for Efficient and Scalable Deep Learning

丰田芝加哥技术学院发布了高效可扩展深度学习论文，使用将离散问题转为连续可微近似的架构设计、模型压缩与优化方法，解决了深度学习模型计算与能源需求高的问题，实现训练和推理效率显著提升且性能保持甚至改善。

### 46. Progressive Element-wise Gradient Estimation for Neural Network Quantization

Oakland University发布了Progressive Element-wise Gradient Estimation (PEGE)论文，使用Progressive Element-wise Gradient Estimation (PEGE)技术，该技术通过对数课程驱动的混合精度替换策略逐步用量化值替代全精度值并将量化感知训练构建为共同优化任务损失与离散化误差的问题，解决了神经网络量化中Straight-Through Estimator (STE)忽略离散化误差导致低比特位宽下模型精度下降的问题，达成了在CIFAR-10和ImageNet数据集上，使ResNet、VGG等架构的低精度量化模型精度匹配甚至超过全精度模型，并持续优于现有反向传播方法的效果。

### 47. Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs

Case Western Reserve University发布了Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs论文，使用Truthful Pruning aligned by Layer-wise Outliers (TPLO)技术，解决了神经网络剪枝破坏LLMs用于谎言检测的内部激活特征导致谎言检测能力下降的问题，达成了在50%稀疏度下幻觉检测准确率达88%并提升TruthfulQA性能的效果

### 48. Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling

德克萨斯大学奥斯汀分校发布了Q-Sched论文，使用量化感知噪声调度器，解决了少步扩散模型的压缩与性能平衡问题，达成了优秀的图像保真度效果

### 49. Quantization Meets OOD: Generalizable Quantization-aware Training from a Flatness Perspective

MMLab发布了Quantization Meets OOD论文，使用分层冻结机制和无序引导的自适应冻结算法（含梯度无序度量）的FQAT技术，解决了量化感知训练（QAT）导致的分布外（OOD）数据泛化性能下降问题，在OOD基准上实现了优越性能。

### 50. Quantum-Optimized Selective State Space Model for Efficient Time Series Prediction

Politehnica University Timisoara发布了《Quantum-Optimized Selective State Space Model for Efficient Time Series Prediction》论文，使用量子优化选择性状态空间模型（Q-SSM，集成状态空间动态与变分量子门，通过参数化量子电路的期望值自适应调节内存更新），解决了长程时间序列预测中捕捉非平稳多尺度依赖、提升噪声鲁棒性、效率及稳定性的问题（如Transformer二次复杂度、S-Mamba训练不稳定），达成了在ETT、Traffic、Exchange Rate等基准上持续优于LSTM、Transformer及S-Mamba等强基线的效果

### 51. RDIT: Residual-based Diffusion Implicit Models for Probabilistic Time Series Forecasting

哈佛大学与麻省理工学院发布了RDIT论文，使用结合点估计与基于残差的条件扩散及双向Mamba网络的技术，解决了概率时间序列预测中的分布建模及训练评估指标不匹配问题，达成了更低CRPS、快速推理和改进覆盖率的效果。

### 52. Scale, Don't Fine-tune: Guiding Multimodal LLMs for Efficient Visual Place Recognition at Test-Time ⋆

香港科技大学发布了《Scale, Don't Fine-tune: Guiding Multimodal LLMs for Efficient Visual Place Recognition at Test-Time》论文，使用零样本Test-Time Scaling (TTS)框架，通过Guidance-based方法和结构化提示生成JSON输出，并结合Uncertainty-Aware Self-Consistency (UASC)，解决了现有VPR方法计算开销高、跨域迁移性有限的问题，达成了跨域VPR性能显著提升且计算效率提升210倍的效果。

### 53. Scaling Legal AI: Benchmarking Mamba and Transformers for Statutory Classification and Case Law Retrieval

研究团队发布了Scaling Legal AI论文，使用Mamba状态空间模型（SSM），解决了Transformer在法律AI中因二次注意力成本导致的效率和可扩展性不足问题，达成了线性缩放支持处理比Transformer长数倍的法律文档且保持或超越分类与检索性能的效果。

### 54. SCOUT: Toward Sub-Quadratic Attention via Segment Compression for Optimized Utility in Transformers

华为诺亚方舟实验室发布了SCOUT论文，使用片段压缩令牌并仅对压缩表示应用注意力的混合架构，解决了Transformers二次注意力复杂度限制长序列扩展性及线性模型远距离信息保留不足的问题，达成在相同计算预算下优于强长序列基线、匹配全注意力Transformers性能、吞吐量高于最先进线性模型且计算和内存成本较全注意力节省超10倍的效果

### 55. Securing Radiation Detection Systems with an Efficient TinyML-Based IDS for Edge Devices

Ontario Tech University发布了基于TinyML的高效边缘设备辐射检测系统入侵检测（IDS）论文，使用TinyML技术优化的XGBoost模型（结合剪枝、量化、特征选择和采样），解决了辐射检测系统（RDSs）易受网络攻击导致测量完整性和可靠性受损的问题，达成了显著减少模型大小和计算需求，实现低资源设备实时入侵检测并平衡效率与准确性的效果。

### 56. Seeing More, Saying More: Lightweight Language Experts are Dynamic Video Token Compressors

香港大学与南方科技大学发布了相关论文，使用LangDC（语言感知动态视频令牌压缩器）技术，解决了现有视频令牌压缩策略因固定压缩比忽略语义密度差异导致的信息丰富片段表示不足与内容贫乏片段计算冗余问题，达成了相比VideoGPT+减少49% FLOPs并保持竞争力性能，且能根据视频片段丰富度自适应调整压缩比的效果。

### 57. SpectMamba: Integrating Frequency and State Space Models for Enhanced Medical Image Detection

南开大学、南方科技大学发布了SpectMamba论文，使用Hybrid Spatial-Frequency Attention (HSFA)块、Visual State-Space Module (VSSM)及Hilbert Curve Scanning技术，解决了CNN感受野有限及Transformer处理高分辨率医学图像计算成本高的问题，达成了在多种医学图像检测任务上高效且有效达到SOTA性能的效果

### 58. TINYMUSICIAN: ON-DEVICE MUSIC GENERATION WITH KNOWLEDGE DISTILLATION AND MIXED PRE-CISION QUANTIZATION

波士顿大学发布了TinyMusician论文，使用融合阶段混合双向与倾斜KL散度及自适应混合精度量化的知识蒸馏技术，解决了Transformer音乐生成模型因参数量大导致计算资源需求高、无法部署在边缘设备的问题，达成保留MusicGen-Small 93%性能且模型大小减少55%、实现首个可移动部署并消除云依赖的效果。

### 59. Towards Adaptive Visual Token Pruning for Large Multimodal Models

Beijing Academy of Artificial Intelligence发布了《Towards Adaptive Visual Token Pruning for Large Multimodal Models》论文，使用基于互信息（保留跨模态对齐）和最大化嵌入空间成对距离（贪婪算法，保持模态内信息多样性）的视觉令牌剪枝策略，解决了大模态模型推理时视觉密集令牌序列导致计算与内存成本高且现有剪枝方法冗余令牌多的问题，达成了在LLaVA-1.5-7B和LLaVA-NEXT-7B等模型上减少88.9%视觉令牌、推理速度提升56.7%且保持性能的效果

### 60. Uirapuru: Timely Video Analytics for High-Resolution Steerable Cameras on Edge Devices

Vrije Universiteit Amsterdam和Paderborn University发布了Uirapuru论文，使用面向边缘设备的高分辨率可转向摄像头及时视频分析技术，解决了边缘设备上高分辨率可转向摄像头视频分析的实时性问题，达成了及时视频分析的效果。

### 61. Variation-aware Vision Token Dropping for Faster Large Vision-Language Models

上海交通大学发布了“Variation-aware Vision Token Dropping for Faster Large Vision-Language Models”论文，使用Variation-aware Vision Token Dropping（V²Drop）技术，解决了大型视觉语言模型（LVLMs）因高分辨率图像/长视频令牌数量多导致的推理效率低及现有内LLM令牌压缩方法的位置偏差与高效算子不兼容问题，达成在图像和视频理解任务中分别保持94.0%和98.6%原始性能，同时减少LLM生成延迟31.5%和74.2%，结合高效算子降低GPU峰值内存的效果。

### 62. Vision-Based Embedded System for Noncontact Monitoring of Preterm Infant Behavior in Low-Resource Care Settings

Soroti University发布了Vision-Based Embedded System for Noncontact Monitoring of Preterm Infant Behavior in Low-Resource Care Settings论文，使用基于视觉的嵌入式系统（集成量化MobileNet模型部署于Raspberry Pi，通过模型量化实现68%尺寸缩减），解决了低资源环境下早产儿行为（睡眠/哭闹）依赖人工或侵入式传感器监测的误差、不实用及皮肤损伤问题，达成了睡眠检测91.8%、哭闹/正常分类97.7%的SOTA准确率且适合边缘实时部署的效果。

### 63. ZeroQAT: Your Quantization-aware Training but Efficient

University of Minnesota发布了ZeroQAT论文，使用零阶优化的QAT框架（前向梯度估计消除反向传播，联合学习量化权重、裁剪阈值和等效变换），解决了现有低比特PTQ精度下降与QAT成本高的问题，达成了PTQ效率与QAT精度兼具的效果。

## 论文详细信息

### 1. A-FloPS: Accelerating Diffusion Sampling with Adaptive Flow Path Sampler

**主要机构**: Tsinghua University, Department of Electronic Engineering
**作者数量**: 3人

**摘要**:
Diffusion models deliver state-of-the-art generative performance across diverse modalities but remain computationally expensive due to their inherently iterative sampling process. Existing training-free acceleration methods typically improve numerical solvers for the reverse-time ODE, yet their effectiveness is fundamentally constrained by the inefficiency of the underlying sampling trajectories. We propose A-FloPS (Adaptive Flow Path Sampler), a principled, training-free framework that reparameterizes the sampling trajectory of any pre-trained diffusion model into a flow-matching form and augments it with an adaptive velocity decomposition. The reparameterization analytically maps diffusion scores to flowcompatible velocities, yielding integration-friendly trajectories without retraining. The adaptive mechanism further factorizes the velocity field into a linear drift term and a residual component whose temporal variation is actively suppressed, restoring the accuracy benefits of high-order integration even in extremely low-NFE regimes. Extensive experiments on conditional image generation and text-to-image synthesis show that A-FloPS consistently outperforms state-of-the-art training-free samplers in both sample quality and efficiency. Notably, with as few as 5 function evaluations, A-FloPS achieves substantially lower FID and generates sharper, more coherent images. The adaptive mechanism also improves native flow-based generative models, underscoring its generality. These results position A-FloPS as a versatile and effective solution for high-quality, low-latency generative modeling.

### 2. Acoustic Interference Suppression in Ultrasound images for Real-Time HIFU Monitoring Using an Image-Based Latent Diffusion Model

**主要机构**: State Key Laboratory of Ultrasound in Medicine and Engineering, School of Microelectronics, College of Biomedical Engineering, Chongqing Medical University, NMPA Key Laboratory for Quality Evaluation of Ultrasonic Surgical Equipment, Tianjin University
**作者数量**: 9人

**摘要**:
High-Intensity Focused Ultrasound (HIFU) is a non-invasive therapeutic technique widely used for treating various diseases. However, the success and safety of HIFU treatments depend on real-time monitoring, which is often hindered by interference when using ultrasound to guide HIFU treatment. To address these challenges, we developed HIFU-ILDiff, a novel deep learning-based approach leveraging latent diffusion models to suppress HIFU-induced interference in ultrasound images. The HIFU-ILDiff model employs a Vector Quantized Variational Autoencoder (VQ-VAE) to encode noisy ultrasound images into a lower-dimensional latent space, followed by a latent diffusion model that iteratively removes interference. The denoised latent vectors are then decoded to reconstruct high-resolution, interference-free ultrasound images. We constructed a comprehensive dataset comprising 18,872 image pairs from in vitro phantoms, ex vivo tissues, and in vivo animal across multiple imaging modalities and HIFU power levels to train and evaluate the model. Experimental results demonstrate that HIFU-ILDiff significantly outperforms the commonly used Notch Filter method, achieving a Structural Similarity Index (SSIM) of 0.796 ± 0.008 and Peak Signal-to-Noise Ratio (PSNR) of 23.780 ± 0.001 compared to SSIM of 0.443 and PSNR of 14.420 for the Notch Filterunder in vitro scenarios. Additionally, HIFU-ILDiff achieves realtime processing at 15 frames per second, markedly faster than the Notch Filter's 5 seconds per frame. These findings indicate that HIFU-ILDiff is able to denoise HIFU interference in the ultrasound guiding images for realtime monitoring during HIFU therapy, which will greatly improve the treatment precision in current clinical applications.

### 3. -A Multi-level Bias Elimination through a Decoding Approach with Knowledge Augmentation for Robust Constitutional Alignment of Language Models

**主要机构**: Artificial Intelligence Institute, University of South Carolina, BITS Pilani Goa, Indian Institute of Information Technology, IIT Madras
**作者数量**: 8人

**摘要**:
Large Language Models (LLMs) can inadvertently reflect societal biases present in their training data, leading to harmful or prejudiced outputs. In the Indian context, our empirical evaluations across a suite of models reveal that biases around caste and religion are particularly salient. Yet, most existing mitigation strategies are Western-centric and fail to address these local nuances. We propose AMBEDKAR, a framework inspired by the egalitarian vision of Dr. B. R. Ambedkar, architect of the Indian Constitution, to guide LLM outputs toward fairness, neutrality, and inclusion in line with Articles 14 to 17. Our approach introduces a Constitution-Aware Decoding Layer, guided by the AI Constitution of India and applied only at inference time, without any parameter updates to the base model. We incorporate a speculative decoding algorithm that proactively reduces casteist and communal bias during generation. This mitigation layer operates directly within the decoding process, avoiding changes to model internals and lowering the computational and infrastructural costs associated with retraining. We reinterpret speculative decoding not merely as an efficiency tool but as a mechanism for fairness. In this framework, a Small Language Model (SLM) acts as a potentially biased generator, while a constitutionally guided Large Language Model (LLM) serves as the verifier. Rather than accelerating generation, the LLM enforces bias-robust trajectories in the SLM's outputs. This inversion of roles gives rise to a fairness-by-speculation paradigm. Our approach yields an absolute reduction of bias upto 26.41% compared to baseline. Our source code, datasets, and results are available at: https://anonymous.4open. science/r/AMBEDKAR-983B/ *

### 4. AMMKD: Adaptive Multimodal Multi-teacher Distillation for Lightweight Vision-Language Models

**主要机构**: The City University of New York, Institute of Computing Technology, University of Southern California, University of Hong Kong, Nanyang Technological University, Chinese Academy of Sciences
**作者数量**: 9人

**摘要**:
The success of large-scale visual language pretraining (VLP) models has driven widespread adoption of image-text retrieval tasks. However, their deployment on mobile devices remains limited due to large model sizes and computational complexity. We propose Adaptive Multi-Modal Multi-Teacher Knowledge Distillation (AMMKD), a novel framework that integrates multi-modal feature fusion, multi-teacher distillation, and adaptive optimization to deliver lightweight yet effective retrieval models. Specifically, our method begins with a feature fusion network that extracts and merges discriminative features from both the image and text modalities. To reduce model parameters and further improve performance, we design a multi-teacher knowledge distillation framework to pre-train two CLIP teacher models. We decouple modalities by pre-computing and storing text features as class vectors via the teacher text encoder to enhance efficiency. To better align teacher and student outputs, we apply KL scatter for probability distribution matching. Finally, we design an adaptive dynamic weighting scheme that treats multi-teacher distillation as a multi-objective optimization problem. By leveraging gradient space diversity, we dynamically adjust the influence of each teacher, reducing conflicts and guiding the student toward more optimal learning directions. Extensive experiments on three benchmark datasets demonstrate that AMMKD achieves superior performance while significantly reducing model complexity, validating its effectiveness and flexibility.

### 5. ANY-ORDER FLEXIBLE LENGTH MASKED DIFFUSION

**主要机构**: Institute for Artificial Intelligence and Fundamental Interactions
**作者数量**: 12人

**摘要**:
Masked diffusion models (MDMs) have recently emerged as a promising alternative to autoregressive models over discrete domains. MDMs generate sequences in an any-order, parallel fashion, enabling fast inference and strong performance on non-causal tasks. However, a crucial limitation is that they do not support token insertions and are thus limited to fixed-length generations. To this end, we introduce Flexible Masked Diffusion Models (FlexMDMs), a discrete diffusion paradigm that simultaneously can model sequences of flexible length while provably retaining MDMs' flexibility of any-order inference. Grounded in an extension of the stochastic interpolant framework, FlexMDMs generate sequences by inserting mask tokens and unmasking them. Empirically, we show that FlexMDMs match MDMs in perplexity while modeling length statistics with much higher fidelity. On a synthetic maze planning task, they achieve ≈ 60% higher success rate than MDM baselines. Finally, we show pretrained MDMs can easily be retrofitted into FlexMDMs: on 16 H100s, it takes only three days to fine-tune LLaDA-8B into a FlexMDM, achieving superior performance on math (GSM8K, 58%→67%) and code infilling performance (52%→65%).

### 6. AN EFFICIENT GNNS-TO-KANS DISTILLATION VIA SELF-ATTENTION DYNAMIC SAMPLING WITH POTENTIAL FOR CONSUMER ELECTRONICS EDGE DEPLOYMENT A PREPRINT

**主要机构**: The School of Railway Intelligent Engineering Dalian Jiaotong University Dalian, University of China Tianjin, The College of Electronic Information and Automation Civil Aviation
**作者数量**: 6人

**摘要**:
Knowledge distillation (KD) is crucial for deploying deep learning models in resource-constrained edge environments, particularly within the consumer electronics sector, including smart home devices, wearable technology, and mobile terminals. These applications place higher demands on model compression and inference speed, necessitating the transfer of knowledge from Graph Neural Networks (GNNs) to more efficient Multi-Layer Perceptron (MLP) models. However, due to their fixed activation functions and fully connected architecture, MLPs face challenges in rapidly capturing the complex neighborhood dependencies learned by GNNs, thereby limiting their performance in edge environments. To address these limitations, this paper introduces an innovative from GNNs to Kolmogorov-Arnold Networks (KANs) knowledge distillation framework-Self-Attention Dynamic Sampling Distillation (SA-DSD). This study improved Fourier KAN (FR-KAN) and replaced MLP with the improved FR-KAN+ as the student model. Through the incorporation of learnable frequency bases and phase-shift mechanisms, along with algorithmic optimization, FR-KAN significantly improves its nonlinear fitting capability while effectively reducing computational complexity. Building on this, a margin-level sampling probability matrix, based on teacher-student prediction consistency, is constructed, and an adaptive weighted loss mechanism is designed to mitigate performance degradation in the student model due to the lack of explicit neighborhood aggregation. Extensive experiments conducted on six real-world datasets demonstrate that SA-DSD achieves performance improvements of 3.05%-3.62% over three GNN teacher models and 15.61% over the FR-KAN+ model. Moreover, when compared with key benchmark models, SA-DSD achieves a 16.96x reduction in parameter count and a 55.75% decrease in inference time.

### 7. An End-to-End Framework for Video Multi-Person Pose Estimation

**主要机构**: University of Science and Technology of China
**作者数量**: 1人

**摘要**:
Video-based human pose estimation models aim to address scenarios that cannot be effectively solved by static image models such as motion blur, out-of-focus and occlusion. Most existing approaches consist of two stages: detecting human instances in each image frame and then using a temporal model for single-person pose estimation. This approach separates the spatial and temporal dimensions and cannot capture the global spatio-temporal context between spatial instances for end-to-end optimization. In addition, it relies on separate detectors and complex post-processing such as RoI cropping and NMS, which reduces the inference efficiency of the video scene. To address the above problems, we propose VEPE (Video Endto-End Pose Estimation), a simple and flexible framework for end-to-end pose estimation in video. The framework utilizes three crucial spatio-temporal Transformer components: the Spatio-Temporal Pose Encoder (STPE), the Spatio-Temporal Deformable Memory Encoder (STDME), and the Spatio-Temporal Pose Decoder (STPD). These components are designed to effectively utilize temporal context for optimizing human body pose estimation. Furthermore, to reduce the mismatch problem during the cross-frame pose query matching process, we propose an instance consistency mechanism, which aims to enhance the consistency and discrepancy of the cross-frame instance query and realize the instance tracking function, which in turn accurately guides the pose query to perform cross-frame matching. Extensive experiments on the Posetrack dataset show that our approach outperforms most two-stage models and improves inference efficiency by 300%.

### 8. 

**主要机构**: Renmin University of China ♦ Modelbest Inc, Gaoling School of Artificial Intelligence, Tsinghua University, School of Artificial Intelligence, Shanghai Jiao Tong University, Department of Computer Science and Technology
**作者数量**: 13人

**摘要**:
With the raid evolution of large language models and multimodal foundation models, the mobile-agent landscape has proliferated without converging on the fundamental challenges. This paper identifies four core problems that must be solved for mobile agents to deliver practical, scalable impact: (1) generalization across tasks, modalities, apps, and devices; (2) accuracy, specifically precise on-screen interaction and click targeting; (3) long-horizon capability for sustained, multi-step goals; and (4) efficiency, specifically high-performance runtime on resourceconstrained devices. We present AppCopilot, a multimodal, multi-agent, general-purpose on-device assistant that operates across applications and constitutes a full-stack, closed-loop system from data to deployment. AppCopilot operationalizes this position through an end-to-end autonomous pipeline spanning data collection, training, deployment, high-quality and efficient inference, and PC/mobile application development. At the model layer, it integrates multimodal foundation models with robust Chinese-English support. At the reasoning and control layer, it combines chain-ofthought reasoning, hierarchical task planning and decomposition, and multi-agent collaboration. At the execution layer, it enables user personalization and experiential adaptation, voice interaction, function/tool calling, cross-app and cross-device orchestration, and comprehensive mobile app support. The system design incorporates profiling-driven optimization for latency, memory, and energy across heterogeneous hardware. Empirically, AppCopilot achieves significant improvements along all four dimensions: stronger generalization, higher-precision on-screen actions, more reliable long-horizon task completion, and faster, more resource-efficient runtime. By articulating a cohesive position and a reference architecture that closes the loop from "data collection-training and deployment-high-quality, efficient inference-application development", this paper offers a concrete roadmap for general-purpose digital assistants and provides actionable guidance for both academic research and industrial adoption. For updates and forthcoming releases, see the project page: https://github.com/OpenBMB/ AppCopilot.

### 9. A Continuous Encoding-Based Representation for Efficient Multi-Fidelity Multi-Objective Neural Architecture Search

**主要机构**: Institute of High Performance Computing (IHPC), Agency for Science, Technology and Research (A*STAR), Centre for Frontier AI Research (CFAR)
**作者数量**: 3人

**摘要**:
Neural architecture search (NAS) is an attractive approach to automate the design of optimized architectures but is constrained by high computational budget, especially when optimizing for multiple, important conflicting objectives. To address this, an adaptive Co-Krigingassisted multi-fidelity multi-objective NAS algorithm is proposed to further reduce the computational cost of NAS by incorporating a clustering-based local multi-fidelity infill sampling strategy, enabling efficient exploration of the search space for faster convergence. This algorithm is further accelerated by the use of a novel continuous encoding method to represent the connections of nodes in each cell within a generalized cell-based U-Net backbone, thereby decreasing the search dimension (number of variables). Results indicate that the proposed NAS algorithm outperforms previously published state-of-the-art methods under limited computational budget on three numerical benchmarks, a 2D Darcy flow regression problem and a CHASE_DB1 biomedical image segmentation problem. The proposed method is subsequently used to create a wind velocity regression model with application in urban modelling, with the found model able to achieve good prediction with less computational complexity. Further analysis revealed that the NAS algorithm independently identified principles undergirding superior U-Net architectures in other literature, such as the importance of allowing each cell to incorporate information from prior cells.

### 10. Bidirectional Sparse Attention for Faster Video Diffusion Training

**主要机构**: 
**作者数量**: 7人

**摘要**:
Video diffusion Transformer (DiT) models excel in generative quality but hit major computational bottlenecks when producing high-resolution, long-duration videos. The quadratic complexity of full attention (O(L 2)) leads to prohibitively high training and inference costs. Full attention inefficiency stems from two key challenges: excessive computation due to the inherent sparsity of Queries and Key-Value pairs, and redundant computation as fixed sparse patterns fail to leverage DiT's dynamic attention. To overcome this limitation, we propose a Bidirectional Sparse Attention (BSA) framework for faster video DiT training, the first to dynamically sparsify both Queries and Key-Value pairs within 3D full attention, thereby substantially improving training and inference efficiency. BSA addresses these issues through two key components. Query sparsity is optimized by selecting the most informative query tokens via semantic similarity and with a dynamic spatial-time training strategy, while KV sparsity is achieved by computing a statistical dynamic threshold to retain only the most salient KV blocks for computation. Extensive experiments demonstrate that BSA significantly accelerates DiT training across long sequences, reducing FLOPs by up to 20× and achieving 17.79× faster attention training, while preserving or even surpassing the generative quality of full attention.

### 11. Cache Management for Mixture-of-Experts LLMs -extended version Spyros Angelopoulos 1[0000-0001-9819-9158] , Loris Marchal 1[0000-0002-5519-9913] , Adrien Obrecht 2[0009-0007-6037-9787] , and

**主要机构**: CNRS, International Laboratory on Learning Systems
**作者数量**: 1人

**摘要**:
Large language models (LLMs) have demonstrated remarkable capabilities across a variety of tasks. One of the main challenges towards the successful deployment of LLMs is memory management, since they typically involve billions of parameters. To this end, architectures based on Mixture-of-Experts have been proposed, which aim to reduce the size of the parameters that are activated when producing a token. This raises the equally critical issue of efficiently managing the limited cache of the system, in that frequently used experts should be stored in the fast cache rather than in the slower secondary memory. In this work, we introduce and study a new paging problem that models expert management optimization. Our formulation captures both the layered architecture of LLMs and the requirement that experts are cached efficiently. We first present lower bounds on the competitive ratio of both deterministic and randomized algorithms, which show that under mild assumptions, LRU-like policies have good theoretical competitive performance. We then propose a layer-based extension of LRU that is tailored to the problem at hand. Extensive simulations on both synthetic datasets and actual traces of MoE usage show that our algorithm outperforms policies for the classic paging problem, such as the standard LRU.

### 12. Communication-Aware Knowledge Distillation for Federated LLM Fine-Tuning over Wireless Networks

**主要机构**: Department of Engineering, King's College London
**作者数量**: 5人

**摘要**:
Federated learning (FL) for large language models (LLMs) offers a privacy-preserving scheme, enabling clients to collaboratively fine-tune locally deployed LLMs or smaller language models (SLMs) without exchanging raw data. While parameter-sharing methods in traditional FL models solves number of technical challenges, they still incur high communication overhead and struggle with adapting to heterogeneous model architectures. Federated distillation, a framework for mutual knowledge transfer via shared logits, typically offers lower communication overhead than parameter-sharing methods. However, transmitting logits from LLMs remains challenging for bandwidth-limited clients due to their high dimensionality. In this work, we focus on a federated LLM distillation with efficient communication overhead. To achieve this, we first propose an adaptive Top-k logit selection mechanism, dynamically sparsifying logits according to real-time communication conditions. Then to tackle the dimensional inconsistency introduced by the adaptive sparsification, we design an adaptive logits aggregation scheme, effectively alleviating the artificial and uninformative inputs introduced by conventional zero-padding methods. Finally, to enhance the distillation effect, we incorporate LoRA-adapted hidden-layer projection from LLM into the distillation loss, reducing the communication overhead further while providing richer representation. Experimental results demonstrate that our scheme achieves superior performance compared to baseline methods while effectively reducing communication overhead by approximately 50%.

### 13. CSFMAMBA: CROSS STATE FUSION MAMBA OPERATOR FOR MULTIMODAL REMOTE SENSING IMAGE CLASSIFICATION

**主要机构**: Shanghai Jiao Tong University
**作者数量**: 3人

**摘要**:
Multimodal fusion has made great progress in the field of remote sensing image classification due to its ability to exploit the complementary spatial-spectral information. Deep learning methods such as CNN and Transformer have been widely used in these domains. State Space Models recently highlighted that prior methods suffer from quadratic computational complexity. As a result, modeling longer-range dependencies of spatial-spectral features imposes an overwhelming burden on the network. Mamba solves this problem by incorporating time-varying parameters into ordinary SSM and performing hardware optimization, but it cannot perform feature fusion directly. In order to make full use of Mamba's low computational burden and explore the potential of internal structure in multimodal feature fusion, we propose Cross State Fusion Mamba (CSFMamba) Network. Specifically, we first design the preprocessing module of remote sensing image information for the needs of Mamba structure, and combine it with CNN to extract multi-layer features. Secondly, a cross-state module based on Mamba operator is creatively designed to fully fuse the feature of the two modalities. The advantages of Mamba and CNN are combined by designing a more powerful backbone. We capture the fusion relationship between HSI and LiDAR modalities with stronger full-image understanding. The experimental results on two datasets of MUUFL and Houston2018 show that the proposed method outperforms the experimental results of Transformer under the premise of reducing the network training burden.

### 14. DaMoC: Efficiently Selecting the Optimal Large Language Model for Fine-tuning Domain Tasks Based on Data and Model Compression

**主要机构**: Ant Group
**作者数量**: 3人

**摘要**:
Large language models (LLMs) excel in general tasks but struggle with domain-specific ones, requiring fine-tuning with specific data. With many open-source LLMs available, selecting the best model for fine-tuning downstream tasks is challenging, primarily focusing on how to quickly identify the optimal LLM. We introduce a Data and Model Compression Framework (DaMoC) that addresses this challenge by: 1) Data Level: A systematic categorization of data filtering methodologies for LLMs is first established, classifying them into three distinct paradigms: (1) distribution-aware methods, (2) quality-aware methods, and (3) hybrid approaches considering both dimensions. Further, we enhance the density of key tokens in the text achieving token compression. Subsequently, we use an LLM to iterative rewrite the text to optimize its expression. 2) Model Level: We use layer similarity scores to assess each layer's importance and remove those with lower importance. Then, we introduce a sparse merging paradigm to preserve as much of the original model's capability as possible. Extensive experiments on four datasets, medical Q&A, financial Q&A, general Q&A, and reading comprehension, show that we can select the optimal LLM while saving approximately 20-fold in training time.

### 15. Democratizing Agentic AI with Fast Test-Time Scaling on the Edge

**主要机构**: Microsoft Research Beijing, Imperial College London London
**作者数量**: 7人

**摘要**:
Deploying agentic AI on edge devices is crucial for privacy and responsiveness, but memory constraints typically relegate these systems to smaller Large Language Models (LLMs) with inferior reasoning capabilities. Test-Time Scaling (TTS) can bridge this reasoning gap by dedicating more compute during inference, but existing methods incur prohibitive overhead on edge hardware. To overcome this, we introduce FlashTTS, a serving system that makes TTS practical for memory-constrained LLM reasoning. FlashTTS introduces three synergistic optimizations: (i) Speculative Beam Extension to mitigate system stragglers from irregular reasoning paths; (ii) Asymmetric Multi-Model Memory Allocation to dynamically balance memory between generation and verification; and (iii) Dynamic Prefix-Aware Scheduling to maximize KV-cache reuse. Built as a plug-and-play library for vLLM, FlashTTS enables edge LLMs (≤ 7B) on a single consumer GPU (24 GB) to match the accuracy and latency of large cloud models. Our evaluation demonstrates that FlashTTS achieves an average 2.2× higher goodput and reduces latency by 38%-68% compared to a vLLM baseline, paving the way for democratized, high-performance agentic AI on edge devices.

### 16. Domain Adaptation-Based Crossmodal Knowledge Distillation for 3D Semantic Segmentation

**主要机构**: Peking University, School of Intelligence Science and Technology
**作者数量**: 3人

**摘要**:
Semantic segmentation of 3D LiDAR data plays a pivotal role in autonomous driving. Traditional approaches rely on extensive annotated data for point cloud analysis, incurring high costs and time investments. In contrast, realworld image datasets offer abundant availability and substantial scale. To mitigate the burden of annotating 3D LiDAR point clouds, we propose two crossmodal knowledge distillation methods: Unsupervised Domain Adaptation Knowledge Distillation (UDAKD) and Feature and Semantic-based Knowledge Distillation (FSKD). Leveraging readily available spatio-temporally synchronized data from cameras and LiDARs in autonomous driving scenarios, we directly apply a pretrained 2D image model to unlabeled 2D data. Through crossmodal knowledge distillation with known 2D-3D correspondence, we actively align the output of the 3D network with the corresponding points of the 2D network, thereby obviating the necessity for 3D annotations. Our focus is on preserving modality-general information while filtering out modality-specific details during crossmodal distillation. To achieve this, we deploy self-calibrated convolution on 3D point clouds as the foundation of our domain adaptation module. Rigorous experimentation validates the effectiveness of our proposed methods, consistently surpassing the performance of state-of-the-art approaches in the field. Code is available at https://github.com/KangJialiang/DAKD.

### 17. DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving

**主要机构**: Cloud Research Team, Samsung SDS
**作者数量**: 5人

**摘要**:
Speculative decoding accelerates large language model inference, but its reliance on a fixed speculation length is suboptimal in large-batch serving environments with diverse requests. This paper explores a new direction for dynamic adaptation by investigating a novel class of post-hoc, diagnostic signals. We propose Dynamic Speculative Decoding Engine (DSDE), a training-free framework built on two primary components: (1) a predictive signal based on the variance of the Kullback-Leibler (KLD) divergence, which diagnoses the generation's regional stability, and (2) an adaptive speculation length cap to mitigate the straggler problem in per-sequence decoding. Experiments demonstrate the potential of using KLD-based stability signals for dynamic adaptation. An algorithm guided by these signals achieves end-to-end latency competitive with leading baselines and exhibits superior robustness across diverse workloads. This robustness is particularly valuable in challenging low-acceptancerate regimes, where the proposed signal maintains its diagnostic utility. Collectively, these findings validate post-hoc signals as a valuable component for building more robust and intelligent LLM inference systems, and highlight a promising direction for future research on dynamic speculation length adaptation.

### 18. DTRNet: Dynamic Token Routing Network to Reduce Quadratic Costs in Transformers

**主要机构**: Huawei Noah's Ark Lab, Advanced Micro Devices, Inc, University of Alberta
**作者数量**: 9人

**摘要**:
Transformers achieve state-of-the-art results across many tasks, but their uniform application of quadratic self-attention to every token at every layer makes them computationally expensive. We introduce DTRNet (Dynamic Token Routing Network), an improved Transformer architecture that allows tokens to dynamically skip the quadratic cost of cross-token mixing while still receiving lightweight linear updates. By preserving the MLP module and reducing the attention cost for most tokens to linear, DTRNet ensures that every token is explicitly updated while significantly lowering overall computation. This design offers an efficient and effective alternative to standard dense attention. Once trained, DTRNet blocks routes only 10% of tokens through attention at each layer while maintaining performance comparable to a full Transformer. It consistently outperforms routing-based layer skipping methods such as MoD and D-LLM in both accuracy and memory at matched FLOPs, while routing fewer tokens to full attention. Its efficiency gains, scales with sequence length, offering significant reduction in FLOPs for long-context inputs. By decoupling token updates from attention mixing, DTRNet substantially reduces the quadratic share of computation, providing a simple, efficient, and scalable alternative to Transformers. 1

### 19. Efficient Large Language Models with Zero-Shot Adjustable Acceleration

**主要机构**: Sharif University of Technology Tehran, Department of Electrical Engineering
**作者数量**: 2人

**摘要**:
Using Large Language Models (LLMs) in realworld applications presents significant challenges, particularly in balancing computational efficiency and performance. Optimizing acceleration after the fine-tuning phase and during inference is crucial for building an efficient architecture. This paper introduces Zero-Shot Adjustable Acceleration, a novel training and inference method that dynamically adjusts hardware usage during inference without requiring additional fine-tuning. The proposed approach is applied to newly developed models and evaluated across multiple classification and text generation tasks. Experimental results demonstrate that the method enables a wide range of acceleration in a zero-shot manner and achieves up to a 11× speedup compared to the baseline.

### 20. Efficient Pyramidal Analysis of Gigapixel Images on a Decentralized Modest Computer Cluster

**主要机构**: IP Paris, SAMOVAR, Télécom SudParis, EPFL, Inria Saclay, Télécom SudParis, SAMOVAR
**作者数量**: 6人

**摘要**:
Analyzing gigapixel images is recognized as computationally demanding. In this paper, we introduce PyramidAI, a technique for analyzing gigapixel images with reduced computational cost. The proposed approach adopts a gradual analysis of the image, beginning with lower resolutions and progressively concentrating on regions of interest for detailed examination at higher resolutions. We investigated two strategies for tuning the accuracy-computation performance trade-off when implementing the adaptive resolution selection, validated against the Camelyon 16 dataset of biomedical images. Our results demonstrate that Pyra-midAI substantially decreases the amount of processed data required for analysis by up to 2.65×, while preserving the accuracy in identifying relevant sections on a single computer. To ensure democratization of gigapixel image analysis, we evaluated the potential to use mainstream computers to perform the computation by exploiting the parallelism potential of the approach. Using a simulator, we estimated the best data distribution and load balancing algorithm according to the number of workers. The selected algorithms were implemented and highlighted the same conclusions in a real-world setting. Analysis time is reduced from more than an hour to a few minutes using 12 modest workers, offering a practical solution for efficient large-scale image analysis.

### 21. Encoder-Only Image Registration

**主要机构**: 
**作者数量**: 8人

**摘要**:
Learning-based techniques have significantly improved the accuracy and speed of deformable image registration. However, challenges such as reducing computational complexity and handling large deformations persist. To address these challenges, we analyze how convolutional neural networks (ConvNets) influence registration performance using the Horn-Schunck optical flow equation. Supported by prior studies and our empirical experiments, we observe that ConvNets play two key roles in registration: linearizing local intensities and harmonizing global contrast variations. Based on these insights, we propose the Encoder-Only Image Registration (EOIR) framework, designed to achieve a better accuracy-efficiency trade-off. EOIR separates feature learning from flow estimation, employing only a 3-layer ConvNet for feature extraction and a set of 3-layer flow estimators to construct a Laplacian feature pyramid, progressively composing diffeomorphic deformations under a large-deformation model. Results on five datasets across different modalities and anatomical regions demonstrate EOIR's effectiveness, achieving superior accuracy-efficiency and accuracy-smoothness trade-offs. With comparable accuracy, EOIR provides better efficiency and smoothness, and vice versa. The source code of EOIR is publicly available on Github.

### 22. Energy Efficient Exact and Approximate Systolic Array Architecture for Matrix Multiplication

**主要机构**: Indian Institute of Technology
**作者数量**: 3人

**摘要**:
Deep Neural Networks (DNNs) require highly efficient matrix multiplication engines for complex computations. This paper presents a systolic array architecture incorporating novel exact and approximate processing elements (PEs), designed using energy-efficient positive partial product and negative partial product cells, termed as PPC and NPPC, respectively. The proposed 8-bit exact and approximate PE designs are employed in a 8x8 systolic array, they achieve energy savings of 22% and 32%, respectively, compared to the existing design. To demonstrate their effectiveness, the proposed PEs are integrated into a systolic array (SA) for Discrete Cosine Transform (DCT) computation, achieving high output quality with a PSNR of 38.21 dB. Furthermore, in an edge detection application using convolution, the approximate PE achieves a PSNR of 30.45 dB. These results highlight the potential of the proposed design to deliver significant energy efficiency while maintaining competitive output quality, making it well-suited for error-resilient image and vision processing applications.

### 23. Entropy-based Coarse and Compressed Semantic Speech Representation Learning

**主要机构**: Zhejiang University
**作者数量**: 8人

**摘要**:
Discrete speech representation learning has recently attracted increasing interest in both acoustic and semantic modeling. Existing approaches typically encode 16 kHz waveforms into discrete tokens at a rate of 25-50 tokens per second. However, given that speech generally conveys only 2-5 words per second, such fine-grained tokenization introduces redundancy and hinders efficiency in downstream training and inference. Moreover, semantic speech representations at this frequency primarily capture phonetic-level information, while semantic understanding may not require such detailed token-level resolution. To address these limitations, we propose an entropy-based dynamic aggregation framework for learning compressed semantic speech representations. A speech language model is first pre-trained via next-token prediction on large-scale unlabeled data to capture frequent token patterns. Predictive entropy is then used to adaptively determine aggregation boundaries, followed by a cross-attention module that fuses information within each segment. By adjusting the entropy threshold, the granularity and compression ratio of the representations can be flexibly controlled. Experiments on ASR, speech-to-text translation, and voice conversion tasks demonstrate that the compressed representations perform on par with or better than dense token sequences, demonstrating the effectiveness of the proposed approach.

### 24. Faster and Better: Reinforced Collaborative Distillation and Self-Learning for Infrared-Visible Image Fusion

**主要机构**: Beijing Institute of Technology, School of Automation
**作者数量**: 5人

**摘要**:
Infrared and visible image fusion plays a critical role in enhancing scene perception by combining complementary information from different modalities. Despite recent advances, achieving high-quality image fusion with lightweight models remains a significant challenge. To bridge this gap, we propose a novel collaborative distillation and self-learning framework for image fusion driven by reinforcement learning. Unlike conventional distillation, this approach not only enables the student model to absorb image fusion knowledge from the teacher model, but more importantly, allows the student to perform self-learning on more challenging samples to enhance its capabilities. Particularly, in our framework, a reinforcement learning agent explores and identifies a more suitable training strategy for the student. The agent takes both the student's performance and the teacher-student gap as inputs, which leads to the generation of challenging samples to facilitate the student's self-learning. Simultaneously, it dynamically adjusts the teacher's guidance strength based on the student's state to optimize the knowledge transfer. Experimental results demonstrate that our method can significantly improve student performance and achieve better fusion results compared to existing techniques.

### 25. FastVGGT: Training-Free Acceleration of Visual Geometry Transformer

**主要机构**: Ministry of Education of China, Unmerge VGGT： FastVGGT：, School of Artificial Intelligence, AutoLab, Shanghai Jiao Tong University, Xiamen University, Key Laboratory of Multimedia Trusted Perception and Efficient Computing
**作者数量**: 4人

**摘要**:


### 26. From TLinFormer to TConstFormer: The Leap to Constant-Time Transformer Attention Achieving O(1) Computation and O(1) KV Cache during Autoregressive Inference

**主要机构**: 
**作者数量**: 1人

**摘要**:
Although the Transformer has become the cornerstone of modern AI, its autoregressive inference suffers from a linearly growing KV Cache and a computational complexity of O(N 2 d), severely hindering its ability to process ultra-long sequences. To overcome this limitation, this paper introduces the TConstFormer architecture, building upon our previous work, TLinFormer. TConstFormer employs an innovative periodic state update mechanism to achieve a truly constant-size O(1) KV Cache. The computational complexity of this mechanism is also O(1) in an amortized sense: it performs purely constant-time computations for k-1 consecutive steps (e.g., k = 256) and executes a single linear-time global information synchronization only on the k-th step. Theoretical calculations and experimental results demonstrate that TConstFormer exhibits an overwhelming advantage over baseline models in terms of speed, memory efficiency, and overall performance on long-text inference tasks. This breakthrough paves the way for efficient and robust streaming language model applications.

### 27. Gated Associative Memory: A Parallel O(N) Architecture for Efficient Sequence Modeling

**主要机构**: Independent Researcher
**作者数量**: 1人

**摘要**:
The Transformer architecture, underpinned by the self-attention mechanism, has become the de facto standard for sequence modeling tasks. However, its core computational primitive scales quadratically with sequence length (O(N 2)), creating a significant bottleneck for processing long contexts. In this paper, we propose the Gated Associative Memory (GAM) network, a novel, fully parallel architecture for sequence modeling that exhibits linear complexity (O(N)) with respect to sequence length. The GAM block replaces the self-attention layer with two parallel pathways: a causal convolution to efficiently capture local, position-dependent context, and a parallel associative memory retrieval mechanism to model global, content-based patterns. These pathways are dynamically fused using a gating mechanism, allowing the model to flexibly combine local and global information for each token. We implement GAM from scratch and conduct a rigorous comparative analysis against a standard Transformer model and a modern linear-time baseline (Mamba) on the WikiText-2 benchmark, as well as against the Transformer on the TinyStories dataset. Our experiments demonstrate that GAM is consistently faster, outperforming both baselines on training speed, and achieves a superior or competitive final validation perplexity across all datasets, establishing it as a promising and efficient alternative for sequence modeling.

### 28. GraphKV: Breaking the Static Selection Paradigm with Graph-Based KV Cache Eviction

**主要机构**: Shanghai Jiao Tong University, EPIC Lab
**作者数量**: 3人

**摘要**:
Efficient Key-Value (KV) cache management is essential for processing long text sequences in large language models (LLMs), where memory constraints often limit performance. Conventional KV eviction strategies, such as top-k selection based on attention scores, depend on static heuristics that fail to capture the evolving implicit dependencies among tokens during inference. To overcome this, we propose GraphKV, a graph-based framework that redefines token selection for KV cache compression. In GraphKV, tokens are modeled as nodes with importance scores, and edges represent their similarity relationships. Through a decaysignal-propagation mechanism, token importance is dynamically updated by propagating information across the graph, enabling adaptive retention of the most contextually significant tokens. GraphKV can be seamlessly utilized in existing KV cache eviction methods such as SnapKV and PyramidKV in a plug-and-play manner. Codes will be released on Github.

### 29. Guidance and Control Neural Network Acceleration using Memristors

**主要机构**: ESA Advanced Concepts Team, Delft University of Technology
**作者数量**: 6人

**摘要**:
In recent years, the space community has been exploring the possibilities of Artificial Intelligence (AI), specifically Artificial Neural Networks (ANNs), for a variety of on board applications. However, this development is limited by the restricted energy budget of smallsats and cubesats as well as radiation concerns plaguing modern chips. This necessitates research into neural network accelerators capable of meeting these requirements whilst satisfying the compute and performance needs of the application. This paper explores the use of Phase-Change Memory (PCM) and Resistive Random-Access Memory (RRAM) memristors for on-board in-memory computing AI acceleration in space applications. A guidance and control neural network (G&CNET) accelerated using memristors is simulated in a variety of scenarios and with both device types to evaluate the performance of memristorbased accelerators, considering device non-idealities such as noise and conductance drift. We show that the memristive accelerator is able to learn the expert actions, though challenges remain with the impact of noise on accuracy. We also show that retraining after degradation is able to restore performance to nominal levels. This study provides a foundation for future research into memristorbased AI accelerators for space, highlighting their potential and the need for further investigation.

### 30. Guided Model-based LiDAR Super-Resolution for Resource-Efficient Automotive scene Segmentation

**主要机构**: Athena Research Center, Industrial Systems Institute
**作者数量**: 3人

**摘要**:
High-resolution LiDAR data is essential for 3D semantic segmentation in autonomous driving. However, the high cost of advanced LiDAR sensors limits their widespread adoption. Affordable alternatives like 16-channel LiDAR generate sparse point clouds, resulting in reduced segmentation performance. To address this, we present the first end-to-end framework that integrates LiDAR super-resolution (SR) and segmentation tasks. This framework achieves context awareness through a novel joint optimization during training, ensuring that the SR process integrates semantic guidance from the segmentation task and preserves the details of smaller classes. Furthermore, the proposed framework incorporates a novel SR loss function to enhance the network's focus on regions of interest. The lightweight model-based SR network achieves a significant reduction in number of parameters compared to state-of-the-art LiDAR SR models, enabling efficient integration with segmentation networks. Experimental results demonstrate that our method delivers segmentation performance comparable to networks using high-resolution and high-cost 64channel LiDAR data.

### 31. ENHANCING IMAGE QUALITY AND ANOMALY DETECTION FOR SMALL AND DENSE INDUSTRIAL OBJECTS IN NUCLEAR RECYCLING

**主要机构**: LGF, Siléane Group, Mines Saint-Etienne, Orano Group, Univ Lyon, Centre SPIN, CNRS, UMR 5307
**作者数量**: 7人

**摘要**:
This paper tackles two key challenges: detecting small, dense, and overlapping objects (a major hurdle in computer vision) and improving the quality of noisy images, especially those encountered in industrial environments. [1, 2]. Our focus is on evaluating methods built on supervised deep learning. We perform an analysis of these methods, using a newly developed dataset comprising over 10k images and 120k instances. By evaluating their performance, accuracy, and computational efficiency, we identify the most reliable detection systems and highlight the specific challenges they address in industrial applications. This paper also examines the use of deep learning models to improve image quality in noisy industrial environments. We introduce a lightweight model based on a fully connected convolutional network. Additionally, we suggest potential future directions for further enhancing the effectiveness of the model.

### 32. Knowledge distillation as a pathway toward next-generation intelligent ecohydrological modeling systems

**主要机构**: The University of Hong Kong, The University of Arizona, Department of Civil Engineering, University of Washington, Thornwell Labs, Department of Hydrology and Atmospheric Sciences, Puget Sound Institute
**作者数量**: 6人

**摘要**:
Simulating ecohydrological processes is essential for understanding complex environmental systems and guiding sustainable management amid accelerating climate change and human pressures. Processbased models provide physical realism but can suffer from structural rigidity, high computational costs, and complex calibration, while machine learning (ML) methods are efficient and flexible yet often lack interpretability and transferability. We propose a unified three-phase framework that integrates processbased models with ML and progressively embeds them into artificial intelligence (AI) through knowledge distillation. Phase I, behavioral distillation, enhances process models via surrogate learning and model simplification to capture key dynamics at lower computational cost. Phase II, structural distillation, reformulates process equations as modular components within a graph neural network (GNN), enabling multiscale representation and seamless integration with ML models. Phase III, cognitive distillation, embeds expert reasoning and adaptive decision-making into intelligent modeling agents using the Eyes-Brain-Hands-Mouth architecture. Demonstrations for the Samish watershed highlight the framework's applicability to ecohydrological modeling, showing that it can reproduce process-based model outputs, improve predictive accuracy, and support scenario-based decision-making. The framework offers a scalable and transferable pathway toward next-generation intelligent ecohydrological modeling systems, with the potential extension to other process-based domains.

### 33. KVComp: A High-Performance, LLM-Aware, Lossy Compression Framework for KV Cache

**主要机构**: University of Houston Houston, Temple University Philadelphia, Xubin He Temple University Philadelphia
**作者数量**: 11人

**摘要**:
Transformer-based large language models (LLMs) demonstrate impressive potential in various practical applications. However, long context inference poses a significant challenge due to the enormous memory requirements of the key-value (KV) cache, which can scale to multiple gigabytes as sequence length and batch size increase. In this paper, we present KVComp, a generic and efficient KV cache management framework optimized for long-text generation that synergistically works with both latency-critical and throughput-critical inference systems. KVComp employs novel lossy compression techniques specifically designed for KV cache data characteristics, featuring careful co-design of compression algorithms and system architecture. Our approach maintains compatibility with the growing nature of KV cache while preserving high computational efficiency. Experimental results show that KV-Comp achieves on average 47% and up to 83% higher memory reduction rate compared to existing methods with little/no model accuracy degradation. Furthermore, KVComp achieves extremely high execution throughput, effectively reducing decompression overhead and, in some cases, even accelerating the matrix-vector multiplication operation and outperform cuBLAS-based attention kernels with less data movement.

### 34. LatentEdit: Adaptive Latent Control for Consistent Semantic Editing

**主要机构**: Southern University of Science and Technology
**作者数量**: 1人

**摘要**:
Diffusion-based Image Editing has achieved significant success in recent years. However, it remains challenging to achieve highquality image editing while maintaining the background similarity without sacrificing speed or memory efficiency. In this work, we introduce La-tentEdit, an adaptive latent fusion framework that dynamically combines the current latent code with a reference latent code inverted from the source image. By selectively preserving source features in high-similarity, semantically important regions while generating target content in other regions guided by the target prompt, LatentEdit enables fine-grained, controllable editing. Critically, the method requires no internal model modifications or complex attention mechanisms, offering a lightweight, plug-and-play solution compatible with both UNet-based and DiT-based architectures. Extensive experiments on the PIE-Bench dataset demonstrate that our proposed LatentEdit achieves an optimal balance between fidelity and editability, outperforming the state-of-the-art method even in 8-15 steps. Additionally, its inversion-free variant further halves the number of neural function evaluations and eliminates the need for storing any intermediate variables, substantially enhancing real-time deployment efficiency.

### 35. Learning to Shard: RL for Co-optimizing the Parallelism Degrees and Per-operator Sharding Dimensions in Distributed LLM Inference

**主要机构**: Yale University, Microsoft Azure
**作者数量**: 6人

**摘要**:
Distributed LLM inference requires careful coordination of parallelization strategies across hundreds to thousands of NPUs to meet production SLOs. Current systems like Megatron-LM rely on static heuristics that separately configure parallelism degrees and per-operator sharding dimensions, leaving significant performance on the table as models scale and hardware topologies diversify. We introduce Learn to Shard, to our knowledge, the first RL-based approach to co-optimize both coarse-grained parallelism degrees and fine-grained per-operator sharding dimensions for distributed LLM inference. Our method employs an attention-based policy over an elite history that learns from high-performing strategies to efficiently navigate the vast combinatorial search space. Evaluated on H100 clusters with MoE models up to 1.6T parameters, Learn to Shard achieves up to 3.5× throughput improvement over metaheuristic baselines and 1.06× over Megatron heuristics.

### 36. LightVLM: Acceleraing Large Multimodal Models with Pyramid Token Merging and KV Cache Compression

**主要机构**: College of Intelligence and Computing, Tianjin University
**作者数量**: 4人

**摘要**:
In this paper, we introduce LightVLM, a simple but effective method that can be seamlessly deployed upon existing Vision-Language Models (VLMs) to greatly accelerate the inference process in a training-free manner. We divide the inference procedure of VLMs into two stages, i.e., encoding and decoding, and propose to simultaneously accelerate VLMs in both stages to largely improve model efficiency. During encoding, we propose pyramid token merging to reduce tokens of different LLM layers in a hierarchical manner by finally only keeping a few dominant tokens to achieve high efficiency. During decoding, aimed at reducing the high latency of outputting long sequences, we propose KV Cache compression to remove unnecessary caches to increase the network throughput. Experimental results show that LightVLM successfully retains 100% performance when only preserving 35% image tokens, and maintains around 98% performance when keeping only 3% image tokens. LightVLM could 2.02× the network throughput and reduce the prefilling time by 3.65×. LightVLM also makes large VLMs faster again by enabling a heavy model (e.g., InternVL2.5 26B) to infer faster than significantly smaller models (e.g., InternVL2.5 8B), hopefully facilitating the real-world deployment. When generating long text sequences (e.g., 4096 tokens), LightVLM could reduce the inference time by 3.21×, largely outperforming existing methods.

### 37. LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

**主要机构**: ByteDance Seed Seattle, ByteDance Seed Shanghai, Shanghai Jiao Tong University Shanghai, ByteDance Seed Beijing
**作者数量**: 20人

**摘要**:
Quantization is a critical technique for accelerating LLM inference by reducing memory footprint and improving computational efficiency. Among various schemes, 4-bit weight and 8-bit activation quantization (W4A8) offers a strong balance between accuracy and performance. However, existing W4A8 GEMM kernels fall short in practice due to inefficient dequantization on CUDA Cores, which cannot keep pace with the high throughput of Tensor Cores. In this paper, we present LiquidGEMM, a hardware-efficient W4A8 GEMM kernel for efficient LLM serving. LiquidGEMM designs two key techniques: LiquidQuant, a hardware-efficient quantization method that enables fast, overflow-safe dequantization using just two arithmetic instructions per four elements; and an implicit finegrained pipeline that fully overlaps weight loading, dequantization, and MMA across warp groups without software synchronization or redundant memory traffic. Experimental results show that Liq-uidGEMM achieves up to 2.90x speedup over state-of-the-art W4A8 kernels and up to 4.94x end-to-end system-level speedup. Compared to various quantized GEMM kernels in NVIDIA TensorRT-LLM, LiquidGEMM delivers 1.12-1.63x performance gains, and achieves up to 1.63x system-level speedup.

### 38. 

**主要机构**: 
**作者数量**: 0人

**摘要**:
We introduce LongCat-Flash, a 560-billion-parameter Mixture-of-Experts (MoE) language model designed for both computational efficiency and advanced agentic capabilities. Stemming from the need for scalable efficiency, LongCat-Flash adopts two novel designs: (a) Zero-computation Experts, which enables dynamic computational budget allocation and activates 18.6B-31.3B (27B on average) per token depending on contextual demands, optimizing resource usage. (b) Shortcut-connected MoE, which enlarges the computation-communication overlap window, demonstrating notable gains in inference efficiency and throughput compared to models of a comparable scale. We develop a comprehensive scaling framework for large models that combines hyperparameter transfer, modelgrowth initialization, a multi-pronged stability suite, and deterministic computation to achieve stable and reproducible training. Notably, leveraging the synergy among scalable architectural design and infrastructure efforts, we complete model training on more than 20 trillion tokens within 30 days, while achieving over 100 tokens per second (TPS) for inference at a cost of $0.70 per million output tokens. To cultivate LongCat-Flash towards agentic intelligence, we conduct a large-scale pre-training on optimized mixtures, followed by targeted mid-and post-training on reasoning, code, and instructions, with further augmentation from synthetic data and tool use tasks. Comprehensive evaluations demonstrate that, as a non-thinking foundation model, LongCat-Flash delivers highly competitive performance among other leading models, with exceptional strengths in agentic tasks. The model checkpoint of LongCat-Flash is open-sourced to foster community research.

### 39. LUT-Fuse: Towards Extremely Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up Tables

**主要机构**: Electronic Information School, Wuhan University, Southeast University, School of Automation
**作者数量**: 6人

**摘要**:
Current advanced research on infrared and visible image fusion primarily focuses on improving fusion performance, often neglecting the applicability on real-time fusion devices. In this paper, we propose a novel approach that towards extremely fast fusion via distillation to learnable lookup tables specifically designed for image fusion, termed as LUT-Fuse. Firstly, we develop a look-up table structure that utilizing low-order approximation encoding and highlevel joint contextual scene encoding, which is well-suited for multi-modal fusion. Moreover, given the lack of ground truth in multi-modal image fusion, we naturally proposed the efficient LUT distillation strategy instead of traditional quantization LUT methods. By integrating the performance of the multi-modal fusion network (MM-Net) into the MM-LUT model, our method achieves significant breakthroughs in efficiency and performance. It typically requires less than one-tenth of the time compared to the current lightweight SOTA fusion algorithms, ensuring high operational speed across various scenarios, even in low-power mobile devices. Extensive experiments validate the superiority, reliability, and stability of our fusion approach. The code is available at https://github.com/zyb5/LUT-Fuse.

### 40. MAMBA-CNN: A HYBRID ARCHITECTURE FOR EFFICIENT AND ACCURATE FACIAL BEAUTY PREDICTION

**主要机构**: Scientific and Technical Research Centre for Arid Areas
**作者数量**: 1人

**摘要**:
The computational assessment of facial attractiveness, a challenging subjective regression task, is dominated by architectures with a critical trade-off: Convolutional Neural Networks (CNNs) offer efficiency but have limited receptive fields, while Vision Transformers (ViTs) model global context at a quadratic computational cost. To address this, we propose Mamba-CNN, a novel and efficient hybrid architecture. Mamba-CNN integrates a lightweight, Mamba-inspired State Space Model (SSM) gating mechanism into a hierarchical convolutional backbone. This core innovation allows the network to dynamically modulate feature maps and selectively emphasize salient facial features and their long-range spatial relationships, mirroring human holistic perception while maintaining computational efficiency. We conducted extensive experiments on the widely-used SCUT-FBP5500 benchmark, where our model sets a new state-of-the-art. Mamba-CNN achieves a Pearson Correlation (PC) of 0.9187, a Mean Absolute Error (MAE) of 0.2022, and a Root Mean Square Error (RMSE) of 0.2610. Our findings validate the synergistic potential of combining CNNs with selective SSMs and present a powerful new architectural paradigm for nuanced visual understanding tasks.

### 41. 

**主要机构**: 
**作者数量**: 0人

**摘要**:


### 42. MoPEQ: Mixture of Mixed Precision Quantized Experts

**主要机构**: Illinois Institute of Technology, Argonne National Laboratory
**作者数量**: 3人

**摘要**:
Large Language and Vision Models using a Mixture-of-Experts (MoE) architecture pose significant challenges for deployment due to their computational and memory demands. Mixed Precision Quantization assigns different precisions to different layers of an LLM/VLM based on layer sensitivity and importance within the model. In this work, we propose a Post Training Quantization algorithm, MoPEQ, that assigns optimal bit width to each expert. Our method balances accuracy and model size by analyzing each expert's sensitivity using Hessian trace approximation instead of relying on the activation frequency of the expert. This per-expert granularity approach clusters similar experts to maintain model performance while reducing memory requirements. The experimental results on VLMEvalKit benchmark datasets using State-of-the-art VLMs Deepseek-VL2-tiny,-small,-base, and MolmoE models demonstrate that our mixed precision quantized MoEs achieve competitive accuracy with substantial improvements in memory footprint compared to uniform-precision baseline methods. We perform a comprehensive study to analyze the impact of expert activation frequency and sensitivity using Hessian trace approximation at both layer-wise and model-wide expert precision allocation of 2, 3, and 4 bits to provide a thorough understanding of mixed precision quantization of VLM-MoEs. The code is available here.

### 43. OmniReason: A Temporal-Guided Vision-Language-Action Framework for Autonomous Driving

**主要机构**: Li Auto Inc, The Hong Kong University of Science and Technology
**作者数量**: 9人

**摘要**:
Recent advances in vision-language models (VLMs) have demonstrated impressive spatial reasoning capabilities for autonomous driving, yet existing methods predominantly focus on static scene understanding while neglecting the essential temporal dimension of real-world driving scenarios. To address this critical limitation, we propose the Om-niReason framework, which establishes robust spatiotemporal reasoning by jointly modeling dynamic 3D environments and their underlying decision-making processes. Our work makes two fundamental advances: (1) We introduce OmniReason-Data, two large-scale vision-language-action (VLA) datasets with dense spatiotemporal annotations and natural language explanations, generated through a novel hallucination-mitigated auto-labeling pipeline that ensures both physical plausibility and temporal coherence; (2) We develop the OmniReason-Agent architecture, which integrates a sparse temporal memory module for persistent scene context modeling and an explanation generator that produces human-interpretable decision rationales, facilitated by our spatiotemporal knowledge distillation approach that effectively captures spatiotemporal causal reasoning patterns. Comprehensive experiments demonstrate state-of-the-art performance, where OmniReason-Agent achieves significant improvements in both open-loop planning tasks and visual question answering (VQA) benchmarks, while establishing new capabilities for interpretable, temporally-aware autonomous vehicles operating in complex, dynamic environments.

### 44. Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption

**主要机构**: CNRS
**作者数量**: 12人

**摘要**:
In contemporary cloud-based services, protecting users' sensitive data and ensuring the confidentiality of the server's model are critical. Fully homomorphic encryption (FHE) enables inference directly on encrypted inputs, but its practicality is hindered by expensive bootstrapping and inefficient approximations of non-linear activations. We introduce SAFHIRE, a hybrid inference framework that executes linear layers under encryption on the server while offloading non-linearities to the client in plaintext. This design eliminates bootstrapping, supports exact activations, and significantly reduces computation. To safeguard model confidentiality despite client access to intermediate outputs, SAFHIRE applies randomized shuffling, which obfuscates intermediate values and makes it practically impossible to reconstruct the model. To further reduce latency, SAFHIRE incorporates advanced optimizations such as fast ciphertext packing and partial extraction. Evaluations on multiple standard models and datasets show that SAFHIRE achieves 1.5×-10.5× lower inference latency than ORION, a state-of-the-art baseline, with manageable communication overhead and comparable accuracy, thereby establishing the practicality of hybrid FHE inference.

### 45. Principled Approximation Methods for Efficient and Scalable Deep Learning

**主要机构**: TOYOTA TECHNOLOGICAL, INSTITUTE AT CHICAGO Chicago
**作者数量**: 1人

**摘要**:
Recent progress in deep learning has been driven by increasingly larger models. However, their computational and energy demands have grown proportionally, creating significant barriers to their deployment and to a wider adoption of deep learning technologies. This thesis investigates principled approximation methods for improving the efficiency of deep learning systems, with a particular focus on settings that involve discrete constraints and non-differentiability. We study three main approaches toward improved efficiency: architecture design, model compression, and optimization. For model compression, we propose novel approximations for pruning and quantization that frame the underlying discrete problem as continuous and differentiable, enabling gradient-based training of compression schemes alongside the model's parameters. These approximations allow for fine-grained sparsity and precision configurations, leading to highly compact models without significant fine-tuning. In the context of architecture design, we design an algorithm for neural architecture search that leverages parameter sharing across layers to efficiently explore implicitly recurrent architectures. Finally, we study adaptive optimization, revisiting theoretical properties of widely used methods and proposing an adaptive optimizer that allows for quick hyperparameter tuning. Our contributions center on tackling computationally hard problems via scalable and principled approximations. Experimental results on image classification, language modeling, and generative modeling tasks show that the proposed methods provide significant improvements in terms of training and inference efficiency while maintaining, or even improving, the model's performance.

### 46. Progressive Element-wise Gradient Estimation for Neural Network Quantization

**主要机构**: Oakland University
**作者数量**: 1人

**摘要**:
Neural network quantization aims to reduce the bit-widths of weights and activations, making it a critical technique for deploying deep neural networks on resource-constrained hardware. Most Quantization-Aware Training (QAT) methods rely on the Straight-Through Estimator (STE) to address the non-differentiability of discretization functions by replacing their derivatives with that of the identity function. While effective, STE overlooks discretization errors between continuous and quantized values, which can lead to accuracy degradation-especially at extremely low bitwidths. In this paper, we propose Progressive Elementwise Gradient Estimation (PEGE), a simple yet effective alternative to STE, which can be seamlessly integrated with any forward propagation methods and improves the quantized model accuracy. PEGE progressively replaces full-precision weights and activations with their quantized counterparts via a novel logarithmic curriculum-driven mixed-precision replacement strategy. Then it formulates QAT as a co-optimization problem that simultaneously minimizes the task loss for prediction and the discretization error for quantization, providing a unified and generalizable framework. Extensive experiments on CIFAR-10 and ImageNet across various architectures (e.g., ResNet, VGG) demonstrate that PEGE consistently outperforms existing backpropagation methods and enables low-precision models to match or even outperform the accuracy of their fullprecision counterparts.

### 47. Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs

**主要机构**: Case Western Reserve University
**作者数量**: 7人

**摘要**:
Neural network pruning has emerged as a promising approach for deploying LLMs in low-resource scenarios while preserving downstream task performance. However, for the first time, we reveal that such pruning disrupts LLMs' internal activation features crucial for lie detection, where probing classifiers (typically small logistic regression models) trained on these features assess the truthfulness of LLM-generated statements. This discovery raises a crucial open question: how can we prune LLMs without sacrificing these critical lie detection capabilities? Our investigation further reveals that naively adjusting layer-wise pruning sparsity based on importance inadvertently removes crucial weights, failing to improve lie detection performance despite its reliance on the most crucial LLM layer. To address this issue, we propose Truthful Pruning aligned by Layer-wise Outliers (TPLO), which places greater emphasis on layers with more activation outliers and stronger discriminative features simultaneously. This preserves LLMs' original performance while retaining critical features of inner states needed for robust lie detection. Moreover, we introduce a prompting rule to enrich the TruthfulQA benchmark for better calibrating LLM pruning. Empirical results show that our approach improves the hallucination detection 1 for pruned LLMs (achieving 88% accuracy at 50% sparsity) and enhances their performance on TruthfulQA. Codes and data are available here.

### 48. Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling

**主要机构**: The University of Texas at Austin
**作者数量**: 2人

**摘要**:
Q-Sched introduces a quantization-aware noise scheduler to few-step diffusion backbones and achieves excellent image fidelity. We find quantization and few-step diffusions to be complementary model compression strategies.

### 49. Quantization Meets OOD: Generalizable Quantization-aware Training from a Flatness Perspective

**主要机构**: Department of Computer Science and Technology, MMLab, Ministry of Education Department of Computer Science and Technology, SIGS, Tsinghua University Shenzhen, The Chinese University of Hong Kong Hong Kong, Key Laboratory of Pervasive Computing, Tsinghua University Beijing
**作者数量**: 14人

**摘要**:
Current quantization-aware training (QAT) methods primarily focus on enhancing the performance of quantized models on indistribution (I.D) data, while overlooking the potential performance degradation on out-of-distribution (OOD) data. In this paper, we first substantiate this problem through rigorous experiment, showing that QAT can lead to a significant OOD generalization performance degradation. Further, we find the contradiction between the perspective that flatness of loss landscape gives rise to superior OOD generalization and the phenomenon that QAT lead to a sharp loss landscape, can cause the above problem. Therefore, we propose a flatness-oriented QAT method, FQAT, to achieve generalizable QAT. Specifically, i) FQAT introduces a layer-wise freezing mechanism to mitigate the gradient conflict issue between dual optimization objectives (i.e., vanilla QAT and flatness). ii) FQAT proposes an disorder-guided adaptive freezing algorithm to dynamically determines which layers to freeze at each training step, effectively addressing the challenges caused by interference between layers. A gradient disorder metric is designed to help the algorithm identify unstable layers during training. Extensive experiments on influential OOD benchmark demonstrate the superiority of our method † Corresponding author.

### 50. Quantum-Optimized Selective State Space Model for Efficient Time Series Prediction

**主要机构**: Politehnica University Timis ¸oara Timis ¸oara, Department of Computer and Information Technology
**作者数量**: 3人

**摘要**:
Long-range time series forecasting remains challenging, as it requires capturing non-stationary and multiscale temporal dependencies while maintaining noise robustness, efficiency, and stability. Transformer-based architectures such as Autoformer and Informer improve generalization but suffer from quadratic complexity and degraded performance on very long time horizons. State space models, notably S-Mamba, provide linear-time updates but often face unstable training dynamics, sensitivity to initialization, and limited robustness for multivariate forecasting. To address such challenges, we propose the Quantum-Optimized Selective State Space Model (Q-SSM), a hybrid quantum-optimized approach that integrates state space dynamics with a variational quantum gate. Instead of relying on expensive attention mechanisms, Q-SSM employs a simple parametrized quantum circuit (RY-RX ansatz) whose expectation values regulate memory updates adaptively. This quantum gating mechanism improves convergence stability, enhances the modeling of long-term dependencies, and provides a lightweight alternative to attention. We empirically validate Q-SSM on three widely used benchmarks, i.e., ETT, Traffic, and Exchange Rate. Results show that Q-SSM consistently improves over strong baselines (LSTM, TCN, Reformer), Transformer-based models, and S-Mamba. These findings demonstrate that variational quantum gating can address current limitations in long-range forecasting, leading to accurate and robust multivariate predictions.

### 51. RDIT: Residual-based Diffusion Implicit Models for Probabilistic Time Series Forecasting

**主要机构**: Dept. of EECS, Harvard University Cambridge, MIT Cambridge
**作者数量**: 6人

**摘要**:
Probabilistic Time Series Forecasting (PTSF) plays a critical role in domains requiring accurate and uncertainty-aware predictions for decision-making. However, existing methods offer suboptimal distribution modeling and suffer from a mismatch between training and evaluation metrics. Surprisingly, we found that augmenting a strong point estimator with a zero-mean Gaussian, whose standard deviation matches its training error, can yield state-of-the-art performance in PTSF. In this work, we propose RDIT, a plug-and-play framework that combines point estimation and residual-based conditional diffusion with a bidirectional Mamba network. We theoretically prove that the Continuous Ranked Probability Score (CRPS) can be minimized by adjusting to an optimal standard deviation and then derive algorithms to achieve distribution matching. Evaluations on eight multivariate datasets across varied forecasting horizons demonstrate that RDIT achieves lower CRPS, rapid inference, and improved coverage compared to strong baselines.

### 52. Scale, Don't Fine-tune: Guiding Multimodal LLMs for Efficient Visual Place Recognition at Test-Time ⋆

**主要机构**: Hong Kong University of Science and Technology, Normal University, South China, University of Science and Technology Beijing, Shenzhen Technology University
**作者数量**: 11人

**摘要**:
Visual Place Recognition (VPR) has evolved from handcrafted descriptors to deep learning approaches, yet significant challenges remain. Current approaches, including Vision Foundation Models (VFMs) and Multimodal Large Language Models (MLLMs), enhance semantic understanding but suffer from high computational overhead and limited cross-domain transferability when fine-tuned. To address these limitations, we propose a novel zero-shot framework employing Test-Time Scaling (TTS) that leverages MLLMs' vision-language alignment capabilities through Guidance-based methods for direct similarity scoring. Our approach eliminates two-stage processing by employing structured prompts that generate length-controllable JSON outputs. The TTS framework with Uncertainty-Aware Self-Consistency (UASC) enables real-time adaptation without additional training costs, achieving superior generalization across diverse environments. Experimental results demonstrate significant improvements in crossdomain VPR performance with up to 210× computational efficiency gains.

### 53. Scaling Legal AI: Benchmarking Mamba and Transformers for Statutory Classification and Case Law Retrieval

**主要机构**: 
**作者数量**: 1人

**摘要**:
The rapid growth of statutory corpora and judicial decisions requires scalable legal AI systems capable of classification and retrieval over extremely long contexts. Transformerbased architectures (e.g., Longformer, DeBERTa) dominate current legal NLP benchmarks but struggle with quadratic attention costs, limiting efficiency and scalability. In this work, we present the first comprehensive benchmarking of Mamba, a state-space model (SSM) with linear-time selective mechanisms, against leading transformer models for statutory classification and case law retrieval. We evaluate models on open-source legal corpora including LexGLUE, EUR-Lex, and ILDC, covering statutory tagging, judicial outcome prediction, and case retrieval tasks. Metrics include accuracy, recall@k, mean reciprocal rank (MRR), and NDCG, alongside throughput (tokens/sec) and maximum context length. Results show that Mamba's linear scaling enables processing of legal documents several times longer than transformers, while maintaining or surpassing retrieval and classification performance. This study introduces a new legal NLP benchmark suite for long-context modeling, open source code and datasets to support reproducibility. Our findings highlight trade-offs between statespace models and transformers, providing guidance for deploying scalable legal AI in statutory analysis, judicial decision support, and policy research.

### 54. SCOUT: Toward Sub-Quadratic Attention via Segment Compression for Optimized Utility in Transformers

**主要机构**: Huawei Noah's Ark Lab
**作者数量**: 6人

**摘要**:
Transformers have demonstrated strong performance across a wide range of sequence modeling tasks, but their quadratic attention complexity limits scalability to long sequences. Linear models such as Mamba and sliding-window attention (SWA) address this by mixing tokens through recurrent or localized operations with fixed-size memory, achieving efficient inference. However, these methods risk degrading performance on long sequences due to their inability to retain detailed information from distant tokens. We propose SCOUT (Segment Compression for Optimized Utility in Transformers), a hybrid architecture that compresses tokens locally within fixed-size segments and applies attention only over these compressed representations. Each token embedding is first enriched via a linear local mixer, Mamba or SWA, that integrates recent context. Then, instead of attending to all previous tokens, each token sparsely attends to a small number of compressed checkpoint tokens that summarize the input history. This design retains much of the expressivity of full attention while substantially reducing the computational and memory cost. By attending to compressed history rather than all previous tokens, SCOUT incurs slightly higher memory than purely linear models, but its growth rate remains subquadratic and far more scalable than that of full Transformers. We analyze SCOUT's computational and memory efficiency and evaluate it empirically on long-context language modeling and reasoning tasks. SCOUT with both Mamba and SWA mixers outperforms strong long-sequence baselines under the same computational budget, matches full-attention Transformers on language modeling and common-sense reasoning tasks at 400M and 1.3B scales. Moreover, our SCOUT achieves higher end-to-end throughput than state-of-the-art linear models, while delivering comparable results on Long sequence benchmarks. These findings establish SCOUT as a practical, scalable solution for long-range sequence modeling, offering more than 10× savings in compute and memory over full attention. 1

### 55. Securing Radiation Detection Systems with an Efficient TinyML-Based IDS for Edge Devices

**主要机构**: Ontario Tech University
**作者数量**: 5人

**摘要**:
1 Radiation Detection Systems (RDSs) play a vital role in ensuring public safety across various settings, from nuclear facilities to medical environments. However, these systems are increasingly vulnerable to cyber-attacks such as data injection, man-in-the-middle (MITM) attacks, ICMP floods, botnet attacks, privilege escalation, and distributed denial-of-service (DDoS) attacks. Such threats could compromise the integrity and reliability of radiation measurements, posing significant public health and safety risks. This paper presents a new synthetic radiation dataset and an Intrusion Detection System (IDS) tailored for resource-constrained environments, bringing Machine Learning (ML) predictive capabilities closer to the sensing edge layer of critical infrastructure. Leveraging TinyML techniques, the proposed IDS employs an optimized XGBoost model enhanced with pruning, quantization, feature selection, and sampling. These TinyML techniques significantly reduce the size of the model and computational demands, enabling real-time intrusion detection on low-resource devices while maintaining a reasonable balance between efficiency and accuracy.

### 56. Seeing More, Saying More: Lightweight Language Experts are Dynamic Video Token Compressors

**主要机构**: Southern University of Science and Technology, The University of Hong
**作者数量**: 5人

**摘要**:
Recent advancements in large video-language models have revolutionized video understanding tasks. However, their efficiency is significantly constrained by processing high volumes of visual tokens. Existing token compression strategies apply a fixed compression ratio, ignoring the variability in semantic density among different video clips. Consequently, this lead to inadequate representation of information-rich clips due to insufficient tokens and unnecessary computation on static or content-poor ones. To address this, we propose LangDC, a Language-aware Dynamic Token Compressor. LangDC leverages a lightweight language model to describe video clips, converting them into soft caption tokens as visual representations. Trained with our proposed semantic density-aware supervision, LangDC aims to 1) cover key visual cues necessary for downstream task reasoning and 2) dynamically adjust compression ratios based on scene richness, reflected by descriptions length. Our design mimics how humans dynamically express what they see: complex scenes (seeing more) elicit more detailed language to convey nuances (saying more), whereas simpler scenes are described with fewer words. Experimental results show that our method reduces FLOPs by 49% compared to VideoGPT+ while maintaining competitive performance. Furthermore, qualitative results demonstrate our approach adaptively adjusts the token compression ratio based on video segment richness.

### 57. SpectMamba: Integrating Frequency and State Space Models for Enhanced Medical Image Detection

**主要机构**: United-Imaging Research Institute of Intelligent Imaging, Nankai University, Southern University of Science and Technology
**作者数量**: 6人

**摘要**:
Abnormality detection in medical imaging is a critical task requiring both high efficiency and accuracy to support effective diagnosis. While convolutional neural networks (CNNs) and Transformer-based models are widely used, both face intrinsic challenges: CNNs have limited receptive fields, restricting their ability to capture broad contextual information, and Transformers encounter prohibitive computational costs when processing high-resolution medical images. Mamba, a recent innovation in natural language processing, has gained attention for its ability to process long sequences with linear complexity, offering a promising alternative. Building on this foundation, we present SpectMamba, the first Mamba-based architecture designed for medical image detection. A key component of SpectMamba is the Hybrid Spatial-Frequency Attention (HSFA) block, which separately learns high-and low-frequency features. This approach effectively mitigates the loss of high-frequency information caused by frequency bias and correlates frequency-domain features with spatial features, thereby enhancing the model's ability to capture global context. To further improve long-range dependencies, we propose the Visual State-Space Module (VSSM) and introduce a novel Hilbert Curve Scanning technique to strengthen spatial correlations and local dependencies, further optimizing the Mamba framework. Comprehensive experiments show that SpectMamba achieves state-of-the-art performance while being both effective and efficient across various medical image detection tasks.

### 58. TINYMUSICIAN: ON-DEVICE MUSIC GENERATION WITH KNOWLEDGE DISTILLATION AND MIXED PRE-CISION QUANTIZATION

**主要机构**: Metropolitan College, Department of Computer Science, Boston University, School of Engineering & Technology, Duy Tan University
**作者数量**: 3人

**摘要**:
The success of the generative model has gained unprecedented attention in the music generation area. Transformer-based architectures have set new benchmarks for model performance. However, their practical adoption is hindered by some critical challenges: the demand for massive computational resources and inference time, due to their large number of parameters. These obstacles make them infeasible to deploy on edge devices, such as smartphones and wearables, with limited computational resources. In this work, we present TinyMusician, a lightweight music generation model distilled from MusicGen (a State-of-the-art music generation model). TinyMusician integrates two innovations: (i) Stage-mixed Bidirectional and Skewed KL-Divergence and (ii) Adaptive Mixed-Precision Quantization. The experimental results demonstrate that TinyMusician retains 93% of the MusicGen-Small performance with 55% less model size. TinyMusician is the first mobile-deployable music generation model that eliminates cloud dependency while maintaining high audio fidelity and efficient resource usage. 1

### 59. Towards Adaptive Visual Token Pruning for Large Multimodal Models

**主要机构**: Beijing Academy of Artificial Intelligence
**作者数量**: 5人

**摘要**:
Large Multimodal Models (LMMs) have achieved significant success across various tasks. These models usually encode visual inputs into dense token sequences, which are then concatenated with textual tokens and jointly processed by a language model. However, the increased token count substantially raises computational and memory costs during inference. Token pruning has emerged as a promising approach to address this issue. Existing token pruning methods often rely on costly calibration or suboptimal importance metrics, leading to redundant retained tokens. In this paper, we analyze the redundancy differences between visual and textual tokens and propose pruning exclusively on visual tokens. Based on this, we propose a visual token pruning strategy that explicitly preserves both cross-modal alignment and intra-modal informational diversity. We introduce a mutual information-based token pruning strategy that removes visual tokens semantically misaligned with textual tokens, effectively preserving the alignment between the visual and textual modalities. To further improve the representational quality of the retained tokens, we additionally prune redundant visual tokens by maximizing the expected pairwise distances in the embedding space, which is solved efficiently with a greedy algorithm. Extensive experiments demonstrate that our method maintains strong performance while reducing tokens by 88.9% on models such as LLaVA-1.5-7B and LLaVA-NEXT-7B, resulting in a 56.7% improvement in inference speed.

### 60. Uirapuru: Timely Video Analytics for High-Resolution Steerable Cameras on Edge Devices

**主要机构**: Vrije Universiteit Amsterdam, Paderborn University
**作者数量**: 5人

**摘要**:


### 61. Variation-aware Vision Token Dropping for Faster Large Vision-Language Models

**主要机构**: Shanghai Jiao Tong University, EPIC Lab, Sichuan University, Zhejiang University
**作者数量**: 6人

**摘要**:
Large vision-language models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding tasks. However, the increasing demand for high-resolution image and long-video understanding results in substantial token counts, leading to reduced inference efficiency. Token compression offers a direct solution by reducing the number of tokens to be processed, thereby improving computational efficiency. Through extensive analysis, we identify two critical limitations in existing inner-LLM token compression methods: positional bias and incompatibility with efficient operators, which hinder their practical deployment for LVLM acceleration. This paper presents the first approach from a token variation perspective, revealing that visual token variations within LLMs exhibit task-agnostic properties. We propose Variation-aware Vision Token Dropping (i.e., V 2 Drop), which progressively removes visual tokens with minimal variation during LVLM inference, thereby enhancing computational efficiency. Extensive experiments across multiple models and benchmarks demonstrate that our V 2 Drop is able to maintain 94.0% and 98.6% of the original model performance for image and video understanding tasks respectively, while reducing LLM generation latency by 31.5% and 74.2%. When combined with efficient operators, V 2 Drop further reduces GPU peak memory usage.

### 62. Vision-Based Embedded System for Noncontact Monitoring of Preterm Infant Behavior in Low-Resource Care Settings

**主要机构**: Electrical and Energy Engineering, Electronics and Computer Engineering, Soroti University
**作者数量**: 3人

**摘要**:
Preterm birth remains a leading cause of neonatal mortality, disproportionately affecting low-resource settings with limited access to advanced neonatal intensive care units (NICUs). Continuous monitoring of infant behavior, such as sleep/awake states and crying episodes, is critical but relies on manual observation or invasive sensors, which are prone to error, impractical, and can cause skin damage. This paper presents a novel, noninvasive, and automated visionbased framework to address this gap. We introduce an embedded monitoring system that utilizes a quantized MobileNet model deployed on a Raspberry Pi for real-time behavioral state detection. When trained and evaluated on public neonatal image datasets, our system achieves state-of-the-art accuracy (91.8% for sleep detection and 97.7% for crying/normal classification) while maintaining computational efficiency suitable for edge deployment. Through comparative benchmarking, we provide a critical analysis of the trade-offs between model size, inference latency, and diagnostic accuracy. Our findings demonstrate that while larger architectures (e.g., ResNet152, VGG19) offer marginal gains in accuracy, their computational cost is prohibitive for real-time edge use. The proposed framework integrates three key innovations: model quantization for memory-efficient inference (68% reduction in size), Raspberry Pi-optimized

### 63. ZeroQAT: Your Quantization-aware Training but Efficient

**主要机构**: University of Minnesota, Stevens Institute of Technology, University of North Texas, University of Georgia, University of Virginia, Northeastern University
**作者数量**: 12人

**摘要**:
Quantization is an effective technique to reduce the deployment cost of large language models (LLMs), and post-training quantization (PTQ) has been widely studied due to its efficiency. However, existing low-bit PTQ methods suffer from accuracy degradation because their layer-wise optimization introduces cumulative error propagation and misalignment between local reconstruction objectives and downstream performance. While quantization-aware training (QAT) provides a principled solution, its reliance on backpropagation incurs prohibitive data, time, and memory costs, limiting its practicality. To address these challenges, we propose ZeroQAT, a zeroth-order optimization-based QAT framework. ZeroQAT leverages forward-only gradient estimation to eliminate the need for backpropagation, significantly reducing computational and memory overhead while retaining the benefits of end-to-end optimization. Moreover, ZeroQAT jointly learns quantized weights, weight clipping thresholds, and equivalent transformations to mitigate quantization error and handle activation outliers. Experiments demonstrate that ZeroQAT achieves the efficiency of PTQ while retaining the accuracy of QAT, offering a practical solution for high-quality low-bit quantization of LLMs.
