# AI推理加速技术论文分析报告
生成时间: 2025-09-04 10:30:49
分析论文数量: 8篇

## 论文技术简报

### 1. Binary Weight Multi-Bit Activation Quantization for Compute-in-Memory CNN Accelerators

研究团队发布了Binary Weight Multi-Bit Activation Quantization论文，使用二进制权重多比特激活（BWMA）量化技术（含权重量化闭式解与激活量化可微函数），解决了现有Compute-in-Memory CNN加速器量化方法在硬件效率与精度间的权衡问题，达成了CIFAR-10和ImageNet数据集上1.44%-5.46%及0.35%-5.37%的精度提升，且4比特激活量化实现硬件成本与性能最优平衡。

### 2. Efficient Diffusion-Based 3D Human Pose Estimation with Hierarchical Temporal Pruning

School of Cyber Science and Engineering发布了Efficient Diffusion-Based 3D Human Pose Estimation with Hierarchical Temporal Pruning论文，使用Hierarchical Temporal Pruning (HTP)策略，解决了扩散模型在3D人体姿态估计中因迭代性和多假设导致的高计算成本问题，达成了训练MACs减少38.5%、推理MACs减少56.8%、推理速度平均提升81.1%并实现SOTA性能的效果

### 3. ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion

字节跳动发布了ERTACache论文，使用误差纠正与时间步调整的特征缓存框架（含离线残差分析、动态积分间隔调整及闭式残差线性化模型），解决了扩散模型迭代推理计算开销大且缓存复用导致质量下降的问题，达成了高达2倍推理加速同时保持或提升视觉质量（在Wan2.1视频模型上2倍加速且VBench退化极小）的效果。

### 4. Failure Prediction Is a Better Performance Proxy for Early-Exit Networks Than Calibration

阿姆斯特丹大学发布了《Failure Prediction Is a Better Performance Proxy for Early-Exit Networks Than Calibration》论文，使用失败预测作为性能代理技术，解决了校准指标误导早期退出模型性能评估的问题，达成了与效率提升强相关、更可靠设计评估早期退出模型的效果。

### 5. Full-Frequency Temporal Patching and Structured Masking for Enhanced Audio Classification

University of Alabama at Birmingham发布了Enhanced Audio Classification论文，使用Full-Frequency Temporal Patching (FFTP)和SpecMask技术，解决了现有模型方形分块破坏频率模式、补丁过多导致计算量大的问题，达成了在AudioSet-18k上mAP提升+6.76、SpeechCommandsV2准确率提升+8.46，同时计算量减少83.26%的效果。

### 6. Neural Network Acceleration on MPSoC board: Integrating SLAC's SNL, Rogue Software and Auto-SNL

SLAC发布了关于MPSoC板上神经网络加速的论文，使用SLAC Neural Network Library (SNL)及Auto-SNL技术，解决了高吞吐量数据流下传统机器学习推理延迟过高的问题，达成了在多数神经网络架构中实现有竞争力或更优延迟、部分情况下节省FPGA资源的效果

### 7. NSPDI-SNN: An efficient lightweight SNN based on nonlinear synaptic pruning and dendritic integration

天津大学发布了NSPDI-SNN论文，使用非线性突触修剪与非线性树突整合技术，解决了现有SNN因缺乏复杂树突结构导致的效率与轻量级不足问题，达成了在多个任务上实现高稀疏性且性能下降最小，尤其在事件流数据集上取得最佳结果并提升突触信息传递效率的效果。

### 8. Spiking Decision Transformers: Local Plasticity, Phase-Coding, and Dendritic Routing for Low-Power Sequence Control

Purdue University发布了Spiking Decision Transformers论文，使用嵌入LIF神经元、生物启发的三因素可塑性、相移脉冲位置编码和轻量级树突路由模块的Spiking Decision Transformer（SNN-DT）技术，解决了传统Transformer架构因依赖密集矩阵运算而难以适应能量受限边缘平台的问题，达成了在经典控制基准上匹配或超过标准Decision Transformer性能，且每个决策发射少于10个脉冲、推断能量降低超四个数量级的效果。

## 论文详细信息

### 1. Binary Weight Multi-Bit Activation Quantization for Compute-in-Memory CNN Accelerators

**主要机构**: 
**作者数量**: 4人

**摘要**:
Compute-in-memory (CIM) accelerators have emerged as a promising way for enhancing the energy efficiency of convolutional neural networks (CNNs). Deploying CNNs on CIM platforms generally requires quantization of network weights and activations to meet hardware constraints. However, existing approaches either prioritize hardware efficiency with binary weight and activation quantization at the cost of accuracy, or utilize multi-bit weights and activations for greater accuracy but limited efficiency. In this paper, we introduce a novel binary weight multi-bit activation (BWMA) method for CNNs on CIM-based accelerators. Our contributions include: deriving closed-form solutions for weight quantization in each layer, significantly improving the representational capabilities of binarized weights; and developing a differentiable function for activation quantization, approximating the ideal multi-bit function while bypassing the extensive search for optimal settings. Through comprehensive experiments on CIFAR-10 and ImageNet datasets, we show that BWMA achieves notable accuracy improvements over existing methods, registering gains of 1.44%-5.46% and 0.35%-5.37% on respective datasets. Moreover, hardware simulation results indicate that 4-bit activation quantization strikes the optimal balance between hardware cost and model performance.

### 2. Efficient Diffusion-Based 3D Human Pose Estimation with Hierarchical Temporal Pruning

**主要机构**: School of Cyber Science and Engineering, Engineering Research Center of Blockchain Application, Department of Computer and Information Science, Supervision And Management, Purple Mountain Laboratories, Faculty of Science and Technology, School of Computer Science and Engineering, Ministry of Education, School of Cyber Science and Engi- neering, Laboratory of New Generation Artificial Intelligence Technology and Its Interdisciplinary Applications, Southeast University, Wuhan University, School of Remote Sensing and Information Engineering and the Collaborative Innovation Center of Geospatial Technology, University of Macau, UOW College Hong Kong
**作者数量**: 13人

**摘要**:
Diffusion models have demonstrated strong capabilities in generating high-fidelity 3D human poses, yet their iterative nature and multi-hypothesis requirements incur substantial computational cost. In this paper, we propose an Efficient Diffusion-Based 3D Human Pose Estimation framework with a Hierarchical Temporal Pruning (HTP) strategy, which dynamically prunes redundant pose tokens across both frame and semantic levels while preserving critical motion dynamics. HTP operates in a staged, top-down manner: (1) Temporal Correlation-Enhanced Pruning (TCEP) identifies essential frames by analyzing interframe motion correlations through adaptive temporal graph construction; (2) Sparse-Focused Temporal MHSA (SFT MHSA) leverages the resulting frame-level sparsity to reduce attention computation, focusing on motion-relevant tokens; and (3) Mask-Guided Pose Token Pruner (MGPTP) performs fine-grained semantic pruning via clustering, retaining only the most informative pose tokens. Experiments on Human3.6M and MPI-INF-3DHP show that HTP reduces training MACs by 38.5%, inference MACs by 56.8%, and improves inference speed by an average of 81.1% compared to prior diffusion-based methods, while achieving state-of-the-art performance.

### 3. ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion

**主要机构**: 
**作者数量**: 10人

**摘要**:
Diffusion models suffer from substantial computational overhead due to their inherently iterative inference process. While feature caching offers a promising acceleration strategy by reusing intermediate outputs across timesteps, naïve reuse often incurs noticeable quality degradation. In this work, we formally analyze the cumulative error introduced by caching and decompose it into two principal components: feature shift error, caused by inaccuracies in cached outputs, and step amplification error, which arises from error propagation under fixed timestep schedules. To address these issues, we propose ERTACache, a principled caching framework that jointly rectifies both error types. Our method employs an offline residual profiling stage to identify reusable steps, dynamically adjusts integration intervals via a trajectory-aware correction coefficient, and analytically approximates cacheinduced errors through a closed-form residual linearization model. Together, these components enable accurate and efficient sampling under aggressive cache reuse. Extensive experiments across standard image and video generation benchmarks show that ERTACache achieves up to 2× inference speedup while consistently preserving or even improving visual quality. Notably, on the state-of-the-art Wan2.1 video diffusion model, ERTACache delivers 2× acceleration with minimal VBench degradation, effectively maintaining baseline fidelity while significantly improving efficiency. The code is available at https://github.com/bytedance/ ERTACache.

### 4. Failure Prediction Is a Better Performance Proxy for Early-Exit Networks Than Calibration

**主要机构**: University of Amsterdam, Jagiellonian University † Warsaw, University of Technology
**作者数量**: 4人

**摘要**:
Early-exit models speed up inference by attaching internal classifiers to intermediate layers of the model and allowing computation to stop once a prediction satisfies an exit criterion. Most early-exit methods rely on confidence-based exit strategies, which motivated some works to calibrate intermediate classifiers to improve the performance of the entire model. In this paper, we show that calibration measures can be misleading indicators of the performance of multi-exit models: a wellcalibrated classifier may still waste computation, and common calibration methods do not preserve the sample ranking within a classifier. We demonstrate empirical cases where miscalibrated networks outperform calibrated ones. As an alternative, we propose to use failure prediction as a more useful proxy for early-exit model performance. Unlike calibration, failure prediction accounts for changes in the ranking of samples and shows a strong correlation with efficiency improvements, making it a more dependable basis for designing and evaluating early-exit models.

### 5. Full-Frequency Temporal Patching and Structured Masking for Enhanced Audio Classification

**主要机构**: University of Alabama at Birmingham Birmingham, Department of Computer Science
**作者数量**: 3人

**摘要**:
Transformers and State-Space Models (SSMs) have advanced audio classification by modeling spectrograms as sequences of patches. However, existing models such as the Audio Spectrogram Transformer (AST) and Audio Mamba (AuM) adopt square patching from computer vision, which disrupts continuous frequency patterns and produces an excessive number of patches, slowing training, and increasing computation. We propose Full-Frequency Temporal Patching (FFTP), a patching strategy that better matches the time-frequency asymmetry of spectrograms by spanning full frequency bands with localized temporal context, preserving harmonic structure, and significantly reducing patch count and computation. We also introduce SpecMask, a patch-aligned spectrogram augmentation that combines full-frequency and localized time-frequency masks under a fixed masking budget, enhancing temporal robustness while preserving spectral continuity. When applied on both AST and AuM, our patching method with SpecMask improves mAP by up to +6.76 on AudioSet-18k and accuracy by up to +8.46 on SpeechCommandsV2, while reducing computation by up to 83.26%, demonstrating both performance and efficiency gains.

### 6. Neural Network Acceleration on MPSoC board: Integrating SLAC's SNL, Rogue Software and Auto-SNL

**主要机构**: 
**作者数量**: 7人

**摘要**:
The LCLS-II Free Electron Laser (FEL) will generate X-ray pulses for beamline experiments at rates of up to 1 MHz, with detectors producing data throughputs exceeding 1 TB/s. Managing such massive data streams presents significant challenges, as transmission and storage infrastructures become prohibitively expensive. Machine learning (ML) offers a promising solution for real-time data reduction, but conventional implementations introduce excessive latency, making them unsuitable for high-speed experimental environments. To address these challenges, SLAC developed the SLAC Neural Network Library (SNL), a specialized framework designed to deploy real-time ML inference models on Field-Programmable Gate Arrays (FPGA). SNL's key feature is the ability to dynamically update model weights without requiring FPGA resynthesis, enhancing flexibility for adaptive learning applications. To further enhance usability and accessibility, we introduce Auto-SNL, a Python extension that streamlines the process of converting Python-based neural network models into SNL-compatible high-level synthesis code. This paper presents a benchmark comparison against hls4ml, the current state-of-the-art tool, across multiple neural network architectures, fixed-point precisions, and synthesis configurations targeting a Xilinx ZCU102 FPGA. The results showed that SNL achieves competitive or superior latency in most tested architectures, while in some cases also offering FPGA resource savings. This adaptation demonstrates SNL's versatility, opening new opportunities for researchers and academics in fields such as high-energy physics, medical imaging, robotics, and many more.

### 7. NSPDI-SNN: An efficient lightweight SNN based on nonlinear synaptic pruning and dendritic integration

**主要机构**: Tianjin University, Academy of Medical Engineering and Translational Medicine, Clinical Hospital of Chengdu Brain Science Institute, School of Life Science and Technology, University of Electronic Science and Technology of China, MOE Key Lab for NeuroInformation, China-Cuba Belt and Road Joint Laboratory on Neurotechnology and Brain-Apparatus Communication
**作者数量**: 8人

**摘要**:
Spiking neural networks (SNNs) are artificial neural networks based on simulated biological neurons and have attracted much attention in recent artificial intelligence technology studies. The dendrites in biological neurons have efficient information processing ability and computational power; however, the neurons of SNNs rarely match the complex structure of the dendrites. Inspired by the nonlinear structure and highly sparse properties of neuronal dendrites, in this study, we propose an efficient, lightweight SNN method with nonlinear pruning and dendritic integration (NSPDI-SNN). In this method, we introduce nonlinear dendritic integration (NDI) to improve the representation of the spatiotemporal information of neurons. We implement heterogeneous state transition ratios of dendritic spines and construct a new and flexible nonlinear synaptic pruning (NSP) method to achieve the high sparsity of SNN. We conducted systematic experiments on three benchmark datasets (DVS128 Gesture, CIFAR10-DVS, and CI-FAR10) and extended the evaluation to two complex tasks (speech recognition and reinforcement learning-based maze navigation task). Across all tasks, NSPDI-SNN consistently achieved high sparsity with minimal performance degradation. In particular, our method achieved the best experimental results on all three event stream datasets. Further analysis showed that NSPDI significantly improved the efficiency of synaptic information transfer as sparsity increased. In conclusion, our results indicate that the complex structure and nonlinear computation of neuronal dendrites provide a promising approach for developing efficient SNN methods.

### 8. Spiking Decision Transformers: Local Plasticity, Phase-Coding, and Dendritic Routing for Low-Power Sequence Control

**主要机构**: Purdue University (Fort Wayne, Independent Researcher London, Department of Computer Science
**作者数量**: 2人

**摘要**:
Reinforcement learning agents based on Transformer architectures have achieved impressive performance on sequential decision-making tasks, but their reliance on dense matrix operations makes them ill-suited for energy-constrained, edge-oriented platforms. Spiking neural networks promise ultra-low-power, event-driven inference, yet no prior work has seamlessly merged spiking dynamics with return-conditioned sequence modeling. We present the Spiking Decision Transformer (SNN-DT), which embeds Leaky Integrate-and-Fire neurons into each self-attention block, trains end-to-end via surrogate gradients, and incorporates biologically inspired three-factor plasticity, phase-shifted spike-based positional encodings, and a lightweight dendritic routing module. Our implementation matches or exceeds standard Decision Transformer performance on classic control benchmarks (CartPole-v1, MountainCar-v0, Acrobot-v1, Pendulum-v1) while emitting fewer than ten spikes per decision, an energy proxy suggesting over four orders-of-magnitude reduction in per inference energy. By marrying sequence modeling with neuromorphic efficiency, SNN-DT opens a pathway toward real-time, low-power control on embedded and wearable devices.
