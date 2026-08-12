# ComfyUI MiniMax H3 优化套件

[English](README.md) | **简体中文**

这是面向 ComfyUI 原生 MiniMax H3 音视频模型的一组模块化优化节点，不修改
ComfyUI 核心文件，分别解决三类问题：

- 不减少模型调用次数，加速全步数 NVFP4 推理；
- 降低峰值显存，使高分辨率或长时长任务能够完成；
- 主动降低步数时，提高每次模型调用所能保留的画质。

每个组件都是独立的自定义节点，可以单独启用、绕过和测量。完整数据和限制见
[中文评估报告](EVALUATION_REPORT.zh-CN.md)及
[英文评估报告](EVALUATION_REPORT.md)。

> 目前验证数据来自一台 RTX 5070 Ti 16 GB 电脑以及有限的提示词和种子。
> 表中数字只证明对应测试负载的结果，不代表所有硬件和工作流都能获得相同收益。

## 应该使用哪个节点？

| 目标 | 建议起点 | 主要代价 |
|---|---|---|
| 官方风格20步，当前验证最快的精确路径 | `H3 NVFP4 Fused MLP` + KJ Sage2 | 仅支持Blackwell/NVFP4 |
| 官方风格20步，降低峰值显存 | `H3 NVFP4 Fused MLP` + `H3 Low-Memory Sage2` | 实测比Fused + 普通Sage2慢约2% |
| 16 GB上运行原本会OOM的15秒/高分辨率任务 | `H3 Fused Kernels` + `H3 Low-Memory Sage2` + `H3 Long-Sequence VRAM Optimizer` | `16gb_chunked`以成功完成为目标，不是速度节点，也不保证bit-exact |
| 减少模型调用次数 | `H3 CAB Sampler`，从CAB-2、theta 0.20、14步开始 | 增加历史缓存显存，并逐渐偏离20步轨迹 |
| 诊断工作流瓶颈 | `H3 Step Profiler` | 测量本身有开销，生产生成时不要保留 |

官方20步工作流仍然能够从Fused MLP和Low-Memory Sage2获益，因为它们在每次
模型计算内部生效。CAB只有在实际减少步数时才会减少耗时。

## 节点说明

### H3 NVFP4 Fused MLP

融合MiniMax H3 MLP中的SwiGLU激活与FC2输入NVFP4量化，避免建立最大的
BF16/FP16中间张量，同时继续调用comfy-kitchen原生NVFP4 GEMM。它不会跳过
block，也不会减少采样步数。

- 最适合：Blackwell显卡和受支持的H3 NVFP4 checkpoint上的默认优化。
- 1280×736、5秒、20步实测：denoise `208.756 s -> 194.079 s`，
  peak allocated VRAM `5122.455 -> 4410.209 MiB`。
- 该次验证的数值结果：latent hash及解码视频完全一致。
- 适用范围：这是Blackwell/NVFP4专用的融合节点。

### H3 Fused Kernels

将H3按segment分别执行的AdaLN调制和门控残差操作，改为覆盖整个packed
sequence的kernel。它保留全部50个block、attention、MLP和采样步数。

- 推荐先使用`auto + accurate`。它保留原生RMSNorm及multiply/add舍入边界，
  只融合segment级工作。
- `fast`还会尝试融合RMSNorm + AdaLN，需要单独进行画质A/B。
- Triton不是加载插件的必要条件；`auto`遇到不支持的路径会回退至PyTorch
  reference。
- 它与 **H3 NVFP4 Fused MLP** 不同：本节点优化block调制和门控，后者优化
  NVFP4 MLP激活/量化边界。
- 实际吞吐收益受sequence形状和kernel launch开销影响。目前报告不声称它能
  在所有配置中加速。

### H3 Low-Memory Sage2

重新组织Sage2的Q/K量化与PV路径，降低attention临时workspace，不减少步数。

- 在Fused MLP基础上的实测峰值：`4410.209 -> 3843.612 MiB`，减少
  `566.597 MiB`。
- 该次测试的代价：denoise `194.079 -> 198.212 s`，慢2.13%。
- 验证路径的combined latent hash完全一致。
- 最适合：高分辨率、长视频或接近独立显存上限的工作流。

### H3 Long-Sequence VRAM Optimizer

用于超长或高分辨率H3 sequence的容量保护。它组合了DynamicVRAM activation
reserve提示、专用H3 Turbo LoRA存在时的bypass delta分块，以及可选的base MLP
分块。短sequence自动保持原始forward路径。

- `auto` / `16gb`：不分块base NVFP4 MLP，应当优先尝试。
- `16gb_chunked`：16 GB最后一级回退。在验证的15秒任务中，4096行分块将
  单次调用的FC1临时张量理论上界从约`5488.6 MiB`降至`224.0 MiB`。
- 它不会切割视频时间轴，也不会减少采样步数。
- base MLP分块会让NVFP4动态输入scale按chunk计算，因此
  `16gb_chunked`不与未分块路径bit-exact。
- 它可能更慢，目标是让原本OOM的任务成功完成。

### H3 CAB Sampler

这是适配ComfyUI denoised-output约定和H3 packed音视频latent的training-free
Corrected Adams-Bashforth求解器。它复用近期velocity历史并进行defect correction，
不会在单个名义step中增加额外模型调用。

- 验证起点：`CAB-2`、`theta=0.20`、`stock_simple` sigmas。
- 在1280×736、5秒测试中，CAB-2的14、12、10步相对20步分别减少30%、40%、
  50%的模型调用。
- CAB历史在736p实测中增加约`112 MiB`峰值显存。
- 同为6步时，CAB在当前样本中比`res_multistep`更接近参考；这不表示低步数
  输出能与官方20步完全一致。

### H3 Low-Step Sigmas

生成H3 sigma schedule。`stock_simple`复现ComfyUI simple scheduler，是当前
生产默认值。late-biased schedule保留为研究对照，但在现有样本中没有优于
`stock_simple`。

### H3 Optimization Controller与H3 Optimized Sampling

这是验证过的5秒路径的简化入口。Controller负责应用所选Fused MLP和Sage2；
Sampling节点同时输出同步的`SAMPLER`和`SIGMAS`，避免求解器与schedule使用
不同的步数。

| Preset | 组成 |
|---|---|
| `exact_speed` | NVFP4 Fused MLP + KJ Sage2 `auto` |
| `exact_low_vram` | NVFP4 Fused MLP + Low-Memory Sage2 |
| `off` | 不修改模型，只透传步数 |

不要在Controller之前放置另一个attention patch。它会明确拒绝冲突，而不是静默
替换已有override。

### H3 Step Profiler

用于受控A/B测试，记录step耗时、CUDA峰值分配和输出fingerprint。Profiler会
同步CUDA，因此会改变计时；它是诊断节点，不是推理优化节点。

## 实测结果

### 1280×736、5秒、20步

RTX 5070 Ti 16 GB上固定prompt、seed、模型和stock-simple schedule：

| 配置 | Denoise | 完整prompt | Peak allocated VRAM | 输出 |
|---|---:|---:|---:|---|
| KJ Sage2 baseline | 208.756 s | 244.969 s | 5122.455 MiB | 参考 |
| Fused MLP + Sage2 | 194.079 s | 223.674 s | 4410.209 MiB | hash完全一致 |
| Fused MLP + Low-Memory Sage2 | 198.212 s | 228.359 s | 3843.612 MiB | hash完全一致 |

### 736×1280、15秒、CAB-14容量验证

`Fused Kernels accurate -> Low-Memory Sage2 -> Long-Sequence 16gb_chunked
-> CAB-2`链路在同一张16 GB显卡上完成全部14步denoise以及音频、视频VAE解码。
Denoise约940秒，完整prompt为1064秒。该结果只证明任务能够完成，不证明速度
更快，也不证明与同shape的20步输出画质等价。

![15秒成功输出的抽帧](benchmark_artifacts/media/cab14_long_15s_contact.png)

## 安装

将[`plugins/`](plugins)中所需的目录分别复制到`ComfyUI/custom_nodes/`，保持
每个插件都是独立目录，然后重启ComfyUI。

使用简化的20步工作流，需要安装：

- `ComfyUI-H3-Optimization-Controller`
- `ComfyUI-H3-NVFP4-Fused-MLP`
- `ComfyUI-H3-Low-Memory-Sage2`
- `ComfyUI-H3-CAB-Sampler`
- `ComfyUI-H3-Low-Step-Sigmas`
- `exact_speed`还需要
  [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

使用15秒容量示例，还需要安装：

- `ComfyUI-H3-Fused-Kernels`
- `ComfyUI-H3-Long-Sequence`

Sage preset需要与当前Python、PyTorch、CUDA和GPU兼容的SageAttention版本。
仓库不包含模型权重或外部依赖。

## 示例工作流

### 保守的20步工作流

[`MiniMax_H3_Unified_Optimization_20step.json`](example_workflows/MiniMax_H3_Unified_Optimization_20step.json)
是一个5秒文生视频工作流，配置为：

```text
exact_speed + res_multistep + stock_simple + 20 steps
```

### 15秒全球通用连续性与容量测试

[`MiniMax_H3_CAB14_LongSequence_15s.json`](example_workflows/MiniMax_H3_CAB14_LongSequence_15s.json)
沿用官方单节点MiniMax H3布局，设置为1280×736、15秒、固定seed、CAB-2 14步
以及16 GB长序列回退。英文提示词描述一名旅客、一只狗和一张红色车票跨越四个
镜头，主要测试：

- 人物身份、服装、道具和空间连续性；
- 低机位跟拍与环绕运镜；
- 可读文字`"PLATFORM 4"`和简短英文对白；
- 环境音、动作音、对白、列车声音及配乐同步。

该工作流刻意设置得比较吃资源。`16gb_chunked`优先保证成功完成，而不是速度和
bit-exact。显存更多的用户应先尝试`auto`或`16gb`。

## 硬件适用范围

| 组件 | Blackwell SM 12.x | Ada SM 8.9 | 其他GPU |
|---|---:|---:|---:|
| NVFP4 Fused MLP | 已验证 | 不支持 | 不支持 |
| H3 Fused Kernels | 已验证 | 兼容Triton时理论可用，未实测 | PyTorch回退或兼容Triton，未实测 |
| Low-Memory Sage2 | 已验证 | 设计兼容，未完整benchmark | 需要兼容的SageAttention路径 |
| Long-Sequence VRAM Optimizer | 已验证 | 预计可用，未实测 | 预计可用，未实测 |
| CAB sampler / sigma schedule | 已验证 | 预计可移植 | 预计可移植 |
| Controller | 已验证 | 部分支持，不支持的组件会被禁用 | 取决于所选组件 |

## 仓库包含的插件

- `ComfyUI-H3-Optimization-Controller`
- `ComfyUI-H3-NVFP4-Fused-MLP`
- `ComfyUI-H3-Fused-Kernels`
- `ComfyUI-H3-Low-Memory-Sage2`
- `ComfyUI-H3-Long-Sequence`
- `ComfyUI-H3-CAB-Sampler`
- `ComfyUI-H3-Low-Step-Sigmas`
- `ComfyUI-H3-Step-Profiler`

早期MLP调用跳过实验和已经否定的Turbo/4步示例不在本仓库发布。

## 外部参考

- [CAB论文](https://arxiv.org/abs/2605.16736)
- [CAB官方参考实现](https://github.com/Anuska-Roy/CAB)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)

## 许可证

本仓库代码使用[MIT License](LICENSE)。外部依赖和模型文件不随仓库发布，并
保留各自的许可证及使用条款。
