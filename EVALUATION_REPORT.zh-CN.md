# MiniMax H3 ComfyUI 优化节点评估报告

评估日期：2026-08-11  
状态：实验性公开预览

## 1. 结论摘要

本轮开发得到的优化并不是单一“万能加速开关”，而是三条可以独立组合的路线：

1. **全步数精确优化**：NVFP4 Fused MLP 在不减少模型调用、不跳过 MLP
   block 的前提下减少中间张量与显存流量。10 步暖机测试约有 2% 采样加速。
2. **显存优化**：Low-Memory Sage2 将实测 denoise 峰值 allocated VRAM 从
   4409.0 MiB 降至 3842.4 MiB，减少 566.6 MiB，latent SHA-256 保持一致；
   代价是约 1% 的轻微速度损失。
3. **低步数优化**：CAB-2 在 6 次模型调用下，相比同为 6 步的
   `res_multistep`，更接近 10 步参考结果；它的主要意义是减少模型调用数，
   而不是让相同的 20 步本身更快。

Hybrid/Sage3 路线在 10 步测试中取得约 6% 的采样加速，但属于近似计算，
需要逐项目检查人物、运动、镜头和音频质量。

对于官方默认 20 步，推荐优先使用：

- 稳妥速度：`exact_speed + res_multistep + stock_simple + 20 steps`
- 显存压力：`exact_low_vram + res_multistep + stock_simple + 20 steps`
- 近似加速：`balanced_fast + res_multistep + stock_simple + 20 steps`

CAB 更适合单独评估 14、12、10 步能否达到可接受的 20 步观感，而不是在
20 步上期待直接加速。

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti 16 GB，Blackwell SM 12.0 |
| 系统内存 | 约 48 GiB，可用共享 GPU 内存约 24 GB |
| 操作系统 | Windows |
| ComfyUI | 0.31.0 |
| Python | 3.12.9 |
| PyTorch | 2.10.0+cu130 |
| comfy-kitchen | 0.2.28 |
| Triton | 3.6.0 |
| UNet | `MiniMax_H3_FL2VA_pruned_nvfp4.safetensors` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |

主要测试为文生视频，最多一张图像参考。测试分辨率覆盖约 480p 与
1280×736，典型时长 5 秒。不同实验尽量固定 prompt、seed、分辨率、时长、
模型和 sigma 调度。

## 3. 测试方法

### 3.1 时间测量

- 区分冷启动、模型初始化和暖机采样。
- 主要比较进度条中的 denoise 时间，桌面显示的完整时间还包含文本编码、
  模型换入、VAE 解码和视频封装。
- 小幅差异需要多次交错测试，避免动态显存加载、编译缓存和后台任务造成误判。

### 3.2 数值一致性

- 对声频与视频 latent 的所有 tensor 生成独立 SHA-256，并生成 combined hash。
- “精确”表示在当前硬件、软件版本和选定 Sage2 路径下 latent hash 一致，
  不代表跨 GPU、跨 PyTorch 或跨 kernel 实现必然 bit-exact。

### 3.3 低步数质量

- 以相同 prompt/seed 的 10 步输出作为轨迹参考。
- 比较解码视频帧的 SSIM 和 PSNR。
- 指标只能描述对本次参考输出的接近程度，不能替代对运动、人物结构、剪辑、
  音频和提示词遵循度的主观评估。

## 4. 分项结果

### 4.1 H3 NVFP4 Fused MLP

原生 H3 MLP 会先形成较大的 BF16/FP16 SwiGLU 激活，再为 FC2 进行 NVFP4
量化。Fused MLP 将激活与 FC2 输入量化融合，继续调用 comfy-kitchen 的原生
NVFP4 GEMM，不跳过 block，也不减少采样步数。

实测结果：

- 1280×736 workload：每次 MLP 调用避免约 952.7 MiB 的大型 BF16/FP16
  中间激活。
- 约 0.2 MP 烟雾测试：每次 MLP 调用避免约 82.7 MiB 中间激活。
- 暖机 10 步采样约提升 2%。
- 已验证打包值和最终 decoded/latent hash 一致。

判断：20 步下仍然有效。其百分比收益预计接近 10 步结果，而绝对节省时间会
随调用次数增加。该结论的 20 步专门重复基准尚未完成，因此不能把 2% 当成
固定保证。

### 4.2 H3 Low-Memory Sage2

该实现重新组织 Sage2 attention 的 Q/K 量化和 PV 路径，降低临时 workspace，
不改变采样步数。

| 指标 | 原 Sage2 | Low-Memory Sage2 | 差值 |
|---|---:|---:|---:|
| denoise peak allocated | 4409.0 MiB | 3842.4 MiB | -566.6 MiB |

- 降幅约 12.9%。
- combined latent SHA-256 一致。
- 实测速度约慢 1%。
- 显存减少是峰值 headroom，不会因为 20 步变成 20 倍；它主要降低高分辨率、
  长视频或系统共享显存介入时的失败风险。

判断：显存临界时价值高；显存充足、只追求速度时使用普通 Sage2 更合适。

### 4.3 H3 Blackwell Hybrid Attention

`balanced_fast` 在首尾 denoise calls 使用较保守的 Sage2，中段使用
SageAttention3 FP4。`maximum_speed` 则全程使用 Sage3。

- 10 步、首尾各 1 次 Sage2 的干净对照中，采样加速约 6%。
- 20 步且 edge calls=1 时，理论调度为 2 次 Sage2 + 18 次 Sage3；因此 Sage3
  覆盖比例会从 10 步时的 80% 上升到 90%。
- 上述 20 步收益是结构性推断，不是已经完成的 20 步实测数据。
- Sage3 是近似 attention，累计误差和内容敏感性必须通过视频 A/B 判断。

判断：适合完整 20 步下进一步追求速度，但不能标为“无损”。

### 4.4 H3 CAB Low-Step Sampler

CAB-2/CAB-3 是 training-free 多步求解器，复用历史 velocity 并进行 defect
correction，不增加每一步中的模型调用。该节点按照 CAB 论文方程独立适配了
ComfyUI denoised-output 约定和 H3 的音视频 NestedTensor。

以 10 步输出为参考：

| 分辨率/时长 | 6步方法 | SSIM | PSNR |
|---|---|---:|---:|
| 约 480p / 5秒 | res_multistep | 0.7902 | 19.62 dB |
| 约 480p / 5秒 | CAB-2 | 0.8155 | 20.63 dB |
| 1280×736 / 5秒 | res_multistep | 0.8017 | 20.09 dB |
| 1280×736 / 5秒 | CAB-2 | 0.8270 | 21.26 dB |

- 两种 6 步方法都是 6 次模型调用；相对 10 步减少 40% NFE。
- CAB 自身算术开销低于 denoise 时间的 1%。
- CAB 历史状态在 736p 下比 `res_multistep` 多约 112 MiB 峰值占用。
- CAB-2 在此次样本上略优于 CAB-3，`theta=0.20` 是当前验证默认值。
- CAB-2 6 步仍不等于 10 步质量，更不能由此推导为等于官方 20 步质量。

判断：下一阶段应以 20 步为参考，系统测试 CAB-2 的 14、12、10 步，而不是
直接把 6 步结论扩大到所有内容。

### 4.5 H3 Low-Step Sigmas

- `stock_simple` 精确复现 ComfyUI simple scheduler 的离散索引和 H3 原生 shift。
- `balanced_late`、`strong_late` 和自定义 late bias 在当前样本上均未优于
  `stock_simple`。

判断：保留为研究接口，但生产默认应使用 `stock_simple`。

### 4.6 两个统一入口节点

新增：

- `H3 Optimization Controller`
- `H3 Optimized Sampling`

控制器只编排独立插件，不复制 CUDA/Triton kernel。它会检测重复 attention
override，并将 `steps` 直接输出给采样节点，防止 Hybrid 调度步数与 sigma
步数不一致。

真实烟雾测试：

- 约 0.2 MP、约 1.6 秒、2 步、CAB-2、stock simple。
- `exact_low_vram` 完整执行成功，冷启动完整耗时 27.13 秒。
- `exact_speed` 暖机完整耗时 1.07 秒。
- 两次 combined latent SHA-256 均为
  `68379847125b82e3fb35b3f6c9f7a75d2b635f7e1293a2421c28ef3a071d4d3c`。

27.13 秒与 1.07 秒不能作为两个预设的速度对比，因为前者包含模型冷初始化；
该测试的意义是验证端到端编排以及两条精确路径的数值一致性。

## 5. 官方 20 步下的适用性

| 组件 | 20 步作用 | 建议 |
|---|---|---|
| Fused MLP | 每一步生效，绝对节省时间累计增加 | 默认开启 |
| exact Sage2 | 保持测试路径数值一致 | 日常默认 |
| Low-Memory Sage2 | 峰值显存下降，但不是按步数累计 | 显存临界时开启 |
| Hybrid/Sage3 | Sage3 覆盖比例更高，潜在加速更明显 | 必须 A/B |
| CAB-2 | 同为 20 步时不减少 NFE | 用于尝试降低到 14/12/10 步 |
| biased sigmas | 当前没有正向证据 | 保持 stock simple |

## 6. 推荐配置

### 6.1 官方质量基准

```text
preset: off
sampler: res_multistep
sigmas: stock_simple
steps: 20
```

### 6.2 精确速度优先

```text
preset: exact_speed
sampler: res_multistep
sigmas: stock_simple
steps: 20
```

### 6.3 高分辨率/长时长显存优先

```text
preset: exact_low_vram
sampler: res_multistep
sigmas: stock_simple
steps: 20
```

### 6.4 近似全步数加速

```text
preset: balanced_fast
sage2_edge_calls: 1
sampler: res_multistep
sigmas: stock_simple
steps: 20
```

### 6.5 低步数实验

```text
preset: exact_speed
sampler: CAB-2
theta: 0.20
sigmas: stock_simple
steps: 14 -> 12 -> 10，逐档对比20步参考
```

低步数测试先使用精确 attention，确认 CAB 差异后再叠加 Hybrid，避免同时引入
两个近似变量而无法判断画质变化来源。

## 7. 局限与后续工作

当前结果不能视为通用 benchmark，原因包括：

- 质量对比主要来自一个 prompt 和一个 seed。
- SSIM/PSNR 衡量的是对参考轨迹的接近度，不等同于审美质量。
- 尚未形成官方 20 步参考下 CAB 10/12/14 步的大样本统计。
- 尚未覆盖对白、复杂手部、多人交互、快速剪辑、强音画同步等高难度内容。
- Blackwell 以外仅验证了 Low-Memory Sage2 的设计兼容范围，没有完整实机矩阵。
- 2% 和 6% 都属于容易受到冷启动、动态权重加载和后台负载影响的小幅收益，
  应至少进行多次交错重复测试。

建议后续建立 8–12 个固定 prompt、至少 3 个 seed 的回归集，分别记录：

- denoise time、完整 prompt time、NFE；
- peak allocated/reserved VRAM 与 Windows shared GPU memory；
- latent hash、逐帧 SSIM/PSNR/VMAF；
- 人物结构、运动连续性、提示词遵循、音画同步的盲评结果。

## 8. 外部参考

- CAB paper: <https://arxiv.org/abs/2605.16736>
- CAB official implementation: <https://github.com/Anuska-Roy/CAB>
- SageAttention: <https://github.com/thu-ml/SageAttention>
- ComfyUI: <https://github.com/Comfy-Org/ComfyUI>

