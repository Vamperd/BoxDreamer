# MyNet PCK2 提升实验顺序

目标：优先提升当前真实视频上的 8 角点稳定性，尤其关注 `pck_2`、`corner_px`、不可见角点稳定性和完整视频可视化效果。

## 已完成工程项

- [x] `subpixel` 解码与 `evaluate.py` 评估入口。
- [x] 单图推理和视频推理接入 `--decode-method` / `--subpixel-window`。
- [x] bbox-only 记录补标 8 角点脚本。
- [x] 修复不可见角点监督缺失：coarse heatmap 全通道监督，fine 坐标损失只监督可见角点。

> 说明：这里的“已完成”指代码实现已完成，不代表已经在 Ubuntu 训练机上完成真实 checkpoint 的对比实验。

## 1. [x] 只改解码，先验证已有 checkpoint

- 做法：在不重新训练的情况下，将 heatmap 解码从 hard argmax 切到 `subpixel` 局部加权质心。
- 当前状态：代码侧已完成 `subpixel` 解码、`evaluate.py`、单图推理和视频推理接入。
- 待验证：真实 checkpoint 的 `argmax` / `subpixel` 指标对比仍需在训练机运行。
- 目的：降低 `64x64 heatmap -> 256 crop` 带来的 4 像素 stride 量化误差。
- 验证：
  ```bash
  python -m src.mynet.evaluate \
    --data-root data/dji_action4_video_annot/mynet \
    --val-index data/dji_action4_video_annot/mynet/val.json \
    --checkpoint models/checkpoints/mynet_video_ft/best.pt \
    --decode-method argmax

  python -m src.mynet.evaluate \
    --data-root data/dji_action4_video_annot/mynet \
    --val-index data/dji_action4_video_annot/mynet/val.json \
    --checkpoint models/checkpoints/mynet_video_ft/best.pt \
    --decode-method subpixel \
    --subpixel-window 5
  ```
- 预期：`pck_2` 上升，`corner_px` 不明显变差。
- 回退：如果 subpixel 被背景响应拖偏，推理时使用 `--decode-method argmax`。

## 2. [x] 修复不可见角点监督缺失（严重性：P0 / 高）

- 当前问题：不可见角点 heatmap 在数据中是全 0，但训练 loss 又用 `corner_valid` mask 掉不可见通道，导致模型没有学习“不可见角点应无响应”。
- 当前状态：代码侧已完成 `L_coarse + 2.0 * L_fine`，仍需在训练机重新训练并对比不可见角点假峰指标。
- 严重性判断：这是训练目标缺失问题，优先级高于真实视频微调、ResNet50 和 heatmap 分辨率升级。
- 影响：
  - 不可见角点可能产生随机峰值。
  - 视频中角点会闪烁或跳到背景/相邻角点。
  - 后续真实微调可能被错误不可见监督污染。
- 推荐改法：
  ```text
  L = L_coarse + 2.0 * L_fine
  ```
- `L_coarse`：对全部 8 个 heatmap 通道监督，不可见角点目标为全 0，让模型学习“该角点不存在于可见区域”。
- `L_fine`：只对 `corner_valid=1` 的可见角点做坐标回归损失，不可见角点不参与坐标损失。
- 验证指标：
  - `pck_2`、`pck_5`、`corner_px`
  - `invisible_false_peak_rate`
  - debug heatmap 中不可见通道是否接近全 0
- 回退：如果可见角点 heatmap 变弱，降低 fine loss 权重，或提高可见角点 heatmap 权重。

## 3. [ ] 使用真实视频标注帧低学习率微调

- 做法：补充当前视频中的真实 bbox 和 8 角点标注，用 `lr=1e-5~3e-5` 从已有 checkpoint 微调。
- 目的：解决合成数据到真实视频的域差异。
- 验证：真实视频验证段的 `pck_2/pck_5` 和完整视频可视化是否提升。
- 回退：如果过拟合，减少 epoch、降低真实帧重复采样比例，或冻结 backbone 只训 head。

## 4. [ ] 加 crop jitter 与真实视频风格增强

- 做法：训练时随机扰动 bbox 中心、尺度、padding，并加入亮度、对比度、gamma、噪声、JPEG、motion blur、defocus blur。
- 目的：让 MyNet 适应 detector bbox 抖动和真实视频画质。
- 验证：detector 输出 bbox 下的推理稳定性，而不是只看手工 bbox。
- 回退：如果角点漂移增大，减小几何扰动幅度，保留 photometric 增强。

## 5. [ ] 调优 heatmap + coordinate mixed loss

- 做法：在已实现的 `L_coarse + 2.0 * L_fine` 基础上，调优 fine loss 权重、soft-argmax temperature 和可见角点 heatmap 权重。
- 目的：让训练目标直接对齐 `corner_px` 和 PCK。
- 验证：`pck_2` 是否继续上升，且 heatmap debug 中仍保持单峰。
- 回退：如果热力图变散，降低 coordinate loss 权重。

## 6. [ ] 替换 ResNet50

- 做法：保持数据、loss、解码不变，只将 backbone 从 ResNet34 换成 ResNet50。
- 目的：验证更强特征提取能力是否带来稳定收益。
- 验证：与同设置 ResNet34 做 ablation，而不是同时改变其它超参。
- 回退：如果真实验证集变差，说明瓶颈更可能是数据/增强/解码而非 backbone。

## 7. [ ] 尝试 128 heatmap 或 384 crop

- 做法：将输出 heatmap 提升到 `128x128`，或输入 crop 提升到 `384x384`。
- 目的：减少小目标和细角点的空间量化误差。
- 验证：显存、速度、`pck_2` 是否共同可接受。
- 回退：如果收益小或推理慢，保留 `256 crop + 64 heatmap + subpixel`。

## 8. [ ] 推理端加入视频时序平滑

- 做法：对同一实例的 bbox 和角点做 3-5 帧 EMA/median 平滑，低置信角点可回退上一帧。
- 目的：单视频最终结果更稳，降低逐帧闪烁。
- 验证：完整视频可视化是否更连续，不能只看单帧 PCK。
- 回退：快速运动或遮挡时降低平滑窗口。
