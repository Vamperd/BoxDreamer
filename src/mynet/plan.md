# MyNet PCK2 提升实验顺序

目标：优先提升当前真实视频上的 8 角点稳定性，尤其关注 `pck_2`、`corner_px` 和完整视频可视化效果。

## 1. 只改解码，先验证已有 checkpoint

- 做法：在不重新训练的情况下，将 heatmap 解码从 hard argmax 切到 `subpixel` 局部加权质心。
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

## 2. 使用真实视频标注帧低学习率微调

- 做法：补充当前视频中的真实 bbox 和 8 角点标注，用 `lr=1e-5~3e-5` 从已有 checkpoint 微调。
- 目的：解决合成数据到真实视频的域差异。
- 验证：真实视频验证段的 `pck_2/pck_5` 和完整视频可视化是否提升。
- 回退：如果过拟合，减少 epoch、降低真实帧重复采样比例，或冻结 backbone 只训 head。

## 3. 加 crop jitter 与真实视频风格增强

- 做法：训练时随机扰动 bbox 中心、尺度、padding，并加入亮度、对比度、gamma、噪声、JPEG、motion blur、defocus blur。
- 目的：让 MyNet 适应 detector bbox 抖动和真实视频画质。
- 验证：detector 输出 bbox 下的推理稳定性，而不是只看手工 bbox。
- 回退：如果角点漂移增大，减小几何扰动幅度，保留 photometric 增强。

## 4. 加 heatmap + coordinate mixed loss

- 做法：保留 heatmap loss，同时从 soft/subpixel 解码坐标加 `SmoothL1Loss`。
- 目的：让训练目标直接对齐 `corner_px` 和 PCK。
- 验证：`pck_2` 是否继续上升，且 heatmap debug 中仍保持单峰。
- 回退：如果热力图变散，降低 coordinate loss 权重。

## 5. 替换 ResNet50

- 做法：保持数据、loss、解码不变，只将 backbone 从 ResNet34 换成 ResNet50。
- 目的：验证更强特征提取能力是否带来稳定收益。
- 验证：与同设置 ResNet34 做 ablation，而不是同时改变其它超参。
- 回退：如果真实验证集变差，说明瓶颈更可能是数据/增强/解码而非 backbone。

## 6. 尝试 128 heatmap 或 384 crop

- 做法：将输出 heatmap 提升到 `128x128`，或输入 crop 提升到 `384x384`。
- 目的：减少小目标和细角点的空间量化误差。
- 验证：显存、速度、`pck_2` 是否共同可接受。
- 回退：如果收益小或推理慢，保留 `256 crop + 64 heatmap + subpixel`。

## 7. 推理端加入视频时序平滑

- 做法：对同一实例的 bbox 和角点做 3-5 帧 EMA/median 平滑，低置信角点可回退上一帧。
- 目的：单视频最终结果更稳，降低逐帧闪烁。
- 验证：完整视频可视化是否更连续，不能只看单帧 PCK。
- 回退：快速运动或遮挡时降低平滑窗口。
