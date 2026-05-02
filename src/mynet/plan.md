# MyNet 当前主线计划：真实相机数据适配 + 全角点监督

## 结论

老师的建议是当前主线：不要继续把重点放在“可见/不可见角点分开检测”上，而应优先生成更接近真实视频的数据，并减小送入 MyNet 的 ROI 背景比例。

当前采用：

- 全 8 角点监督：默认所有角点都生成 heatmap 高斯峰并参与坐标 loss。
- 更紧 ROI crop：减少背景，让网络更关注 DJI Action4 本体。
- 数据分布优先：相机内参、视角、距离、成像风格、bbox 分布要向真实视频靠拢。

## 已完成代码调整

- [x] `scripts/build_mynet_dataset.py` 默认回退到全角点监督：
  - `--corner-visibility-source all_corners`
  - 不再默认依赖 `scene_gt_corners.json`
  - `corner_valid` 主要由角点是否落在 crop/heatmap 内决定，正常情况下应接近全 1

- [x] `scripts/build_mynet_dataset.py` 默认使用更紧的 crop：
  - `--bbox-source mask_projected_union`
  - `--padding-ratio 0.10`
  - bbox 同时考虑 mask/bbox 与 8 个投影角点，避免只用 mask 时裁掉 3D bbox 角点

- [x] 保留已有可见性实验代码：
  - `scene_gt_corners` 路线仍可用于实验
  - 但不作为当前主线训练方式

## 当前推荐数据生成命令

```bash
python scripts/build_mynet_dataset.py \
  --bop-root dji-action4-real-wrist-jacket-occlusion \
  --output-root data/dji_action4_real_camera_tight_allcorners \
  --corners-meta dji_action4_real_wrist_jacket_occlusion_mynet/meta.json \
  --corner-visibility-source all_corners \
  --bbox-source mask_projected_union \
  --padding-ratio 0.10 \
  --overwrite
```

如果发现 crop 仍然背景过多，可以尝试：

```bash
--padding-ratio 0.05
```

如果发现角点被裁掉或 heatmap 缺失，则回到：

```bash
--padding-ratio 0.15
```

## 当前推荐训练命令

```bash
python -m src.mynet.train \
  --data-root data/dji_action4_real_camera_tight_allcorners \
  --output-dir models/checkpoints/mynet_resnet34_tight_allcorners \
  --epochs 300 \
  --batch-size 128 \
  --num-workers 8 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --amp \
  --tensorboard
```

在这一路线下，`invisible_*` 指标不再作为主要判断依据。核心看：

- `val/corner_px`
- `val/pck_2`
- `val/pck_5`
- 真实视频可视化效果
- 少量真实标注帧上的 PCK

## 数据生成需要继续优化的方向

1. 相机更真实
   - 使用真实视频接近的 `K`
   - 匹配真实视频分辨率、焦距、主点和目标尺度
   - 让渲染中 DJI 在画面中的大小接近 detector crop 中的大小

2. 姿态更真实
   - 统计真实视频中 DJI 的常见朝向、俯仰角、旋转角
   - 减少真实视频中几乎不会出现的极端角度
   - 保留两个 DJI 同屏的场景，因为测试视频中确实同时出现两个目标

3. crop 更接近推理
   - 训练 crop 不应包含过多背景
   - 训练 crop 的 padding 应接近 `predict_video.py` 推理时的 `--bbox-padding`
   - 建议训练和推理先统一使用 `0.10`

4. 图像风格更接近真实视频
   - 加入运动模糊、压缩噪声、曝光变化、白平衡变化
   - 加强高光、反光、暗部噪声和背景复杂度
   - 使用真实视频帧背景或相似背景替代过于干净的渲染背景

5. 用真实小验证集校准
   - 从真实视频抽 30-50 帧
   - 标注 bbox 和 8 角点
   - 固定为真实验证集，不参与训练或只少量微调
   - 用它判断数据生成是否真的更接近真实视频

## 验证清单

- [ ] 检查新数据集 `corner_valid` 是否基本为 8 个 1。
- [ ] 随机查看 `debug_vis`，确认 8 个角点都在 crop 内。
- [ ] 对比旧 crop 与新 crop，确认背景明显减少。
- [ ] 用新模型跑真实视频，分别测试：
  - `--bbox-padding 0.10`
  - `--bbox-padding 0.15`
  - `--bbox-padding 0.25`
- [ ] 用真实标注帧计算 PCK，避免只相信合成验证集。

## 暂时降级的路线

以下内容保留为工具，但不作为当前主线：

- 不可见角点全 0 heatmap 监督
- `scene_gt_corners.json` 外部遮挡估计
- 根据 heatmap score 隐藏低置信度角点
- 推理端 affine 补全低置信度角点

这些功能仍可用于诊断视频结果，但当前提升泛化的第一优先级是：**训练数据更真实、ROI 更紧、全角点稳定监督。**
## ROI padding 可调与紧 crop 流程

- [x] `scripts/build_mynet_dataset.py` 支持两种训练 crop padding：
  - `--padding-ratio 0.10`：按 bbox 最大边比例扩边，适合目标尺寸变化较大时使用。
  - `--padding-pixels 8`：每边固定扩 8 像素，传入后覆盖 `--padding-ratio`，适合希望严格控制背景比例时使用。
- [x] `src/mynet/infer_image.py` 与 `src/mynet/predict_video.py` 同步支持：
  - `--bbox-padding 0.10`
  - `--bbox-padding-pixels 8`

当前仍推荐保持正方形 ROI，不把网络改成任意矩形输入。原因是 heatmap 坐标、ResNet 输入尺寸和视频推理流程都假设 `crop_size x crop_size`。因此“贴合 bbox”的含义是：取能包住 bbox/投影角点的最小正方形，再只加很小 padding。

推荐实验顺序：

1. 用 `--bbox-source mask_projected_union` 保证 crop 同时包住 mask、bbox 和 8 个投影角点。
2. 生成 2-3 组小样本，分别测试 `--padding-ratio 0.05`、`--padding-ratio 0.10`、`--padding-pixels 8`。
3. 检查 `debug_vis`：8 个角点都应在 crop 内，DJI Action4 在 crop 中占主要面积，背景不应明显多于物体主体。
4. 选择最紧且不裁掉角点的一组，生成完整数据集并训练。
5. 视频推理时使用与训练集一致的 padding 方式；如果训练用 `--padding-pixels 8`，推理也用 `--bbox-padding-pixels 8`。
