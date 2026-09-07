

Train/test script examples
- `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 tools/train.py -c path/to/config &> train.log 2>&1 &`
- `-r path/to/checkpoint`
- `--amp`
- `--test-only` 


Tuning script examples
- `torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth` 


Export script examples
- `python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check`


GPU do not release memory
- `ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9`


Save all logs
- Appending `&> train.log 2>&1 &` or `&> train.log 2>&1`


## 可视化 OXPID_S 的全部测试结果（单类别 object）

训练结束后，在项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/visualize_test.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_oxpid_s.yml \
  -r output/rtdetr_r50vd_6x_ldxray/best.pth \
  --device cuda:0 \
  --conf 0.3 \
  -o output/oxpid_s_test_vis
```

该命令读取配置中的 `val_dataloader`，即 `test_one_object.json` 及其图片目录，
采用与测试评估相同的预处理和后处理，将检测框还原到原图尺寸。
每张测试图片都会保存，包括没有标注或没有预测框超过阈值的图片，最后不足一个 batch 的图片也会保留。
模型按配置选择普通权重或 EMA 权重，与训练中的评估逻辑一致；当前配置 `use_ema: False` 使用普通权重。

输出内容：

- `output/oxpid_s_test_vis/index.html`：用浏览器打开，可以浏览所有结果并点击查看原尺寸。
- `output/oxpid_s_test_vis/predictions/`：原图上的红色预测框，标注 `object` 和置信度。
- `output/oxpid_s_test_vis/comparison/`：左右对照图，左侧为绿色真实标注框，右侧为红色预测框。

文件以 COCO `image_id` 命名，网页显示对应的原始文件名。输出为高质量 JPG，保留原图尺寸；
左右对照图宽度为原图的两倍，并增加说明栏。标注框使用数据集的处理逻辑，会过滤 `iscrowd` 和无效框。
`--conf` 只过滤可视化的预测框，不改变训练或计算 mAP。降低阈值会显示更多候选框，
提高阈值会减少低置信度框；如需查看后处理返回的全部候选框，可设置 `--conf 0`。

`CUDA_VISIBLE_DEVICES=1` 已经选定物理 GPU 1，因此程序内部使用 `--device cuda:0`。
显存不足时可加 `--batch-size 1`；仅使用 CPU 时改为 `--device cpu`。
此脚本使用与现有评估代码相同的全精度推理，不需要传入训练用的 `--amp`。

配置中的数据路径来自 Linux 训练环境。如图片或标注已移动，可在命令末尾追加：

```bash
  --img-folder /实际路径/图片目录 --ann-file /实际路径/test_one_object.json
```

例如本仓库当前本地数据布局对应 `--img-folder data/OXPID_S/val --ann-file data/OXPID_S/test_one_object.json`。

训练命令保持不变：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_oxpid_s.yml --amp
```

每轮评估后，按照 `val_dataloader`（当前为 `test_one_object.json`）上的
**COCO bbox mAP@[0.50:0.95]** 选择最佳模型，分数严格提高时覆盖保存 `best.pth`，相同时保留较早的模型。
第一个有效分数即使为 0，也会保存；无效分数不会被选为最佳。
`output/rtdetr_r50vd_6x_ldxray` 来自当前训练配置的 `output_dir`，其中：

- `best.pth`：最佳轮次的完整训练检查点，可直接用于上面的可视化命令。
- `checkpoint.pth`：最近完成训练及评估的一轮，可用于断点续训。
- `checkpointXXXX.pth`：按 `checkpoint_step` 保存的指定轮次。

检查点包含模型、优化器、学习率调度器、AMP/EMA 状态（如已启用），以及最佳分数和对应轮次。
使用 `-r .../checkpoint.pth` 续训时会恢复最佳记录，后续分数下降不会覆盖原有最佳模型。
旧版检查点没有最佳记录时，从恢复后的首次有效评估开始选择。
`log.txt` 同时记录 `best_epoch` 和 `best_coco_eval_bbox`；轮次编号从 0 开始。
如果要检查最后一轮或其他轮次，把可视化命令的 `-r` 换成相应检查点。
