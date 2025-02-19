

# YOLO-DualSpectral 双光融合目标检测模型

📌 **GitHub地址**: [https://github.com/your_username/YOLOv10-DynamicFusion](https://github.com/your_username/YOLOv10-DynamicFusion)  
*基于YOLOv10改进的动态双光融合模型，支持阶段性跨模态特征增强，适用于低光场景目标检测。*

---

## 📖 项目简介
本项目基于**YOLO**框架，针对低光场景提出动态双光（可见光+红外）融合策略。核心改进包括：
1. **跨模态动态融合模块**：通过Transformer实现跨模态注意力（交互）与模态内部注意力（增强），结合前馈网络生成融合特征。
2. **阶段性训练策略**：前n个epoch仅用红外数据，后续按比例融合可见光，提升模型对低光场景的鲁棒性。
3. **动态融合卷积**：替代传统特征拼接，实现更高效的多模态信息融合。

---

## 🛠️ 模型架构
### 1. 动态跨模态融合模块
- **跨模态注意力**：通过交叉注意力机制对齐可见光与红外特征。
- **模态内部注意力**：增强单模态特征表示能力。
- **前馈网络**：整合双模态信息，输出优化后的融合特征。

```python
class DynamicFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 跨模态注意力机制
        self.cross_attn_ir = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn_vi = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # 模态内部自注意力
        self.self_attn_ir = nn.MultiheadAttention(d_model, nhead//2, batch_first=True)
        self.self_attn_vi = nn.MultiheadAttention(d_model, nhead//2, batch_first=True)
         # 前馈网络
        self.ffn_ir = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_vi = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # 归一化层
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
      
```

### 2. 阶段性训练策略
```yaml
task_stages:
  - epochs: 49       # 前50epoch
    ir_ratio: 1.0    # 仅红外
  - epochs: 79       # 50-80epoch
    ir_ratio: 0.8    # 红外80% + 可见光20%
  - epochs: 100      # 80-100epoch
    ir_ratio: 0.6    # 红外60% + 可见光40%
```

---

## 📊 数据集准备
### LLVIP 数据集
- **简介**：包含30,976张严格对齐的可见光-红外图像对，标注行人目标。
- **下载**：[LLVIP GitHub](https://github.com/bingqixuan/LLVIP) | [预处理版本](https://blog.csdn.net/2301_77697936/article/details/142790634)
- **结构**：
  ```
  datasets/LLVIP/
    ├── images/         # 图像对
    │   ├── visible/    # 可见光
    │   └── infrared/   # 红外
    └── labels/         # YOLO格式标注
  ```

---

## ⚙️ 训练配置
参考`dota8.yaml`修改数据集路径：
```yaml
# ultralytics/cfg/datasets/llvip.yaml
path: ../datasets/LLVIP
train: images/train
val: images/val
names:
  0: person
```

启动训练：
```bash
yolo detect train data=llvip.yaml model=yolov10n-dynamicfusion.pt epochs=100 imgsz=640
```

---


## 📚 参考文献
1. DOTA8数据集配置参考 [Ultralytics Docs](https://docs.ultralytics.com/zh/datasets/obb/dota8/)
2. 动态融合理论 [BLVD数据集论文](https://arxiv.org/pdf/1903.06405.pdf)
3. LLVIP数据集细节 [GitHub](https://github.com/lovepreeminence/Image-Fusion)
4. 数据格式转换工具 [Datumaro](https://www.51openlab.com/article/453/)

---

## 💡 注意事项
- 数据路径需根据实际位置调整。
- 阶段性融合比例可通过修改`task_stages`调整。
- 推荐使用马赛克增强提升小目标检测。

欢迎提交Issue或PR共同改进！🚀