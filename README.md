# full_combo

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

**一个把经典模型摊开来看的地方。** CV 与 NLP 两棵目录树，每个模型一个文件，结构完整、
可直接 `import` 或复制进自己的工程——写论文对比 baseline、面试前复习结构、
临时需要一个 backbone，翻这一个仓库就够了。

不是框架，不追求工程完备，只追求**打开文件就能看懂这个模型长什么样**。

---

## 目录

- [CV：视觉模型](#cv视觉模型)
- [NLP：语言模型](#nlp语言模型)
- [怎么用](#怎么用)
- [说明](#说明)

---

## CV：视觉模型

`CV/` 下每个 `.py` 都是一个独立可用的 `nn.Module`，覆盖分类骨干、注意力模块与分割网络。

### 经典分类骨干

| 模型 | 文件 | 备注 |
| --- | --- | --- |
| ResNet | `CV/ResNet.py` | 含 ResNet18/34/50/101/152 与 Wide-ResNet |
| ResNeXt | `CV/ResNeXt.py` | 分组卷积版本 |
| SE-ResNet | `CV/SE_ResNet.py` | 加入 Squeeze-and-Excitation 模块 |
| VGG16 | `CV/VGG16.py` | |
| GoogleNet | `CV/GoogleNet.py` | Inception 模块的早期形态 |
| Inception | `CV/Inception.py`、`InceptionV1.py`、`InceptionV2.py`、`InceptionV3.py` | 四个版本并列，方便对比演进 |
| DenseNet | `CV/Densenet.py` | 密集连接 |
| MobileNet | `CV/mobilenetV1.py`、`mobilenetV2.py`、`MobileNetV3.py`、`MobilenetV3_ver2.py` | 轻量级主力，v3 有两种实现 |
| MobileNetXt | `CV/mobilenetXt.py` | |
| ShuffleNet | `CV/ShuffleNet.py` | 通道混洗 |
| EfficientNet | `CV/Efficientnet.py` | 复合缩放 |
| RegNet | `CV/RegNet.py` | 设计空间搜索得到的规则网络 |
| RepVGG | `CV/RepVGG.py` | 训练/推理结构重参数化 |
| MixNet | `CV/MixNet.py` | 混合卷积核 |
| CoAtNet | `CV/CoAtNet.py` | 卷积 + 注意力的混合架构 |

### Transformer 与轻量化变体

| 模型 | 文件 | 备注 |
| --- | --- | --- |
| Vision Transformer | `CV/VisionTransformer.py` | ViT，patch embedding + 多头自注意力 |
| Swin Transformer | `CV/Swin.py` | 层次化窗口注意力 |
| MobileViT | `CV/MobileViT.py` | 移动端 ViT |

### 注意力模块与分割

| 模块 | 文件 | 备注 |
| --- | --- | --- |
| Triplet Attention | `CV/TripletAttention.py` | 即插即用的三路注意力，可直接嵌进任意 backbone |
| U-Net | `CV/UNet_model.py` | 完整分割网络 |
| U-Net 组件 | `CV/UNet_part.py` | 单独的编码/解码块，便于拼装 |
| HRNet | `CV/HRnet/` | 高分辨率保持网络，含 COCO / MPII 人体姿态实验配置与 demo |

## NLP：语言模型

`NLP/` 按「表示学习 → 分类 → 生成 → 序列标注」的顺序编号，基本是照着论文复现的路径。

### 词表示

| # | 模型 | 文件 | 一句话 |
| --- | --- | --- | --- |
| 1 | Skip-Gram + Negative Sampling | `NLP/1-skip-gram+neg.py` | word2vec 的经典训练方式 |
| 2 | GloVe | `NLP/2-Glove.py` | 全局共现矩阵分解 |
| 3 | C2W | `NLP/3-C2W.py` | 基于上下文的词向量 |

### 文本分类

| # | 模型 | 文件 | 一句话 |
| --- | --- | --- | --- |
| 4 | TextCNN | `NLP/4-TextCNN.py` | 多尺度卷积核提取 n-gram 特征 |
| 5 | CharTextCNN | `NLP/5-CharTextCNN.py` | 字符级版本，抗 OOV |
| 6 | FastText | `NLP/6-FastText.py` | 词袋 + 层次 Softmax，速度极快 |
| 9 | HAN | `NLP/9-HAN_Attention.py` | 词级 + 句级双层注意力 |
| 11 | Transformer | `NLP/11-Transformer.py` | 纯注意力编码器 |
| 12 | BERT | `NLP/12-Bert.py` | 预训练 + 微调范式 |

### 生成与序列标注

| # | 模型 | 文件 | 一句话 |
| --- | --- | --- | --- |
| 7 | Seq2Seq | `NLP/7-Seq2Seq.py` | Encoder-Decoder 基础结构 |
| 8 | Seq2Seq + Attention | `NLP/8-Seq2Seq+Attention.py` | Bahdanau 注意力复现 |
| 13 | BiLSTM-CRF | `NLP/13-BiLSTM_CRF.py` | 序列标注标配，CRF 约束标签转移 |
| 14 | Conformer | `NLP/14-conformer/` | 卷积 + 自注意力，语音识别 SOTA 结构 |
| 10 | SGM | `NLP/10-SGM/` | 带完整训练/推理脚本的独立子项目 |

### 任务级实现

| 目录 | 内容 |
| --- | --- |
| `NLP/情感分析/` | rnn、lstm、bi-lstm、treelstm、td_lstm、tc_lstm、atae_lstm、memnet、bert_spc |
| `NLP/文本匹配/` | DSSM、SiameseNet、Compare-Aggregate |
| `NLP/pytorch_example/` | mnist、language_model、dcgan、vae 四个入门示例 |
| `NLP/paper/` | 上述模型对应的原始论文 PDF，对着代码读更快 |

## 怎么用

单个模型文件通常自带 `if __name__ == '__main__':`，可以直接跑一遍看输出尺寸：

```bash
python CV/ResNet.py
```

也可以复制到工程里直接用：

```python
from CV.Swin import SwinTransformer
model = SwinTransformer(...)
```

> `CV/` 下的模型实现多取自 PyTorch / torchvision 官方源码与论文参考实现，为便于阅读整理成了
> 单文件版本；部分文件仍保留了 torchvision 内部的相对导入（如 `from ..transforms._presets import ...`），
> 单独运行时把这几个 import 换成 `torchvision` 的公开接口，或直接删掉即可。

## 说明

- **定位是「查阅」而不是「框架」**：没有统一的 registry、没有配置系统，模型之间不共享训练流程。
  需要成套训练框架时，用 `timm`、`torchvision` 或 `transformers` 更合适。
- 各文件依赖的 PyTorch 版本不完全一致（部分来自较新的 torchvision），遇到 API 报错时按报错
  调整一两个参数即可，结构本身是稳的。
- `NLP/10-SGM`、`NLP/14-conformer` 是带训练脚本的完整子项目，用法见各自目录下的 README。

## 相关仓库

- [TMG-NNet](https://github.com/yiyiliu0415/TMG-NNet) —— 传输矩阵引导的无训练神经网络（OLEN 2026）
- [ReID_using-openvino](https://github.com/yiyiliu0415/ReID_using-openvino) —— OpenVINO 行人重识别检索
