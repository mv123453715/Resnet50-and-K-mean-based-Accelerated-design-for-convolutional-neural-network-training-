# Resnet50-and-K-mean-based-Accelerated-design-for-convolutional-neural-network-training-

```
基於Resnet50與Kmean之卷積神經網路訓練加速設計
```

使用工具 Anaconda、Python、Pytorch^

1.摘要

卷積神經網路訓練加速一直是深度學習十分熱門的項目，本研究提出將
訓練圖片透過ResNet50卷積到FC層後tensor進行K-mean分群，然後剔除
群中相似的訓練圖片，挑選出Global中最具代表性的訓練資料，以工研究
AOI 瑕疵分類資料集為例，透過本方法可剔除73.6 %訓練資料，也就是訓練
時間加速3.79倍，準確度卻相同。

2.建立VGG 19 模型

3.設定Loss Function, Optimizer, Learning rate, Epochs

4.訓練 2228 張圖片與使用ResNet50和、k-mean後訓練 250 張圖片，提升訓

練速度3.79倍，準確度相同。
