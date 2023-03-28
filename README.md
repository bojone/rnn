# 线性RNN的相关变体

用bert4keras实现三个快速可并行的RNN变体：LRU、SLRU和RWKV。

## 简介

- 中文博客：https://kexue.fm/archives/9554
- LRU论文：https://arxiv.org/abs/2303.06349
- RWKV链接：https://github.com/BlinkDL/RWKV-LM

## 并行

线性RNN支持并行算法，可以将O(L)的运算降低到O(log L)，本项目利用的是prefix sum问题的“Upper/Lower算法”来实现RNN并行。

具体细节可以参考中文博客的“[并行化](https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96)”一节

## 交流
QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
