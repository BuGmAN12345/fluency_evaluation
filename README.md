# Multilingual Fluency Scorer

这是一个基于BERT多语言模型的文本流畅度评估工具，通过智能滑动窗口技术和概率修正机制，能够快速准确地评估多语言文本的流畅程度。特别适合处理混合语言文本和长文本场景，虽然在某些情况下可能会对混合语言给出偏高的评分，但整体效果较好。可以用于语言分析以及LLM安全检测上

**主要功能亮点**：  
✨ 采用动态滑动窗口（默认窗口半径5个词）智能捕捉上下文关系，有效消除文本长度对评分的影响  
✨ 独创UNK字符概率补偿算法，通过随机权重调整降低未知词对评分的干扰  
✨ 支持50+语言的流畅度评估，特别优化了中/英/法/德等常用语言的处理逻辑  
✨ 内置本地模型自动加载机制，网络异常时无缝切换本地预训练模型  
✨ 批量推理加速技术（默认batch_size=20）使处理速度提升3-5倍  

**快速上手**：  
```python
from multilingual_fluency_scorer import MultilingualFluencyScorer

# 初始化评估器（自动检测GPU加速）
scorer = MultilingualFluencyScorer(
    window_size=5,  # 上下文窗口半径
    batch_size=20   # 并行计算量
)

# 评估混合语言文本
mixed_text = "深度学习是AI领域的重要方向，需要大量data和算力支持。"
score = scorer.calculate_fluency_score(
    mixed_text,
    max_sample_num=100,    # 最大采样token数
    low_boundary=0.1,      # 低概率判定阈值
    weight_low_basic=1.5,  # 初始补偿权重
    asy_e=1                # 权重增长系数
)
print(f"文本通顺度：{score}/100")
```
使用注意：
⚠️ 由于mBERT模型的跨语言特性，中英混合文本可能获得比纯中文更高的评分
⚠️ 遇到特殊符号或罕见词时，建议适当提高low_boundary参数（0.15-0.2）
⚠️ 长文本处理时设置max_sample_num=150-200可获得速度与精度的最佳平衡
⚠️ 首次使用会自动下载约1.7GB的多语言BERT模型（可通过local_path参数指定本地路径）

技术优势：

处理100字文本仅需0.4秒（3050笔记本版本，CUDA GPU环境）
动态随机采样机制使长文本评估耗时不再线性增长
智能padding策略完美适配512token的BERT输入限制
概率加权算法自动修正UNK字符的评分偏差
