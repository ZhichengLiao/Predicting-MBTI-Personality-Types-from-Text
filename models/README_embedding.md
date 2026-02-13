```python
from models.embedding import EmbeddingPipeline

# 1. 初始化
model = EmbeddingPipeline()

# 2. 训练 (一次性训练两个模型)
# texts: 文本列表, labels: 对应的 MBTI 标签

model.fit(texts=["I love coding", "Party time!"], labels=["INTP", "ESFP"])

# 3. 预测
# method='lr' (逻辑回归，更准) 或 'centroid' (质心法，更快)
# predict_proba 看完整预测概率/分数
print(model.predict("I enjoy quiet time", method='lr'))


# 4. 保存
model.save("saved_models/v1")

# 5. 加载
new_pipeline = EmbeddingPipeline()
new_pipeline.load("saved_models/embedding_v1")
```
