```
mbti-project/
├── data/
│   ├── raw/                  # A: 原始数据
│   │   └── mbti_1.csv
│   ├── processed/            # A: 清洗后数据
│   │   ├── train.csv
│   │   └── test.csv
│   └── eda_report.html       # A: EDA 报告
├── models/
│   ├── bm25_retrieval.py     # B: BM25 检索
│   ├── tfidf_classifiers.py  # B: TF-IDF + LR/NB
│   ├── embedding_pipeline.py # C: Dense retrieval
│   └── dichotomy_classifiers.py  # D: 四维度二分类
├── evaluation/
│   ├── evaluator.py          # D: 统一评估框架
│   └── confusion_analysis.py # D: Dichotomy 混淆分析
├── app/
│   └── streamlit_app.py      # E: 前端
├── results/                  # 所有人的输出结果
├── requirements.txt
└── README.md
```

