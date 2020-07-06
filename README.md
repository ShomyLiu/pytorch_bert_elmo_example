## 测试Bert/ELMo词向量
- 任务: 文本分类 (text classification)
- 数据集： 电影评论情感分类
- 模型 word embeddings + encoder:
    - word embeddings:
        - Bert
        - ELMo
        - GloVe
    - encoder:
        - CNN+MaxPooling
        - RNN+Last Hidden States
        - PositionEncoding+Transformer+Average Pooling
        - Average all words

博客总结：[Bert/ELMo文本分类](http://shomy.top/2020/07/06/bert-elmo-cls/)


## 使用方法 Usage
- 环境:
    - python3.6+
    - pytorch 1.4+
    - transformers
    - AllenNLP
    - sklearn
    - fire
- 克隆代码到本地, 依据`data/readme.md`说明 下载Bert/ELMo/GloVe的词向量文件
- 运行代码：
    ```
    python main.py train --emb_method='elmo' --enc_method='cnn'
    ```
- 可配置项:
    - emb_method: [elmo, glove, bert]
    - enc_method: [cnn, rnn, transformer, mean]

其余可参看`config.py`, 比如使用`--gpu_id=1`来指定使用的GPU

## 结果

运行环境:
- GPU: 1080Ti
- CPU: E5 2680 V4

此外，实验中我们将Bert与ELMo的参数固定住. 

| Embedding | Encoder | Acc | Second |
| - | - | - | - |
| **Bert** | MEAN | 0.8031 | 17.98s |
| **Bert** | CNN | 0.8397 | 18.35s |
| **Bert** | RNN | 0.8444 | 18.93s |
| **Bert** | Transformer | **0.8472** | 20.95s |
| *ELMo* | Mean | 0.7572 | 25.05s |
| *ELMo* | CNN | 0.8172 | 25.53s |
| *ELMo* | RNN | **0.8219** | 27.18s |
| *ELMo* | Transformer | 0.8051 | 26.209 |
| GloVe | Mean | 0.8003 | 0.60s |
| GloVe | CNN | 0.8031 | 0.76s |
| GloVe | RNN | **0.8219** | 1.45s |
| GloVe | Transformer | 0.8153 | 1.71s |
