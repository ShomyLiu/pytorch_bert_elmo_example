## 测试ELMo词向量
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

## 使用方法 Usage
- 环境:
    - python3.6+
    - pytorch 1.4+
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


| Embedding | Encoder |  Acc| Second|
| ---- | ---- | ----| ---- |
| Bert  | Mean        |     |     |
| Bert  | CNN         |     |     |
| Bert  | RNN         |     |     |
| Bert  | Transformer |     |     |
| ELMo  | Mean        |     |     |
| ELMo  | CNN         |     |     |
| ELMo  | RNN         |     |     |
| ELMo  | Transformer |     |     |
| GloVe | Mean        |     |     |
| GloVe | CNN         |     |     |
| GloVe | RNN         |     |     |
| GloVe | Transformer |     |     |


