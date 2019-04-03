## 测试ELMo词向量
- 任务: 文本分类 (text classification)
- 数据集： 电影评论情感分类
- 模型： TextCNN
- 词向量:
    - ELMo
    - GloVe
    - ELMo + GloVe

## 使用方法 Usage
- 环境:
    - pytorch 1.0
    - AllenNLP
    - sklearn
    - fire
- 克隆代码到本地 (包含ELMo 512维的参数文件, 以及GloVe向量文件, 总大小共100M左右)
- 解压data/elmo.7z 
- 运行代码：
    ```
    python main.py train --emb_method='elmo'
    ```
- 可配置项: 参看`config.py`, 比如使用`--gpu_id=1`来指定使用的GPU

## 结果
ELMo速度要慢，效果提升不太明显

TODO

