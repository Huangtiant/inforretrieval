import gensim
from gensim.models import Word2Vec

# 加载语料库
sentences = gensim.models.word2vec.Text8Corpus('D:\Sec_sem_ju\inforretrieval\data\mydata\\train.txt')

# 训练模型
model = Word2Vec(sentences, vector_size=300, window=5, min_count=5, workers=4)

# 保存模型为二进制文件
model.save('w2v_model.bin')