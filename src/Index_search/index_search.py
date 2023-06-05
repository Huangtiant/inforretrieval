import sys
sys.path.append(r'D:\Sec_sem_ju\inforretrieval\src')

from PDF_parsers import ChineseParser
import math 
import jieba
from gensim.models import Word2Vec
import os
import time

class IndexSearch:
    def __init__(self,pdf_path,txt_path):
        self.inverted_index = ChineseParser(pdf_path, txt_path).parseChinese()
        # 加载停用词和词向量模型
        self.stop_words = self.load_stop_words('data/mydata/stopwords.txt')
        self.w2v_model = self.load_w2v_model('data/mydata/w2v_model.bin')

    def vector_search(self, query):
        # 将查询分割成单词
        query_words = jieba.lcut(query)
        # 初始化查询向量
        query_vector = {word: 1 for word in query_words}
        
        # 计算查询向量的长度
        query_vector_length = math.sqrt(sum(w**2 for w in query_vector.values()))
        
        # 初始化结果字典
        results = {}
        
        # 对于每个查询词，计算其文档向量和查询向量之间的余弦相似度
        for word in query_words:
            if word in self.inverted_index:
                # 获取包含该词的文档集合
                documents = self.inverted_index[word]
                # 对于每个包含该词的文档，计算其文档向量和查询向量之间的余弦相似度
                for document, f in documents.items():
                    # 初始化文档向量
                    document_vector = {w: self.inverted_index[w].get(document, 0) for w in self.inverted_index.keys() if w != word}

                    # 计算文档向量的长度
                    document_vector_length = math.sqrt(sum(w**2 for w in document_vector.values()))
                    
                    # 计算文档向量和查询向量之间的余弦相似度
                    similarity = sum(query_vector.get(w, 0) * document_vector.get(w, 0) for w in self.inverted_index.keys() if w !=word) / (query_vector_length * document_vector_length)
                    
                    # 将文档和相似度添加到结果字典中
                    if document in results:
                        results[document] += similarity
                    else:
                        results[document] = similarity
        
        # 对结果字典进行排序并返回前K个结果
        k = 10
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return sorted_results

    def boolean_search(self, query):
        # 将查询分割成单词
        query_words = jieba.lcut(query)
        
        # 初始化结果集
        result_set = set()
        
        # 对于每个查询词
        for word in query_words:
            if word in self.inverted_index:
                # 获取包含该词的文档集合
                documents = self.inverted_index[word]
                
                # 将文档集合添加到结果集中
                for document in documents:
                    result_set.add(document)
    
        # 返回查询结果
        return result_set
    
    def semantic_search(self, query):
        # 分词并去除停用词
        words = [word for word in jieba.cut(query) if word not in self.stop_words]
        
        # 查找相关文档
        # 第一个查询词出现的文档
        doc_ids = set(self.inverted_index[words[0]].keys())

        # 与后续的查询词取交集
        for word in words[1:]:
            if word in self.inverted_index:
                # intersection 方法是 Python 内置的集合操作方法之一，它的作用是求两个集合的交集
                doc_ids = doc_ids.intersection(set(self.inverted_index[word].keys()))
            else:
                continue  # 如果该词不在倒排索引中，则跳过

        # 计算相似度并排序
        results = []
        for doc_id in doc_ids:
            doc_words = self.load_document_words(doc_id)
            score = self.compute_similarity(words, doc_words, doc_id)
            results.append((score, doc_id))

        return sorted(results, reverse=True)[:10]

    def load_stop_words(self, file_path):
        stop_words = []
        with open(file_path, 'r', encoding='utf-8') as f:
            stop_words = f.read().split()
        return stop_words

    def load_w2v_model(self, file_path):
        return Word2Vec.load(file_path)

    # 读取文档
    def load_document_words(self, doc_id):
        with open(doc_id, 'r', encoding='utf-8') as f:
            content = f.read()
            words = [word for word in jieba.cut(content) if word not in self.stop_words]
            return words
    
    # 查询词和文档中的词进行相似度比较
    def compute_similarity(self, words1, words2, document_id):
        # 建一个空列表，用于存储各个单词的相似度。
        similarity = []
        #  words1 和 words2 中的所有单词对。对于每个单词对，
        for word1 in words1:
            for word2 in words2:
                try:
                    # 获取出现次数
                    freq1 = self.inverted_index[word1][document_id]
                    freq2 = self.inverted_index[word2][document_id]
                    # 使用训练好的词向量模型计算两个单词的相似度。
                    sim = self.w2v_model.wv.similarity(word1, word2)
                    # 当前单词对的相似度乘以它们在文档中出现的频次，然后将它们的乘积添加到 similarity 中
                    similarity.append(sim * freq1 * freq2)  # 考虑词汇在文章中的频次
                except KeyError:
                    pass
        # 如果 similarity 列表不为空，则计算它们的平均值并返回，否则返回 0.0。
        if similarity:
            return sum(similarity) / len(similarity)
        else:
            return 0.0
    
if __name__ == '__main__':
    mysearch = IndexSearch("data/pdfdata", "data/txtdata")

    query = input("请输入要查询的文本：")
    start_time = time.time()

    # results = mysearch.boolean_search(query)
    # results = mysearch.vector_search(query)
    results = mysearch.semantic_search(query)
    
    end_time = time.time()
    interval = end_time - start_time
    print("查询运行时间为：", interval, "秒")
    # print(results)
    for score, doc_id in results:
        print(f"Score: {score}, Document ID: {doc_id}")
