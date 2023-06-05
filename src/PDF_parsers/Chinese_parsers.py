import glob
import os
import json
import jieba
import time
from PyPDF2 import PdfReader
import numpy as np
import sys
sys.path.append(r'D:\Sec_sem_ju\inforretrieval\src\PDF_parsers')
from mytools import is_chinese

class ChineseParser:
    def __init__(self, pdf_dir, txt_dir):
        # 接收的pdf目录
        self.pdf_dir = pdf_dir
        # 存储pdf文件路径的列表
        self.pdf_dir_ids = list()

        # 接收的pdf目录
        self.txt_dir = txt_dir
        # 存储pdf文件路径的列表
        self.txt_dir_ids = list()

        # self.vocabulary是一个列表用于存储文本中出现的所有单词
        self.vocabulary = list()
        self.inverted_index = None

    def read_inverted_index(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                self.inverted_index = json.load(f)
            f.close()
        else:
            # do something else
            pass

    def parseChinese(self):
        
        self.read_inverted_index("data/mydata/inverted_index.json")
        if self.inverted_index:

            return self.inverted_index
        # 获取pdf文件地址，并扩展至self.pdf_dir_ids
        pdf_files = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
        for index in range(len(pdf_files)):
            self.pdf_dir_ids.append(pdf_files[index])

        # 逐个打开PDF文件并提取文本
        # 构建词汇表
        for index in range(len(self.pdf_dir_ids)):
            with open(self.pdf_dir_ids[index], 'rb') as pdf:
                pdf_reader = PdfReader(pdf)
                # 获取PDF文件中的页数
                pages = len(pdf_reader.pages)
                # 提取每一页的文本，并逐页处理文本
                for page_number in range(pages):
                    page = pdf_reader.pages[page_number]
                    # 从PDF文件的当前页中提取文本并将其存储到text变量中
                    # 进行了编码转换，使用的编码方式是'latin1'，解码方式是'gbk'
                    text = page.extract_text().encode('latin1', 'ignore').decode('gbk', 'ignore')
                    # 处理提取的文本信息
                    # print(text)
                    # 分词测试，用raw_seg_list存储本次分词结果，是个列表
                    raw_seg_list = jieba.lcut(text, cut_all=True)

                    # 过滤非中文
                    seg_list = list()
                    for token_index in range(len(raw_seg_list)):
                        if is_chinese(raw_seg_list[token_index]):
                            seg_list.append(raw_seg_list[token_index])
                    # print("|".join(seg_list))
                # 每处理完一页就加入self.vocabulary
                self.vocabulary.extend(seg_list)
                # 去重复操作
                self.vocabulary = list(set(self.vocabulary))
        
            # print(self.vocabulary)
        
        txt_files = glob.glob(os.path.join(self.txt_dir, "*.txt"))
        for index in range(len(txt_files)):
            self.txt_dir_ids.append(txt_files[index])

        for index in range(len(self.txt_dir_ids)):
            with open(self.txt_dir_ids[index], 'rb') as txt:
                text = txt.read()
                # 将文本转换为包含所有中文的列表
                raw_seg_list = jieba.lcut(text, cut_all=True)
                # 过滤非中文
                seg_list = list()
                for token_index in range(len(raw_seg_list)):
                    if is_chinese(raw_seg_list[token_index]):
                        seg_list.append(raw_seg_list[token_index])
            # 每处理完txt就加入self.vocabulary
            self.vocabulary.extend(seg_list)
            # 去重复操作
            self.vocabulary = list(set(self.vocabulary))

        # 读取停用词
        with open('data/mydata/stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = f.read().split()
        # print(stopwords)
        # os.system("PAUSE")
        # 去除停用词
        new_vocabulary = [word for word in self.vocabulary if word not in stopwords]
        self.vocabulary = new_vocabulary
        # 词汇表生成成功
        # print(self.vocabulary)

        # 术语文档关联矩阵，其中矩阵元素的值为0或1。
        # 倒排索引中每个词对应的文档集合初始化为空集合
        self.inverted_index = {word: {} for word in self.vocabulary}

        # 遍历所有PDF文件，查找每个文件出现过的中文词汇并相应地更新倒排索引
        for index in range(len(self.pdf_dir_ids)):
            with open(self.pdf_dir_ids[index], 'rb') as pdf:
                pdf_reader = PdfReader(pdf)
                # 获取PDF文件中的页数
                pages = len(pdf_reader.pages)
                # 提取每一页的文本，并逐页处理文本
                for page_number in range(pages):
                    page = pdf_reader.pages[page_number]
                    # 从PDF文件的当前页中提取文本并将其存储到text变量中
                    text = page.extract_text().encode('latin1', 'ignore').decode('gbk', 'ignore')
                    # 将文本转换为包含所有中文的列表
                    raw_seg_list = jieba.lcut(text, cut_all=True)
                    seg_list = [w for w in raw_seg_list if is_chinese(w)]

                    # 统计每个词在文档中出现的次数
                    word_count = {}
                    for word in seg_list:
                        if word in self.vocabulary:
                            if word in word_count:
                                word_count[word] += 1
                            else:
                                word_count[word] = 1

                    # 遍历每个中文词汇，更新对应的文档集合
                    # 更新倒排索引
                    for word, count in word_count.items():
                        if word in self.inverted_index:
                            if self.pdf_dir_ids[index] in self.inverted_index[word]:
                                self.inverted_index[word][self.pdf_dir_ids[index]] += count
                            else:
                                self.inverted_index[word][self.pdf_dir_ids[index]] = count

        # 遍历所有txt文件，查找每个文件出现过的中文词汇并相应地更新倒排索引
        for index in range(len(self.txt_dir_ids)):

            with open(self.txt_dir_ids[index], 'rb') as txt:
                text = txt.read()
                # 将文本转换为包含所有中文的列表
                raw_seg_list = jieba.lcut(text, cut_all=True)

                seg_list = [w for w in raw_seg_list if is_chinese(w)]

                # 统计每个词在文档中出现的次数
                word_count = {}
                for word in seg_list:
                    if word in self.vocabulary:
                        if word in word_count:
                            word_count[word] += 1
                        else:
                            word_count[word] = 1

                # 遍历每个中文词汇，更新对应的文档集合
                for word, count in word_count.items():
                    if word in self.inverted_index:
                        self.inverted_index[word][self.txt_dir_ids[index]] = count
        # 成功构建倒排索引
        # print(self.inverted_index)
        # os.system("PAUSE")
        with open('data/mydata/inverted_index.json', 'w') as f:
            json.dump(self.inverted_index, f)
        return self.inverted_index


if __name__ == '__main__':
    # start_time = time.time()
    ChineseParser("data/pdfdata", "data/txtdata").parseChinese()
    # end_time = time.time()
    # interval = end_time - start_time
    # print("程序总体运行时间为：", interval, "秒")
