# -*-coding:utf-8-*-
import jieba
import matplotlib.pyplot as plt

from functools import reduce
from wordcloud import WordCloud

TEXT_PATH = ""
STOPWORDS_PATH = ""
OUT_PATH = " "

def main():
    stopwords = set(
        line.strip() 
            for line in open(STOPWORDS_PATH, 'r', encoding='utf-8')
    )
    

    with open(TEXT_PATH, 'r', encoding='utf-8') as fp:
        txt = fp.read()
        # 将jieba分词换成你所实现的分词算法
        words = jieba.cut(txt)
        words = [
            filter(lambda x: x not in stopwords, words)
        ]
        
        sentences = reduce(
            lambda a, b: a + ' ' + b,
            words
        )

        # ⽣成词云就这⼀步
        wordcloud = WordCloud(
            background_color='white',
            font_path="Library/SourceHanSerif-Heavy.ttc",
            width=2000,
            height=2000,
        ).generate(sentences)
        # 输出词云图⽚，⾃⾏学习matplotlib.pyplot如何使⽤
        
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.savefig(OUT_PATH)
        # plt.show()

if __name__ == '__main__':
    main()