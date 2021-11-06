'''
Author: wbs2788
Date: 2021-11-04 19:59:34
LastEditTime: 2021-11-05 14:13:52
LastEditors: wbs2788
Description: file information
'''
import main
from wordcloud import WordCloud
import matplotlib.pyplot as plt
stopwords = [line.strip() for line in open('data/stop_word.txt', 'r', encoding
='utf-8')]
with open('wbs.txt', 'r', encoding='utf-8') as f:
    txt = f.read()
    txt = txt.strip()

    # 将jieba分词换成你所实现的分词算法
    fmm = main.FMM(txt)
    bmm = main.BMM(txt)
    words = main.BM(txt, fmm, bmm)
    words = main.list2str(words, txt)
    sentences = ""
    print(len(words))
    for word in words:
        if word in stopwords:
            continue
        sentences += str(word)+' '
    # ⽣成词云就这⼀步
    wordcloud = WordCloud(background_color='white',
        font_path="SourceHanSerifCN-Heavy.otf",
        width=2000,
        height=2000,).generate(sentences)
    # 输出词云图⽚，⾃⾏学习matplotlib.pyplot如何使⽤
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig("Constitution")
    plt.show()