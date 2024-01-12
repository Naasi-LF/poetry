import numpy as np
import re
import matplotlib.pyplot as plt
# 使用 with 语句打开文件
# with open('baijing.txt', 'r') as file:
#     # 读取文件内容到变量 data 中
#     data = file.read()
# （莎士比亚的《哈姆雷特》独白）
data = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""
# data = """
# It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way – in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.
# """
# 分词
# 使用正则表达式来处理标点符号和单词
words = re.findall(r'\b\w+\b', data.lower())

# 创建词到索引的映射
unique_words = list(set(words))
vocab_size = len(unique_words)
word_to_index = {w: i for i, w in enumerate(unique_words)}
index_to_word = {i: w for i, w in enumerate(unique_words)}

# 构建输入输出数据
X = []
y = []
for i in range(len(words) - 1):
    X.append(word_to_index[words[i]])
    y.append(word_to_index[words[i + 1]])

# 将数据转换为numpy数组
X = np.array(X)
y = np.array(y)


# RNN参数
input_size = vocab_size
hidden_size = 100
output_size = vocab_size
learning_rate = 0.01

# 参数初始化
Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))

# 定义前向传播和反向传播函数
def forward_backward(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # 前向传播
    for t in range(len(inputs)):
        xs[t] = np.zeros((input_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])

    # 反向传播
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
def train(data, iter_num):
    hprev = np.zeros((hidden_size, 1))
    loss_history = []  # 用于存储每次迭代的损失值

    for i in range(iter_num):
        inputs = X
        targets = y

        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = forward_backward(inputs, targets, hprev)
        loss_history.append(loss)  # 将当前损失添加到历史记录中

        for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
            param += -learning_rate * dparam

        print(f"Epoch {i + 1}, Loss: {loss}")

    # 绘制损失历史
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss during training')
    plt.show()


train(data, iter_num=1500)  # 使用相同的train函数和参数

import random

def generate_poem_with_punctuation(start_word, word_to_index, index_to_word, hidden_size, min_length=5, max_length=8):
    word = start_word
    h = np.zeros((hidden_size, 1))
    poem = [word]

    # 随机确定诗歌长度
    length = random.randint(min_length, max_length)

    for _ in range(length - 1):  # 减1因为已经包含起始词
        x = np.zeros((input_size, 1))
        x[word_to_index[word]] = 1
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        idx = np.random.choice(range(vocab_size), p=p.ravel())
        word = index_to_word[idx]
        poem.append(word)


    poem[-1] += '.'  

    return ' '.join(poem)

generated_poem = generate_poem_with_punctuation(start_word='the', word_to_index=word_to_index, index_to_word=index_to_word, hidden_size=hidden_size)
print(generated_poem)
