import jieba
import numpy as np
import csv
import random
import re
import matplotlib.pyplot as plt
# 清洗文本，移除空格和特殊符号
def clean_text(text):
    text = re.sub(r'\s+', '', text)  # Remove all spaces and non-word characters
    return text

# 使用jieba进行分词
def tokenize_poem(poem):
    return list(jieba.cut(poem))

# 读取和处理诗歌数据
poems = []
with open('data.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        # if random.random() < 0.03:
            poem_content = clean_text(row['content'])
            poems.append(tokenize_poem(poem_content))

# 构建词汇表
vocab = set()
for poem in poems:
    for word in poem:
        vocab.add(word)

vocab_size = len(vocab)
char_to_index = {char: i for i, char in enumerate(vocab)}
index_to_char = {i: char for i, char in enumerate(vocab)}

# 模型参数初始化
input_size = vocab_size
hidden_size = 5 # 可以调整这个值以优化模型
output_size = vocab_size
rate = 0.01

Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))

# 前向和后向传播
def forward_backward(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    # RNN通过上一次的h与x共同作用
    hs[-1] = np.copy(hprev)
    loss = 0

    for t in range(len(inputs)):
        xs[t] = np.zeros((input_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])

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

# 每五个为一个序列
length = 5 
# 训练参数
num = 500000  # 迭代次数
patience = 800   # 耐心值

def train(data, num, patience=500):
    global Wxh, Whh, Why, bh, by  # 全局
    words = data.split()  # 将字符串分割成词的列表
    lowest_loss = np.inf # 迭代找最小
    best = {}
    counter = 0

    n, p = 0, 0
    hprev = np.zeros((hidden_size, 1))
    losses = [] # 记录loss
    lossess = []
    for i in range(num):
        if p + length + 1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))
            p = 0  

        inputs = [char_to_index[words[p+j]] for j in range(length)]
        # targets = [char_to_index[words[p+j+1]] for j in range(length)]
        targets = [char_to_index[words[p+j+1]] if p+j+1 < len(words) else char_to_index[words[0]] for j in range(length)]
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = forward_backward(inputs, targets, hprev)

        # 梯度下降
        for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
            param += -rate * dparam

        p += length  
        n += 1  
        if n%10 ==0:
            lossess.append(loss)
        if n % 200 == 0:
            print(f"Epoch {n}, Loss: {loss}")
            losses.append(loss)
        
        if loss < lowest_loss:
            lowest_loss = loss
            best = {
                'Wxh': Wxh.copy(), 
                'Whh': Whh.copy(), 
                'Why': Why.copy(), 
                'bh': bh.copy(), 
                'by': by.copy()
            }
            counter = 0
        else:
            counter += 1

        # if counter > patience:
        #     print(f"Early stopping at iteration {n}, Lowest loss: {lowest_loss}")
        #     break

    Wxh, Whh, Why, bh, by = best.values()
    plt.plot(range(len(losses)),losses)
    plt.xlabel('num')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    plt.scatter(range(len(lossess)), lossess, color='red', marker='o')
    plt.xlabel('num')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    return best, lowest_loss

# 将分词后的诗句转换为字符串
poems = [' '.join(poem) for poem in poems]
poems = [re.sub(r'[^\w\s]', '', poem) for poem in poems]
# poems = [re.sub(r'[\s\n]', '', poem) for poem in poems]
# 现在 `poems` 是一个字符串列表，可以安全地合并为一个长字符串
data = ' '.join(poems)


best_params, lowest_loss = train(data, num, patience)

print(f"lowest loss: {lowest_loss}")

def generate_line(start_char, length=5):

    if start_char not in char_to_index:
        random_char = np.random.choice(list(vocab))
        start_index = char_to_index[random_char]
    else:
        start_index = char_to_index[start_char]

    x = np.zeros((input_size, 1))
    x[start_index] = 1
    h = np.zeros((hidden_size, 1))
    indices = [start_index]  

    for _ in range(length - 1): 
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y_hat = np.dot(Why, h) + by
        prob = np.exp(y_hat) / np.sum(np.exp(y_hat))
        next_char = np.random.choice(range(vocab_size), p=prob.ravel())
        indices.append(next_char)
        x = np.zeros((input_size, 1))
        x[next_char] = 1

    generated_poem = "".join(index_to_char[i] for i in indices)
    return start_char + generated_poem[1:]


chars = input("请输入藏头诗的头: ").split(' ')

for char in chars:
    generated_line = generate_line(char, length=5)
    print(generated_line)