import re
import numpy as np
import csv

# 数据预处理和模型参数初始化的代码保持不变

# 数据预处理
poems = []
with open('data.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        poem_content = row['content']
        poems.append(poem_content)

cleaned_poems = [re.sub(r'[^\w\s]', '', poem) for poem in poems]

vocab = set("".join(cleaned_poems))
vocab_size = len(vocab)
char_to_index = {char: i for i, char in enumerate(vocab)}
index_to_char = {i: char for i, char in enumerate(vocab)}

# 模型参数初始化
input_size = vocab_size
hidden_size = 100  # 可以调整这个值以优化模型
output_size = vocab_size
learning_rate = 0.001

Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))

# 前向和后向传播函数保持不变
def forward_backward(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
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

# 调整后的训练函数
def train(data, iter_num):
    n, p = 0, 0
    hprev = np.zeros((hidden_size, 1))

    for i in range(iter_num):
        hprev = np.zeros((hidden_size, 1))  # 重置隐藏状态

        for line in data:
            if len(line) < 2:  # 跳过空行或单字符行
                continue

            inputs = [char_to_index[line[0]]]  # 使用首个字符作为输入
            targets = [char_to_index[char] for char in line[1:]]  # 目标是整行诗的剩余部分

            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = forward_backward(inputs, targets, hprev)

            for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
                param += -learning_rate * dparam

            if n % 100 == 0:
                print(f"Epoch {n}, Loss: {loss}")
            n += 1

# 文本生成函数保持不变

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

# 训练模型
data = ''.join(cleaned_poems)
train(data, iter_num=2000)

# 生成藏头诗
input_chars = input("请输入藏头诗的头: ").split(' ')

for char in input_chars:
    generated_line = generate_line(char, length=5)
    print(generated_line)

