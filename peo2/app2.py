import torch
import pickle
import random
from flask import Flask, request, render_template
# 定义 LSTM 模型类
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# 载入模型和字典
def load_model_and_dictionaries(model_path, word_to_num_path, num_to_word_path):
    with open(word_to_num_path, 'rb') as f:
        word_to_num = pickle.load(f)
    with open(num_to_word_path, 'rb') as f:
        num_to_word = pickle.load(f)

    model = LSTMModel(len(word_to_num), 100, 128)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, word_to_num, num_to_word

# 生成藏头诗的函数
def generate_acrostic(model, sequence_length, start_words, word_to_num, num_to_word):
    poem = []
    for word in start_words:
        word = find_starting_word(word, word_to_num)
        line = generate_line(model, word, sequence_length, word_to_num, num_to_word)
        poem.append(line)
    return '\n'.join(poem)

def generate_line(model, start_word, sequence_length, word_to_num, num_to_word, length=3):
    model.eval() 
    text = start_word
    for _ in range(length - 1):  
        input_sequence = [word_to_num.get(word, random.choice(list(word_to_num.values()))) for word in text][-sequence_length:]
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        next_word = num_to_word[output.argmax(1).item()]
        text += next_word
    return text

def find_starting_word(char, word_to_num):
    starting_words = [word for word in word_to_num if word.startswith(char)]
    if starting_words:
        return random.choice(starting_words) 
    else:
        return char

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    poem = ""
    if request.method == "POST":
        start_words = request.form.get("start_words")
        if start_words:
            # 加载模型和字典
            model, word_to_num, num_to_word = load_model_and_dictionaries('peo2/lstm_model.pth', 'peo2/word_to_num.pkl', 'peo2/num_to_word.pkl')
            # 生成藏头诗
            poem = generate_acrostic(model, 20, start_words, word_to_num, num_to_word)
    return render_template("index.html", poem=poem)

if __name__ == "__main__":
    app.run(debug=True,port=8080)