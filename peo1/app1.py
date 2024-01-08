from flask import Flask, request, render_template
import numpy as np
import pickle

def load_model():
    Wxh = np.load('Wxh.npy')
    Whh = np.load('Whh.npy')
    Why = np.load('Why.npy')
    bh = np.load('bh.npy')
    by = np.load('by.npy')

    with open('char_to_index.pickle', 'rb') as handle:
        char_to_index = pickle.load(handle)

    with open('index_to_char.pickle', 'rb') as handle:
        index_to_char = pickle.load(handle)

    return Wxh, Whh, Why, bh, by, char_to_index, index_to_char

def generate_line(Wxh, Whh, Why, bh, by, char_to_index, index_to_char, start_char, length=5):
    input_size, hidden_size = Wxh.shape[1], Wxh.shape[0]

    if start_char not in char_to_index:
        random_char = np.random.choice(list(char_to_index.keys()))
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
        next_char_index = np.random.choice(range(len(char_to_index)), p=prob.ravel())
        indices.append(next_char_index)
        x = np.zeros((input_size, 1))
        x[next_char_index] = 1

    generated_poem = "".join(index_to_char[i] for i in indices)
    return generated_poem
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    poem = ""
    if request.method == 'POST':
        input_chars = request.form['input_chars'].split(' ')
        Wxh, Whh, Why, bh, by, char_to_index, index_to_char = load_model()
        for char in input_chars:
            generated_line = generate_line(Wxh, Whh, Why, bh, by, char_to_index, index_to_char, char, length=5)
            poem += generated_line + "\n"
    return render_template('index.html', poem=poem)

if __name__ == '__main__':
    app.run(debug=True)
