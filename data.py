import pandas as pd
import re

df = pd.read_csv('data.csv')

def remove_parentheses(text):
    return re.sub(r'\([^)]*\)', '', text)

df = df.applymap(remove_parentheses)

df.to_csv('data.csv', index=False)