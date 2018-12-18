import pickle
from MEMM import MEMM


with open('model_1.pkl', 'rb') as input:
    model_1 = pickle.load(input)
model_1.create_competition_file(comp_filename = 'comp.words')

with open('model_2.pkl', 'rb') as input:
    model_2 = pickle.load(input)
model_2.create_competition_file(comp_filename = 'comp2.words')
