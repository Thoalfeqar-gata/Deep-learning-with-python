import keras, numpy as np, random, sys
from keras import layers

def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype(np.float64)
    preds = preds ** (1/temperature)
    preds = preds / np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


with open('Chapter 8/Generating text with LSTM/text files/Nietzsche.txt') as file:
    text = file.read().lower()

maxlen = 60
step = 1
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])

chars = sorted(list(set(text))) #get the unique chars
char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorizing...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = bool)
y = np.zeros((len(sentences), len(chars)), dtype = bool)

for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x[i, j, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
        
model = keras.models.Sequential([
    layers.LSTM(128, input_shape = (maxlen, len(chars))),
    layers.Dense(len(chars), activation = 'softmax')
])

model.summary()
model.compile('rmsprop', 'categorical_crossentropy')

for epoch in range(1, 60):
    print('Epoch: ', epoch)
    model.fit(x, y, batch_size = 128, epochs = 1)
    start_index = np.random.randint(0, len(text) - maxlen)
    seed_text = text[start_index : start_index + maxlen]
    
    print('Generating with seed: "'+ seed_text + '"')
    
    for temperature in [0.2, 0.5, 1, 1.2]:
        print('\n\nTemperature is set to:', temperature)
        sys.stdout.write(seed_text)
        
        generated_text = seed_text
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1
                
            preds = model.predict(sampled, verbose = 0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            
            generated_text += next_char
            generated_text = generated_text[1:]
            
            sys.stdout.write(next_char)