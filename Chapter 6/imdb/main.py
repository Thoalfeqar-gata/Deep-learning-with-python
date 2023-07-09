from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Embedding, Flatten, Dense, Dropout
import numpy as np, os, keras

labels = []
texts = []
test_texts = []
test_labels = []
data_dir = 'Chapter 6/imdb/data/aclImdb/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

for label_type in ['neg', 'pos']:
    for directory in [train_dir, test_dir]:
        dir_name = os.path.join(directory, label_type)
        for filename in os.listdir(dir_name):
            if filename.endswith('.txt'):

                try:
                    with open(os.path.join(dir_name, filename)) as file:
                        contents = file.read()

                except UnicodeDecodeError:
                    continue
                
                if directory == train_dir:
                    texts.append(contents)
                        
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)
                        
                elif directory == test_dir:
                    test_texts.append(contents)
                        
                    if label_type == 'neg':
                        test_labels.append(0)
                    else:
                        test_labels.append(1)

total_samples = len(labels)
max_length = 100
validation_samples = 5000
training_samples = total_samples - validation_samples
max_words = 10000
embedding_dim = 100
use_pretrained_embeddings = False

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts) 
data = pad_sequences(sequences, maxlen = max_length, padding = 'post', truncating = 'post')
labels = np.asarray(labels)

if use_pretrained_embeddings:
    glove_dir = 'Chapter 6/imdb/Glove'
    embeddings_index = {}
    with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding = 'utf8') as file:
        for line in file:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype = np.float32)
            embeddings_index[word] = embedding
            
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    

indices = np.arange(data.shape[0], dtype = np.int32)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples : training_samples + validation_samples]
y_val = labels[training_samples : training_samples + validation_samples]

model = keras.models.Sequential([
    Embedding(max_words, embedding_dim, input_length = max_length),
    Flatten(),
    Dense(32, 'relu'),
    Dropout(0.3),
    Dense(1, 'sigmoid')
])

if use_pretrained_embeddings:
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

model.summary()
model.compile('rmsprop', 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 25, batch_size = 32, validation_data = [x_val, y_val])
model.save_weights('Chapter 6/imdb/models/imdb_model.h5')

sequences = tokenizer.texts_to_sequences(test_texts)
x_test = pad_sequences(sequences, maxlen = max_length, padding = 'post', truncating = 'post')
y_test = np.asarray(test_labels)

model.evaluate(x_test, y_test)