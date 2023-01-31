import os
import re
import numpy as np
import argparse
import pickle

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.keras.backend import set_session

## for tensor 1
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
set_session(sess)
print("use-gpu:", tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

corpusFile = "banav2.txt"
corpusSequenceFile = corpusFile[:-4] + "_" + "char_sequences.txt"
seq_length = 10
epochs = 50
# part_size = 1000000
part_size = 10000
#batch_size = 512
batch_size = 256
period = 5

def checkCorpus(string):
    currentDir = os.listdir()
    if (string in currentDir and os.path.isfile(string)):
        return string
    else:
        # print("No folder named %s" % string)
        return -1
def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    bos = "{"
    eos = "}"
    test = []
    temp = 0
    for i in newString.split("\n"):
        i = bos + i + eos
        if temp == 0: 
            print(i)
            temp = 1
        test.append(i)
    newString = " ".join(test).strip()

    # remove punctuations
    # INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    #newString = re.sub("[^abcdefghijklmnopqrstuvwxyzàáâãäéêìíïðóôúýăđĕĩĭŏũŭơư̆ạảấầậắằặẹẽếềểễệỉịốổỗộớờỡợụủữỹ']"," ", newString)
    newString = re.sub("[^{}a-zạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđàáâãäéêìíïðóôúýăđĕĩĭŏũŭơư̆ạảấầậắằặẹẽếềểễệỉịốổỗộớờỡợụủữỹ']", " ", newString)
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=1:
            long_words.append(i)
    return (" ".join(long_words)).strip()

# load doc into memory
def load_data(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_data(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w', encoding='utf-8')
	file.write(data)
	file.close()

#def create_seq(orgText, length, reverted=False):   
def create_seq(orgText, length, reverted=True):
    text = orgText[::-1] if reverted == True else orgText
    sequences = list()
    for i in range(length, len(text)):
        seq = text[i-length:i+1]
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences

def constrain(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x
### neu chua co file ten banav2corpusSequence
# if (not os.path.exists(corpusSequenceFile)):
if True:
    # load text
    raw_text = load_data(corpusFile)

    # clean
    raw_text = text_cleaner(raw_text)

    # organize into sequences of characters
    sequences = create_seq(raw_text, seq_length)
    # sequences = create_seq(raw_text, seq_length,reverted=True)

    # save sequences to file
    save_data(sequences, corpusSequenceFile)

    # load
raw_data = load_data(corpusSequenceFile)
lines = raw_data.split('\n')

chars = sorted(list(set(raw_data)))
mapping = dict((c, i) for i, c in enumerate(chars))

# save the mapping
pickle.dump(mapping, open('name_data_mapping.pkl', 'wb'))

sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)


sequences = np.array(sequences)
X_train, y_train = sequences[:,:-1].copy(), sequences[:,-1].copy()


input_shape = (seq_length, vocab_size)
current_part = 0
max_part = int(len(X_train) / part_size) + 1
lastEpoch = 0
if (os.path.exists('savedEpochs/current_part.txt')):
    with open('savedEpochs/current_part.txt', 'r', encoding='utf8') as f:
        current_part = int(f.read())

# define model

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length = seq_length, trainable=True))
model.add(LSTM(512))
model.add(Dropout(0.15))
model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.load_weights("model-epoch-035.h5")

if (os.path.exists('savedEpochs/part_%d' % current_part)):
    listEpochs = [x for x in os.listdir('savedEpochs/part_%d' % current_part) if x[:12] == 'model-epoch-' and x[-3:] == '.h5']

    if (len(listEpochs) > 0):
        lastEpoch = max([int(x[12:-3]) for x in listEpochs])
        lastEpochFile = 'savedEpochs/part_%d/model-epoch-%03d.h5' % (current_part, lastEpoch)
        # load weights
        model.load_weights(lastEpochFile)
        print("CONTINUE TRAINING FROM PART %d EPOCH %03d......" % (current_part, lastEpoch))
    else:
        lastEpoch = 0
        
print('total-train-data:', len(X_train))
print('total-part:', max_part)
print("current-part: ", current_part)
###########train right#########
for i in range(current_part, max_part):
    with open('savedEpochs/current_part.txt', 'w', encoding='utf8') as f:
        f.write(str(i))

    print("====================================================================")
    print("=                       TRAINING PART %03d                          =" % i)
    print("====================================================================")

    if (not os.path.exists('savedEpochs/part_%d' % i)):
        os.mkdir('savedEpochs/part_%d' % i)

    if (i > current_part):
        lastEpoch = 0

    start_point = i * part_size
    end_point = (i + 1) * part_size
    end_point = constrain(end_point, 0, len(X_train))

    X = X_train[start_point:end_point]
    y = to_categorical(y_train[start_point:end_point], num_classes=vocab_size)

    # continue checkpoint
    checkpoint = ModelCheckpoint('savedEpochs/part_%d/model-epoch-{epoch:03d}.h5' % i, period=period)

    # fit model
    print("co fit")
    model.fit(X, y, epochs=epochs, initial_epoch = lastEpoch, callbacks=[checkpoint], batch_size = batch_size)

#model.save('model.h5')

model.save('model_right.h5')
print("done train")