from google.colab import drive
drive.mount("/content/drive")

import os
import numpy as np
import regex

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path_to_file = "/content/drive/MyDrive/rnn language model/shakespeare.txt"
with open(path_to_file, "r", encoding="utf-8") as file:
  contents = file.read()

print(len(contents))

type(contents)

contents[:250]

chars = list(set(contents))
print("len of vocab : ", len(chars))
# char_to_idx = {c:i for i, c in enumerate(chars)}
# idx_to_char = {i:c for i, c in enumerate(chars)}
# print(len(idx_to_char))
char_to_idx = tf.keras.layers.StringLookup(vocabulary=chars, oov_token="<unk>")
idx_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_idx.get_vocabulary(), invert=True, oov_token="<unk>")
print("size of vocab : ",char_to_idx.vocabulary_size())

print(contents[:30].split(" "))

text = regex.findall(r'\X', contents, regex.U)
data = char_to_idx(text)

data[:10]

text[:10]

unisplit = tf.strings.unicode_split(contents, "UTF-8")

unisplit[:10]

unisplit_data = char_to_idx(unisplit)

unisplit_data[:10]

print("raw text : ", text[:10])
toks = char_to_idx(text[:10])
print("converted to tokens : ", toks)
chars = idx_to_char(toks)
print("back to chars : ", chars)
print("joined sentence : ", tf.strings.reduce_join(chars).numpy().decode("utf-8"))

split = int(0.90*len(unisplit_data))

train_data = unisplit_data[:split]
valid_data = unisplit_data[split:]

tensor_train = tf.data.Dataset.from_tensor_slices(train_data)
tensor_valid = tf.data.Dataset.from_tensor_slices(valid_data)

tensor_train

for i in tensor_train.take(10):
  print(idx_to_char(i).numpy().decode("utf-8"))

max_len = 100
tensor_train = tensor_train.batch(max_len+1, drop_remainder=True)
tensor_valid = tensor_valid.batch(max_len+1, drop_remainder=True)

for i in tensor_train.take(1):
  print(idx_to_char(i))

for seq in tensor_train.take(5):
  print(tf.strings.reduce_join(idx_to_char(seq), axis=-1).numpy())

print(tf.strings.reduce_join(idx_to_char(i)).numpy().decode('utf-8'))

def create_data(tokens):
  input = tokens[:-1]
  output = tokens[1:]
  return input, output

train = tensor_train.map(create_data)
valid = tensor_valid.map(create_data)

for inp, out in train.take(1):
  print("Input : ", tf.strings.reduce_join(idx_to_char(inp)).numpy())
  print("Output : ", tf.strings.reduce_join(idx_to_char(out)).numpy())

batch_size = 64
buffer = 10000

train = (train.shuffle(buffer)
              .batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.experimental.AUTOTUNE))

valid = (valid.shuffle(buffer)
              .batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.experimental.AUTOTUNE))

train

for i, o in valid.take(1):
  print(i)
  print(o)
  print(i.shape, o.shape)
  break

e = layers.Embedding(66, 128)(i)
print("embedding output : ",e.shape)

# sep, h, c = layers.Bidirectional(layers.LSTM(units=1024, return_sequences=True, return_state=True))(e)
# print(sep.shape)

rnn_layer1 = layers.Bidirectional(layers.LSTM(units=1024, return_sequences=True, return_state=True))
rnn_out1= rnn_layer1(e)
print("rnn1 output : ", rnn_out1[0].shape)
rnn_layer2 = layers.Bidirectional(layers.LSTM(units=1024, return_sequences=True, return_state=True))
rnn_out2= rnn_layer2(rnn_out1[0])
print("rnn2 output : ", rnn_out2[0].shape)

dense = layers.Dense(units=66, activation="softmax")(rnn_out2[0])
print("classifier shape : ", dense.shape)

next, hf, cf, hb, cb = rnn_out1

print(hf.shape)
print(hb.shape)
tst_rnn = layers.LSTM(units=1024, return_sequences=True, return_state=True)
ro = tst_rnn(e, initial_state=[hf, cf])

class lm(keras.Model):
  def __init__(self, embed_dim, vocab_size, rnn_units):
    super().__init__()
    self.embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    self.rnn1 = layers.GRU(units=rnn_units, return_sequences=True, return_state=True)
    self.classifier = layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embed(inputs, training=training)
    if states==None:
      states = self.rnn1.get_initial_state(x)
    rnn1_out, states = self.rnn1(x, initial_state=states, training=training)
    dense_out = self.classifier(rnn1_out, training=training)

    if return_state:
      return dense_out, states
    else:
      return dense_out

class lm(keras.Model):
  def __init__(self, embed_dim, vocab_size, rnn_units):
    super().__init__()
    self.embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    self.lstm1 = layers.LSTM(units=rnn_units, return_sequences=True, return_state=True)
    self.classifier = layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embed(inputs, training=training)
    if states==None:
      states = self.lstm1.get_initial_state(x)
    lstm_out, h, c = self.lstm1(x, initial_state=states, training=training)
    dense_out = self.classifier(lstm_out, training=training)

    if return_state:
      return dense_out, [h, c]
    else:
      return dense_out

vocab_size = char_to_idx.vocabulary_size()
dims = 256
units = 1024
rnn_lm = lm(dims, vocab_size, units)

outputs, states = rnn_lm(i, return_state=True)
print(outputs.shape)

print(h.shape)
print(c.shape)

rnn_lm.summary()

# print(np.random.choice(np.arange(vocab_size+1), outputs[1,1,:].numpy().ravel()))

sampled_idx = tf.random.categorical(outputs, num_samples=1)
# for i in sampled_idx.numpy().ravel():
#   print(idx_to_char[i])

opt = keras.optimizers.Adam(learning_rate=1e-3)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
rnn_lm.compile(optimizer=opt, loss=loss)

for batch in valid:
  x, y = batch
  print(x.shape)
  print(y.shape)
  break

x

y

t = [idx_to_char[i] for i in y[0,:].numpy()]
print("".join(t))

# callbacks 
class evalCallback(keras.callbacks.Callback):
  def __init__(self, eval_data, id_to_ch):
    self.data = eval_data
    self.to_char = id_to_ch
  def on_epoch_end(self, epoch, logs=None):
    predictions = []
    target = []
    for b in self.data:
      x, y = b
      bp = rnn_lm.predict(x)
      for i in range(bp.shape[0]):
        # pred_sent = [self.to_char[i] for i in tf.squeeze(tf.random.categorical(bp[i,:,:], num_samples=1), axis=-1).numpy()]
        pred_sent = self.to_char(tf.squeeze(tf.random.categorical(bp[i,:,:], num_samples=1), axis=-1))
        # pred_sent = "".join(pred_sent)
        pred_sent = tf.strings.reduce_join(pred_sent, axis=-1).numpy().decode("utf-8")
        predictions.append(pred_sent)
        break
      for i in range(y.shape[0]):
        tar_sent = self.to_char(y[i,:])
        tar_sent = tf.strings.reduce_join(tar_sent, axis=-1).numpy().decode("utf-8")
        target.append(tar_sent)
        break
    print("target  : ", tar_sent, "\n")
    print("predicted : ", pred_sent, "\n")
      
    
csv_cb = keras.callbacks.CSVLogger("info.csv")
tensorboard_cb = keras.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
eval_cb = evalCallback(valid, idx_to_char)

epochs = 20
history = rnn_lm.fit(train, validation_data=valid, epochs=epochs, callbacks=[csv_cb, tensorboard_cb, eval_cb])

sp = "/content/drive/MyDrive/rnn language model/trained weights and models"
rnn_lm.save(f"{sp}/lm")

rnn_lm.save_weights(f"{sp}/lm_weights.h5")

class generator(keras.Model):
  def __init__(self, model, id_to_ch, ch_to_id):
    super().__init__()
    self.model = model
    self.id_to_ch = id_to_ch
    self.ch_to_id = ch_to_id
  
  @tf.function
  def gen(self, text, states):
    output_text = text
    # for i in range(max_len):
    # tokens = regex.findall(r"\X", text, regex.U)
    tokens = tf.strings.unicode_split(text, "UTF-8")
    # print(tokens.shape)
    t = self.ch_to_id(tokens).to_tensor()
    # print(t.shape)
    # t = tf.expand_dims(t, axis=0)
    # print(t.shape)
    out, states = self.model(t, states=states, return_state=True)
    # print(out.shape)
    tok= tf.squeeze(tf.random.categorical(out[:,-1,:], num_samples=1), axis=-1)
      # print(tok)
    c = self.id_to_ch(tok)
      # # print(c)
      # o = tf.strings.reduce_join(c).numpy().decode("utf-8")
      # # print(o)
      # output_text+=o
      # text = output_text

    return c, states

# o = tf.constant(["ROMEO:"])
o = "ROMEO:"
txt = tf.constant([o])
states = None
output = [o]
g = generator(rnn_lm, idx_to_char, char_to_idx)
max_len = 1000
for i in range(max_len):
  txt, states = g.gen(txt, states=states)
  s = tf.strings.reduce_join(txt).numpy().decode("utf-8")
  output.append(txt)
  o+=s

print(o)

tf.saved_model.save(g, "/content/drive/MyDrive/rnn language model/exported_saved_model")

reloaded = tf.saved_model.load("/content/drive/MyDrive/rnn language model/exported_saved_model")
o = "ROMEO:"
txt = tf.constant([o])
states = None
output = [o]
g = generator(rnn_lm, idx_to_char, char_to_idx)
max_len = 2000
for i in range(max_len):
  txt, states = reloaded.gen(txt, states=states)
  s = tf.strings.reduce_join(txt).numpy().decode("utf-8")
  output.append(txt)
  o+=s

print(o)











