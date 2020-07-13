import tensorflow as tf
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Conv1D,Flatten,Dense,TimeDistributed,LSTM,GRU,Bidirectional,Concatenate,concatenate
import scipy.io
import numpy as np
from utlites import biterr
class Decoder(tf.keras.Model):
    def __init__(self, rnn_size):
        super(Decoder, self).__init__()
        self.rnn_size = rnn_size
        # self.kernal = kernal
        self.gru1= Bidirectional(GRU(rnn_size,return_sequences=True))
        self.gru2 = Bidirectional(GRU(rnn_size, return_sequences=True))
        self.gru3 = Bidirectional(GRU(rnn_size, return_sequences=True,return_state=True))

        self.dense = Dense(1,activation='sigmoid')
        self.concatenate=Concatenate()
    def call(self, input):
        # out=tf.expand_dims(input,0)
        out = self.gru1(input)
        out = self.gru2(out)
        out, forward,backward = self.gru3(out)
        out = self.dense(out)
        # return out
        return out,forward,backward

class Decoder1(tf.keras.Model):
    def __init__(self, rnn_size):
        super(Decoder1, self).__init__()
        self.rnn_size = rnn_size
        # self.kernal = kernal
        self.gru1= Bidirectional(GRU(rnn_size,return_sequences=True))
        self.gru2 = Bidirectional(GRU(rnn_size, return_sequences=True))
        self.gru3 = Bidirectional(GRU(rnn_size, return_sequences=True,return_state=True))

        self.dense = Dense(1,activation='sigmoid')
        self.concatenate=Concatenate()
    def call(self, input,forward,backward):
        out = self.gru1(input,initial_state=[forward,backward])
        out = self.gru2(out)
        out, forward,backward = self.gru3(out)
        out = self.dense(out)
        return out,forward,backward

def loss_func(targets, logits):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(targets, logits)
    return loss
optimizer = tf.keras.optimizers.Adam()
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
with mirrored_strategy.scope():
    @tf.function
    def train_step(source_seq, target_seq_out):
        loss = 0
        with tf.GradientTape() as tape:
            n = np.array([i for i in range(128,4224,128 )])
            decode,forward,backward = Decoder(source_seq[:,0:128,:])
            DECODER=decode
            for i in range(len(n)-1) :
                n0=n[i]
                n1=n[i+1]
                decode1, forward, backward = Decoder1(source_seq[:, n0:n1, :], forward, backward)
                DECODER=concatenate([DECODER,decode1],axis=1)


            # The loss is now accumulated through the whole batchconcatenate([decode,decode1,decode2,decode3,decode4,decode5,decode6,decode7],axis=1)
            loss = loss_func(target_seq_out, DECODER)
            #
            variables = Decoder.trainable_variables+ Decoder1.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return loss
LSTM_SIZE=256
Decoder = Decoder( LSTM_SIZE)
Decoder1 = Decoder1(LSTM_SIZE)


mat = scipy.io.loadmat('data/BPSK/BPSKtrain128data0.mat')
train_ori=mat['trainori']
train_noise=mat['trainnoise']
mat1 = scipy.io.loadmat('data/BPSK/BPSKtest128data4.mat')
test_ori=mat1['testori']
test_noise=mat1['testnoise']
test_ori = tf.dtypes.cast(test_ori, tf.float32)
test_noise = tf.dtypes.cast(test_noise, tf.float32)
BATCH_SIZE=128
dataset = tf.data.Dataset.from_tensor_slices((train_ori, train_noise))
dataset = dataset.batch(BATCH_SIZE)
NUM_EPOCHS=200
BER=[]
P=[]

for e in range(NUM_EPOCHS):

    for batch, (train_ori, train_noise) in enumerate(dataset.take(-1)):
        train_ori = tf.dtypes.cast(train_ori, tf.float32)
        train_noise = tf.dtypes.cast(train_noise, tf.float32)
        loss = train_step(train_noise, train_ori)
        print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))

    # n = np.array([i for i in range(128, 4224, 128)])
    # decode, forward, backward = Decoder(test_noise[:, 0:128, :])
    # DECODER = decode
    # for i in range(len(n) - 1):
    #     n0 = n[i]
    #     n1 = n[i + 1]
    #     decode1, forward, backward = Decoder1(test_noise[:, n0:n1, :], forward, backward)
    #     DECODER = concatenate([DECODER, decode1], axis=1)
    # n = np.array([i for i in range(100, 900, 100)])
    # decode, forward, backward = Decoder(test_noise[:, 0:100, :])
    # DECODER = decode
    # for i in range(len(n) - 1):
    #     n0 = n[i]
    #     n1 = n[i + 1]
    #     decode1, forward, backward = Decoder(test_noise[:, n0:n1, :])
    #     DECODER = concatenate([DECODER, decode1], axis=1)
    # , forward, backward[:, 0:100, :][0:800,:,:]
    n = np.array([i for i in range(512, 41472, 512)])
    decode, forward, backward = Decoder(test_noise[0:512, :, :])
    DECODER = decode
    for i in range(len(n) - 1):
        n0 = n[i]
        n1 = n[i + 1]
        decode1, forward, backward = Decoder(test_noise[n0:n1, :, :])
        DECODER = concatenate([DECODER, decode1], axis=0)
    ber = biterr(test_ori, np.round(DECODER))
    print('Epoch {} ber {:.4f}'.format(e + 1,ber))
    BER =np.append(BER,ber)
postion=np.argmin(BER)
print('epoch',postion+1)


mat = scipy.io.loadmat('data/QPSK/QPSKtest128data9.mat')
test_ori=mat['testori']
test_noise=mat['testnoise']
test_ori = tf.dtypes.cast(test_ori, tf.float32)
test_noise = tf.dtypes.cast(test_noise, tf.float32)
n = np.array([i for i in range(512, 41472, 512)])
decode, forward, backward = Decoder(test_noise[0:512, :, :])
DECODER = decode
for i in range(len(n) - 1):
    n0 = n[i]
    n1 = n[i + 1]
    decode1, forward, backward = Decoder(test_noise[n0:n1, :, :])
    DECODER = concatenate([DECODER, decode1], axis=0)
ber9 = biterr(test_ori, np.round(DECODER))
#
# mat = scipy.io.loadmat('data/16QAM/16QAMtest128data10.mat')
# test_ori=mat['testori']
# test_noise=mat['testnoise']
# test_ori = tf.dtypes.cast(test_ori, tf.float32)
# test_noise = tf.dtypes.cast(test_noise, tf.float32)
# n = np.array([i for i in range(512, 41472, 512)])
# decode, forward, backward = Decoder(test_noise[0:512, :, :])
# DECODER = decode
# for i in range(len(n) - 1):
#     n0 = n[i]
#     n1 = n[i + 1]
#     decode1, forward, backward = Decoder(test_noise[n0:n1, :, :])
#     DECODER = concatenate([DECODER, decode1], axis=0)
# ber10 = biterr(test_ori, np.round(DECODER))

