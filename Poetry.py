import MyRNN
import tensorflow as tf
import numpy as np

print("Hello World")

# Defining Netowrk Parameters
num_input = MyRNN.num_input # No of Inputs / Size of Embedding
num_classes = MyRNN.num_classes # Size of output / Size of Embedding
print(num_classes)
num_layers = 2 # No of Hidden Layers, Since we require Multi-Layer NN
num_hidden = 256 # Number of Hidden Neurons (LSTM Cells)
num_unroll = 100 # No of Times the RNN is unrolled
# dH.setNumUnroll(num_unroll)
batch_size = 64 # No of times the unrolled RNN is replicated for training
# dH.setBatchSize(batch_size)

# Learning Rate
lr = 0.001
# Loop Control To Control Number of Epochs, for simplicity we count the number of batches
# As opposed to epochs
num_batches = 50

checkpoint_path = "checkpoints/Model1.ckpt"
InitPhrase = "And"
with tf.Session() as sess:
    net = MyRNN.MyRNN(num_input, num_classes, num_layers, num_hidden, sess, lr)
    saver = tf.train.Saver(tf.global_variables())
    print("TF Session Created")
    saver.restore(sess=sess, save_path=checkpoint_path)
    print("Checkpoint Restored")
    FinalStr = InitPhrase
    for i in range(len(InitPhrase)):
        InpVec = MyRNN.dH.getOneHotVecGen(InitPhrase[i])
        if(i==0):
            fIter = True
        else:
            fIter = False
        # print(fIter)
        outDist = net.stepPred(InpVec, fIter)
        # print(InpVec.shape)
        # print(InpVec)

    print("Generating Poetry")
    for i in range(5):
        idx = np.random.choice(range(MyRNN.dH.vocabSize), p = outDist)
        next_ch = MyRNN.dH.idx_to_char[idx]
        FinalStr += next_ch
        InpVec = MyRNN.dH.getOneHotVecGen(next_ch)
        outDist = net.stepPred(InpVec, False)

    print(FinalStr)
