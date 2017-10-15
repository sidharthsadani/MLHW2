# Utilities for RNN

import os
import numpy as np
import re
import random

class RNNUtils(object):
    def __init__ (self, inpFile = "FrostExcerpt.txt",
            inpPath = os.getcwd()):
        # self.parseFile
        # getNext
        self.inpPath = inpPath
        self.inpFile = inpFile
        self.inpFilePath = os.path.join(self.inpPath, self.inpFile)
        self.fHandle = open(self.inpFilePath, 'r')
        self.fData = None
        self.vocabSize = 0
        self.dataSize = 0
        self.char_to_idx = None
        self.idx_to_char = None
        self.curIdx = 0
        self.dataEnd = False
        self.iterCt = 0
        self.batchSize = 0
        self.numUnroll = 0
        self.startId = None
        print("RNN Utils Object Created")

    def __iter__ (self):
        return self

    def __next__ (self):
        return self.next()

    def next(self):
        self.iterCt += 1

    def setBatchSize(self, bsize):
        self.batchSize = bsize
    
    def setNumUnroll(self, nsize):
        self.numUnroll = nsize

    def parseData(self):
        self.fData = self.fHandle.read()
        self.dataSize = len(self.fData)
        print("Total Training Characters: ", self.dataSize)
        charset = sorted(set(self.fData))
        self.vocabSize = len(charset)
        print("Vocab Size: ", self.vocabSize)
        self.char_to_idx = { ch:i for i,ch in enumerate(charset)}
        self.idx_to_char = { i:ch for i,ch in enumerate(charset)}
        # print(charset)
        # print(self.vocabSize)
        iterD = re.finditer(r"\s\S", self.fData)
        # Computing Start Id of Words to ensure batch begins at a valid word
        self.startId = [m.start(0)+1 for m in iterD]
        ch = [self.fData[i] for i in self.startId]
        # print(ch[0:20])

    def getOneHotVec(self, ch):
        ohv = np.zeros(shape=(self.vocabSize), dtype=np.float32)
        ohv[self.char_to_idx[ch]] = 1
        return ohv

    def nextBatch(self):
        x = np.zeros(shape=(self.batchSize, self.vocabSize), dtype=np.float32)
        y = np.zeros(shape=(self.batchSize, self.vocabSize), dtype=np.float32)
        for i in range(self.batchSize):
            self.curIdx = self.curIdx % self.dataSize
            nextIdx = (self.curIdx + 1) % self.dataSize
            x[i, :] = self.getOneHotVec(self.fData[self.curIdx])
            y[i, :] = self.getOneHotVec(self.fData[nextIdx])
            self.curIdx = self.curIdx + 1

        return x, y
    
    def nextSample(self):
        x = np.zeros(shape=(self.numUnroll, self.vocabSize), dtype=np.float32)
        y = np.zeros(shape=(1, self.vocabSize), dtype=np.float32)
        cIdx = self.curIdx % self.dataSize
        for i in range(self.numUnroll):
            x[i, :] = self.getOneHotVec(self.fData[cIdx])
            cIdx = (cIdx + 1) % self.dataSize
        nextIdx = (cIdx + 1) % self.dataSize
        y[0, :] = self.getOneHotVec(self.fData[nextIdx])
        self.curIdx = self.curIdx + 1
        if(self.curIdx % 1000 == 0):
            print("1k Characters Consumed")

        return x, y

    def nextBlankSample(self):
        x = np.zeros(shape=(self.numUnroll, self.vocabSize), dtype=np.float32)
        return x

    def getBatchSeeds(self):
        seedsId = random.sample(range(len(self.startId)),self.batchSize)
        seeds = [self.startId[i] for i in seedsId]
        return seeds

    def nextBatchNew(self):
        x = np.zeros(shape=(self.batchSize, self.numUnroll, self.vocabSize), dtype=np.float32)
        y = np.zeros(shape=(self.batchSize, self.numUnroll, self.vocabSize), dtype=np.float32)
        batchSeeds = self.getBatchSeeds()
        for i in range(self.numUnroll):
            for j in range(self.batchSize):
                xidx = (batchSeeds[j] + i) % self.dataSize
                yidx = (xidx + 1) % self.dataSize
                x[j,i,:] = self.getOneHotVec(self.fData[xidx])
                y[j,i,:] = self.getOneHotVec(self.fData[yidx])

        return x, y
    
    def getOneHotVecGen(self, ch):
        ohv = np.zeros(shape=(1, self.vocabSize), dtype=np.float32)
        ohv[0, self.char_to_idx[ch]] = 1
        return ohv



# Test Run
print("Hello World, Utils")
# x = RNNUtils()
# print(x.inpPath)
# print(x.inpFile)
# print(x.inpFilePath)
# x.parseData()
