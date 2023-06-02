#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
import time
import random
random.seed(0)
tf.random.set_seed(0)
np.random.seed(0)
np.set_printoptions(threshold=np.inf,linewidth=np.inf,suppress=True,precision=2)

def main():
    dataSize=int(sys.argv[1])
    testDataSize=int(sys.argv[2])
    dataLength=1
    middleUnitSize=256
    outputClassLength=testDataSize
    sampleSize=1024
    
    def makeDataset(dataSize,dataLength,sampleSize):
        lix,lidSecondx,lidSecondt=[],[],[]
        for i in range(sampleSize):
            dSecondt=np.arange(dataSize)
            tmpSecondt=np.reshape(dSecondt,(-1,1))
            dx=np.asarray(np.random.uniform(low=0,high=dataSize,size=(dataSize,dataLength)),dtype=np.float32)
            x=np.concatenate([dx,tmpSecondt],axis=1)
            x=np.asarray(x,dtype=np.float32)
            s=x[np.argsort(x[:,0])]
            dSecondt=s[:,1]
            dSecondx=np.arange(dataSize)
            dSecondx=np.reshape(dSecondx,(-1,1))
            lix.append(x)
            lidSecondx.append(dSecondx)
            lidSecondt.append(dSecondt)
        lix=np.asarray(lix,dtype=np.float32)
        lidSecondx=np.asarray(lidSecondx,dtype=np.float32)
        lidSecondt=np.asarray(lidSecondt,dtype=np.int32)
        return lix,lidSecondx,lidSecondt
    
    modelE=Encorder(middleUnitSize)
    modelD=Decorder(dataLength,middleUnitSize,outputClassLength)
    facc1=tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer=tf.keras.optimizers.Adam()
    
    checkpointE=tf.train.Checkpoint(model=modelE)
    checkpointE.restore(tf.train.latest_checkpoint("experiment/sort/01/parameter/{}/E".format(dataSize))).expect_partial()
    checkpointD=tf.train.Checkpoint(model=modelD)
    checkpointD.restore(tf.train.latest_checkpoint("experiment/sort/01/parameter/{}/D".format(dataSize))).expect_partial()
    
    ##### 中間層の呼び出し．
    testm=np.load("experiment/sort/01/parameter/{}/middle.npy".format(dataSize))
    testm=testm[0]
    testm=tf.expand_dims(testm,0)
    testm=tf.tile(testm,(sampleSize,1))
    
    for j in range(1):
        testx,testSecondx,testSecondt=makeDataset(testDataSize,dataLength,sampleSize)
        
        for i in range(testx.shape[1]):
            testm=modelE.call(testm,testx[:,i])
        testm=tf.expand_dims(testm,1)
        testm=tf.tile(testm,[1,testDataSize,1])
        
        print(testx[0:10:])
        ty=modelD.call(testm,testSecondx)
        tkai=np.argmax(ty,axis=2)
        print("tm ",np.asarray(testm[0,0]))
        print("ty_50sample",tkai[:10,:])
        print("tt_50sample",testSecondt[:10,:])
        testm=testm[:,0]
        print("Discrete ACC",float(facc1(testSecondt,ty)))
        
        pt=testSecondt
        py=tkai
        tcount=0
        for i in range(len(tkai)):
#            print(pt[i],py[i])
            if list(pt[i])==list(py[i]):
                tcount+=1
        print("Genuine ACC",tcount/len(tkai))

class Encorder(tf.keras.Model):
    def __init__(self,middleUnitSize):
        super(Encorder,self).__init__()
        self.w1=tf.keras.layers.Dense(middleUnitSize,name="w1",trainable=True)
        self.w2=tf.keras.layers.Dense(middleUnitSize,name="w2",trainable=True)
        self.w3=tf.keras.layers.Dense(middleUnitSize,name="w3",trainable=True)
        self.a1=tf.keras.layers.LeakyReLU()
    def call(self,m,x1):
        q=self.w1(x1)
        q=self.a1(q)
        
        p=self.w2(m)
        p=self.a1(p)
        
        m=p+q
        m=self.w3(m)
        m=self.a1(m)
        
        return m

class Decorder(tf.keras.Model):
    def __init__(self,inputCodeLength,middleUnitSize,outputClassLength):
        super(Decorder,self).__init__()
        self.w4=tf.keras.layers.Dense(middleUnitSize,name="w4",trainable=True)
        self.w5=tf.keras.layers.Dense(middleUnitSize,name="w5",trainable=True)
        self.w6=tf.keras.layers.Dense(middleUnitSize,name="w6",trainable=True)
        self.w7=tf.keras.layers.Dense(outputClassLength,name="w7",trainable=True)
        self.concatenate=tf.keras.layers.Concatenate()
        self.a1=tf.keras.layers.LeakyReLU()
    def call(self,m,x2):
        r=self.w4(m)
        r=self.a1(r)
        
        s=self.w5(x2)
        s=self.a1(s)
        
        y=self.concatenate([s,r])
        y=self.w6(y)
        y=self.a1(y)
        
        y=self.w7(y)
        y=tf.nn.softmax(y)
        
        return y

if __name__ == "__main__":
    main()
