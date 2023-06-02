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
    maxDataSize=int(sys.argv[1])
    os.makedirs("experiment/survey/02/parameter/{}/E".format(maxDataSize),exist_ok=True)
    os.makedirs("experiment/survey/02/parameter/{}/D".format(maxDataSize),exist_ok=True)
    dataLength=16
    trainSize=maxDataSize
    validSize=maxDataSize
    middleUnitSize=256
    MaxEpochSize=500000
    outputClassLength=10
    sampleSize=1024
    
    def makeDataset(dataSize,dataLength,sampleSize):
        lix,lidSecondx,lidSecondt=[],[],[]
        for i in range(sampleSize):
            dSecondt=np.random.randint(0,10,(dataSize))
            tmpSecondt=np.reshape(dSecondt,(-1,1))
            dSecondx=np.asarray(np.random.uniform(low=0,high=9,size=(dataSize,dataLength)),dtype=np.float32)
            x=np.concatenate([dSecondx,tmpSecondt],axis=1)
            x=np.asarray(x,dtype=np.float32)
            np.random.shuffle(x)
            lix.append(x)
            lidSecondx.append(dSecondx)
            lidSecondt.append(dSecondt)
        lix=np.asarray(lix,dtype=np.float32)
        lidSecondx=np.asarray(lidSecondx,dtype=np.float32)
        lidSecondt=np.asarray(lidSecondt,dtype=np.int32)
        return lix,lidSecondx,lidSecondt
    
    trainx,trainSecondx,trainSecondt=makeDataset(maxDataSize,dataLength,sampleSize)
    validx,validSecondx,validSecondt=makeDataset(maxDataSize,dataLength,sampleSize)
    
    initialMemoryTrain=tf.random.uniform(shape=(sampleSize,middleUnitSize),minval=-1,maxval=1,dtype=tf.dtypes.float32,seed=0)
    trainm=tf.tile(initialMemoryTrain,(1,1))
    initialMemoryValid=tf.random.uniform(shape=(sampleSize,middleUnitSize),minval=-1,maxval=1,dtype=tf.dtypes.float32,seed=0)
    validm=tf.tile(initialMemoryValid,(1,1)) # tf.tile()で生成されるのは別物のベクトルだから実はこうしなくても良い．つまり，validm=tf.tile(initialMemoryTrain,(1,1))でも問題はない．
    
    modelE=Encorder(middleUnitSize)
    modelD=Decorder(dataLength,middleUnitSize,outputClassLength)
    fscce=cce=tf.keras.losses.SparseCategoricalCrossentropy()
    facc1=tf.keras.metrics.SparseCategoricalAccuracy()
    facc2=tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer=tf.keras.optimizers.Adam()
    
    checkpointE=tf.train.Checkpoint(model=modelE)
    checkpointD=tf.train.Checkpoint(model=modelD)
    
#    @tf.function
    def runModel(tm,tx1,tx2,tt,dataSize,mode):
        with tf.GradientTape() as tape:
            if mode=="train":
                modelE.trainable=True
                modelD.trainable=True
            else:
                modelE.trainable=False
                modelD.trainable=False
            for i in range(tx1.shape[1]):
                tm=modelE.call(tm,tx1[:,i])
            tm=tf.expand_dims(tm,1)
            tm=tf.tile(tm,[1,dataSize,1])
            ty=modelD.call(tm,tx2)
            costValue=fscce(tt,ty)
            if mode=="train":
                accuracy=facc1(tt,ty)
            elif mode=="valid":
                accuracy=facc2(tt,ty)
            gradient=tape.gradient(costValue,modelE.trainable_variables+modelD.trainable_variables)
            optimizer.apply_gradients(zip(gradient,modelE.trainable_variables+modelD.trainable_variables))
            tm=tm[:,0]
            return tm,ty,costValue,accuracy
    
    for epoch in range(1,MaxEpochSize+1):
        dataSize=maxDataSize
        
        trainm,trainy,trainCost,trainAcc=runModel(trainm,trainx,trainSecondx,trainSecondt,dataSize,"train")
        validm,validy,validCost,validAcc=runModel(validm,validx,validSecondx,validSecondt,dataSize,"valid")
        if epoch%10==0:
            print("Epoch {:5d}: Training cost= {:.4f}, Training acc= {:.4f}, Validation cost= {:.4f}, Validation acc= {:.4f}".format(epoch,trainCost,trainAcc,validCost,validAcc))
            if 1 and epoch%100==0:
                print("tm ",np.asarray(trainm[0]))
                tm=tf.expand_dims(trainm,1)
                tm=tf.tile(tm,[1,dataSize,1])
                ty=modelD.call(tm,trainSecondx)
                tkai=np.argmax(ty,axis=2)
                print("ty_50sample",tkai[:10,:])
                print("tt_50sample",trainSecondt[:10,:])
                
                print("vm ",np.asarray(validm[0]))
                vm=tf.expand_dims(validm,1)
                vm=tf.tile(vm,[1,dataSize,1])
                vy=modelD.call(vm,validSecondx)
                vkai=np.argmax(vy,axis=2)
                print("vy_50sample",vkai[:10,:])
                print("vt_50sample",validSecondt[:10,:])
            if trainAcc>0.8:
                checkpointE.save("experiment/survey/02/parameter/{}/E/model".format(maxDataSize))
                checkpointD.save("experiment/survey/02/parameter/{}/D/model".format(maxDataSize))
                np.save("experiment/survey/02/parameter/{}/middle".format(maxDataSize),trainm)
                exit()
            elif epoch==MaxEpochSize:
                checkpointE.save("experiment/survey/02/parameter/{}/E/model".format(maxDataSize))
                checkpointD.save("experiment/survey/02/parameter/{}/D/model".format(maxDataSize))
                np.save("experiment/survey/02/parameter/{}/middle".format(maxDataSize),trainm)

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
