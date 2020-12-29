import math
import numpy as np
class Endpointing:
    def __init__(self):
        self.forgetfactor = 1  # typical value >= 1
        self.level = 0  # first frame
        self.background = 0  # initially set to avg energy of first 10 frames
        self.adjustment = 0.05  # typical value 0.05
        self.threshold = 10  #
        self.frameCount = 0
        self.backgroundOF10 = 0
        self.slienceThreshold = 50 # almost 1/44100*1024*25 = 0.58s
        self.continueSlienceTime = 0
        pass

    def isSlienceTooLong(self):
        return self.continueSlienceTime >= self.slienceThreshold

    def EnergyPerSampleInDecibel(self,audioframe):

        sum = 0
        for item in audioframe:
            # sum += math.log(item * item+1,10)
            sum += item * item
            # print(sum)
        decibel = 10 * math.log(sum,10)
        return decibel


    def classifyFrame(self,audioframe):
        audioframe = audioframe.astype(np.int64)
        current = self.EnergyPerSampleInDecibel(audioframe)
        if(self.frameCount<10):
            self.frameCount += 1
            self.background += current
            if(self.frameCount==1):
                self.level = current
            elif(self.frameCount == 10):
                self.level = ((self.level * self.forgetfactor) + current) / (self.forgetfactor + 1)
                self.background = self.backgroundOF10 / 10
            else: # 2 -- 9
                self.level = ((self.level * self.forgetfactor) + current) / (self.forgetfactor + 1)
            return True

        isSpeech = False
        self.level = ((self.level * self.forgetfactor) + current) / (self.forgetfactor + 1)
        print("current", current)
        print("level", self.level)
        print("background",self.background)
        if (current < self.background):
            self.background = current
        else:
            self.background += (current - self.background) * self.adjustment
        if (self.level < self.background):
            self.level = self.background
        if (self.level - self.background > self.threshold):
            isSpeech = True

        if not isSpeech:
            self.continueSlienceTime += 1
        if isSpeech and self.continueSlienceTime != 0:
            print("连续沉默Frame个数为:",self.continueSlienceTime)
            self.continueSlienceTime = 0


        return isSpeech
        # pass


