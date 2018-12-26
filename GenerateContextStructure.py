# -*- coding:utf-8 -*-

import numpy as np
import cPickle as pickle 
import json
import re
import time

class Context:
    def __init__(self):
        # self.courseDir = '../rawdata/Dataset/ML/'
        # self.captionFilenameList = ['Captions_machine-learning_Stanford.json', 'Captions_machine-learning_Washington.json']
        # self.conceptFilename = 'CoreConcepts_ML'
        # self.dataStructureDir = '../datastructure/20170109/ML/'
        
        self.courseDir = '../rawdata/Dataset/DSA/'
        self.captionFilenameList = ['Captions_data-structure-and-algorithm_UC-San-Diego.json', 'Captions_algorithms_Stanford.json', 'Captions_algorithms_Princeton.json']
        self.conceptFilename = 'CoreConcepts_DSA'
        self.dataStructureDir = '../datastructure/20170109/DSA/'
        
        # self.courseDir = '../rawdata/Dataset/CAL/'
        # self.captionFilenameList = ['Captions_calculus_Pennsylvania.json', 'Captions_calculus1_Ohio.json']
        # self.conceptFilename = 'CoreConcepts_CAL'
        # self.dataStructureDir = '../datastructure/20170109/CAL/'
        
        self.captions = []
        self.conceptList = []
        self.int2concept = {}
        self.concept2int = {}
        self.loadRawData()
        
    def loadRawData(self):
        for filename in self.captionFilenameList:
            with open(self.courseDir+filename, 'r') as frCap:
                for line in frCap.readlines():
                    self.captions.append(json.loads(line.strip()))
        with open(self.courseDir+self.conceptFilename, 'r') as frCon:
            for line in frCon.readlines():
                words = line.strip().split("::;")
                for word in words:
                    self.concept2int[word] = str(len(self.conceptList))
                self.int2concept[str(len(self.conceptList))] = words
                self.conceptList.append(words[0])
        self.Cn = len(self.conceptList)
        self.Vn = len(self.captions)
        print 'Concept list length:', self.Cn
        print 'Captions length:', self.Vn
        # pickle.dump(self.captions, file(self.dataStructureDir+'ML_captions.pkl', 'wb'), True)
        self.ew = pickle.load(file(self.dataStructureDir+ '../../EW+.pkl', 'rb'))


    def countAny(self, text, word):
        cnt = 0.0
        for w in self.int2concept[str(self.concept2int[word])]:
            cnt += text.count(w)
        return cnt
        
        
    def generateVM(self):
        self.VM = np.zeros((self.Cn, self.Cn), dtype=np.float32)
        for ci in self.conceptList:
            for cj in self.conceptList:
                cnt = 0.0
                for vk in self.captions:
                    if self.countAny(vk['text'].lower(), cj) != 0:
                        cnt += self.countAny(vk['text'].lower(), ci)
                self.VM[ self.conceptList.index(ci) ][ self.conceptList.index(cj) ] = cnt
        pickle.dump(self.VM, file(self.dataStructureDir+'VM.pkl', 'wb'), True)
        
    def generateSM(self):
        self.SM = np.zeros((self.Cn, self.Cn), dtype=np.float32)        
        for ci in self.conceptList:
            for cj in self.conceptList:
                cnt = 0.0
                for vk in self.captions:
                    sentences = re.split('\.|\!|\?', vk['text'].lower())[0:-1]
                    for sentence in sentences:
                        if self.countAny(sentence, cj) != 0:
                            cnt += self.countAny(sentence, ci)
                self.SM[ self.conceptList.index(ci) ][ self.conceptList.index(cj) ] = cnt
        pickle.dump(self.SM, file(self.dataStructureDir+'SM.pkl', 'wb'), True)
        
    def generateVI(self):
        self.VI = []
        for ci in self.conceptList:
            Ii = []
            for _ in xrange(len(self.captionFilenameList)):
                Ii.append([])
            for vk in self.captions:
                if self.countAny(vk['text'].lower(), ci) != 0:
                    Ii[vk['course_id']-1].append(vk['video_ID'])
            self.VI.append(Ii)
        pickle.dump(self.VI, file(self.dataStructureDir+'VI.pkl', 'wb'), True)
        
    def generateSN(self):
        self.SN = np.zeros((self.Cn,), dtype=np.float32)
        for ci in self.conceptList:
            cnt = 0.0
            for vk in self.captions:
                sentences = re.split('\.|\!|\?', vk['text'].lower())[0:-1]
                for sentence in sentences:
                    if self.countAny(sentence, ci) != 0:
                        cnt += 1
            self.SN[ self.conceptList.index(ci) ] = cnt
        pickle.dump(self.SN, file(self.dataStructureDir+'SN.pkl', 'wb'), True)
        
    def generateCV(self):
        self.CV = np.zeros((self.Cn, self.Vn), dtype=np.float32)
        for ci in self.conceptList:
            for vj in self.captions:
                self.CV[ self.conceptList.index(ci) ][ self.captions.index(vj) ] = self.countAny(vj['text'].lower(), ci)
        pickle.dump(self.CV, file(self.dataStructureDir+'CV.pkl', 'wb'), True)

    def generateAVGD_N(self): 
        def findSub(str, sub):
            ans = []
            for w in self.int2concept[self.concept2int[sub]]:
                sta = 0
                while True:
                    idx = str.find(w, sta)
                    if idx == -1 or len(str[idx:-1])<len(sub):
                        break
                    ans.append(idx)
                    sta = idx + len(w)
            return ans
        
        self.AVG_D = np.zeros((self.Cn, self.Cn), dtype=np.float32)
        self.Nij = np.zeros((self.Cn, self.Cn), dtype=np.float32)
        cccnt = 0
        for ci in self.conceptList:
            for cj in self.conceptList:
                sigma_dm2 = 0.0
                nij = 0.0
                for vk in self.captions:
                    sentences = re.split('\.|\!|\?', vk['text'].lower())[0:-1]
                    for sentence in sentences:
                        posA = findSub(sentence, ci)
                        posB = findSub(sentence, cj)
                        if len(posA)!=0 and len(posB)!=0:
                            sigma_dm2 += (1.0*sum(posA)/len(posA) - 1.0*sum(posB)/len(posB))**2
                            nij += 1.0
                self.AVG_D[self.conceptList.index(ci), self.conceptList.index(cj)] = (sigma_dm2 / nij) if nij!=0 else 0.0
                self.Nij[self.conceptList.index(ci), self.conceptList.index(cj)] = nij
        pickle.dump({'AVG_D': self.AVG_D, 'Nij': self.Nij}, file(self.dataStructureDir+'AVGD_N.pkl', 'wb'), True)

    def generateW_AVGD_N(self): 
        def findSub(str, sub):
            ans = []
            for w in self.int2concept[self.concept2int[sub]]:
                sta = 0
                while True:
                    idx = str.find(w, sta)
                    if idx == -1 or len(str[idx:-1])<len(sub):
                        break
                    ans.append(idx)
                    sta = idx + len(w)
            return ans
        
        self.W_AVG_D = np.zeros((self.Cn, self.Cn), dtype=np.float32)
        self.W_Nij = np.zeros((self.Cn, self.Cn), dtype=np.float32)
        cccnt = 0
        for ci in self.conceptList:
            for cj in self.conceptList:
                sigma_dm2 = 0.0
                nij = 0.0
                for wiki in self.ew:
                    sentences = re.split('\.|\!|\?', self.ew[wiki]['Article'].lower())[0:-1]
                    for sentence in sentences:
                        posA = findSub(sentence, ci)
                        posB = findSub(sentence, cj)
                        if len(posA)!=0 and len(posB)!=0:
                            sigma_dm2 += (1.0*sum(posA)/len(posA) - 1.0*sum(posB)/len(posB))**2
                            nij += 1.0
                self.W_AVG_D[self.conceptList.index(ci), self.conceptList.index(cj)] = (sigma_dm2 / nij) if nij!=0 else 0.0
                self.W_Nij[self.conceptList.index(ci), self.conceptList.index(cj)] = nij
        pickle.dump({'W_AVG_D': self.W_AVG_D, 'W_Nij': self.W_Nij}, file(self.dataStructureDir+'W_AVGD_N.pkl', 'wb'), True)
    
    
    
    def loadStructure(self, matrixName):
        return pickle.load(file(self.dataStructureDir+ matrixName +'.pkl', 'rb'))

    def showData(self):
        VM = self.loadStructure('VM')
        print 'VM:', VM.shape
        cnt = 0.0
        for i in range(VM.shape[0]):
            for j in range(VM.shape[1]):
                if (VM[i, j] == 0):
                    cnt += 1;
        print 'for VMij is 0:', cnt, '\t\t(%.2f%%)' % (cnt/(VM.shape[0]*VM.shape[1])*100)
        
        SM = self.loadStructure('SM')
        print 'SM:', SM.shape
        cnt = 0.0
        for i in range(SM.shape[0]):
            for j in range(SM.shape[1]):
                if (SM[i, j] == 0):
                    cnt += 1;
        print 'for SMij is 0:', cnt, '\t\t(%.2f%%)' % (cnt/(SM.shape[0]*SM.shape[1])*100)
        
        VI = self.loadStructure('VI')
        print 'VI:', len(VI)
        print VI[0:3]
        
        CV = self.loadStructure('CV')
        print 'CV:', CV.shape
        cnt = 0.0
        for i in range(CV.shape[0]):
            for j in range(CV.shape[1]):
                if (CV[i, j] == 0):
                    cnt += 1;
        print 'for CVij is 0:', cnt, '\t\t(%.2f%%)' % (cnt/(CV.shape[0]*CV.shape[1])*100)
        
        avgd_n = self.loadStructure('AVGD_N')
        AVG_D = avgd_n['AVG_D']
        Nij = avgd_n['Nij']
        print 'AVG_D:', AVG_D.shape, 'Nij:', Nij.shape
        cnt = 0.0
        for i in range(AVG_D.shape[0]):
            for j in range(AVG_D.shape[1]):
                if (AVG_D[i, j] == 0):
                    cnt += 1;
        print 'for AVG_Dij is 0:', cnt, '\t\t(%.2f%%)' % (cnt/(AVG_D.shape[0]*AVG_D.shape[1])*100)
        cnt = 0.0
        for i in range(Nij.shape[0]):
            for j in range(Nij.shape[1]):
                if (Nij[i, j] == 0):
                    cnt += 1;
        print 'for Nij is 0:', cnt, '\t\t(%.2f%%)' % (cnt/(Nij.shape[0]*Nij.shape[1])*100)


if __name__ == "__main__":
    context = Context()
    # context.generateAll()
    #context.generateVM()
    # context.showData()
    # print 'Start generating AVGD_N...'
    # t1 = time.time()
    # context.generateAVGD_N()
    # t2 = time.time()
    # print "\t__cost:", t2-t1, "s."
    # context.generateVI()
    # VI = context.loadStructure('VI')
    # print VI[0:10]
    # context.generateW_AVGD_N()
    
    # print context.findSub('so in logistic regression, we have our familiar form of the hypothesis there and the sigmoid activation function shown on the right', 'activation function')
    # print context.findSub('so in logistic regression, we have our familiar form of the hypothesis there and the sigmoid activation function shown on the right', 'hypothesis')