# -*- coding:utf-8 -*-

import numpy as np
import cPickle as pickle 
import json
import re
import time


class FeatureGenerator:
    """
    Generate feature data from data structures.

    Attributes:
        # Context
            conceptList: All non-repeated Concepts List. {c1 ... cn}
            VM: Video­cooccurrence Matrix, VMij = # ci appears in videos containing cj.
            SM: Sentence­cooccurrence Matrix, SMij = # ci appears in sentences containing cj.
            VI: Video Indexes, Ii = {k|ci appears in video vk}.
            SN: Sentence Number, sni = number of sentences containing ci.
            CV: Concept­Video matrix, CVij = # ci appears in video vj.
        # Wikipedia
			EW: Wiki Article Embeddings and Articles, Format: {"name of ai": {"title": "title of ai", "Vector": "vector of ai", "Article": "article of ai"}}
            CW: Concept Embeddings, Format: {"ci": vector of ci}
        # Similarity
            SC: Concept Similarity Matrix, SCij = cosine similarity between ci and cj.
            WC.npy: Wiki Similarity Matrix, WCij = cosine similarity between ci and aj, should be load with numpy.
        # Class
            dataStructureDir: Data structures location.
            featuresDir: Raw features location.
            K: 
    """

    def __init__(self):
        TYPE = 'ML'
        IF_W = 'W-'
        self.K = 10
        # ML = [111, 548]; DSA = [263, 390, 449]; CAL = [81, 359];
        self.VIDNUM = []
        if TYPE=='ML': self.VIDNUM = [111, 548] # Defined by videos amount of each course [course_1, course_1+2, ...]
        elif TYPE=='DSA': self.VIDNUM = [263, 390, 449]
        elif TYPE=='CAL': self.VIDNUM = [81, 359]
        self.dataStructureDir = '../datastructure/20170109/'+TYPE+'/'
        self.featuresDir = '../features/20170109/'+IF_W+TYPE+'_'
        self.labeledFileDir = '../rawdata/Dataset/'+TYPE+'/'+IF_W+TYPE+'_LabeledFile'
        self.VECDIM = 400
        self.features = {}
        print 'Loading data...'
        t1 = time.time()
        self.loadData()
        t2 = time.time()
        print '\t__cost:', t2-t1, "s."
    
    def loadData(self):
        tem_concepts = pickle.load(file(self.dataStructureDir+ 'conceptDict.pkl', 'rb'))
        self.conceptList = tem_concepts['conceptList']
        self.int2concept = tem_concepts['int2concept']
        self.concept2int = tem_concepts['concept2int']
        self.Cn = len(self.conceptList)
        self.skipList = []
        self.VM = pickle.load(file(self.dataStructureDir+ 'VM.pkl', 'rb'))
        self.SM = pickle.load(file(self.dataStructureDir+ 'SM.pkl', 'rb'))
        self.VI = pickle.load(file(self.dataStructureDir+ 'VI.pkl', 'rb'))
        self.SN = pickle.load(file(self.dataStructureDir+ 'SN.pkl', 'rb'))
        self.CV = pickle.load(file(self.dataStructureDir+ 'CV.pkl', 'rb'))
        
        self.wikiList = pickle.load(file(self.dataStructureDir+ '../../wikiList.pkl', 'rb'))
        self.Wn = len(self.wikiList)
        cw = pickle.load(file(self.dataStructureDir+ 'CW.pkl', 'rb'))
        for concept in self.conceptList:
            bool = False
            for c in self.int2concept[self.concept2int[concept]]:
                if cw.has_key(c)==True:
                    bool = True
            if bool==False:
                self.skipList.append(concept)
        
        print 'skipList length:', len(self.skipList)
        
        self.SC = pickle.load(file(self.dataStructureDir+ 'SC.pkl', 'rb'))
        self.argsortedSC = np.argsort(-self.SC, kind='heapsort')
        self.WC = np.load(self.dataStructureDir+ 'WC.npy')
        self.argsortedWC = np.argsort(-self.WC, kind='heapsort')

    
    def findTopKConceptsInConcepts(self, concept, k):
        return self.argsortedSC[concept, 0:k].tolist()[0]

    def findTopKConceptsInWikipedia(self, concept, k):
        return self.argsortedWC[concept, 0:k].tolist()


    # Video Reference Distance
    def VrefR(self, A, B):
        return self.VM[B, A] / len(sum(self.VI[A], []))
    def VrefD(self, A, B):
        return self.VrefR(B, A) - self.VrefR(A, B)
    
    # Sentence Reference Distance
    def SrefR(self, A, B):
        return self.SM[B, A] / self.SN[A]
    def SrefD(self, A, B):
        return self.SrefR(B, A) - self.SrefR(A, B)
    
    # Generalized Video Reference Distance
    def GVrefD(self, A, B):
        k = self.K
        val_1 = 0.0
        sum_wbi = 0.0
        for bi in self.findTopKConceptsInConcepts(B, k):
            val_1 += self.VrefR(bi, A) * self.SC[bi, B]
            sum_wbi += 1.0*self.SC[bi, B]
        val_1 /= sum_wbi
        
        val_2 = 0.0
        sum_wai = 0.0
        for ai in self.findTopKConceptsInConcepts(A, k):
            val_2 += self.VrefR(ai, B) * self.SC[ai, A]
            sum_wai += self.SC[ai, A]
        val_2 /= sum_wai

        return val_1 - val_2
    
    # Generalized Sentence Reference Distance
    def GSrefD(self, A, B):
        k = self.K
        val_1 = 0.0
        sum_wbi = 0.0
        for bi in self.findTopKConceptsInConcepts(B, k):
            val_1 += self.SrefR(bi, A) * self.SC[bi, B]
            sum_wbi += self.SC[bi, B]
        val_1 /= sum_wbi
        
        val_2 = 0.0
        sum_wai = 0.0
        for ai in self.findTopKConceptsInConcepts(A, k):
            val_2 += self.SrefR(ai, B) * self.SC[ai, A]
            sum_wai += self.SC[ai, A]
        val_2 /= sum_wai

        return val_1 - val_2
    
    # Wikipedia RefD
    def Wref(self, A, B):
        k = self.K
        if A in self.findTopKConceptsInWikipedia(B, k):
            return 1.0
        return 0.0
    def WrefD(self, A, B):
        k = self.K
        val_1 = 0.0
        sum_wb = 0.0
        for b in self.findTopKConceptsInWikipedia(B, k):
            val_1 += self.Wref(b, A) * self.WC[B, b]
            sum_wb += self.WC[B, b]
        val_1 /= sum_wb
        
        val_2 = 0.0
        sum_wa = 0.0
        for a in self.findTopKConceptsInWikipedia(A, k):
            val_2 += self.Wref(a, B) * self.WC[A, a]
            sum_wa += self.WC[A, a]
        val_2 /= sum_wa

        return val_1 - val_2
    
    # TOC Distance
    def TocD(self, A, B):
        sum_ab = 0.0
        num = 0
        for i in range(len(self.VIDNUM)):
            if len(self.VI[A][i]) != 0 and len(self.VI[B][i]) != 0:
                sum_ab += abs(sum(self.VI[A][i])/len(self.VI[A][i]) - sum(self.VI[B][i])/len(self.VI[B][i]))
                num += 1
        if num == 0:
            return 0.0
        else:
            return sum_ab / num
    
    # Distributional Asymmetry Distance
    def AsyD(self, A, B):
        # print '____', A, B
        sum_ab = 0.0
        num = 0
        for i in range(len(self.VIDNUM)):
            if len(self.VI[A][i]) != 0 and len(self.VI[B][i]) != 0:
                num += 1
                Mcnt = 0
                sigmaF = 0.0
                # print '>\t', self.VI[A][i]
                # print '>\t', self.VI[B][i]
                for IiA in self.VI[A][i]:
                    for IjB in self.VI[B][i]:
                        if (IiA < IjB):
                            # print i, IiA, IjB
                            sigmaF += self.CV[A, (IjB-1) if i==0 else (self.VIDNUM[i-1]+ IjB-1)] - self.CV[B, (IiA-1) if i==0 else (self.VIDNUM[i-1]+ IiA-1)]
                            Mcnt += 1

                if Mcnt == 0: sum_ab += 0.0
                else: sum_ab += sigmaF / Mcnt
        if num == 0: return 0.0
        else: return sum_ab / num

    # Complexity Level Distance
    def ClD(self, A, B):
        def AVCmulAST(A):
            sum_vc = 0.0
            sum_st = 0.0
            num = 0
            for i in range(len(self.VIDNUM)):
                if len(self.VI[A][i]) != 0:
                    num += 1
                    sum_vc += 1.0 * len(self.VI[A][i]) / self.VIDNUM[i]
                    sum_st += 1.0 * (self.VI[A][i][-1] - self.VI[A][i][0] + 1) / self.VIDNUM[i]
            if num == 0: return 0.0
            else: return (sum_vc / num) * (sum_st / num)
        return AVCmulAST(A) - AVCmulAST(B)
    
    # Semantic Similariy
    def SimD(self, A, B):
        return self.SC[A, B]
    
    def generateFeatures(self):
        cnt = 0
        with open(self.labeledFileDir, 'r') as fr:
            for line in fr.readlines():
                words = line.strip().split('\t\t') 
                A = words[0]
                B = words[1]
                if A in self.skipList or B in self.skipList:
                    continue
                label = 0                    
                if (words[2]=='1-'):
                    label = 1
                    A, B = B, A
                elif (words[2]=='-1'):
                    label = 1
                
                Aindex = self.conceptList.index(A)
                Bindex = self.conceptList.index(B)
                # print Aindex, '__', Bindex
                features = {'VrefD': self.VrefD(Aindex, Bindex), 'SrefD': self.SrefD(Aindex, Bindex), 'GVrefD': self.GVrefD(Aindex, Bindex), 'GSrefD': self.GSrefD(Aindex, Bindex), 'WrefD': self.WrefD(Aindex, Bindex), 'TocD': self.TocD(Aindex, Bindex), 'AsyD': self.AsyD(Aindex, Bindex), 'SimD': self.SimD(Aindex, Bindex), 'ClD': self.ClD(Aindex, Bindex)}
                self.features[A+'::;'+B] = {'features': features, 'label': label}
                cnt += 1
                if cnt%1000 == 0:
                    print '\t__cnt:', cnt
                # print sample
    
    def saveFeatures(self):
        pickle.dump(self.features, file(self.featuresDir+ 'features_'+ (str)(self.K) +'.pkl', 'wb'), True)
        
    def loadFeatures(self, featuresName='features.pkl'):
        return pickle.load(file(self.featuresDir+ featuresName, 'rb'))

    def debugForOnePair(self, A, B):
        self.loadData()
        print 'Pair: ', str(A), '__', str(B)
        print 'VrefD is:\t', self.VrefD(A, B)
        print 'SrefD is:\t', self.SrefD(A, B)
        print 'GVrefD is:\t', self.GVrefD(A, B)
        print 'GSrefD is:\t', self.GSrefD(A, B)
        print 'WrefD is:\t', self.WrefD(A, B)
        print 'TocD is:\t', self.TocD(A, B)
        print 'AsyD is:\t', self.AsyD(A, B)
        print 'SimD is:\t', self.SimD(A, B)
        print 'ClD is:\t', self.ClD(A, B)
    
    def showStatistics(self):
        features = self.loadFeatures('features_'+ (str)(self.K) +'.pkl')
        VrefD = 0.0
        GVrefD = 0.0
        TocD = 0.0
        WrefD = 0.0
        SimD = 0.0
        GSrefD = 0.0
        AsyD = 0.0
        SrefD = 0.0
        ClD = 0.0
        label_0 = 0.0
        cnt = 0.0
        for key in features:
            cnt += 1.0
            if features[key]['features']['VrefD'] == 0.0:   VrefD += 1.0
            if features[key]['features']['GVrefD'] == 0.0:   GVrefD += 1.0
            if features[key]['features']['TocD'] == 0.0:   TocD += 1.0
            if features[key]['features']['WrefD'] == 0.0:   WrefD += 1.0
            if features[key]['features']['SimD'] == 0.0:   SimD += 1.0
            if features[key]['features']['GSrefD'] == 0.0:   GSrefD += 1.0
            if features[key]['features']['AsyD'] == 0.0:   AsyD += 1.0
            if features[key]['features']['SrefD'] == 0.0:   SrefD += 1.0
            if features[key]['features']['ClD'] == 0.0:   ClD += 1.0
            if features[key]['label'] == 0:   label_0 += 1.0
        
        print '--------------- Statistics -----------------'
        print 'VrefD is 0:\t', VrefD, '\t\t(%.2f%%)' % (VrefD/cnt*100)
        print 'SrefD is 0:\t', SrefD, '\t\t(%.2f%%)' % (SrefD/cnt*100)
        print 'GVrefD is 0:\t', GVrefD, '\t\t(%.2f%%)' % (GVrefD/cnt*100)
        print 'GSrefD is 0:\t', GSrefD, '\t\t(%.2f%%)' % (GSrefD/cnt*100)
        print 'WrefD is 0:\t', WrefD, '\t\t(%.2f%%)' % (WrefD/cnt*100)
        print 'TocD is 0:\t', TocD, '\t\t(%.2f%%)' % (TocD/cnt*100)
        print 'AsyD is 0:\t', AsyD, '\t\t(%.2f%%)' % (AsyD/cnt*100)
        print 'SimD is 0:\t', SimD, '\t\t(%.2f%%)' % (SimD/cnt*100)
        print 'ClD is 0:\t', SimD, '\t\t(%.2f%%)' % (ClD/cnt*100)
        print 'label is 0:\t', label_0, '\t\t(%.2f%%)' % (label_0/cnt*100)
        print 'Total size:\t', cnt

        print '-----Check Matirx-------'
        VM = pickle.load(file(self.dataStructureDir+ 'VM.pkl', 'rb'))
        print VM.shape
        cnt = 0.0
        for i in range(VM.shape[0]):
            for j in range(VM.shape[1]):
                if (VM[i, j] == 0):
                    cnt += 1
        print 'for VMij is 0:', cnt, '\t\t(%.2f%%)' % (cnt/(VM.shape[0]*VM.shape[1])*100)

        SM = pickle.load(file(self.dataStructureDir+ 'SM.pkl', 'rb'))
        print SM.shape
        cnt = 0.0
        for i in range(SM.shape[0]):
            for j in range(SM.shape[1]):
                if (SM[i, j] == 0):
                    cnt += 1
        print 'for SMij is 0:', cnt, '\t\t(%.2f%%)' % (cnt/(SM.shape[0]*SM.shape[1])*100)


if __name__ == "__main__":
    
    fg = FeatureGenerator()

    print 'Generating features...'
    t1 = time.time()
    fg.generateFeatures()
    t2 = time.time()
    print '\t__cost:', t2-t1, "s."
    print 'Saving features...'
    t1 = time.time()
    fg.saveFeatures()
    t2 = time.time()
    print '\t__cost:', t2-t1, "s."
    
    # fg.showStatistics()
    