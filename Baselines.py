# -*- coding:utf-8 -*-

import numpy as np
import cPickle as pickle
import re, time
from sklearn import metrics

class SupervisedRI:
    def __init__(self):
        TYPE = 'CAL'
        IF_W = 'W-'
        self.K = 10
        self.dataStructureDir = '../datastructure/20170109/'+TYPE+'/'
        self.labeledFileDir = '../rawdata/Dataset/'+TYPE+'/'+IF_W+TYPE+'_LabeledFile'
        self.featuresDir = '../features/20170109/'+IF_W+TYPE+'_'
        self.skipList = []
        # ML = [111, 548]; DSA = [263, 390, 449]; CAL = [81, 359];
        self.VIDNUM = []
        if TYPE=='ML': self.VIDNUM = [111, 548] # Defined by videos amount of each course [course_1, course_1+2, ...]
        elif TYPE=='DSA': self.VIDNUM = [263, 390, 449]
        elif TYPE=='CAL': self.VIDNUM = [81, 359]
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
        avgd_n = pickle.load(file(self.dataStructureDir+ 'AVGD_N.pkl', 'rb'))
        self.AVG_D = avgd_n['AVG_D']
        self.Nij = avgd_n['Nij']
        self.maxAvg_d = np.max(self.AVG_D)
        self.maxNij = np.max(self.Nij)
        # w_avgd_n = pickle.load(file(self.dataStructureDir+ 'W_AVGD_N.pkl', 'rb'))
        # self.W_AVG_D = w_avgd_n['W_AVG_D']
        # self.W_Nij = w_avgd_n['W_Nij']
        # self.maxW_Avg_d = np.max(self.W_AVG_D)
        # self.maxW_Nij = np.max(self.W_Nij)
        self.VI = pickle.load(file(self.dataStructureDir+ 'VI.pkl', 'rb'))
        self.skipList = []
        # 生成 ML_sriFeatures_10.pkl 需注释以下
        self.ew = pickle.load(file(self.dataStructureDir+ '../../EW+.pkl', 'rb'))
        self.EW = []
        for concept in self.conceptList:
            has = False
            for w in self.int2concept[str(self.concept2int[concept])]:
                if self.ew.has_key(w):
                    has = True
                    self.EW.append(self.ew[w])
                    break
            if has == False:
                self.skipList.append(concept)
                self.EW.append({})
        print 'self.skipList length:', len(self.skipList)
        #
        self.WC = np.load(self.dataStructureDir+ 'WC.npy')
        self.argsortedWC = np.argsort(-self.WC, kind='heapsort')

    def countAny(self, text, word):
        cnt = 0.0
        for w in self.int2concept[str(self.concept2int[word])]:
            cnt += text.count(w)
        return cnt
        

    def WikipediaLinkBasedJaccardSimilarity(self, A, B):
        inUnionSet = list(set(self.EW[A]['Inlinks']).union(set(self.EW[B]['Inlinks'])))
        inIntersectionSet = list(set(self.EW[A]['Inlinks']).intersection(set(self.EW[B]['Inlinks'])))
        outUnionSet = list(set(self.EW[A]['Outlinks']).union(set(self.EW[B]['Outlinks'])))
        outIntersectionSet = list(set(self.EW[A]['Outlinks']).intersection(set(self.EW[B]['Outlinks'])))
        return (1.0*len(inIntersectionSet)/len(inUnionSet)) if len(inUnionSet)!=0 else 1.0, (1.0*len(outIntersectionSet)/len(outUnionSet)) if len(outUnionSet)!=0 else 1.0
    
    # [Wikipedia features]
    def WikipediaLinkBasedSemanticSimilarity(self, A, B):
        Q_i = []
        for link in self.EW[A]['Inlinks']:
            if link in self.concept2int.keys():
                Q_i.append(link)
        Q_j = []
        for link in self.EW[B]['Inlinks']:
            if link in self.concept2int.keys():
                Q_j.append(link)
        if len(Q_i)==0:
            Q_i.append(self.conceptList[A])
        if len(Q_j)==0:
            Q_j.append(self.conceptList[B])
        return (1.0 - (max(np.log(len(Q_i)), np.log(len(Q_j))) - np.log(len(list(set(Q_i).intersection(set(Q_j)))))) / (np.log(len(self.conceptList)-len(self.skipList)) - min(np.log(len(Q_i)), np.log(len(Q_j)))) ) if (len(list(set(Q_i).intersection(set(Q_j))))!=0) else 0.0

    def RelationalStrengthInTextbookWikipedia(self, A, B):
        return np.log((self.W_Nij[A, B]/self.maxW_Nij)/(self.W_AVG_D[A, B]/self.maxW_Avg_d)) if self.W_AVG_D[A, B]!=0 else 0.0

    def SupportiveRelationshipInConceptDefinition(self, A, B):
        sentenceA = re.split('\.|\!|\?', self.EW[A]['Article'].lower())[0]
        sentenceB = re.split('\.|\!|\?', self.EW[B]['Article'].lower())[0]
        a = 0.0
        b = 0.0
        if self.conceptList[A] in sentenceB:
            a = 1.0
        if self.conceptList[B] in sentenceA:
            b = 1.0
        if a != b: return (a-b)
        else: return 0.0
    
    def RefD(self, A, B):
        def findTopKConceptsInWikipedia(concept, k):
            return self.argsortedWC[concept, 0:k].tolist()
        def Wref(A, B, k):
            if A in findTopKConceptsInWikipedia(B, k):
                return 1.0
            return 0.0
        k = self.K
        val_1 = 0.0
        sum_wb = 0.0
        for b in findTopKConceptsInWikipedia(B, k):
            val_1 += Wref(b, A, k) * self.WC[B, b]
            sum_wb += self.WC[B, b]
        val_1 /= sum_wb
        val_2 = 0.0
        sum_wa = 0.0
        for a in findTopKConceptsInWikipedia(A, k):
            val_2 += Wref(a, B, k) * self.WC[A, a]
            sum_wa += self.WC[A, a]
        val_2 /= sum_wa
        return val_1 - val_2

    def WikiContentCosineSimilarity(self, A, B):
        va = np.array(self.EW[A]['Vector'])
        vb = np.array(self.EW[B]['Vector'])
        return np.dot(va, vb.T) / (np.sqrt(np.dot(va, va.T))*np.sqrt(np.dot(vb, vb.T)))

    def ConceptCo_occurrenceInWikipedia(self, A, B):
        return self.W_Nij[A, B]

    # [Textbook features]
    def TOCDistance(self, A, B):
        sum_ab = 0.0
        num = 0
        for i in range(len(self.VIDNUM)):
            if len(self.VI[A][i]) != 0 and len(self.VI[B][i]) != 0:
                sum_ab += 1.0*abs(sum(self.VI[A][i])/len(self.VI[A][i]) - 1.0*sum(self.VI[B][i])/len(self.VI[B][i]))
                num += 1
        if num == 0:
            return 0
        else:
            return sum_ab / num

    def ConceptCo_occurrenceInTextbook(self, A, B):
        return self.Nij[A, B]

    def RelationalStrengthInTextbook(self, A, B):
        return np.log((self.Nij[A, B]/self.maxNij)/(self.AVG_D[A, B]/self.maxAvg_d)) if self.AVG_D[A, B]!=0 else 0.0
        

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
                # inWLBJS, outWLBJS = self.WikipediaLinkBasedJaccardSimilarity(Aindex, Bindex)
                features = {'CCT': self.ConceptCo_occurrenceInTextbook(Aindex, Bindex), 'RST': self.RelationalStrengthInTextbook(Aindex, Bindex), 'TocD': self.TOCDistance(Aindex, Bindex),}# 'CCW': self.ConceptCo_occurrenceInWikipedia(Aindex, Bindex), 'WCCS': self.WikiContentCosineSimilarity(Aindex, Bindex), 'RefD': self.RefD(Aindex, Bindex), 'inWLBJS': inWLBJS, 'outWLBJS': outWLBJS, 'WLBSS': self.WikipediaLinkBasedSemanticSimilarity(Aindex, Bindex), 'RS': self.RelationalStrengthInTextbookWikipedia(Aindex, Bindex), 'SR': self.SupportiveRelationshipInConceptDefinition(Aindex, Bindex)}
                self.features[A+'::;'+B] = {'features': features, 'label': label}
                cnt += 1
                if cnt%1000 == 0:
                    print '\t__cnt:', cnt
                # print sample
        pickle.dump(self.features, file(self.featuresDir+ 'sriFeatures_'+ (str)(self.K) +'.pkl', 'wb'), True)



if __name__ == "__main__":

    sri = SupervisedRI()
    sri.generateFeatures()
