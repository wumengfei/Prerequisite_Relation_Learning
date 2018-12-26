# -*- coding:utf-8 -*-

import numpy as np
import cPickle as pickle
import re
from sklearn import metrics

TYPE = 'CAL'
dataStructureDir = '../datastructure/20170109/'+TYPE+'/'
labeledFileDir = '../rawdata/Dataset/'+TYPE+'/'+TYPE+'_LabeledFile'
featuresDir = '../features/20170109/'+TYPE+'_'

tem_concepts = pickle.load(file(dataStructureDir+ 'conceptDict.pkl', 'rb'))
conceptList = tem_concepts['conceptList']
int2concept = tem_concepts['int2concept']
concept2int = tem_concepts['concept2int']

ew = pickle.load(file(dataStructureDir+ '../../EW+.pkl', 'rb'))
print ew[0:5]
exit()
cnt = 0
with open(labeledFileDir.replace('/'+TYPE+'_', '/W-'+TYPE+'_'), 'w') as fw:
    with open(labeledFileDir, 'r') as fr:
        for line in fr.readlines():
            words = line.strip().split('\t\t') 
            A = words[0]
            B = words[1]
            
            hasA = False
            hasB = False
            for w in int2concept[str(concept2int[A])]:
                if ew.has_key(w):
                    hasA = True
                    break
            for w in int2concept[str(concept2int[B])]:
                if ew.has_key(w):
                    hasB = True
                    break
            
            if hasA and hasB:
                cnt += 1
                fw.write(line)

print 'W-'+TYPE+' size is:', cnt
