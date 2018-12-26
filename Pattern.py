

    

def Baseline2(self):
    self.patterns = [['such as', '+', '>'], 
                    ['belong to', '+', '<'],
                    ['given','+', '<'],
                    ['type of','+'],
                    ['belongs to', '+', '<']]
    def FAppears(text, word):
        minInd = 9999999
        for w in self.int2concept[str(self.concept2int[word])]:
            ind = text.find(w)
            if ind>=0 and ind<minInd:
                minInd = ind
        return minInd

    def LAppears(text, word):
        maxInd = -1
        for w in self.int2concept[str(self.concept2int[word])]:
            ind = text.rfind(w)
            if ind>=0 and ind>maxInd:
                maxInd = ind
        return maxInd
    def matcher(A, B, M):
        FlagCount = 0
        for vk in self.captions:
            text = vk['text'].lower()
            sentences = re.split('\.|\!|\?', text)[0:-1]
            L = len(sentences)
            if L<M:
                continue
            for i in range(0,L-M+1):
                sgroup = ""
                for j in range(0,M):
                    sgroup += sentences[i+j]
                if self.countAny(sgroup,A)!=0 and self.countAny(sgroup,B)!=0:
                    for p in patterns:
                        ind = sgroup.find(p[0])
                        if ind>=0:
                            if p[1]=='+':
                                p1 = FAppears(sgroup[ind:],A)
                                p2 = FAppears(sgroup[ind:],B)
                                if p1>p2:
                                    if p[2]=='<':
                                        FlagCount -= 1
                                    elif p[2]=='>':
                                        FlagCount += 1
                                elif p1<p2:
                                    if p[2]=='<':
                                        FlagCount += 1
                                    elif p[2]=='>':
                                        FlagCount -= 1
                            elif p[1]=='-':
                                p1 = LAppears(sgroup[0:ind],A)
                                p2 = LAppears(sgroup[0:ind],B)
                                if p1>p2:
                                    if p[2]=='<':
                                        FlagCount += 1
                                    elif p[2]=='>':
                                        FlagCount -= 1
                                elif p1<p2:
                                    if p[2]=='<':
                                        FlagCount -= 1
                                    elif p[2]=='>':
                                        FlagCount += 1
        if FlagCount>0:
            return 1.0
        else:
            return 0.0
    
    ypred = []
    for i in range(len(self.Ltest)):
        [A, B] = self.Ltest[i].split('::;')
        ypred.append(matcher(A, B))
    
    cnt = 0
    for i in ypred:
        if i == 0: cnt += 1
    print 'match num:', cnt
    
    ypred = np.asarray(ypred)
    A = metrics.accuracy_score(self.Ytest, ypred)
    P = metrics.precision_score(pc.Ytest, ypred)
    R = metrics.recall_score(pc.Ytest, ypred)
    F1 = metrics.f1_score(pc.Ytest, ypred)
    print "accuracy:", A
    print "precision:", P
    print "recall:", R
    print "f1 score:", F1
    print metrics.classification_report(self.Ytest, ypred, target_names=['Non', 'Pre'])
    return P, R