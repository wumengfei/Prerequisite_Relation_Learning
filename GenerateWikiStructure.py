# -*- coding: cp936 -*-
#潘亮铭
#2016.11.23

import json
import re
import logging
import os.path
import sys
import multiprocessing
import codecs

from sklearn.externals import joblib
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import cPickle as pickle

'''
生成EW，和W，格式为：
{"mentionTitle": {"Title":"WikiTitle", "Article":"WikiArticle", "Vector":"xxx"}}
保存成二进制格式
'''
def GenerateEW(mention,article,modelpath):
    model = Word2Vec.load(modelpath)
    print 'Model Loaded!'
    ArticleDic = {}
    articleF = codecs.open(article,"r","utf-8")
    for line in articleF:
        x = line.split("::;")
        ArticleDic[x[0]] = x[1].strip()
    print 'Article Loaded!'
    mentionF = codecs.open(mention,"r","utf-8")
    EW = {}
    cc = 0
    suc = 0
    for line in mentionF:
        cc += 1
        if cc%100000==0:
            print cc
        x = line.split("\t\t")
        Mention = x[0]
        Title = x[1].split("::;")[0]
        if not ArticleDic.has_key(Title):
            continue
        Article = ArticleDic[Title]
        success = 0
        try:
            Vector = model[unicode(Mention.replace(" ","_"))]
            success = 1
        except KeyError,e:
            success = 0
        if success==1:
            suc += 1
            EW[Mention] = {"Title":Title,"Article":Article,"Vector":Vector}
    print 'Outputing...PKL',suc
    #输出二进制格式
    pickle.dump(EW,file(r"../datastructure/EW.pkl","wb"),True)
    print 'Outputing...Readable'
    #输出Human Readable的格式
    outputF = codecs.open(r"../datastructure/EW.readable","w","utf-8")
    for (key,value) in EW.items():
        OUT = {"Mention":key ,"Title":value["Title"],"Article":value["Article"],"Vector":value["Vector"].tolist()}
        encodedjson_utf = json.dumps(OUT,ensure_ascii=False)
        outputF.write(encodedjson_utf+"\n")
    outputF.close()
	
'''
为EW.pkl增加Inlink和Outlink字段，输出EW+.pkl
"Inlinks": set("Mention1","Mention2"), "Outlinks": "Inlinks": set("Mention1","Mention2")
'''
def EnrichEW(EWF,Inlink,Outlink):
	F = file(EWF,"rb")
	EW = pickle.load(F)
	Title2Mention = {}
	for (key,value) in EW.items():
		Title2Mention[value["Title"]] = key
	for (key,value) in EW.items():
		EW[key]["Inlinks"] = []
		EW[key]["Outlinks"] = []
	InlinkF = codecs.open(Inlink,"r","utf-8")
	for line in InlinkF:
		x = line.split("\t\t")
		Title = x[0]
		Inlinks = x[1].strip().split(";")
		if Title2Mention.has_key(Title):
			Mention = Title2Mention[Title]
			EW[Mention]["Inlinks"] = Inlinks
	OutlinkF = codecs.open(Outlink,"r","utf-8")
	for line in OutlinkF:
		x = line.split("\t\t")
		Title = x[0]
		Outlinks = x[1].strip().split(";")
		if Title2Mention.has_key(Title):
			Mention = Title2Mention[Title]
			EW[Mention]["Outlinks"] = Outlinks
	print 'Outputing...PKL'
    #输出二进制格式
	pickle.dump(EW,file(r"../datastructure/EW+.pkl","wb"),True)

'''
测试一下EW的Inlink和OutLink是否正确
'''
def testEW(EWF):
	F = file(EWF,"rb")
	EW = pickle.load(F)
	ci = 0
	co = 0
	for (key,value) in EW.items():
		if not len(value["Inlinks"])==0:
			ci += 1
		if not len(value["Outlinks"])==0:
			co += 1
	print ci,co,len(EW)
	
'''
读取EW,输出Mention和Article的列表
'''	
def ListArticleandMention(EWF):
	#EW = np.load(EWF)
	F = file(EWF,"rb")
	EW = pickle.load(F)
	outputF = codecs.open(r"../datastructure/EW.articlelist","w","utf-8")
	for (key,value) in EW.items():
		outputF.write(value["Title"]+"\t\t"+key+"\n")
	outputF.close()
	
'''
读取EW.articlelist，统计课程概念有多少是Wiki概念
'''
def PrintWikiConcepts():
	conceptlist = joblib.load(r"../datastructure/conceptList.pkl")
	WikiMentionSet = set()
	EWF = codecs.open(r"../datastructure/EW.articlelist","r","utf-8")
	for line in EWF:
		WikiMentionSet.add(line.split("\t\t")[1].strip())
	count = 0
	for concept in conceptlist:
		if concept in WikiMentionSet:
			print concept
			count += 1
	print count,len(conceptlist),float(count)/len(conceptlist)

'''
生成CW，格式为{"MainConcept":[vector]}
这里的MainConcept是一个同义词集的第一个代表的，顺序与conceptDic['conceptList']对应
'''
def GenerateCW(modelpath,conceptpath):
	model = Word2Vec.load(modelpath)
	CW = {}
	print 'Model Loaded!'
	conceptDic = pickle.load(file(conceptpath,"rb"))
	conceptlist = conceptDic['conceptList']
	concept2int = conceptDic['concept2int']
	int2concept = conceptDic['int2concept']
	#这里的规则是：如果有Wiki词条，就直接用它的Embedding，否则取第一个词的Embedding
	for mainconcept in conceptlist:
		HasWiki = False
		syntoms = int2concept[concept2int[mainconcept]]
		for concept in syntoms:
			if concept.find("-")>=0:
				concept = concept.replace("-"," ")
			num = len(concept.split(" "))
			grams = concept.split(" ")
			if num==1:
				try:
					CW[mainconcept] = model[unicode(concept)]
					HasWiki = True
					break
				except KeyError,e:
					a = 1
			elif num==2:
				try:
					CW[mainconcept] = model[unicode(grams[0]+"_"+grams[1])]
					HasWiki = True
					break
				except KeyError,e:
					a = 1
			elif num==3:
				try:
					CW[mainconcept] = model[unicode(grams[0]+"_"+grams[1]+"_"+grams[2])]
					HasWiki = True
					break
				except KeyError,e:
					a = 1
		if HasWiki==False:
			concept = mainconcept
			#特殊处理一下带-的词组
			if concept.find("-")>=0:
				concept = concept.replace("-"," ")
			num = len(concept.split(" "))
			grams = concept.split(" ")
			if num==1:
				try:
					CW[mainconcept] = model[unicode(concept)]
				except KeyError,e:
					print concept
			elif num==2:
				try:
					CW[mainconcept] = model[unicode(grams[0]+"_"+grams[1])]
				except KeyError,e:
					try:
						CW[mainconcept] = (model[unicode(grams[0])] + model[unicode(grams[1])])
					except KeyError,e:
						print concept
			elif num==3:
				try:
					CW[mainconcept] = model[unicode(grams[0]+"_"+grams[1]+"_"+grams[2])]
				except KeyError,e:
					try:
						CW[mainconcept] = (model[unicode(grams[0]+"_"+grams[1])] + model[unicode(grams[2])])
					except KeyError,e:
						try:
							CW[mainconcept] = (model[unicode(grams[0])] +  model[unicode(grams[1]+"_"+grams[2])])
						except KeyError,e:
							try:
								CW[mainconcept] = (model[unicode(grams[0])] + model[unicode(grams[1])] +model[unicode(grams[2])])
							except KeyError,e:
								print concept
	pickle.dump(CW,file(OUTPUT_DIR+r"CW.pkl","wb"),True)


'''
生成CW，格式为{"MainConcept":[vector]}
这里的MainConcept是一个同义词集的第一个代表的，顺序与conceptDic['conceptList']对应
'''
def GenerateCWOld(modelpath,conceptpath):
	model = Word2Vec.load(modelpath)
	CW = {}
	print 'Model Loaded!'
	conceptDic = pickle.load(file(conceptpath,"rb"))
	conceptlist = conceptDic['conceptList']
	concept2int = conceptDic['concept2int']
	int2concept = conceptDic['int2concept']
	#这里的规则是：取平均，这是老版本
	for mainconcept in conceptlist:
		FinalVector = np.array(400*[0.0])
		syntoms = int2concept[concept2int[mainconcept]]
		for concept in syntoms:
			#特殊处理一下带-的词组
			if concept.find("-")>=0:
				concept = concept.replace("-"," ")
			num = len(concept.split(" "))
			grams = concept.split(" ")
			if num==1:
				try:
					FinalVector += model[unicode(concept)]
				except KeyError,e:
					print mainconcept,concept
			elif num==2:
				try:
					FinalVector += model[unicode(grams[0]+"_"+grams[1])]
				except KeyError,e:
					try:
						FinalVector += (model[unicode(grams[0])] + model[unicode(grams[1])])
					except KeyError,e:
						print mainconcept,concept
			elif num==3:
				try:
					FinalVector += model[unicode(grams[0]+"_"+grams[1]+"_"+grams[2])]
				except KeyError,e:
					try:
						FinalVector += (model[unicode(grams[0]+"_"+grams[1])] + model[unicode(grams[2])])
					except KeyError,e:
						try:
							FinalVector += (model[unicode(grams[0])] +  model[unicode(grams[1]+"_"+grams[2])])
						except KeyError,e:
							try:
								FinalVector += (model[unicode(grams[0])] + model[unicode(grams[1])] +model[unicode(grams[2])])
							except KeyError,e:
								print mainconcept,concept
		if sum(FinalVector**2)!=0:
			CW[mainconcept] = FinalVector
		else:
			print "Missing",mainconcept
	pickle.dump(CW,file(OUTPUT_DIR+r"CW.pkl","wb"),True)
	
	
'''
生成SC，格式为一个n*n的矩阵，矩阵的index与conceptList的顺序对应
'''
def GenerateSC(CWpath, conceptpath):
	CW = pickle.load(file(CWpath,"rb"))
	conceptlist = pickle.load(file(conceptpath,"rb"))['conceptList']
	n = len(conceptlist)
	SCList = []
	for c in conceptlist:
		row = []
		if CW.has_key(c):
			vc = np.array(CW[c])
			nc = np.dot(vc,vc.T)**0.5
			for k in conceptlist:
				if CW.has_key(k):
					vk = np.array(CW[k])
					nk = np.dot(vk,vk.T)**0.5
					csim = 0.5 + 0.5*np.dot(vc,vk)/(nk*nc)
					row.append(csim)
				else:
					row.append(0.0)
			SCList.append(row)
		else:
			SCList.append(n*[0.0])
	SC = np.matrix(SCList)
	pickle.dump(SC,file(OUTPUT_DIR+r"SC.pkl","wb"),True)
	print SC.shape
	
'''
测试一下SC的效果，输出每个Concept最接近的Top N个概念
'''
def testSC(SCpath,conceptpath):
	SC = pickle.load(file(SCpath,"rb"))
	conceptlist = pickle.load(file(conceptpath,"rb"))['conceptList']
	MSC = np.argsort(-SC)
	i = 0
	for c in conceptlist:
		print c+"\t\t"
		for k in range(0,10):
			print conceptlist[MSC[i,k]]+", "+str(SC[i,MSC[i,k]])
		print "\n"
		i += 1

'''
生成Wiki Article的List
'''
def generateWikiArticleList(EWpath):
	EW = pickle.load(file(EWpath,"rb"))
	WikiArticleList = []
	for key in EW:
		WikiArticleList.append(key)
	pickle.dump(WikiArticleList,file(r"../datastructure/wikiList.pkl","wb"),True)

'''
生成WC，即concept与WikiArticle的矩阵
'''
def generateWC(CWpath, EWpath, conceptpath, wikipath):
	CW = pickle.load(file(CWpath,"rb"))
	EW = pickle.load(file(EWpath,"rb"))
	conceptlist = pickle.load(file(conceptpath,"rb"))['conceptList']
	wikilist = pickle.load(file(wikipath,"rb"))
	n = len(conceptlist)
	m = len(wikilist)
	WCList = []
	for c in conceptlist:
		print c
		row = []
		if CW.has_key(c):
			vc = np.array(CW[c])
			nc = np.dot(vc,vc.T)**0.5
			for w in wikilist:
				vw = np.array(EW[w]["Vector"])
				nw = np.dot(vw,vw.T)**0.5
				csim = 0.5 + 0.5*np.dot(vc,vw)/(nc*nw)
				row.append(csim)
			WCList.append(row)
		else:
			WCList.append(m*[0.0])
	WC = np.matrix(WCList)
	print WC.shape
	#pickle.dump(WC,file(r"../datastructure/WC.pkl","wb"),True)
	np.save(OUTPUT_DIR+r"WC.npy",WC)
	
'''
测试一下WC的效果，输出每个Concept最接近的Top N个Wiki文章
'''
def testWC(WCpath, conceptpath, wikipath):
	print "Reading WC..."
	WC = np.load(WCpath)
	conceptlist = pickle.load(file(conceptpath,"rb"))['conceptList']
	wikilist = pickle.load(file(wikipath,"rb"))
	print "Sorting WC..."
	MWC = np.argsort(-WC)
	i = 0
	for c in conceptlist:
		print c+"\t\t"
		for k in range(0,10):
			print wikilist[MWC[i,k]]+", "+str(WC[i,MWC[i,k]])
		print "\n"
		i += 1
	


if __name__ == "__main__":
    #GenerateEW(r"../rawdata/Wiki/MentionWikiCounts",r"../rawdata/Wiki/wiki_article",r"../rawdata/Wiki/wiki_pharse_corpus.model")
    #EnrichEW(r"../datastructure/EW.pkl",r"../rawdata/Wiki/enwiki-inlink.dat",r"../rawdata/Wiki/enwiki-outlink.dat")
	#testEW(r"../datastructure/EW+.pkl")
	#ListArticleandMention(r"../datastructure/EW.pkl")
    #PrintWikiConcepts()
    
    INPUT_DIR = r'../datastructure/20170109/CAL/'
    OUTPUT_DIR = r'../datastructure/20170109/CAL/'

    GenerateCW(r"../rawdata/Wiki/wiki_pharse_corpus.model", INPUT_DIR+r"conceptDict.pkl")
    GenerateSC(INPUT_DIR+r"CW.pkl", INPUT_DIR+r"conceptDict.pkl")
    #testSC(r"../datastructure/20161221/SC.pkl",r"../datastructure/20161221/conceptDict.pkl")
    #generateWikiArticleList(r"../datastructure/EW.pkl")
    generateWC(INPUT_DIR+r"CW.pkl", r"../datastructure/EW.pkl", INPUT_DIR+r"conceptDict.pkl", r"../datastructure/wikiList.pkl")
    # testWC(r"../datastructure/20161221/WC.npy", r"../datastructure/20161221/conceptDict.pkl",r"../datastructure/wikiList.pkl")
