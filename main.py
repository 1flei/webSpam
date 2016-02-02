import pprint
import sys, os
import logging
import lamp
import random
import time
from lamp import LAMP
from lshCache import LSHCache
from lshCache import KNNCache
from feat_extract import yieldFeatAndLabel
from feat_extract import testTime
from feat_extract import printTime
from sklearn import svm

# valData = "../res/webspam_val.dat/webspam_val.dat"
# valLabel = "../res/webspam_train.lab"
trainData = "../res/webspam_train.dat"
trainLabel = "../res/webspam_train.lab"
outpFilePath = "../res/webspam_splite.dat"




if __name__ == "__main__":
    logging.basicConfig(filename='lsh.log',level=logging.DEBUG)
    
    s1 = svm.SVC(C=2**3,gamma=2**-6)
    s2 = svm.SVC()
    lamp = LAMP(KNNCache(LSHCache(b=50,r=1,min_shingle=3,max_shingle=4),k = 200),classifier = s2,db='lamp.db')
    
    start = time.time()
    last = time.time()
    
#     resFile = open('output.txt','w')
    
    id = 0
    cnt0 = 1
    cnt1 = 0
    
    testoutp = open("testout3.txt","w")
    for w,f,l in yieldFeatAndLabel(trainData, trainLabel):
        w.extend(['$' for i in xrange(max(3-len(w),0))])
        
        testoutp.write('***%d\n%s\n%s\n%s\n'%(id,w,f,l))
        
        if id%1000<700:
            if id%100==99:
                print id,time.time()-start
                print f,l
#             print 'a',time.time()-last
#             last = time.time()
#             testTime('')
            lamp.insert(id=id, docs=w, feature=f, label=l)
#             testTime('a')
#             printTime()
#             print 'b',time.time()-last
#             last = time.time()
        else:
            if id%1000==700:
                cnt0=0
                cnt1=0
                cnt2=0
                cnt3=0
            if id%100==99:
                print cnt0,cnt1,cnt2,cnt3
            if id%1000==999:
                logging.debug('%d %d %d %d'%(cnt0,cnt1,cnt2,cnt3))
                print 'timeCost:',time.time()-last
                last = time.time()
            predictedLabel = lamp.predict(docs=w, feature=f)
#             print 'and its label is', l
            if predictedLabel==-1 and l==-1:
                cnt0+=1
            elif predictedLabel==1 and l==-1:
                cnt1+=1
            elif predictedLabel==-1 and l==1:
                cnt2+=1
            else:
                cnt3+=1
        id+=1
#     cnt0=0
#     cnt1=0
#     for (id,docs,feature,label) in yieldFeatAndLabel(5000, plabel, pp):
#         print id
#         predictedLabel = lamp.predict(docs=docs, feature=feature)
#         if predictedLabel==label :
#             cnt0+=1
#             print 'correct'
#         else:
#             cnt1+=1
#     print 1.*cnt0/(cnt0+cnt1)
        