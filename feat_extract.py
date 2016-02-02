from HTMLParser import HTMLParserimport sysimport urllibimport reimport zlibimport numpy as npimport math#ADDfrom multiprocessing import Processimport os, time, randomfrom time import timefrom lamp import *#ADDendtagdict={'br':0,'hr':1,'img':2,'input':3,'param':4,'meta':5,'link':6,        'area':7,'base':8,'col':9,'command':10,'embed':11,'keygen':12,        'param':13,'source':14,'track':15,'wbr':16}mediadict={'img':0,'video':1,'audio':2,'source':3,'embed':4}tagstack = []attrstack = []Dtag=['script','style']feat_vector=[]class MyFeatures:    def __init__(self):        self.cpw = 0 # number of words in page        self.ctw = 0 # number of words in title        self.average_wl = 0.0 # average word length        self.anchor_frac = 0.0 # precentage of anchor text        self.visible_frac = 0.0 # percentage of visible text        self.compress_rate = 0.0 # compression rate        self.cprecision100 = 0.0 # corpus precision top=100        self.cprecision200 = 0.0 # corpus precision top=200        self.cprecision500 = 0.0 # corpus precision top=500        self.cprecision1000 = 0.0 # corpus precision top=1000        self.crecall100 = 0.0 # corpus recall top=100        self.crecall200 = 0.0 # corpus recall top=200        self.crecall500 = 0.0 # corpus recall top=500        self.crecall1000 = 0.0 # corpus recall top=1000        self.LH=0.0 # Independent Likelihood        self.CLH=0.0 # Conditional Likelihood        self.tagnum=0 # number of tags in page        self.mediafrac=0.0 # number of media tag in page        #self.inputfrac=0.0 # fraction of input tag in page    def computeCTW(self, string):        s=re.split('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\s]+',string)        self.ctw=len([x for x in range(len(s)) if x != ''])    def likelihood(self,words,size=3):        words=words.strip()        words=re.split('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\s]+',words)#        print words#         print "len:",len(words)        if len(words)<size:            return#         if len(words)<50:#             print words        i=0        j=i+size        grams=dict()        g_counts=[]        #for conditional likelihood        grams_c=dict()        gc_counts=[]        while(j<=len(words)):            temp=' '.join(words[i:j])            tempc=' '.join(words[i:j-1])            if grams.get(temp,None)!=None:                grams[temp]+=1            else:                grams[temp]=1            if grams_c.get(tempc,None)!=None:                grams_c[tempc]+=1            else:                grams_c[tempc]=1            i+=1            j+=1#        print grams        k=len(grams)        kc=len(grams_c)        k=float(k)        kc=float(kc)        for i in grams:            g_counts.append(grams[i])        for i in grams_c:            gc_counts.append(grams_c[i])        g_counts=np.array(g_counts)        gc_counts=np.array(gc_counts)#        print "k,kc:",k,kc#        print g_counts,gc_counts        Prob=g_counts/k        Prob_c=gc_counts/kc#        print "Prob:",Prob        #Independent LikeliHood        self.LH=-sum(np.log(Prob))/k#         print "LH:",self.LH        #Conditional LikeliHood        Prob=[]        for i in grams:            temp=i.split(' ')[0:size-1]            temp=' '.join(temp)#            print temp            Prob.append(grams[i]/float(grams_c[temp]))#         print "Prob:",Prob        self.CLH=-sum(np.log(Prob))/k#         print "CLH:",self.CLH    def computeFeat(self, parser):        self.cpw = parser.wordcount        self.computeCTW(parser.title)        if parser.wordcount!=0:            self.average_wl = 1.0 * parser.charcount / parser.wordcount            self.anchor_frac = 1.0 * parser.anchorcount / parser.wordcount            self.visible_frac = 1 - 1.0 * parser.hiddencount / parser.wordcount#             self.cprecision100 = 1.0 * parser.word100 / parser.wordcount#             self.cprecision200 = 1.0 * parser.word200 / parser.wordcount#             self.cprecision500 = 1.0 * parser.word500 / parser.wordcount#             self.cprecision1000 = 1.0 * parser.word1000 / parser.wordcount        if len(parser.words)!=0:            self.compress_rate = 1.0 * len(zlib.compress(parser.words,9)) / len(parser.words)#         self.crecall100 = 1.0 * len(parser.popterm100) / 100#         self.crecall200 = 1.0 * len(parser.popterm200) / 200#         self.crecall500 = 1.0 * len(parser.popterm500) / 500#         self.crecall1000 = 1.0 * len(parser.popterm1000) / 1000        self.likelihood(parser.words)        self.tagnum = parser.tagnum        self.mediafrac = 1.*parser.mediatag/parser.tagnum        def get(self):        return [self.cpw, self.ctw, self.average_wl, self.anchor_frac, self.compress_rate,                 self.LH, self.CLH, self.tagnum, self.mediafrac]class MyHTMLParser(HTMLParser):       def __init__(self):        self.wordcount=0        self.title=''        self.word=[]        self.charcount=0        self.anchorcount=0        self.hiddencount=0        """self.word100 = 0        self.word200 = 0        self.word500 = 0        self.word1000 = 0        self.popterm100 = []        self.popterm200 = []        self.popterm500 = []        self.popterm1000 = []"""        self.tagnum = 0        self.mediatag = 0        self.inputtag = 0        #only consists of words        self.words=''        HTMLParser.__init__(self)    def handle_starttag(self,tag,attrs):        #if tag == 'a':        #    print (tag,len(attrs))        self.tagnum += 1        if tag not in tagdict:            #print (tag,len(attrs))            tagstack.append((tag,len(attrs)))            if len(attrs):                attrstack.extend(attrs)        else:            if tag == 'img':                self.mediatag += 1            elif tag == 'input':                self.inputtag += 1         #print tagstack        #print attrstack        #print '----------------------------------------'#        print "tag:",tag#        sys.stdout.write('<'+tag+' '+str(attrs)+'>')        def handle_endtag(self,tag):        #if tag == 'a':        #    print '/%s, %s' % (tag,tagstack[-1][0])        if len(tagstack) and tagstack[-1][0] == tag:            attrsnum=tagstack.pop()            #print '/%s' % tag            for i in range(attrsnum[1]):                attrstack.pop()    #        print "tag_end:",tag#        sys.stdout.write('<'+'\\'+tag+'>')##    def handle_comment(self, data):##        print('<!-- -->')##        print data    def handle_data(self,data):        #print data        if data.strip()!='':            if len(tagstack):                s=tagstack[-1]                #find title                if s[0]=='title':                    self.title+=data                    #print self.title                #self.content+=data                if s[0] in mediadict:                    self.mediatag += 1            data=data.strip()            if len(tagstack) == 0 or s[0] not in Dtag:                self.words+=data                self.words+=' '                #print data                tmp=re.split('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\s]+',data)                tmpcount=0                for i in range(len(tmp)):                    if tmp[i] != '':#                         testTime('')                        tmpcount += 1                        self.word.append(tmp[i])                        self.charcount+=len(tmp[i])                self.wordcount+=tmpcount                if len(tagstack) and s[1]: #attributes are not null                    for j in range(len(attrstack)-1,len(attrstack)-1-s[1],-1):                        if s[0] == 'a' and attrstack[j][0]=='href':                            self.anchorcount+=tmpcount                        if attrstack[j][0]=='hidden':                            self.hiddencount+=tmpcountvalData = "../data/webspam_val.dat/webspam_val.dat"#trainData = "../data/test.txt"trainData = "../res/webspam_train.dat.bz2/webspam_train.dat.bz2"trainLabel = "../res/webspam_train.lab/webspam_train.lab"outpFilePath = "../data/webspam_splite.dat"def initialize():    n = len(tagstack)    for i in xrange(n):        tagstack.pop()    m = len(attrstack)    for j in xrange(m):        attrstack.pop()#ADDdef handle_HTML(html):    initialize()    parser = MyHTMLParser()    parser.feed(html)    feat=MyFeatures()    if parser.wordcount:        feat.computeFeat(parser)    else:        print parser.wordcount    #print parser.tagnum    #print tagstack    tmp = [feat.cpw, feat.ctw, feat.average_wl, feat.anchor_frac, feat.visible_frac, feat.compress_rate, feat.LH, feat.CLH, feat.tagnum, feat.mediafrac]    #print tmp    return tmp#ADDenddef readInpFile(filePath, mode):    count = 0    if mode == 1:        featdb = MyDb('train_feat.db')        sampdb = MyDb('sample100000.db')    else:        featdb = MyDb('test_feat.db')    inpFile = open(filePath,"rb")    html = ''    starttime = time()    while True:        s = inpFile.read(1)        if s=='\0':            print count            #print len(tagstack), len(attrstack)            #print tagstack#             if count % 1000 == 0:#                 print 'running time: ',time()-start##            with open('../data/file/'+str(count)+".txt", 'a+') as f:##                f.write(html)            #print html            #p = Process(target=handle_HTML, args=(html,))            #p.start()            feat_vector.append((count,handle_HTML(html)))                        #if mode == 1 and count <= 50000:            #    sampdb.importFeatures([(count,feat_vector)])            #featdb.importFeatures([(count,feat_vector)])#            print parser.wordcount            #print parser.charcount            # compute feature vector            html=''            #print "length of tags: ", len(tagstack)            #print "length of attrs: ", len(attrstack)            #print tagstack            count+=1        else:            html+=s        if count == 100000:            sampdb.importFeatures(feat_vector)            #featdb.importFeatures(feat_vector)            #break        if s=='':            featdb.importFeatures(feat_vector)            breaktc = {}    lastTime = time.time()def testTime(k):    global lastTime    if k in tc:        tc[k]+=time.time()-lastTime    else:        tc[k] = 0    lastTime = time.time()    def printTime():    for k in tc:        print k,tc[k]        def yieldFeatAndLabel(dataPath,labelPath):    count=0    datafp = open(dataPath,"rb")    labelfp = open(labelPath,"r")    html = ''#     testTime('')    while 1:#         testTime('aa')        s = datafp.read(1)#         s = datafp.readline()        if s == '\0':            initialize()                        parser = MyHTMLParser()            parser.feed(html)                        feat=MyFeatures()            feat.computeFeat(parser)                        l = labelfp.readline()                        yield parser.word,feat.get(),float(l)                        html = ''            count += 1#             testTime('')        else:#             testTime('')            html += s#             testTime('s')        if s == '':            break#         testTime('')if __name__ == "__main__":#     cleanup('features.txt')#     readFreqWords('freq_word.txt')    #readInpFile(trainData)    for f,l in yieldFeatAndLabel(trainData,trainLabel):        print f,l    #inpFile = open(inpFilePath,"r")    #outpFile = open(outpFilePath,"w")       #s=inpFile.read(300000)    #outpFile.write(s)