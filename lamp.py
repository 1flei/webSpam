from lshCache import LSHCache
from lshCache import KNNCache
from sklearn import svm
import sqlite3
#from shove import Shove
import time

class MyDb:
    def __init__(self, fileName):
        self._conn = sqlite3.connect(fileName)
        self._conn.execute('create table if not exists webpages(id INTEGER,webpage TEXT)')
#         self.execute('create index if not exists webpagesIndex on webpages(id)')
        self._conn.execute('create table if not exists features(id INTEGER,fno INTEGER,feature REAL)')
        self.createIndex()
        
    
#     def insertWebpages(self, id, webpage):
#         res = self.execute('insert into webpages(id,webpage) values (?,?)',id,webpage)
#         return res
#     
#     def getWebpagesById(self, id):
#         cursor = self.execute('select id,webpage from webpages where id=?', id)
#         row = cursor.fetchone()
#         if row is None:
#             return None
#         return row[1]

    def createIndex(self):
        self._conn.execute('create index if not exists featuresIndex on features(id)')
    
    def dropIndex(self):
        self._conn.execute('drop index featuresIndex')
    
    def insertFeatures(self, id, feature):
        for i,r in enumerate(feature):
            res = self._conn.execute('insert into features(id,fno,feature) values (?,?,?)',(id,i,r))
        return res
    
    def getFeaturesById(self, id):
        cursor = self._conn.execute('select id,feature from features where id=:what', {'what':id})
        feature = []
        for row in cursor:
            feature.append(row[2])
        return feature
    
    def importFeatures(self, inps):
        self.dropIndex()
        for (id,feature) in inps:
            self.insertFeatures(id, feature)
        self.createIndex()
        
class MyMemCache:
    """
    indexing for features
    """
    def __init__(self,k=3,size=2**25,db="mci.db"):
        self._k = k
        self._size = size
        self._curind = 0
        self._dict = [{} for i in xrange(k)]
        self._db = MyDb(db)
        
    def insert(self,id,features):
#         print 'insert',id,features
        if len(self._dict[self._curind])<self._size:
            self._dict[self._curind][id] = features
        else:
            self._curind = (self._curind+1)%self._k
            self._db.importFeatures(self._dict[self._curind])
            self._dict[self._curind].clear()
            self._dict[self._curind][id] = features
            
    def get(self,id):
        for i in xrange(self._k):
#             for k in self._dict[i]:
#                 print k,self._dict[k]
            if id in self._dict[i]:
                return self._dict[i][id]
        return self._db.getFeaturesById(id)

class LAMP:
    
    def __init__(self, knn=KNNCache(), classifier=svm.SVC(), db="lamp.db"):
        self._knn = knn
        
        self._labels = {}
        self._feature = MyMemCache(db=db)
        self._classifier = classifier
    
#     def _getDocsById(self, id):
#         docs = []
#         return docs
    
    def _getLabelById(self,id):
        label = self._labels[id]
        return label
    
    def _getFeaturesById(self, id):
        return self._feature.get(id)
        
    def insert(self, docs, id, label, feature):
#         print 'lampInsert',id,feature
        self._knn.insert(docs, id)
        self._labels[id] = label
        self._feature.insert(id, feature)
        
    def predict(self, docs, feature):
        dups = self._knn.getKnn(docs)
        features = []
        labels = []
        for dupid in dups:
            #build a svm on dups
            X = self._getFeaturesById(dupid)
            features.append(X)
            y = self._getLabelById(dupid)
            labels.append(y)
#         print 'f',features
#         print 'l',labels
        isAllTheSame = True
        for l in labels:
            if l!=labels[0]:
                isAllTheSame = False
                break
        if isAllTheSame:
            return labels[0]
        else:
            s = svm.SVC()
            s.fit(features,labels)
            res = s.predict(feature)
#             self._classifier.fit(features,labels)
#             res = self._classifier.predict(feature)
            return res
        