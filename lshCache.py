#refer to https://github.com/embr/lsh
#we recode almost all methods of the class because of some errors and weakness

from collections import defaultdict
import numpy as np
import random
import sys
import time
import logging

logging.getLogger().setLevel(logging.INFO)

class LSHCache:

    def __init__(self, b=20, r=5, min_shingle=3, max_shingle=4, shingle_vec_bucket_size=2**31):
        # assign it
        self._n = b*r
        self._b = b
        self._r = r
        self._max_shingle = max_shingle
        self._min_shingle = min_shingle
        self._sbucket_size = shingle_vec_bucket_size

        # check it
        assert self._min_shingle >= 0, '_min_shingle must be greater and equal than 0.  Current _min_shingle=%d' % (self._min_shingle)
        assert self._max_shingle > self._min_shingle, '_max_shingle must be greater than _min_shingle.  Current _max_shingle=%d and _min_shingle=%d' % (self._max_shingle,self.min_shingle)

        # make it
        self._memomask = [] # stores the random 32 bit sequences for each hash function
        self._num_docs = 0
        self._most_recent_insert = 0
        self._init_hash_masks(self._n)
        self._cache = [defaultdict(list) for i in range(self._b)]
        
        #logging.debug('memomask: %s',self._memomask)


    def _init_hash_masks(self,num_hash):
        """
        This initializes the instance variable _memomask which is a list of the 
        random 32 bits associated with each hash function
        """
        for i in range(num_hash):
            random.seed(i)
            self._memomask.append(int(random.getrandbits(32)))
        
    def _get_shingle_vec(self, doc):
        """
        Takes a sequence of tokenized words and maps each shingle to a unique id.
        These unique ids, are then added to the shingle_vec object which is just a sparse
        vector implemented as a dict with v[id]=1 when a shingle id is present
        """
        #logging.debug('entering with len(doc)=%d', len(doc))
        v = {}
        for n in range(self._min_shingle,self._max_shingle):
            #doc.insert(0,'<start>')
            for j in range(len(doc) - n + 1):
                s = doc[j:j+n]
#                 if not self._shingles.has_key(tuple(s)):
#                     self._shingles[tuple(s)] = hash(tuple(s))%self._sbucket_size
#                     self._counter += 1
#                 v[self._shingles[tuple(s)]] = 1
                v[hash(tuple(s))%self._sbucket_size] = 1
                #logging.debug('docs: %s & id %d',str(s), self._shingles[tuple(s)])
        return v
    
    def _get_sig(self,shingle_vec,num_perms):
        """
        recoded version of _get_sig
        """
        sig = [self._sbucket_size]*num_perms
        keys = sorted(shingle_vec.keys())
        for r in keys:
            #logging.debug('r=%d', r)
            h = np.array([hash((r,mask)) % self._sbucket_size for mask in self._memomask])
            #logging.debug('h=%s',h)
            for i in range(num_perms):
                if (h[i] < sig[i]):
                    sig[i] = h[i]
            #logging.debug('mhash=%s',sig)
        return sig

    def _get_lsh(self,sig,b,r):
        """
        Takes an n-dimensional minhash signature and computes b hashes for each of
        b bands of r rows in the signature.  These hashes can take on any value that
        can be stored in the 32bit integer.
        """
        lsh = []
        for i,band in enumerate(range(b)):
            lsh.append(hash(tuple(sig[i*r:i*r+r])))
        #logging.debug('hashed signature: %s\n[get_lsh]\tto bins: %s',sig,lsh)
        return lsh
    
    def _get_lsh_from_doc(self, doc):
        """
        given an iterable of hashable items, returns a list of bucket ids
        """
        #logging.debug('got tokenized doc: len(doc)=%d %s', len(doc), doc)
        shingle_vec = self._get_shingle_vec(doc)
        #logging.debug('got shingle_vec: len(shingle_vec)=%d %s', len(shingle_vec), shingle_vec)
        sig = self._get_sig(shingle_vec,self._n) # n-dimensional min-hash signiture
        #logging.debug('got minhash sig: len(sig)=%d %s', len(sig), sig)
        lsh = self._get_lsh(sig,self._b,self._r) # r-dimensional list of bucket ids
        return lsh

    def _insert_lsh(self,lsh,doc_id):
        """
        Given an LSH vector of bucket indices, this method inserts the current doc
        id in the corresponding bucket for each of the _b tables
        """
        #dup_buckets = []
        self._num_docs += 1
        for i,band_bucket in enumerate(lsh):
            if doc_id not in self._cache[i][band_bucket]:
                #dup_buckets.append(self._cache[i][band_bucket])
                self._cache[i][band_bucket].append(doc_id)
        #return dup_buckets

    @classmethod
    def prepare_dup_buckets(cls, buckets, id=None):
#         logging.debug('buckets: %s', buckets)
        all = list(set(reduce(list.__add__, buckets, [])))
        #if id:
        #    all.remove(id)
        #all.remove(id)
        return all

    # public methods

    def get_dup_buckets(self, doc):
        """
        Returns a list of buckets (which are themselves lists) that contain the ids
        of any matching documents.  If the cache was built in chronological order
        then buckets are also in chronological order
        """
        if (not doc):
            print '[process_doc]\tfound empty doc, skipping'
            return
        lsh = self._get_lsh_from_doc(doc)
        dups = []
        for i,band_bucket in enumerate(lsh):
            dups.append(self._cache[i][band_bucket])
        return dups

    def get_dups(self, doc):
        return self.prepare_dup_buckets(self.get_dup_buckets(doc))

    def insert(self, doc, id):
        lsh = self._get_lsh_from_doc(doc)
        #logging.debug('id: %d lsh: %s', id, lsh)
        self._insert_lsh(lsh, id)
        #logging.debug('dup_buckets: %s', dup_buckets)
        #return self.prepare_dup_buckets(dup_buckets, id=id)

    #def insert_batch(self, doc_tuples):
    #    """Batch method for adding db docs to cache"""
    #    print '[add_docs]\tentering with len(docs)=%d' % (len(docs))
    #    for i, doc_tuple in enumerate(doc_tupless):
    #        if (i % 100 == 0):
    #            print '\r[add_docs]\tprocessed %d / %d docs:' % (i,len(docs)),
    #        dup_buckets[i] = self.insert(*doc_tuple)
    #    return dup_buckets

    def num_docs(self):
        return self._num_docs

    def most_recent_insert(self):
        return self._most_recent_insert

    #def num_shingles(self):
    #    return self._counter
    


class KNNCache:
    def __init__(self, lsh=LSHCache(), k=1000):
        self._lsh = lsh
        self._k = k
        self._sampledPnt = []
        self._id2idx = {}
        self._pntCount = 0
        
    def insert(self, doc, id):
        self._lsh.insert(doc,id)
        if len(self._sampledPnt) < 5*self._k:
            self._id2idx[id] = len(self._sampledPnt)
            self._sampledPnt.append(id)
        else:
            if random.random()<1.*self._pntCount/(self._pntCount+1):
                r = random.randint(0,5*self._k-1)
                del self._id2idx[self._sampledPnt[r]]
                self._id2idx[id] = r
                self._sampledPnt[r] = id
        self._pntCount += 1
            
        
    def getKnn(self, doc):
        tmp = self._lsh.get_dups(doc)
#          print 'get elems in range', len(tmp), tmp
        if len(tmp) < self._k:
            tmp.extend(random.sample(self._sampledPnt,self._k-len(tmp)))
        elif len(tmp) > self._k:
            tmp=random.sample(tmp,self._k)
#         print tmp
#         tmp.extend(random.sample(self._sampledPnt,20))
        return tmp