#coding:utf-8
import numpy as np
import sys
import math
import random
sys.path.append('../prepare-for-model/')
from ReadWholeGraph import ReadWholeGraph
tripleTotal = 1
headTotal = 1
midTotal = 1
tailTotal = 1
entityTotal = 1
rwg = ReadWholeGraph('../data-prepared/graph.node_mp_small1', '../data-prepared/graph.edge_mp_small1')
data, author2paperId, paperId2venueId, paperId2authorId, authorId2paperId, _, author2id = rwg.ReadGraph()
with open('../data/googlescholar.8area.author.label.txt') as f:
    label_author = {}
    author_label = {}
    for line in f:
        line = line.strip().split(' ')
        a, l = line[0], int(line[1])
        if a not in author2id:
            continue
        author_label[a] = l
        if l not in label_author:
            label_author[l] = []
        label_author[l].append(a)
with open('../data/googlescholar.8area.venue.label.txt') as f:
    label_venue = {}
    venue_label = {}
    for line in f:
        line = line.strip().split(' ')
        v, l = line[0], int(line[1])
        venue_label[v] = l
        if l not in label_venue:
            label_venue[l] = []
        label_venue[l].append(v)
def min(a, b):
	if a > b:
		return b
	return a
#get global values: tripleTotal, entityTotal, tagTotal

#load triples from file
class init():
    def __init__(self):
        pass
    def getGlobalValues(self):
        return tripleTotal, entityTotal, headTotal, midTotal, tailTotal
    def getTriples(self, path):
        headList = []
        midList = []
        tailList = []
        headSet = {}
        midSet = {}
        tailSet = {}
        headSetList = set()
        midSetList = set()
        tailSetList = set()
        f = open(path, "r")
        content = f.readline()
        global tripleTotal, entityTotal, headTotal, midTotal, tagTotal
        tripleTotal, entityTotal, headTotal, midTotal, tailTotal = [int(i) for i in content.strip().split()]
        for content in f:
            values = content.strip().split()
            values = [(int)(i) for i in values]
            headList.append(values[0])
            midList.append(values[1])
            tailList.append(values[2])
            headSetList.add(values[0]) 
            midSetList.add(values[1])
            tailSetList.add(values[2])
            if str(values[0]) +' '+ str(values[1]) not in headSet: 
                headSet[str(values[0]) +' '+ str(values[1])] = [values[2]]
            else:
                headSet[str(values[0]) +' '+ str(values[1])].append(values[2])
            if str(values[0]) +' '+ str(values[2]) not in midSet: 
                midSet[str(values[0]) + ' ' + str(values[2])] = [values[1]]
            else:
                midSet[str(values[0]) + ' ' + str(values[2])].append(values[1])
            if str(values[1]) +' '+ str(values[2]) not in tailSet: 
                tailSet[str(values[1]) + ' ' + str(values[2])] = [values[0]]
            else:
                tailSet[str(values[1]) + ' ' + str(values[2])].append(values[0])
        f.close()
        return headList, midList, tailList, headSet, midSet, tailSet, list(headSetList), list(midSetList), list(tailSetList)
#generate transNet training batches
    def batch_iter(self, headList, midList, tailList, headSet, midSet, tailSet, headSetList, midSetList, tailSetList, batch_size, isTest = False, precent = 0.7):
        data_size = len(headList)
        entity_size = entityTotal
        # Shuffle the data at each epoch
#    if not isTest:
#        shuffle_indices = np.random.permutation(np.arange(data_size))
#    else:
        shuffle_indices = np.arange(data_size)
        start_index = 0
        batch_id = 0
        if not isTest:
            data_size = int(precent * data_size)
        else:
            start_index = int(precent * data_size)
        end_index = min(start_index+batch_size, data_size)
        while start_index < data_size:
            pos_h = []
            pos_m = []
            pos_t = []
            neg_h = []
            neg_m = []
            neg_t = []
            for i in range(start_index, end_index):
                cur_h = headList[shuffle_indices[i]]
                cur_t = tailList[shuffle_indices[i]]
                cur_m = midList[shuffle_indices[i]]
                #replace head
                pos_h.append(cur_h)
                pos_t.append(cur_t)
                pos_m.append(cur_m)
                a_name = data[cur_h].getValue()
                label = author_label[a_name]
                authorset = label_author[label]
                rand = random.randint(0, len(authorset)-1)
                rand_h = author2id[authorset[rand]]
                while(rand_h in tailSet[str(cur_m) + ' ' + str(cur_t)]):
                    rand = random.randint(0, len(authorset)-1)
                    rand_h = author2id[authorset[rand]]
                neg_h.append(rand_h)
                neg_t.append(cur_t)
                neg_m.append(cur_m)
                if isTest:
                    continue
                #repalce tail
                pos_h.append(cur_h)
                pos_t.append(cur_t)
                pos_m.append(cur_m)
                
                a_name = data[cur_t].getValue()
                label = author_label[a_name]
                authorset = label_author[label]
                rand = random.randint(0, len(authorset)-1)
                rand_t = author2id[authorset[rand]]

                while(rand_t in headSet[str(cur_h) + ' ' + str(cur_m)]):
                    rand = random.randint(0, len(authorset)-1)
                    rand_t = author2id[authorset[rand]]
                neg_h.append(cur_h)
                neg_t.append(rand_t)
                neg_m.append(cur_m)
                #replace relation
                pos_h.append(cur_h)
                pos_t.append(cur_t)
                pos_m.append(cur_m)
                rand = random.randint(0, len(midSetList)-1)
                rand_m = midSetList[rand]
                while(rand_m in midSet[str(cur_h) + ' ' + str(cur_t)]):
                    rand = random.randint(0, len(midSetList)-1)
                    rand_m = midSetList[rand]
                neg_h.append(cur_h)
                neg_t.append(cur_t)
                neg_m.append(rand_m)
            batch_id += 1
            yield np.array(pos_h, dtype = np.int32), np.array(pos_t, dtype = np.int32), np.array(pos_m, dtype = np.int32),\
            np.array(neg_h, dtype = np.int32), np.array(neg_t, dtype = np.int32), np.array(neg_m,  dtype = np.int32)
            start_index = end_index
            end_index = min(start_index+batch_size, data_size)
    def batch_autoencoder(self, ae_list, batch_size):
        data_size = len(ae_list)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        start_index = 0
        end_index = min(start_index + batch_size, data_size)
        batch_id = 0
        while start_index < data_size:
            vecs = []
            for i in range(start_index, end_index):
                vec_index = ae_list[shuffle_indices[i]]
                vecs.append(vec_index)
            batch_id += 1
            yield np.array(vecs)
            start_index = end_index
            end_index = min(start_index + batch_size, data_size)

