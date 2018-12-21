import scipy.io as sio
import numpy as np
import random
import copy
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix

class Graph(object):
    def __init__(self, file_path):
        self.st = 0
        self.is_epoch_end = False
        fin = open(file_path, "r")
        firstLine = fin.readline().strip().split()
        self.N = int(firstLine[0])
        self.E = int(firstLine[1])
        self.__is_epoch_end = False
        self.adj_matrix = dok_matrix((self.N, self.N), np.int_)
        count = 0
        for line in fin.readlines():
            line = line.strip().split()
            x = int(line[0])
            y = int(line[1])
            if self.adj_matrix[x, y] == 1:
                continue
            self.adj_matrix[x, y] += 1
            self.adj_matrix[y, x] += 1
            count += 1
        fin.close()
        self.adj_matrix = self.adj_matrix.tocsr()
        print("getData done")
   
