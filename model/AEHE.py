# coding: utf-8
#带高阶信息版本  最终版 2018-5-24 09:03:17
import sys
import random
sys.path.append('/home/fsh/code/shichuan/AHIN/Aminer/data-prepared')
#APA
import tensorflow as tf
from init5 import *
from classification_model import *
from graph2 import *
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
import time
import numpy
graph_data = Graph('../data-prepared/author_instance1')
adj_matrix = graph_data.adj_matrix
print('adj_matrix:', adj_matrix.todense())
length = adj_matrix.shape[1]
ae_d = 5
precent = float(sys.argv[1])
print('ae_d:', ae_d)
print('precent:', precent)
def precisionK(y_true, y_pred, k):
    sort_id = np.argsort(y_pred)[::-1]
    y = np.array(y_true)[sort_id[:k]]
    z = 0.0
    for s in y:
        if s == 1:
            z += 1.0
    return z / k
    
class Config(object):
    def __init__(self):
        self.learning_rate = 0.001
        self.training_epochs = 30
        self.batch_size = 128
        self.gamma = 1
        self.delta = 1
        self.alpha = 0.1
        self.l2_lambda = 0.001
        self.keep_prob = 0.8
        self.step_num = 3
        self.hidden_num = 12
        self.elem_num = 12
        self.author_layers = []
        self.paper_layers = []
        self.entityTotal = 0
        self.no_weight = False
        self.warm_up_ae_epochs = 10
        self.pos_br = 10.0

class OutlierNet(object):
    def __init__(self, config):
        author_layers = config.author_layers
        paper_layers = config.paper_layers
        author_layers_length = len(author_layers)
        paper_layers_length = len(paper_layers)
        author_input_size = author_layers[0]
        paper_input_size = paper_layers[0]
        l2_lambda = config.l2_lambda
        keep_prob = config.keep_prob
        gamma = config.gamma
        delta = config.delta
        alpha = config.alpha
        step_num = config.step_num
        hidden_num = config.hidden_num
        elem_num = config.elem_num
        pos_br = config.pos_br

        self.pos_h = tf.placeholder(tf.float32, [None, author_input_size], name = 'pos_h')
        self.pos_m = tf.placeholder(tf.float32, [None, paper_input_size], name = 'pos_m')
        self.pos_t = tf.placeholder(tf.float32, [None, author_input_size], name = 'pos_t')
        self.neg_h = tf.placeholder(tf.float32, [None, author_input_size], name = 'neg_h')
        self.neg_m = tf.placeholder(tf.float32, [None, paper_input_size], name = 'neg_m')
        self.neg_t = tf.placeholder(tf.float32, [None, author_input_size], name = 'neg_t')
        
        self.batch_size = tf.placeholder(tf.int32, [None], name = 'batch_size')
        self.pos_h_ids = tf.sparse_placeholder(tf.float32)
        self.pos_t_ids = tf.sparse_placeholder(tf.float32)
        self.neg_h_ids = tf.sparse_placeholder(tf.float32)
        self.neg_t_ids = tf.sparse_placeholder(tf.float32)

#***********************************************************************************************
        with tf.name_scope("ae"):
            cur_seed = random.getrandbits(32)
            encode = tf.get_variable(name = "encode", shape = [length, ae_d], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
            encode_b = tf.get_variable(name="encode_b", initializer = tf.zeros([ae_d]))
            decode = tf.get_variable(name = "decode", shape = [ae_d, length], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
            decode_b = tf.get_variable(name="decode_b", initializer = tf.zeros([length]))

            self.pos_h_ids1 = tf.sparse_tensor_to_dense(self.pos_h_ids, validate_indices = False)
            self.pos_t_ids1 = tf.sparse_tensor_to_dense(self.pos_t_ids, validate_indices = False)
            self.neg_h_ids1 = tf.sparse_tensor_to_dense(self.neg_h_ids, validate_indices = False)
            self.neg_t_ids1 = tf.sparse_tensor_to_dense(self.neg_t_ids, validate_indices = False)

            encode_pos_h  = tf.nn.relu(tf.sparse_matmul(self.pos_h_ids1, encode, a_is_sparse = True) + encode_b)
            decode_pos_h  = tf.nn.relu(tf.sparse_matmul(encode_pos_h, decode) + decode_b)
            encode_pos_t  = tf.nn.relu(tf.sparse_matmul(self.pos_t_ids1, encode, a_is_sparse = True) + encode_b)
            decode_pos_t  = tf.nn.relu(tf.sparse_matmul(encode_pos_t, decode) + decode_b)
            encode_neg_h  = tf.nn.relu(tf.sparse_matmul(self.neg_h_ids1, encode, a_is_sparse = True) + encode_b)
            decode_neg_h  = tf.nn.relu(tf.sparse_matmul(encode_neg_h, decode) + decode_b)
            encode_neg_t  = tf.nn.relu(tf.sparse_matmul(self.neg_t_ids1, encode, a_is_sparse = True) + encode_b)
            decode_neg_t  = tf.nn.relu(tf.sparse_matmul(encode_neg_t, decode) + decode_b)

            self.ae_loss = 0.0

            self.ae_loss_init = tf.reduce_sum(abs(tf.multiply(self.pos_h_ids1 - decode_pos_h,  tf.multiply(pos_br, tf.sign(self.pos_h_ids1)))))
            self.ae_loss = self.ae_loss_init + tf.reduce_sum(abs(tf.multiply(self.pos_t_ids1 - decode_pos_t,  tf.multiply(pos_br, tf.sign(self.pos_t_ids1)))))
            self.ae_loss += tf.reduce_sum(abs(tf.multiply(self.neg_h_ids1 - decode_neg_h,  tf.multiply(pos_br, tf.sign(self.neg_h_ids1)))))
            self.ae_loss += tf.reduce_sum(abs(tf.multiply(self.neg_t_ids1 - decode_neg_t,  tf.multiply(pos_br, tf.sign(self.neg_t_ids1)))))
        with tf.name_scope("mlp"):
            self.relation_W_a = []
            self.relation_b_a = []
            self.relation_W_p = []
            self.relation_b_p = []
            self.pos_a1_hidden = []
            self.pos_a2_hidden = []
            self.pos_p_hidden = []
            self.pos_r_hidden_test = []
            self.neg_a1_hidden = []
            self.neg_a2_hidden = []
            self.neg_p_hidden = []
            self.mlp_l2_loss = 0.0

            # author mlp
            for i in range(author_layers_length - 1):
                cur_seed = random.getrandbits(32)
                self.relation_W_a.append(tf.get_variable(name = "relation_W_a"+str(i), shape = [author_layers[i], author_layers[i+1]], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed)))
                self.relation_b_a.append(tf.get_variable(name="relation_b_a"+str(i), initializer = tf.zeros([author_layers[i+1]])))
                self.mlp_l2_loss += tf.nn.l2_loss(self.relation_W_a[i])+tf.nn.l2_loss(self.relation_b_a[i])
                # feed pos_h into mlp
                if i == 0:
                    layers_pos_a1  = tf.nn.relu(tf.matmul(self.pos_h, self.relation_W_a[i]) + self.relation_b_a[i])
                    layers_neg_a1  = tf.nn.relu(tf.matmul(self.neg_h, self.relation_W_a[i]) + self.relation_b_a[i])
                elif i == author_layers_length - 2:
                    layers_pos_a1 = tf.matmul(self.pos_a1_hidden[i-1], self.relation_W_a[i]) + self.relation_b_a[i]
                    layers_neg_a1 = tf.matmul(self.neg_a1_hidden[i-1], self.relation_W_a[i]) + self.relation_b_a[i]
                else:
                    layers_pos_a1 = tf.nn.relu(tf.matmul(self.pos_a1_hidden[i-1], self.relation_W_a[i]) + self.relation_b_a[i])
                    layers_neg_a1 = tf.nn.relu(tf.matmul(self.neg_a1_hidden[i-1], self.relation_W_a[i]) + self.relation_b_a[i])
                if i == (author_layers_length-3)/2:
                    cur_seed = random.getrandbits(32)
                    self.pos_a1_rep = tf.nn.dropout(layers_pos_a1, keep_prob, seed=cur_seed)
                    cur_seed = random.getrandbits(32)
                    self.neg_a1_rep = tf.nn.dropout(layers_neg_a1, keep_prob, seed=cur_seed)
                    self.pos_a1_hidden.append(self.pos_a1_rep)
                    self.neg_a1_hidden.append(self.neg_a1_rep)
                else:
                    self.pos_a1_hidden.append(layers_pos_a1)
                    self.neg_a1_hidden.append(layers_neg_a1)                  
            for i in range(author_layers_length - 1):
                cur_seed = random.getrandbits(32)
                
                # feed pos_h into mlp
                if i == 0:
                    layers_pos_a2  = tf.nn.relu(tf.matmul(self.pos_t, self.relation_W_a[i]) + self.relation_b_a[i])
                    layers_neg_a2  = tf.nn.relu(tf.matmul(self.neg_t, self.relation_W_a[i])+ self.relation_b_a[i])
                elif i == author_layers_length - 2:
                    layers_pos_a2 = tf.matmul(self.pos_a2_hidden[i-1], self.relation_W_a[i]) + self.relation_b_a[i]
                    layers_neg_a2 = tf.matmul(self.neg_a2_hidden[i-1], self.relation_W_a[i]) + self.relation_b_a[i]
                else:
                    layers_pos_a2 = tf.nn.relu(tf.matmul(self.pos_a2_hidden[i-1], self.relation_W_a[i])+self.relation_b_a[i])
                    layers_neg_a2 = tf.nn.relu(tf.matmul(self.neg_a2_hidden[i-1], self.relation_W_a[i])+self.relation_b_a[i])
                if i == (author_layers_length-3)/2:
                    cur_seed = random.getrandbits(32)
                    self.pos_a2_rep = tf.nn.dropout(layers_pos_a2, keep_prob, seed=cur_seed)
                    cur_seed = random.getrandbits(32)
                    self.neg_a2_rep = tf.nn.dropout(layers_neg_a2, keep_prob, seed=cur_seed)
                    self.pos_a2_hidden.append(self.pos_a2_rep)
                    self.neg_a2_hidden.append(self.neg_a2_rep)
                else:
                    self.pos_a2_hidden.append(layers_pos_a2)
                    self.neg_a2_hidden.append(layers_neg_a2)
            for i in range(paper_layers_length - 1):
                cur_seed = random.getrandbits(32)
                self.relation_W_p.append(tf.get_variable(name = "relation_W_p"+str(i), shape = [paper_layers[i], paper_layers[i+1]], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed)))
                self.relation_b_p.append(tf.get_variable(name="relation_b_p"+str(i), initializer = tf.zeros([paper_layers[i+1]])))
                self.mlp_l2_loss += tf.nn.l2_loss(self.relation_W_p[i])+tf.nn.l2_loss(self.relation_b_p[i])      
                # feed pos_m into mlp
                if i == 0:
                    layers_pos_p  = tf.nn.relu(tf.matmul(self.pos_m, self.relation_W_p[i]) + self.relation_b_p[i])
                    layers_neg_p  = tf.nn.relu(tf.matmul(self.neg_m, self.relation_W_p[i]) + self.relation_b_p[i])
                elif i == paper_layers_length - 2:
                    layers_pos_p = tf.matmul(self.pos_p_hidden[i-1], self.relation_W_p[i])+ self.relation_b_p[i]
                    layers_neg_p = tf.matmul(self.neg_p_hidden[i-1], self.relation_W_p[i]) + self.relation_b_p[i]
                else:
                    layers_pos_p = tf.nn.relu(tf.matmul(self.pos_p_hidden[i-1], self.relation_W_p[i]) + self.relation_b_p[i])
                    layers_neg_p = tf.nn.relu(tf.matmul(self.neg_p_hidden[i-1], self.relation_W_p[i]) + self.relation_b_p[i])
  
                if i == (paper_layers_length-3)/2:
                    cur_seed = random.getrandbits(32)
                    self.pos_p_rep = tf.nn.dropout(layers_pos_p, keep_prob, seed=cur_seed)
                    cur_seed = random.getrandbits(32)
                    self.neg_p_rep = tf.nn.dropout(layers_neg_p, keep_prob, seed=cur_seed)
                    self.pos_p_hidden.append(self.pos_p_rep)
                    self.neg_p_hidden.append(self.neg_p_rep)
                else:
                    self.pos_p_hidden.append(layers_pos_p)
                    self.neg_p_hidden.append(layers_neg_p)
            self.a1_embed = self.pos_a1_hidden[-1]
            self.p_embed = self.pos_p_hidden[-1]
            self.a2_embed = self.pos_a2_hidden[-1]
        with tf.name_scope('concate'):
            self.encode_pos_h = encode_pos_h
            con_pos_h = tf.concat([self.pos_a1_hidden[-1], encode_pos_h], 1)
            con_pos_t = tf.concat([self.pos_a2_hidden[-1], encode_pos_t], 1)
            con_neg_h = tf.concat([self.neg_a1_hidden[-1], encode_neg_h], 1)
            con_neg_t = tf.concat([self.neg_a2_hidden[-1], encode_neg_t], 1)
            self.con_pos_h = con_pos_h

            '''
        with tf.name_scope("node_lookup"):
            cur_seed = random.getrandbits(32)
            embeddings = tf.get_variable(name = "embeddings", shape = [entityTotal, author_layers[-1]], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
            pos_h_e = tf.nn.embedding_lookup(embeddings, self.pos_h_ids)
            pos_m_e = tf.nn.embedding_lookup(embeddings, self.pos_m_ids)
            pos_t_e = tf.nn.embedding_lookup(embeddings, self.pos_t_ids)
            neg_h_e = tf.nn.embedding_lookup(embeddings, self.neg_h_ids)
            neg_m_e = tf.nn.embedding_lookup(embeddings, self.neg_m_ids)
            neg_t_e = tf.nn.embedding_lookup(embeddings, self.neg_t_ids)
        with tf.name_scope("max"):
            new_pos_h = tf.maximum(pos_h_e, self.pos_a1_hidden[-1])
            new_pos_m = tf.maximum(pos_m_e, self.pos_p_hidden[-1])
            new_pos_t = tf.maximum(pos_t_e, self.pos_a2_hidden[-1])
            new_neg_h = tf.maximum(neg_h_e, self.neg_a1_hidden[-1])
            new_neg_m = tf.maximum(neg_m_e, self.neg_p_hidden[-1])
            new_neg_t = tf.maximum(neg_t_e, self.neg_a2_hidden[-1])

            self.update_pos_h = tf.scatter_update(embeddings, self.pos_h_ids, new_pos_h)
            self.update_pos_m = tf.scatter_update(embeddings, self.pos_m_ids, new_pos_m)
            self.update_pos_t = tf.scatter_update(embeddings, self.pos_t_ids, new_pos_t)
            self.update_neg_h = tf.scatter_update(embeddings, self.neg_h_ids, new_neg_h)
            self.update_neg_m = tf.scatter_update(embeddings, self.neg_m_ids, new_neg_m)
            self.update_neg_t = tf.scatter_update(embeddings, self.neg_t_ids, new_neg_t)
            self.update = [self.update_pos_h, self.update_pos_m, self.update_pos_t,
            self.update_neg_h, self.update_neg_m, self.update_neg_t]
            '''
        with tf.variable_scope("ape", reuse = None) as vs:
        #    vs.reuse_variables()

            inputs_pos = tf.concat([tf.reduce_sum(tf.multiply(con_pos_h, self.pos_p_hidden[-1]),axis = 1, keep_dims = True), tf.reduce_sum(tf.multiply(con_pos_h, con_pos_t), axis = 1, keep_dims = True), tf.reduce_sum(tf.multiply(self.pos_p_hidden[-1], con_pos_t),axis = 1,  keep_dims = True)], 1)
            inputs_neg = tf.concat([tf.reduce_sum(tf.multiply(con_neg_h, self.neg_p_hidden[-1]),axis = 1, keep_dims = True), tf.reduce_sum(tf.multiply(con_neg_h, con_neg_t), axis = 1, keep_dims = True), tf.reduce_sum(tf.multiply(self.neg_p_hidden[-1], con_neg_t),axis = 1,  keep_dims = True)], 1)
            self.inputs_pos = inputs_pos
            self.inputs_neg = inputs_neg

            if config.no_weight:
                merge_layers = tf.contrib.keras.layers.Dense(1, kernel_initializer = tf.ones_initializer(), trainable = False, name = 'merge_pos')
                merge_pos = merge_layers(inputs_pos)
                merge_pos = merge_layers(inputs_neg)
                b_pos = tf.get_variable(name="b_pos", initializer = tf.zeros(1))
                merge_pos_new = merge_pos + b_pos
                
                merge_neg = tf.contrib.keras.layers.Dense(1, kernel_initializer = tf.ones_initializer(), trainable = False, name = 'merge_pos')(inputs_pos)
                b_neg = tf.get_variable(name="b_pos", initializer = tf.zeros(1))
                merge_neg_new = merge_neg + b_neg
            else:
                print('inputs_pos', inputs_pos)
                merge_layers = tf.contrib.keras.layers.Dense(1, kernel_initializer = tf.ones_initializer(),kernel_constraint = 'NonNeg', trainable = True, name = 'merge_pos')
                self.merge_pos_new = merge_layers(inputs_pos)
                self.merge_neg_new = merge_layers(inputs_neg)
                print(self.merge_pos_new)
            outlier_loss_pos = tf.reduce_sum(tf.log(tf.clip_by_value(tf.sigmoid(tf.multiply(self.merge_pos_new, tf.ones_like(self.merge_pos_new))), 1e-8, 1.0 )))
            outlier_loss = outlier_loss_pos + tf.reduce_sum(tf.log(tf.clip_by_value(tf.sigmoid(tf.multiply(self.merge_neg_new, tf.multiply(-1.0, tf.ones_like(self.merge_neg_new)))), 1e-8, 1.0)))
            self.outlier_loss = tf.multiply(-1.0, outlier_loss)
        #    pos_h_test = tf.nn.embedding_lookup(embeddings, self.pos_h_ids)
        #    pos_m_test = tf.nn.embedding_lookup(embeddings, self.pos_m_ids)
        #    pos_t_test = tf.nn.embedding_lookup(embeddings, self.pos_t_ids)
            inputs_test = tf.concat([tf.reduce_sum(tf.multiply(con_pos_h, self.pos_p_hidden[-1]),axis = 1, keep_dims = True), tf.reduce_sum(tf.multiply(con_pos_h, con_pos_t), axis = 1, keep_dims = True), tf.reduce_sum(tf.multiply(self.pos_p_hidden[-1], con_pos_t),axis = 1,  keep_dims = True)], 1)
            self.outlier_score = merge_layers(inputs_test)
            self.loss = self.outlier_loss + alpha * self.ae_loss + l2_lambda * self.mlp_l2_loss
 
with open('../data-prepared/outputfeatures_mp1') as f:
    feature_dic = {}
    n = 0
    for line in f:
        line = line.strip().split('\t')[1].split(' ')
        ll = []
        for l in line:
            try:
                l = float(l)
            except:
                l =0.0
            ll.append(l)
        feature_dic[n] = ll
        n += 1
with open('../data-prepared/map1') as f:
    map_id = {}
    for line in f:
        line = line.strip().split()
        map_id[line[0]] = line[1]
import time
if __name__=="__main__":
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4 #设置设置tf占用显存的比例
    config.gpu_options.allow_growth=True  #设置tf模式为按需赠长模式
    session = tf.Session(config=config)

    def match_fea(input_idx):
        return_fea = []
        for n in input_idx:
            return_fea.append(feature_dic[n])
        return np.array(return_fea, dtype = 'float')
    def match_id(input_idx):
        t1 = time.time()
        return_fea = []
        for n in input_idx:
            b = adj_matrix[int(map_id[str(n)])].todense().reshape((-1))
            return_fea.append(b.tolist())
        z = np.array(return_fea).reshape((-1, length))
        re_matix = sparse.dok_matrix(z)
        t2 = time.time()
        return re_matix


    config = Config()
    config.author_layers = [58,  50, 40, 30]
    config.paper_layers = [108,  90, 70, 35]
    config.step_num = 3
    config.hidden_num = 12
    config.elem_num = 12
    init1 = init()
    init2 = init()
    headList, midList, tailList, headSet, midSet, tailSet, \
    headSetList, midSetList, tailSetList = init1.getTriples('../data-prepared/sampleinstance_mp1')
    tripleTotal, entityTotal, headTotal, midTotal, tailTotal = init1.getGlobalValues()
#    print('1',tripleTotal)
#    headList_t, midList_t, tailList_t, headSet_t, midSet_t, tailSet_t, \
#    headSetList_t, midSetList_t, tailSetList_t = getTriples('../data-prepared/sampleinstance_mp_test')
#    print('2',tripleTotal)
    config.entityTotal = entityTotal
    print('alpha:', config.alpha)
    def train():
        cur_seed = random.getrandbits(32)
       # initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed)
        with tf.variable_scope("model", reuse=None):
            model = OutlierNet(config = config)
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            train_op = optimizer.minimize(model.loss)
            train_ae = optimizer.minimize(model.ae_loss_init)
        Train = True
        display_step = 5
        with tf.Session() as sess:
            if Train:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                for epoch in range(config.warm_up_ae_epochs):
                    sum_loss = 0.0
                    batches = init1.batch_autoencoder(headList + tailList, 256)
                    for batch in batches:
                        vecs = batch
                        tmp1 = match_id(vecs)
                        tf1 = tf.SparseTensorValue(np.array(list(tmp1.keys())), np.array(list(tmp1.values())), np.array(tmp1.shape))
                        _, c = sess.run([train_ae, model.ae_loss_init], feed_dict = {
                            model.pos_h_ids: tf1
                            })
                        sum_loss += c
                    print('sum_loss', sum_loss)
                fw = open('../data-prepared/log3'+ str(precent), 'w')

                for epoch in range(config.training_epochs):
                    avg_cost = 0
                    batches = init1.batch_iter(headList, midList,tailList, headSet, midSet,tailSet, headSetList, midSetList, tailSetList, config.batch_size, precent = precent)
                    total_batch = int(tripleTotal/config.batch_size)
                    for batch in batches:
                        t3 = time.time()
                        pos_h_1, pos_t_1, pos_m_1, neg_h_1, neg_t_1, neg_m_1 = batch
                        tmp1 = match_id(pos_h_1)
                        tmp2 = match_id(pos_t_1)
                        tmp3 = match_id(neg_h_1)
                        tmp4 = match_id(neg_t_1)
                        tf1 = tf.SparseTensorValue(np.array(list(tmp1.keys())), np.array(list(tmp1.values())), np.array(tmp1.shape))
                        tf2 = tf.SparseTensorValue(np.array(list(tmp2.keys())), np.array(list(tmp2.values())), np.array(tmp2.shape))
                        tf3 = tf.SparseTensorValue(np.array(list(tmp3.keys())), np.array(list(tmp3.values())), np.array(tmp3.shape))
                        tf4 = tf.SparseTensorValue(np.array(list(tmp4.keys())), np.array(list(tmp4.values())), np.array(tmp4.shape))

                        _, c = sess.run([train_op, model.loss], feed_dict={
                                model.pos_h: match_fea(pos_h_1),
                                model.pos_m: match_fea(pos_m_1),
                                model.pos_t: match_fea(pos_t_1),
                                model.neg_h: match_fea(neg_h_1),
                                model.neg_m: match_fea(neg_m_1),
                                model.neg_t: match_fea(neg_t_1),
                                model.pos_h_ids: tf1,
                                model.pos_t_ids: tf2,
                                model.neg_h_ids: tf3,
                                model.neg_t_ids: tf4,
                                model.batch_size: np.array(len(pos_h_1)).reshape((1,))}
                                )
                        t4 = time.time()
                        avg_cost += c/total_batch
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                    if epoch % display_step:
                        batches = init2.batch_iter(headList, midList, tailList, headSet, midSet, tailSet, headSetList, midSetList, tailSetList, batch_size = 128, isTest = True, precent = precent)
                        output_embedding = {}
                        n = 0
                        outlier_score = None
                        label = []
                        for batch in batches:
                            n += 1
                            pos_h_test, pos_t_test, pos_m_test, neg_h_test, neg_t_test, neg_m_test = batch
                            tmp1 = match_id(pos_h_test)
                            tmp2 = match_id(pos_t_test)
                            tmp3 = match_id(neg_h_test)
                            tmp4 = match_id(neg_t_test)
                            tf1 = tf.SparseTensorValue(np.array(list(tmp1.keys())), np.array(list(tmp1.values())), np.array(tmp1.shape))
                            tf2 = tf.SparseTensorValue(np.array(list(tmp2.keys())), np.array(list(tmp2.values())), np.array(tmp2.shape))
                            tf3 = tf.SparseTensorValue(np.array(list(tmp3.keys())), np.array(list(tmp3.values())), np.array(tmp3.shape))
                            tf4 = tf.SparseTensorValue(np.array(list(tmp4.keys())), np.array(list(tmp4.values())), np.array(tmp4.shape))

                            outlier_score_pos = sess.run([model.outlier_score],feed_dict = {
                                model.pos_h: match_fea(pos_h_test),
                                model.pos_m: match_fea(pos_m_test),
                                model.pos_t: match_fea(pos_t_test),
                                model.pos_h_ids: tf1,
                                model.pos_t_ids: tf2,
                                })
                            outlier_score_neg = sess.run([model.outlier_score],feed_dict = {
                                model.pos_h: match_fea(neg_h_test),
                                model.pos_m: match_fea(neg_m_test),
                                model.pos_t: match_fea(neg_t_test),
                                model.pos_h_ids: tf3,
                                model.pos_t_ids: tf4,
                                })
                            outlier_score_pos = outlier_score_pos[0]
                            outlier_score_neg = outlier_score_neg[0]
                            if outlier_score is None:
                                outlier_score = -1 * outlier_score_pos
                                outlier_score = np.concatenate((outlier_score, -1 * outlier_score_neg))
                            else:
                                outlier_score = np.concatenate((outlier_score, -1 * outlier_score_pos, -1 * outlier_score_neg))
                            label += [-1] * outlier_score_pos.shape[0]
                            label += [1] * outlier_score_neg.shape[0]
                        fw.write('AP:'+ str(average_precision_score(label, outlier_score.reshape((-1)))))
                        fw.write('\n')
                        fw.write('AUC:'+ str(roc_auc_score(label, outlier_score.reshape((-1)))))
                        fw.write('\n')
                        fw.write('precision@5:'+str(precisionK(label, outlier_score.reshape((-1)), 5)))
                        fw.write('\n')
                        fw.write('precision@10:'+str(precisionK(label, outlier_score.reshape((-1)), 10)))
                        fw.write('\n')
                        fw.write('precision@30:'+str(precisionK(label, outlier_score.reshape((-1)), 30)))
                        fw.write('\n')
                        fw.write('precision@50:'+str(precisionK(label, outlier_score.reshape((-1)), 50)))
                        fw.write('\n')
                        fw.write('precision@100:'+str(precisionK(label, outlier_score.reshape((-1)), 100)))
                        fw.write('\n')
                fw.close()
                '''
                batches = init2.batch_iter(headList, midList, tailList, headSet, midSet, tailSet, headSetList, midSetList, tailSetList, batch_size = 512, isTest = True, precent = 0)
                n = 0
                outlier_score = None
                for batch in batches:
                    n += 1
                    pos_h_test, pos_t_test, pos_m_test, neg_h_test, neg_t_test, neg_m_test = batch
                    tmp1 = match_id(pos_h_test)
                    tmp2 = match_id(pos_t_test)
                    tmp3 = match_id(neg_h_test)
                    tmp4 = match_id(neg_t_test)
                    tf1 = tf.SparseTensorValue(np.array(list(tmp1.keys())), np.array(list(tmp1.values())), np.array(tmp1.shape))
                    tf2 = tf.SparseTensorValue(np.array(list(tmp2.keys())), np.array(list(tmp2.values())), np.array(tmp2.shape))
                    tf3 = tf.SparseTensorValue(np.array(list(tmp3.keys())), np.array(list(tmp3.values())), np.array(tmp3.shape))
                    tf4 = tf.SparseTensorValue(np.array(list(tmp4.keys())), np.array(list(tmp4.values())), np.array(tmp4.shape))
                    outlier_score_pos = sess.run([model.outlier_score],feed_dict = {
                        model.pos_h_ids: pos_h_test,
                        model.pos_m_ids: pos_m_test,
                        model.pos_t_ids: pos_t_test,
                        model.pos_h_ids: tf1,
                        model.pos_t_ids: tf2,
                        })
                
                    outlier_score_pos = outlier_score_pos[0]
                    if outlier_score is None:
                        outlier_score = -1 * outlier_score_pos
                    else:
                        outlier_score = np.concatenate((outlier_score, -1 * outlier_score_pos))
                outlier_score = outlier_score.reshape((-1))
                sort = np.argsort(-outlier_score)
                f = open('../data-prepared/sort0515'+time.time(), 'w')
                for s in sort:
                    f.write(str(s))
                    f.write('\n')
                f.close()
                '''
    train()
