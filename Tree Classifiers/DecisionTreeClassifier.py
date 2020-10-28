from DecisionTreeNode import DecisionTreeNode
import time

import numpy as np
import pandas as pd
import json

from queue import Queue, PriorityQueue
from copy import deepcopy

from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
    
def printPQ(pq) :
    print("elements in queue\n", pq.queue, "***********\n")

  
class DecisionTreeClassifier :
    
    def __init__(self, min_samples_split = 2, min_samples_leaf = 1, d = None, max_depth = None, alpha = 0.0) :
        # Init method to set initial attributes
        # min_samples_split : minimum number if node before it's considered by default as leaf
        # alpha : parameter for pruning
        # d : default value for target
        
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.d = d
        self.max_depth = max_depth
    
    
    def _read_file(self, path) :
        # utility method that reads data from path if specified
        
        self.data = pd.read_csv(path)
    
    
    def _get_attributes(self, data) :
        # utility method to get attributes and their values from the data and store them
        # in a dictionary attribute
        
        self.attributes = {}
        for col in self.data.columns :
            if col != self.target :
                self.attributes[col] = sorted(list(set(data[col])))
         
        
    def _Gini(self, data):
        # utility method to calculate the gini value for a set of records
        
        n = len(data)
        if n == 0 :
            return 1
        
        count0 = len(data[data[self.target] == 0])
        count1 = n - count0
        
        return (1 - (count0 / n)**2 - (count1 / n)**2)
    
    
    def _gini_split(self, left, right) :
        # utility method to calculate the gini split value for a certain leaf and constraint
        # (already specified in the outer method), it's inputs are the left and right records 
        # of a constraint
        
        g_left = self._Gini(left)
        g_right = self._Gini(right)
        n = len(left) + len(right)
        gini_split = (len(left) / n) * g_left + (len(right) / n) * g_right
        
        return gini_split
    
    
    def _optimal_gini_split(self, node_data) :
        # utility method to get the optimal gini split for a certain node, which iterates through all the possible
        # nodes, calculate the gini split value for each one and then return the optimal feature and constraint
        
        mini = 2
        feature = ""
        opt_constr = 1e9
        
        for col in self.attributes.keys() :
            for constr in self.attributes[col][1:] :
                node_data_left = node_data[node_data[col] < constr]
                node_data_right = node_data[node_data[col] >= constr]
                
                gini_split = self._gini_split(node_data_left, node_data_right)
                
                if gini_split < mini :
                    mini = gini_split
                    feature = col
                    opt_constr = constr
                    
        return feature, opt_constr
    
    def _build_node(self, node, indices = None) :
        # utility function to build the tree one node at a time
        
#         print(indices)
        
        # get indices if given else it's a root node and gets all data
        if indices is not None :
            node_data = self.data.loc[indices]
        else :
            node_data = self.data
        
#         print(node_data)
        
        # if node has no data returns None
        if len(node_data) == 0 :
            return None
        
        # compute gini of node
        node.gini = self._Gini(node_data)
        
        # if data is homogenous (only has one class) then give it the majority class
        # and mark it as a leaf node
        if len(set(node_data[self.target])) == 1 :
            
            Class = list(node_data[self.target])[0]
            node.set_class(Class)
            node.node_type = "Leaf"
            
            return node
        
        # if node has fewer records that the minimum number then give it the default class 
        # and mark it as a leaf node
        if len(node_data[self.target]) < self.min_samples_split :
            
            node.set_class(self.d)
            node.node_type = "Leaf"
            
            return node
        
        # compute the optimal constraint and feature for the gini split
        feature, opt_constr = self._optimal_gini_split(node_data)
              
#         print(node)
        
        
        # get the indices of the left and right nodes according to the optimal 
        # constraint already found
        indices_left = node_data[node_data[feature] < opt_constr].index
        indices_right = node_data[node_data[feature] >= opt_constr].index
        
        # associate the appropriate class to the current node, which is the majority one 
        # or the default in case of equal numbers
        # while this is not necessary for all the nodes at this stage, it is especially
        # useful for the post-pruning part later on
        num0 = node_data[self.target].value_counts()[0]
        num1 = node_data[self.target].value_counts()[1]

        Class = 0 if num0 > num1 else 1
        if num0 == num1 :
            Class = self.d

        node.set_class(Class)
        
        # handle the case that one of the nodes has all the records and the other 
        # doesn't get any, in which case we shouldn't split and instead mark the node
        # as a leaf node
        if len(indices_left) == len(node_data) or len(indices_right) == len(node_data) :
            
            node.node_type = "Leaf"
            
            return node
        
        # gove the optimal feature and constraint attributes to the current node
        node.feature = feature
        node.constr = opt_constr
        
        # initialize the left and right nodes
        left = DecisionTreeNode(level = node.level + 1)
        right = DecisionTreeNode(level = node.level + 1)
        
        # recursively build the tree by building the left and right nodes
        # for the current node and return them
        node.left = self._build_node(left, indices_left)
        node.right = self._build_node(right, indices_right)
        
        # give the right node type to the current node
        if node.node_type != "Root" :
            node.node_type = "Intermediate"
        
        return node
    
    
    def _propagate(self, node, row) :
        # utility method that propagate a prediction through the nodes of the tree
        # according to the constraints, until it hits a leaf node which then returns its result
        
        if node.node_type == "Leaf" :
            return node.Class
        
        if row[node.feature] < node.constr :
            return self._propagate(node.left, row)
        else :
            return self._propagate(node.right, row)
    
    
    def _get_preds(self, X, root = None) :
        # utility method to make prediction on records, X being a pandas dataframe containing the data
        # the root to be used is specified depending on the usage (prediction, or pruning for example)
        
        if root is None :
            root = self.root
        
        y = X.apply(lambda x : self._propagate(root, x), axis=1)
        
        return y
    
    
    def predict(self, X) :
        # method to make prediction on records, X being a pandas dataframe containing the data
        
        return self._get_preds(X, self.root)
    
    
    def _get_leaves_number(self, root) :
        
        if root.node_type == "Leaf" :
            return 1
        
        return self._get_leaves_number(root.left) + self._get_leaves_number(root.right)
    
    
    def _generalization_error(self, root = None) :
        
        if root is None :
            root = self.root
            
        target_col = self.data[self.target]
        y = self._get_preds(self.data.drop([self.target], axis=1), root)
        
        train_error = (y == target_col).value_counts()[False]
        num_leaves = self._get_leaves_number(root)
        
        gen_error = (train_error + self.alpha * num_leaves) / len(target_col)
        
        return gen_error
    
    
    def _bfs_traversal(self, root = None) :
        # BFS traversal to get tree nodes in the correct order
        
        if root is None :
            root = self.root
        
        q = Queue()
        q.put((root, None))
        
        while not q.empty() :
            node, parent = q.get()
            
            if node == None :
                continue
            
            yield node, parent
            
            q.put((node.left, node.index))
            q.put((node.right, node.index))
            
    
    def _set_index(self, root = None) :
        
        if root is None :
            root = self.root
        
        index = 1
        for node, _ in self._bfs_traversal() :
            node.set_index(index)
            index += 1
    
    
    def _post_prune(self) :
        
        aux_root = deepcopy(self.root)
        vis = {}
        pq = PriorityQueue()
        
        while True :
            
            if not pq.empty() :
                prune_node = pq.get().item
                prune_node.node_type = "Leaf"
                prune_node.left = None
                prune_node.right = None

                err = self._generalization_error()
                new_err = self._generalization_error(aux_root)

                #print("old error : {}, new error : {}\n\n".format(err, new_err))

                if err > new_err :
                    self.root = deepcopy(aux_root)
                    modified = True
                else :
                    aux_root = deepcopy(self.root)
                    
            # BFS add potenial prune to pq
            
            for node, _ in self._bfs_traversal(aux_root) :

                if node.node_type == "Leaf" :
                    continue

                if node.left.node_type == "Leaf" and node.right.node_type == "Leaf" and node.index not in vis.keys() :
                    pq.put(PrioritizedItem(-node.level, node))
                    vis[node.index] = True
                                    
            #printPQ(pq)
            #print(self._bfs_traversal(aux_root))
            #print(vis, "\n")
            
            if pq.empty() :
                break
                
        self._set_index()
            
    
    def fit(self, target, data = None, data_path = None, verbose = False) :
        # method that handles the logic of building the tree from the data given to it
        
        assert(data is not None or data_path is not None)
        if data is not None :
            self.data = data
        else :
            self._read_file(path)
            
        self.target = target
        self._get_attributes(data)
        
        root = DecisionTreeNode(node_type = "Root", level = 0)
        
        start_time = time.time()
        self.root = self._build_node(root)
        if verbose :
            print("Fit tree in --- %s seconds ---\n" % (time.time() - start_time))
            
        self._set_index()
        
        if verbose :
            print("=========== Tree before pruning ============\n\n" + self.__repr__())
        start_time = time.time()
        self._post_prune()
        if verbose :
            print("Pruned tree in --- %s seconds ---\n" % (time.time() - start_time))
            print("=========== Tree after pruning ============\n\n" + self.__repr__())
        
    
    def __repr__(self) :
        # method for printing the decision tree
        
        def_str = "Decision Tree Classifier \n"
        
        attrts_str = "minimum number of samples to split node : {}, alpha : {}, default : {}, target variable : {}\n\n" \
            .format(self.min_samples_split, self.alpha, self.d, self.target)
        
        attr_str = "Attributes : " + json.dumps(self.attributes) + "\n\n"
        
        tree_str = ""
        for node, parent in self._bfs_traversal() :
            tree_str += "{}; ".format(node.index)   
            if parent : 
                tree_str += "child of {}, ".format(parent)
            tree_str += node.__repr__() + "\n************\n"
        
        return def_str + attrts_str + attr_str + tree_str