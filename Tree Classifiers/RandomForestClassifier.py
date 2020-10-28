from DecisionTreeClassifier import DecisionTreeClassifier
import time
import numpy as np
import pandas as pd 

class RandomForestClassifier :

    def __init__(self, d = None, n_estimators = 100, min_samples_split = 2, min_samples_leaf = 1, max_depth = None, alpha = 0.0, max_features = 'auto', bootstrap = True, max_samples = None) :
        # Init method to set initial attributes
        # n_estimators : number of trees in the forest
        # min_samples_split : minimum number if node before it's considered by default as leaf
        # alpha : parameter for pruning
        # d : default value for target
        
        self.n_estimators = n_estimators
        
        assert(isinstance(max_features, (int, float)) or max_features in ['auto', 'sqrt', 'log2', None])
            
        self.max_features = max_features
        self.max_samples = max_samples
        self.num_classes = 2
        self.estimators_ = [DecisionTreeClassifier(min_samples_split, min_samples_leaf, d, max_depth, alpha) for _ in range(self.n_estimators)]
        
    def _get_num_samples(self, total_samples) :
    
        if isinstance(self.max_samples, int) :
            return self.max_samples
        elif isinstance(self.max_samples, float) :
            return int(self.max_samples * total_samples)
        else :
            return total_samples  
            
    def _get_num_features(self, total_features) :
    
        if isinstance(self.max_features, int) :
            return self.max_features
            
        elif isinstance(self.max_features, float) :
            return int(self.max_features * total_features)
            
        elif self.max_features == 'auto' or self.max_features == 'sqrt':
            return int(np.sqrt(self.total_features))
            
        elif self.max_features == 'log2' :
            return int(np.log2(self.total_features))
            
        else :
            return total_samples 
    
    def predict_proba(self, X) :
        # method to make prediction on records, X being a pandas dataframe containing the data
        
        results = np.zeros((self.n_estimators, X.shape[0]), dtype = np.int8)
        
        for index, tree in enumerate(self.estimators_) :
            results[index, :] = tree.predict(X)
        
        preds = np.sum(results, axis = 0) / self.n_estimators 
        
        return preds
      
    def predict(self, X) :
        
        preds = self.predict_proba(X)
        
        return (preds >= 0.5).astype(int)
    
    
    def fit(self, target, data = None, data_path = None, verbose = None) :
        # method that handles the logic of building the tree from the data given to it
        
        assert(data is not None or data_path is not None)
        if data is not None :
            self.data = data
        else :
            self._read_file(path)
            
        start_time = time.time()
        
        num_samples = self._get_num_samples(data.shape[0])
        num_features = self._get_num_features(data.shape[1])
        print(num_samples)
        print(num_features)
        
        for index, tree in enumerate(self.estimators_, start = 1) :
            # add code to take certain number of features
            
            sample_rows = data.sample(num_samples)
            cur_tree_data = pd.concat([sample_rows.drop(target, axis=1).sample(num_features, axis='columns'), sample_rows[target]], axis=1)
            
            tree.fit(data = cur_tree_data, target = target, verbose = False)
            
            if verbose :
                if isinstance(verbose, int) and index % verbose == 0 :
                    print("Estimator {}/{}".format(index, self.n_estimators))
        
        for tree in self.estimators_ :
            print(tree)
        
        if verbose :
            print("Finished fitting forest in --- %s seconds ---\n" % (time.time() - start_time))