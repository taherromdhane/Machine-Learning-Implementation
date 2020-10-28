class DecisionTreeNode :
    
    def __init__(self, level, node_type = None, left = None, right = None, gini = None, feature = None, constr = None, index = None) :
        self.left = left
        self.right = right
        self.node_type = node_type
        self.level = level
        self.gini = gini
        self.feature = feature
        self.constr = constr
        self.index = index
        
    # add class for leaf nodes
    
    def set_class(self, Class) :
        self.Class = Class
        
    def get_class(self) :
        if self.Class :
            return self.Class
        else :
            return None
    
    def set_index(self, index) :
        self.index = index
        
    def get_index(self) :
        if self.index :
            return self.index
        else :
            return None
    
    def __repr__(self) :
        
        node_str = "Level {}, {}, ".format(self.level, self.node_type)
        if self.node_type != "Leaf" :
            node_str += "Feature: {} < {}, ".format(self.feature, self.constr)
        else :
            node_str += "Class : {}, ".format(self.Class)
        
        node_str += "Gini : {}".format(self.gini)
        
        return node_str