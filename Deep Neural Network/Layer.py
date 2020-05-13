class Layer :

    def __init__(self, activation='linear', dim) :
        
        self.dim = dim
        self.activation = activation

    def __sigmoid(self, z):
    
        # This method computes the sigmoid of z, a scalar or numpy array of any size.

        s = 1 / (1 + np.exp(-z))

        return s
            

    def update(self) :
        
        
    def propag