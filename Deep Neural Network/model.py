class DenseModel :

    def __init__(self, input_dim, layer_dims, learning_rate, optimizer, loss) :
        self.history = {}

    def fit(X, y, batch_size, epochs, eval_set=None, Verbose)