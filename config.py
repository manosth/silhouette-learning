import numpy as np
class Params():
        def __init__(self):
            # optimizer and training params
            self.lr = 1e-3
            self.eps = 1e-8

            self.batch_size = 64
            self.epochs = 200

            self.init_mode = "data_whole"
            self.tensorboard = True

            # model params
            self.num_groups = 320
            self.group_size = 1

            self.num_layers = 15

            self.group_tau = 0.01
            self.group_lambda = 0.5

            # data params
            self.n_classes = 10
            self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

            self.n_channels = 1
            self.input_size = 1024
            self.input_width = 32
            self.input_height = 32