class paramsTrain():
    def __init__(self, **kwargs):
        self.dataPath = "" if ('dataPath' not in kwargs.keys()) else kwargs['dataPath']
        self.class_n = 5
        self.h = 650
        self.w = 1250
        self.patch_size = 128
        self.num_ims_train = 1002
        self.learning_rate = 0.0001

        self.batch_size = 16
        