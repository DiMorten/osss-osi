class paramsTrain():
    def __init__(self, **kwargs):
        self.dataPath = "" if ('dataPath' not in kwargs.keys()) else kwargs['dataPath']
        self.mode = "train" if ('mode' not in kwargs.keys()) else kwargs['mode']
        self.modelId = "" if ('modelId' not in kwargs.keys()) else kwargs['modelId']
        self.loss = "weighted_categorical_crossentropy" if ('loss' not in kwargs.keys()) else kwargs['loss']

        self.class_n = 5
        self.h = 650
        self.w = 1250
        self.patch_size = 128
        self.num_ims_train = 1002
        self.learning_rate = 1e-3
        self.channel_n = 1

        self.batch_size = 16
        