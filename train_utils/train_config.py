import numpy as np


class TicTacTrainConfig(object):
    def __init__(self):
        self.batch_size = 64
        self.num_workers = 4
        self.mean_val = np.array([65.9054])
        self.epoch = 100
        self.lr = 0.001
        self.weight_decay = 0.0001
        self.epoch_lr_decay_milestones = [30, 60]
        dataset_labels = np.hstack((np.zeros(909, dtype=np.int32), np.ones(34744, dtype=np.int32)))
        class_sample_weight = 1000. / np.array([909, 34744])
        self.samples_weight = np.array([class_sample_weight[t] for t in dataset_labels])
        self.loss_growth_factor = 0.00005
        self.fine_tune = False


config = TicTacTrainConfig()
