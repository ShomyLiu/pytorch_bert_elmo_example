# -*- coding: utf-8 -*-

class Config():

    glove_file = "./data/glove/glove_300d.npy"
    word2id_file = "./data/word2id.npy"
    elmo_options_file = "./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
    elmo_weight_file = "./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
    emb_method = 'elmo'  # elmo/glove/elmo_glove

    num_filters = 100
    k = [3, 4, 5]
    vocab_size = 18766
    glove_dim = 300
    elmo_dim = 512

    num_labels = 2

    use_gpu = True
    gpu_id = 0

    dropout = 0.5
    epochs = 50

    test_size = 0.1
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 64
    device = "cuda:0"


def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


Config.parse = parse
opt = Config()
