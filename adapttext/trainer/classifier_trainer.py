from ..evaluator.evaluator import Evaluator
from ...fastai1.text import *
from ...fastai1.basics import *
from ...fastai1.callbacks import SaveModelCallback, ReduceLROnPlateauCallback, OverSamplingCallback
from ..hyperparameter.tuner import HyperParameterTuner
from .trainer import Trainer
from ..optimizer.DiffGradOptimizer import DiffGrad
import copy

class ClassifierTrainer(Trainer):
    """Trainer for Classifier"""
    def __init__(self, data, lm_fns, mdl_path, model_store_path, is_backward=False, drop_mult=0.5, *args, **kwargs):
        super(ClassifierTrainer, self).__init__(*args, **kwargs)
        self.__data = data
        self.__is_backward = is_backward
        self.__drop_mult = drop_mult

    def retrieve_classifier(self) -> 'TextClassifierLearner':
        """
        Setup and Retrieve Classification model
        :rtype: object
        """
        databunch = self.__data
        dropout_probs = dict(input=0.25, output=0.1, hidden=0.15, embedding=0.02, weight=0.2)
        size_of_embedding = 400
        num_of_hidden_neurons = 1550
        num_of_layers = 4

        config = dict(emb_sz=size_of_embedding, n_hid=num_of_hidden_neurons, n_layers=num_of_layers,
                      input_p=dropout_probs['input'], output_p=dropout_probs['output'],
                      hidden_p=dropout_probs['hidden'],
                      embed_p=dropout_probs['embedding'], weight_p=dropout_probs['weight'], pad_token=1, qrnn=False)

        embedding_size = config['emb_sz']
        num_of_classes = databunch.c
        split_function = awd_lstm_clas_split
        back_propagation_through_time_val = 80
        max_input_sequence_length = 80 * 20
        padding_idx = 1
        vocab_size = len(databunch.vocab.itos)

        for dropout_key in config.keys():
            if dropout_key.endswith('_p'): config[dropout_key] *= self.__drop_mult

        linear_features = [60]
        dropout_ps = [0.1] * len(linear_features)
        layers = [embedding_size * 3] + linear_features + [num_of_classes]
        dropout_ps = [config.pop('output_p')] + dropout_ps

        lstm_encoder = AWD_LSTM(vocab_size, **config)

        encoder = MultiBatchEncoder(back_propagation_through_time_val, max_input_sequence_length, lstm_encoder,
                                    pad_idx=padding_idx)
        model = SequentialRNN(encoder, PoolingLinearClassifier(layers, dropout_ps))

        learn = RNNLearner(databunch, model, split_func=split_function)

        return learn

    def train(self, grad_unfreeze=True):
        """
        Train Classification model
        :rtype: object
        """
        learn = self.retrieve_classifier()

        # DiffGrad Optimization
        optar = partial(DiffGrad, betas=(.91, .999), eps=1e-7)
        learn.opt_func = optar

        # Load the LM encoder
        if self.__is_backward:
            learn.load_encoder(f'{self._lang}fine_tuned_enc_bwd')
        else:
            learn.load_encoder(f'{self._lang}fine_tuned_enc')
        # Freeze the model
        learn.freeze()

        # Obtain optimal LR
        tuner = HyperParameterTuner(learn)
        lr = tuner.find_optimized_lr()

        # Train classifier for 12 iterations
        learn.fit_one_cycle(12, lr, callbacks=[SaveModelCallback(learn),
                                                  ReduceLROnPlateauCallback(learn)])

        # store model temporarily
        classifier_initial = copy.deepcopy(learn)

        evaluator = Evaluator()
        classifier_initial_accuracy = evaluator.get_accuracy(classifier_initial).item()

        print('Gradual Unfreezing..')
        # Gradual unfreezing
        if grad_unfreeze:
            # Freeze two layer groups
            learn.freeze_to(-2)
            # Train classifier for eight iterations
            learn.fit_one_cycle(8, lr,
                                callbacks=[SaveModelCallback(learn),
                                           ReduceLROnPlateauCallback(learn)])

            # Freeze three layer groups
            learn.freeze_to(-3)
            # Train classifier for six iterations
            learn.fit_one_cycle(6, lr,
                                callbacks=[SaveModelCallback(learn),
                                           ReduceLROnPlateauCallback(learn)])

        print('Completely Unfreezing..')

        # Completely unfreezing
        learn.unfreeze()

        # Obtain optimal LR
        tuner = HyperParameterTuner(learn)
        lr_unfrozed = tuner.find_optimized_lr()

        if lr_unfrozed:
            lr = lr_unfrozed

        learn.fit_one_cycle(6, lr, callbacks=[SaveModelCallback(learn),
                                              ReduceLROnPlateauCallback(learn)])
        learn.fit_one_cycle(6, lr / 2, callbacks=[SaveModelCallback(learn),
                                                  ReduceLROnPlateauCallback(learn)])

        classifier_unfrozen_accuracy = evaluator.get_accuracy(learn).item()

        if classifier_unfrozen_accuracy < classifier_initial_accuracy:
            print('reverting back to initial model...')
            learn = classifier_initial
            print('The new accuracy is {0} %.'.format(classifier_initial_accuracy))

        return learn
