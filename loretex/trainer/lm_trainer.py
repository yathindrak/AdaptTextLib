from ...fastai1.text import *
from ...fastai1.basics import *
from ...fastai1.callbacks import SaveModelCallback, ReduceLROnPlateauCallback
from ..hyperparameter.tuner import HyperParameterTuner
from .trainer import Trainer
from ..optimizer.DiffGradOptimizer import DiffGrad


class LMTrainer(Trainer):
    def __init__(self, data, lm_fns, mdl_path, model_store_path, is_backward=False, drop_mult=0.9, is_gpu=True, lang='si'):
        self.data = data
        self.lm_fns = lm_fns
        self.mdl_path = mdl_path
        self.model_store_path = model_store_path
        self.is_backward = is_backward
        self.drop_mult = drop_mult
        self.is_gpu = is_gpu
        self.lang = lang

    def retrieve_language_model(self, databunch: DataBunch, config: dict, drop_multi_val: float = 1.,
                                pretrained_file_paths: OptStrTuple = None, metrics=None) -> 'LanguageLearner':
        embedding_size = config['emb_sz']
        split_function = awd_lstm_lm_split
        for dropout_key in config.keys():
            if dropout_key.endswith('_p'): config[dropout_key] *= drop_multi_val
        output_dropout_p, output_bias = map(config.pop, ['output_p', 'out_bias'])
        # Get list of words(itos) in vocab
        vocab_size = len(databunch.vocab.itos)
        lstm_encoder = AWD_LSTM(vocab_size, **config)
        enc_embedding = lstm_encoder.encoder
        lstm_decoder = LinearDecoder(vocab_size, embedding_size, output_dropout_p, tie_encoder=enc_embedding,
                                     bias=output_bias)
        model = SequentialRNN(lstm_encoder, lstm_decoder)
        learn = LanguageLearner(databunch, model, split_function, metrics=metrics)

        if pretrained_file_paths is not None:
            print(pretrained_file_paths)
            data_path = learn.path
            model_path = learn.model_dir

            func_names = [data_path / model_path / f'{func_name}.{extension}' for func_name, extension in
                          zip(pretrained_file_paths, ['pth', 'pkl'])]

            learn = learn.load_pretrained(*func_names)
            learn.freeze()

        return learn

    def train(self):
        dropout_probs = dict(input=0.25, output=0.1, hidden=0.15, embedding=0.02, weight=0.2)
        size_of_embedding = 400
        num_of_hidden_neurons = 1200
        num_of_layers = 4

        config = dict(emb_sz=size_of_embedding, n_hid=num_of_hidden_neurons, n_layers=num_of_layers,
                      input_p=dropout_probs['input'], output_p=dropout_probs['output'],
                      hidden_p=dropout_probs['hidden'], embed_p=dropout_probs['embedding'],
                      weight_p=dropout_probs['weight'], pad_token=1, qrnn=False, out_bias=True)

        lm_fn_1_fwd = self.mdl_path / f'{self.lang}_wt.pth'
        lm_fn_1_bwd = self.mdl_path / f'{self.lang}_wt_bwd.pth'

        if ((not self.is_backward and lm_fn_1_fwd.exists()) or (self.is_backward and lm_fn_1_bwd.exists())):
            if self.is_gpu:
                learn = self.retrieve_language_model(self.data, config=config, drop_multi_val=self.drop_mult,
                                                 pretrained_file_paths=self.lm_fns,
                                                 metrics=[error_rate, accuracy, Perplexity()]).to_fp16()
            else:
                learn = self.retrieve_language_model(self.data, config=config, drop_multi_val=self.drop_mult,
                                                     pretrained_file_paths=self.lm_fns,
                                                     metrics=[error_rate, accuracy, Perplexity()])
        else:
            if self.is_gpu:
                learn = self.retrieve_language_model(self.data, config=config, drop_multi_val=self.drop_mult,
                                                 metrics=[error_rate, accuracy, Perplexity()]).to_fp16()
            else:
                learn = self.retrieve_language_model(self.data, config=config, drop_multi_val=self.drop_mult,
                                                     metrics=[error_rate, accuracy, Perplexity()])

        optar = partial(DiffGrad, betas=(.91, .999), eps=1e-7)
        learn.opt_func = optar

        # Find LR
        tuner = HyperParameterTuner(learn)
        lr = tuner.find_optimized_lr()

        learn.fit_one_cycle(2, lr, moms=(0.8, 0.7),
                            callbacks=[SaveModelCallback(learn), ReduceLROnPlateauCallback(learn, factor=0.8)])

        learn.unfreeze()
        learn.fit_one_cycle(11, lr, moms=(0.8, 0.7),
                            callbacks=[SaveModelCallback(learn), ReduceLROnPlateauCallback(learn, factor=0.8)])

        learn.predict("මේ අතර", n_words=30)

        if self.is_backward:
            # comment below 2 lines of code out to avoid overriding base lm: cause errors otherwise
            # learn.to_fp32().save(self.mdl_path  / self.lm_fns[0], with_opt=False)
            # learn.data.vocab.save(self.model_store_path)

            # learn.data.vocab.save('/content/data/siwiki/models/si_wt_vocab_bwd.pkl')

            learn.save(f'{self.lang}fine_tuned_bwd')
            learn.save_encoder(f'{self.lang}fine_tuned_enc_bwd')
        else:
            # comment below 2 lines of code out to avoid overriding base lm: cause errors otherwise
            # learn.to_fp32().save(self.mdl_path / self.lm_fns[0], with_opt=False)
            # learn.data.vocab.save(self.model_store_path)

            # learn.data.vocab.save('/content/data/siwiki/models/si_wt_vocab.pkl')

            learn.save(f'{self.lang}fine_tuned')
            learn.save_encoder(f'{self.lang}fine_tuned_enc')