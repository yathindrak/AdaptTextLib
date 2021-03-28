from ...fastai1.basics import *
from ...fastai1.callbacks import SaveModelCallback, ReduceLROnPlateauCallback, OverSamplingCallback
from ..hyperparameter.tuner import HyperParameterTuner
from .trainer import Trainer
from ..optimizer.DiffGradOptimizer import DiffGrad
from ...fastai1.tabular import *


class EnsembleTrainer(Trainer):
    def __init__(self, learn_clas_fwd, learn_clas_bwd, drop_mult=0.5, is_imbalanced=False, lang='si'):
        self.learn_clas_fwd = learn_clas_fwd
        self.learn_clas_bwd = learn_clas_bwd
        self.is_imbalanced = is_imbalanced
        self.drop_mult = drop_mult
        self.lang = lang

    def retrieve_classifier(self, metrics=None):
        pred_tensors_fwd, pred_tensors_target_fwd = self.learn_clas_fwd.get_preds(DatasetType.Valid, ordered=True)
        pred_tensors_bwd, pred_tensors_target_bwd = self.learn_clas_bwd.get_preds(DatasetType.Valid, ordered=True)

        preds_fwd = pd.DataFrame(pred_tensors_fwd.numpy()).add_prefix('fwd_')
        preds_textm_bwd = pd.DataFrame(pred_tensors_bwd.numpy()).add_prefix('bwd_')
        preds_target_fwd = pd.DataFrame(pred_tensors_target_fwd.numpy())

        ensemble_df = (preds_fwd
                       .join(preds_textm_bwd)
                       .join(preds_target_fwd).rename(columns={0: "target"})
                       )

        column_names = ensemble_df.columns.values.tolist()
        column_names.pop()

        tabular_processes = [FillMissing, Categorify, Normalize]

        data_ensemble = (TabularList
                         .from_df(ensemble_df, cat_names=[], cont_names=column_names, procs=tabular_processes)
                         .split_by_rand_pct(valid_pct=0.1, seed=42)
                         .label_from_df(cols="target")
                         .databunch())

        learn = tabular_learner(data_ensemble, layers=[1000,500], ps=[0.001, 0.01], metrics=metrics, emb_drop = 0.04)

        return learn

    def train(self):
        learn = self.retrieve_classifier(metrics=[error_rate, accuracy])

        optar = partial(DiffGrad, betas=(.91, .999), eps=1e-7)
        learn.opt_func = optar

        # Find LR
        tuner = HyperParameterTuner(learn)
        lr = tuner.find_optimized_lr()

        if self.is_imbalanced:
            learn.fit_one_cycle(8, lr, callbacks=[SaveModelCallback(learn),OverSamplingCallback(learn),
                                                  ReduceLROnPlateauCallback(learn, factor=0.8)])

            learn.fit_one_cycle(8, lr/2, callbacks=[SaveModelCallback(learn),OverSamplingCallback(learn),
                                                  ReduceLROnPlateauCallback(learn, factor=0.8)])

            learn.fit_one_cycle(8, lr/2,
                                callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy'),
                                           ReduceLROnPlateauCallback(learn, factor=0.8)])
        else:
            learn.fit_one_cycle(8, lr, callbacks=[SaveModelCallback(learn),
                                                  ReduceLROnPlateauCallback(learn, factor=0.8)])

            learn.fit_one_cycle(8, lr / 2, callbacks=[SaveModelCallback(learn),
                                                      ReduceLROnPlateauCallback(learn, factor=0.8)])

            learn.fit_one_cycle(8, lr/2,
                                callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy'),
                                           ReduceLROnPlateauCallback(learn, factor=0.8)])

        return learn
