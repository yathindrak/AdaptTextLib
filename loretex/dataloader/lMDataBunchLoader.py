from fastai1.text import *
from fastai1.basics import *
from ..utils.dropboxhandler import DropboxHandler
from .baseDataBunchLoader import BaseDataBunchLoader

class LMDataBunchLoader(BaseDataBunchLoader):
  def __init__(self, df_train_set, df_val_set, text_col_name, label_col_name, splitting_ratio, app_root, continuous_train=False, bs = 128, is_backward = False, lang = 'si'):
    self.df_train_set = df_train_set
    self.df_val_set = df_val_set
    self.text_col_name = text_col_name
    self.label_col_name = label_col_name
    self.splitting_ratio = splitting_ratio
    self.continuous_train = continuous_train
    self.app_root = app_root
    self.bs = bs
    self.is_backward = is_backward
    self.lang = lang

  def load(self):
    if (self.continuous_train):
      dropboxHandler = DropboxHandler(self.app_root)
      dropboxHandler.upload_file(self.df_train_set[self.text_col_name])

    data = TextLMDataBunch.from_df('.', train_df=self.df_train_set, valid_df=self.df_val_set, text_cols=self.text_col_name, backwards=self.is_backward)

    if self.is_backward:
      data.save(f'{self.lang}_data_lm_bwd.pkl')
      # data = load_data('./', f'{lang}_data_lm_bwd.pkl')
    else:
      data.save(f'{self.lang}_data_lm_fwd.pkl')
      # data = load_data('./', f'{lang}_data_lm_fwd.pkl')
    return data