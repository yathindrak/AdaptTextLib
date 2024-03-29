from ...fastai1.text import *
from ..utils.dropbox_handler import DropboxHandler
from .data_bunch_loader import DataBunchLoader


class LMDataBunchLoader(DataBunchLoader):
    """Provide Databunch for Language Model"""
    def __init__(self, df_train_set, df_val_set, text_col_name, label_col_name, app_root,
                 continuous_train=False, is_backward=False, *args, **kwargs):
        super(LMDataBunchLoader, self).__init__(*args, **kwargs)
        self.__df_train_set = df_train_set
        self.__df_val_set = df_val_set
        self.__text_col_name = text_col_name
        self.__label_col_name = label_col_name
        self.__continuous_train = continuous_train
        self.__app_root = app_root
        self.__is_backward = is_backward

    def load(self):
        """
        Returns databunch for Language Model
        :rtype: object
        """
        if self.__continuous_train:
            dropbox_handler = DropboxHandler(self.__app_root)
            dropbox_handler.upload_text_file(self.__df_train_set[self.__text_col_name])

        # spacy tokenizer
        tokenizer = Tokenizer(SpacyTokenizer, lang="xx")

        data = TextLMDataBunch.from_df('.', train_df=self.__df_train_set, valid_df=self.__df_val_set,
                                       text_cols=self.__text_col_name, tokenizer=tokenizer, backwards=self.__is_backward)

        # Save checkpoints
        if self.__is_backward:
            data.save(f'{self._lang}_data_lm_bwd.pkl')
        else:
            data.save(f'{self._lang}_data_lm_fwd.pkl')
        return data
