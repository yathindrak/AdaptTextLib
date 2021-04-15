from ...fastai1.text import *
from ...fastai1.basics import *
from .data_bunch_loader import DataBunchLoader


class BaseLMDataBunchLoader(DataBunchLoader):
    def __init__(self, path, splitting_ratio, is_backward=False, *args, **kwargs):
        super(BaseLMDataBunchLoader, self).__init__(*args, **kwargs)
        self.__path = path
        self.__splitting_ratio = splitting_ratio
        # self.seed = seed
        # self.bs = bs
        self.__is_backward = is_backward
        # self.lang = lang

    def load(self):
        tokenizer = Tokenizer(SpacyTokenizer, lang="xx")
        data = (TextList.from_folder(Path(self.__path), processor=[OpenFileProcessor(), TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(max_vocab=30000)])
                .split_by_rand_pct(self.__splitting_ratio, seed=self._seed)
                .label_for_lm()
                .databunch(bs=self._bs, num_workers=1, backwards=self.__is_backward))
        # Store data
        data.save(f'{self._lang}_databunch')
        print(len(data.vocab.itos), len(data.train_ds))
        print("Base LM vocab size : " + str(len(data.vocab.itos)))
        # Load data
        # data = load_data(self.path, f'{self.lang}_databunch', bs=self.bs)
        return data
