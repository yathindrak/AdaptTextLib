import zipfile

from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

from ..fastai1.basics import *
from .dataloader.baseLMDataBunchLoader import BaseLMDataBunchLoader
from .dataloader.classificationDataBunchLoader import ClassificationDataBunchLoader
from .dataloader.lMDataBunchLoader import LMDataBunchLoader
from .preprocessor.preprocessor import TextPreProcessor
from .trainer.baseLMTrainer import BaseLMTrainer
from .trainer.classifierTrainer import ClassifierTrainer
from .trainer.lMTrainer import LMTrainer
from .utils.wikihandler import WikiHandler

load_dotenv('.env')


class LoReTex:
    def __init__(self, lang, data_root, bs=128, splitting_ratio=0.1):
        self.lang = lang
        self.data_root = data_root
        self.bs = bs
        self.splitting_ratio = splitting_ratio
        self.data_path = Path(data_root + '/data')
        self.lang = 'si'
        self.name = f'{lang}wiki'
        self.path = self.data_path / self.name
        self.base_lm_data_path = self.path / 'articles'
        self.mdl_path = self.path / 'models'
        self.lm_fns = [self.mdl_path / f'{lang}_wt', self.mdl_path / f'{lang}_wt_vocab']
        self.lm_fns_bwd = [self.mdl_path / f'{lang}_wt_bwd', self.mdl_path / f'{lang}_wt_vocab_bwd']

    def setup_wiki_data(self):
        # making required directories
        self.path.mkdir(exist_ok=True, parents=True)
        self.mdl_path.mkdir(exist_ok=True)

        wikihandler = WikiHandler(self.lang)

        # Getting wiki articles
        wikihandler.retrieve_articles(self.path)
        print("Retrieved wiki articles")

        # Prepare articles
        base_lm_data_path = wikihandler.prepare_articles(self.path)
        print("Completed preparing wiki articles")
        return base_lm_data_path

    def add_external_text(self, txt_filename, filepath, url):
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        r = requests.get(url, stream=True, headers=headers)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(self.data_root)

        shutil.move(self.data_root + "/" + txt_filename, str(self.base_lm_data_path / txt_filename))

    def prepareBaseLMCorpus(self):
        print(os.getenv('TUTORIAL_BOT_TOKEN'))
        self.setup_wiki_data()
        txt_filename = "test-s.txt"
        filepath = Path(self.data_root + "/test-s.zip")
        url = "https://www.dropbox.com/s/cnd985vl1bof50y/test-s.zip?dl=0"
        self.add_external_text(txt_filename, filepath, url)

    def buildBaseLM(self):
        if (not Path(self.base_lm_data_path).exists()):
            print("Please load the corpus before building base LM")
            return

        baseLMDataBunchLoader = BaseLMDataBunchLoader(self.base_lm_data_path, self.splitting_ratio)
        data_lm_fwd = baseLMDataBunchLoader.load()

        baseLMDataBunchLoader = BaseLMDataBunchLoader(self.base_lm_data_path, self.splitting_ratio,
                                                      is_backward=True)
        data_lm_bwd = baseLMDataBunchLoader.load()

        model_store_path = self.mdl_path / Path(f'{self.lang}_wt_vocab.pkl')
        model_store_path_bwd = self.mdl_path / Path(f'{self.lang}_wt_vocab_bwd.pkl')

        # forward
        lmTrainer_fwd = BaseLMTrainer(data_lm_fwd, self.lm_fns, self.mdl_path, model_store_path)
        lmTrainer_fwd.train()

        # backward
        lmTrainer_bwd = BaseLMTrainer(data_lm_bwd, self.lm_fns_bwd, self.mdl_path, model_store_path_bwd)
        lmTrainer_bwd.train()

    def buildClassifier(self, df, text_name, label_name, preprocessor=None):

        func_names = [f'{func_name}.{extension}' for func_name, extension in zip(self.lm_fns, ['pth', 'pkl'])]

        if not Path(func_names[0]).exists():
            return

        custom_model_store_path = self.mdl_path / Path(f'{self.lang}_lm_wt_vocab.pkl')
        custom_model_store_path_bwd = self.mdl_path / Path(f'{self.lang}_lm_wt_vocab_bwd.pkl')

        if preprocessor is None:
            preprocessor = TextPreProcessor(df, text_name)
            preprocessor.preprocess_text()
        else:
            preprocessor.preprocess_text()

        item_counts = df[label_name].value_counts()
        print(item_counts)
        df[label_name].value_counts().plot.bar(rot=30)

        df_trn, df_val = train_test_split(df, stratify=df[label_name], test_size=0.1)

        # forward training
        lmDataBunchLoader = LMDataBunchLoader(df_trn, df_val, text_name, label_name, self.splitting_ratio, self.data_root)
        data_lm = lmDataBunchLoader.load()

        lmDataBunchLoaderBwd = LMDataBunchLoader(df_trn, df_val, text_name, label_name, self.splitting_ratio, self.data_root,
                                                 is_backward=True)
        data_lm_bwd = lmDataBunchLoaderBwd.load()

        vocab = data_lm.train_ds.vocab

        classificationDataBunchLoader = ClassificationDataBunchLoader(df_trn, df_val, text_name, label_name,
                                                                      self.splitting_ratio, vocab)
        data_class = classificationDataBunchLoader.load()

        data_class.show_batch()

        classificationDataBunchLoaderBwd = ClassificationDataBunchLoader(df_trn, df_val, text_name, label_name,
                                                                         self.splitting_ratio, vocab, is_backward=True)
        data_class_bwd = classificationDataBunchLoaderBwd.load()

        data_class_bwd.show_batch()

        classes = data_class.classes

        lmTrainerFwd = LMTrainer(data_lm, self.lm_fns, self.mdl_path, custom_model_store_path, False)
        languageModelFWD = lmTrainerFwd.train()

        classifierTrainerFwd = ClassifierTrainer(data_class, self.lm_fns, self.mdl_path, custom_model_store_path, False)
        classifierModelFWD = classifierTrainerFwd.train()

        lmTrainerBwd = LMTrainer(data_lm_bwd, self.lm_fns_bwd, self.mdl_path, custom_model_store_path_bwd, True)
        languageModelBWD = lmTrainerBwd.train()

        classifierTrainerBwd = ClassifierTrainer(data_class_bwd, self.lm_fns_bwd, self.mdl_path, custom_model_store_path_bwd,
                                                 True)
        classifierModelBWD = classifierTrainerBwd.train()

        return classifierModelFWD, classifierModelBWD, classes
