import zipfile
import requests
from sklearn.model_selection import train_test_split

from .trainer.ensemble_trainer import EnsembleTrainer
from .utils.dropbox_handler import DropboxHandler
from .utils.zip_handler import ZipHandler
from ..fastai1.basics import *
from .dataloader.base_lm_data_bunch_loader import BaseLMDataBunchLoader
from .dataloader.classification_data_bunch_loader import ClassificationDataBunchLoader
from .dataloader.lm_data_bunch_loader import LMDataBunchLoader
from .preprocessor.preprocessor import TextPreProcessor
from .trainer.base_lm_trainer import BaseLMTrainer
from .trainer.classifier_trainer import ClassifierTrainer
from .trainer.lm_trainer import LMTrainer
from .utils.wiki_handler import WikiHandler


class AdaptText:
    def __init__(self, lang, data_root, bs=128, splitting_ratio=0.1, is_imbalanced=False):
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
        self.lm_store_path = [f'{data_root}/data/{lang}wiki/models/si_wt_vocab.pkl',
                              f'{data_root}/data/{lang}wiki/models/si_wt.pth',
                              f'{data_root}/data/{lang}wiki/models/si_wt_vocab_bwd.pkl',
                              f'{data_root}/data/{lang}wiki/models/si_wt_bwd.pth']
        self.lm_store_files = ['si_wt_vocab.pkl', 'si_wt.pth', 'si_wt_vocab_bwd.pkl', 'si_wt_bwd.pth']
        self.classifiers_store_path = ["models/fwd-export.pkl", "models/bwd-export.pkl"]
        self.is_imbalanced = is_imbalanced

        if not torch.cuda.is_available():
            self.is_gpu = False
            warnings.warn(
                'Note that CUDA support is not available for your instance, Hence training will be continued on CPU')
        else:
            self.is_gpu = True

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

    def prepare_base_lm_corpus(self):
        self.setup_wiki_data()
        # txt_filename = "test-s.txt"
        # filepath = Path(self.data_root + "/test-s.zip")
        # url = "https://www.dropbox.com/s/cnd985vl1bof50y/test-s.zip?dl=0"

        # txt_filename = "half-si-dedup.txt"
        # filepath = Path(self.data_root + "/half-si-dedup.zip")
        # url = "https://www.dropbox.com/s/alh6jf4rqxhhzow/half-si-dedup.zip?dl=0"

        txt_filename = "si_dedup.txt"
        filepath = Path(self.data_root + "/si_dedup.txt.zip")

        url = "https://www.dropbox.com/s/sa8jqah8x9p4q2l/si_dedup.txt.zip?dl=0"

        self.add_external_text(txt_filename, filepath, url)

        dropbox_handler = DropboxHandler(self.data_root)
        dropbox_handler.download_articles()

    def prepare_pretrained_lm(self, model_file_name):
        # models-test-s-10-epochs-with-cls.zip
        if (Path(f'{os.getcwd()}{self.data_root}').exists()):
            shutil.rmtree(f'{os.getcwd()}{self.data_root}')
        dropbox_handler = DropboxHandler(self.data_root)
        dropbox_handler.download_pretrained_model(model_file_name)

        zip_handler = ZipHandler()
        zip_handler.unzip(model_file_name)

        if os.path.exists(self.mdl_path):
            shutil.rmtree(str(self.mdl_path))
            os.mkdir(str(self.mdl_path))
            os.mkdir(str(self.base_lm_data_path))
        else:
            os.mkdir(str(self.data_path))
            os.mkdir(str(self.path))
            os.mkdir(str(self.mdl_path))
            os.mkdir(str(self.base_lm_data_path))

        for source in self.lm_store_files:
            source = f'{os.getcwd()}{self.data_root}/data/{self.lang}wiki/models/{source}'
            shutil.move(source, self.mdl_path)

    def build_base_lm(self):
        if (not Path(self.base_lm_data_path).exists()):
            print("Base LM corpus not found, preparing the corpus...")
            self.prepare_base_lm_corpus()

        baseLMDataBunchLoader = BaseLMDataBunchLoader(self.base_lm_data_path, self.splitting_ratio)
        data_lm_fwd = baseLMDataBunchLoader.load()

        baseLMDataBunchLoader = BaseLMDataBunchLoader(self.base_lm_data_path, self.splitting_ratio,
                                                      is_backward=True)
        data_lm_bwd = baseLMDataBunchLoader.load()

        model_store_path = self.mdl_path / Path(f'{self.lang}_wt_vocab.pkl')
        model_store_path_bwd = self.mdl_path / Path(f'{self.lang}_wt_vocab_bwd.pkl')

        # forward
        lmTrainer_fwd = BaseLMTrainer(data_lm_fwd, self.lm_fns, self.mdl_path, model_store_path, is_gpu=self.is_gpu)
        lmTrainer_fwd.train()

        # backward
        lmTrainer_bwd = BaseLMTrainer(data_lm_bwd, self.lm_fns_bwd, self.mdl_path, model_store_path_bwd,
                                      is_gpu=self.is_gpu)
        lmTrainer_bwd.train()

    def build_classifier(self, df, text_name, label_name, grad_unfreeze: bool = True, preprocessor=None):
        df = df[[text_name, label_name]]
        func_names = [f'{func_name}.{extension}' for func_name, extension in zip(self.lm_fns, ['pth', 'pkl'])]

        if not Path(func_names[0]).exists():
            raise Exception("Base LM does not exists, please load pretrained model")

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
        lmDataBunchLoader = LMDataBunchLoader(df_trn, df_val, text_name, label_name, self.splitting_ratio,
                                              self.data_root)
        data_lm = lmDataBunchLoader.load()

        lmDataBunchLoaderBwd = LMDataBunchLoader(df_trn, df_val, text_name, label_name, self.splitting_ratio,
                                                 self.data_root,
                                                 is_backward=True)
        data_lm_bwd = lmDataBunchLoaderBwd.load()

        vocab = data_lm.train_ds.vocab

        classificationDataBunchLoader = ClassificationDataBunchLoader(df_trn, df_val, text_name, label_name,
                                                                      self.splitting_ratio, vocab)
        data_class = classificationDataBunchLoader.load()

        # print(data_class.show_batch())

        classificationDataBunchLoaderBwd = ClassificationDataBunchLoader(df_trn, df_val, text_name, label_name,
                                                                         self.splitting_ratio, vocab, is_backward=True)
        data_class_bwd = classificationDataBunchLoaderBwd.load()

        # print(data_class_bwd.show_batch())

        classes = data_class.classes

        print('Training Forward model...')

        lmTrainerFwd = LMTrainer(data_lm, self.lm_fns, self.mdl_path, custom_model_store_path, False,
                                 is_gpu=self.is_gpu)
        languageModelFWD = lmTrainerFwd.train()

        classifierTrainerFwd = ClassifierTrainer(data_class, self.lm_fns, self.mdl_path, custom_model_store_path, False, is_imbalanced=self.is_imbalanced)
        classifierModelFWD = classifierTrainerFwd.train(grad_unfreeze)

        # eval = Evaluator()
        # classifierFWDAccuracy = eval.get_accuracy(classifierModelFWD)
        # classifierFWDAccuracyNew = 0

        # if(classifierFWDAccuracy.item() < 53.0):
        #     classifierTrainerFwd = ClassifierTrainer(data_class, self.lm_fns, self.mdl_path, custom_model_store_path,
        #                                              False)
        #     classifierModelFWDNew = classifierTrainerFwd.train(grad_unfreeze)
        #     classifierFWDAccuracyNew = eval.get_accuracy(classifierModelFWD)
        #
        # if (classifierFWDAccuracyNew.item() > classifierFWDAccuracy.item()):
        #     classifierModelFWD = classifierModelFWDNew

        print('Training Backward model...')

        lmTrainerBwd = LMTrainer(data_lm_bwd, self.lm_fns_bwd, self.mdl_path, custom_model_store_path_bwd, True,
                                 is_gpu=self.is_gpu)
        languageModelBWD = lmTrainerBwd.train()

        classifierTrainerBwd = ClassifierTrainer(data_class_bwd, self.lm_fns_bwd, self.mdl_path,
                                                 custom_model_store_path_bwd, True, is_imbalanced=self.is_imbalanced)
        classifierModelBWD = classifierTrainerBwd.train(grad_unfreeze)

        # eval = Evaluator()
        # classifierBWDAccuracy = eval.get_accuracy(classifierModelBWD)
        # classifierBWDAccuracyNew = 0
        #
        # if (classifierBWDAccuracy.item() < 53.0):
        #     classifierTrainerBwd = ClassifierTrainer(data_class, self.lm_fns, self.mdl_path, custom_model_store_path,
        #                                              False)
        #     classifierModelBWDNew = classifierTrainerBwd.train(grad_unfreeze)
        #     classifierBWDAccuracyNew = eval.get_accuracy(classifierModelBWDNew)
        #
        # if (classifierBWDAccuracyNew.item() > classifierBWDAccuracy.item()):
        #     classifierModelBWD = classifierModelBWDNew

        ensembleTrainer = EnsembleTrainer(classifierModelFWD, classifierModelBWD)
        ensembleModel = ensembleTrainer.train()

        return classifierModelFWD, classifierModelBWD, ensembleModel, classes

    def store_lm(self, zip_file_name):
        # zip_file_name = "test.zip"
        zip_archive = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
        for item in self.lm_store_path:
            zip_archive.write(item)
        zip_archive.close()

        dropbox_handler = DropboxHandler(self.data_root)
        dropbox_handler.upload_zip_file(zip_file_name, f'/adapttext/models/{zip_file_name}')

