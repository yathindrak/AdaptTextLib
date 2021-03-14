from ...fastai1.text import BaseTokenizer, List
import sentencepiece as spm


class SinhalaTokenizer(BaseTokenizer):
    def __init__(self):
        self.lang = 'si'
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str("/storage/data/siwiki/articles/tmp/spm.model"))

    def tokenizer(self, t: str) -> List[str]:
        return self.sp.EncodeAsPieces(t)
