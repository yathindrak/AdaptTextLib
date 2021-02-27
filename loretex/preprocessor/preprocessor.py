import emoji


class TextPreProcessor:
    def __init__(self, df, text_name):
        self.df = df
        self.text_name = text_name

    def demojize_text(self, text):
        return emoji.demojize(text)

    def preprocess_text(self):
        self.df.dropna()
        self.df[self.text_name] = self.df[self.text_name].apply(self.demojize_text)
        return self.df
