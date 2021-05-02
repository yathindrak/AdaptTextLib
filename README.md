# Getting Started with AdaptText Developer Framework

Make text classification available for everyone !!!

## Setup

In the project directory, first clone the library

`git clone https://gitlab.com/AdaptText/AdaptTextLib.git`

Then run the setup.sh shell script to make everything ready for you.

`cd AdaptTextLib && bash setup.sh`

Then run the setup.sh shell script to make everything ready for you.

## Usage

First initialize the adapttext with the preferred language code (Sinhala) , Root directory, batch size and splitting ratio
```
from AdaptTextLib.adapttext.adapt_text import AdaptText

lang = 'si'
app_root = "/storage"
bs = 128
splitting_ratio = 0.1
adapttext = AdaptText(lang, app_root, bs, splitting_ratio)
```

Then load the pretrained model
```
adapttext.prepare_pretrained_lm("full_si_dedup.zip")
```

Then train the classifier by passing the dataframe, the text and label cols
```
import pandas as pd

pd.set_option('display.max_colwidth', -1)
path_to_csv="my_awesome_dataset.csv"
df = pd.read_csv(path_to_csv)

text_name = "Title"
label_name = "Label"

classifierModelFWD, classifierModelBWD, ensembleModel, classes = adapttext.build_classifier(df, text_name, label_name, grad_unfreeze=True)
```
To get evaluations use below code
```
from AdaptTextLib.adapttext.evaluator.evaluator import Evaluator

evaluator = Evaluator()
evaluator.evaluate(ensembleModel, classifierModelFWD)
```

That's It !!!

Apart from that to build an own pretrained model or setup continous training execute following code.

```
adapttext.build_base_lm()
adapttext.store_lm("full_si_dedup_new.zip")
```
