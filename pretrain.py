from .adapttext.adapt_text import AdaptText

if __name__ == '__main__':
	lang = 'si'
	app_root = "/storage"
	bs = 128
	splitting_ratio = 0.1
	adapttext = AdaptText(lang, app_root, bs, splitting_ratio)
	adapttext.build_base_lm()
	adapttext.store_lm("full_si_dedup_30k.zip")