# Latent Dirichlet Allocation: Uncover Topics in Customer App Reviews

In large parts the code is self-explanatory.

To scrape the App Store reviews I found a nice [App Store scraper package](https://www.freecodecamp.org/news/how-to-use-python-to-scrape-app-store-reviews/) that I just had to adapt for a newer requests package (>=2.31.0) to avoid dependency conflicts with "unstructured", "gradio", "tiktoken" or "jupyterlab-server". Specifically, in this package errors occurred with the Retry-Class of `urllib`. Check the `base.py`-file in the folder `app_store_scraper` for a link with more information on that.

This repo is just considered to be an experiment working with LDA.

Maybe I will broaden the scope of this repo by leveraging on the encoded output of this LDA task in order to train a classifier model.

So far, I can tell that as of now LDA is not suitable the bigger the dataset gets. Once, I scrape 1500 reviews the coherence score as a metric to measure how coherent a topic is by the sampled words that represent this very topic, drops dramatically.