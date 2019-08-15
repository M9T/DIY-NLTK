# install nltk and numpy first (https://www.nltk.org/install.html)
# download the nltk data (save here: C:\Users\<MYUSER>\AppData\Roaming\nltk_data)
import nltk
import matplotlib.pyplot
import numpy as np
# nltk.download("popular")

from nltk.tokenize import word_tokenize
from nltk.text import Text

# Test general nltk methods
#%%
sentence = "Zalando SE is a European e-commerce company based in Berlin, Germany. The company follows a platform approach, offering Fashion and Lifestyle products to customers in 17 European markets. Zalando was founded in Germany in 2008. Swedish company Kinnevik is the largest owner with 32%."
tokens = word_tokenize(sentence)
tokens = [word.lower() for word in tokens]
print(tokens[:3])
t = Text(tokens)

#%%
print(t.count("is"))
#%%
print(t.index("company"))
#%%
print(t.similar("Berlin"))
#%%
print(t.vocab())
#%%
print(t.plot(10))

# Test nltk corpus by reuters
from nltk.corpus import reuters
#%%
print(Text(reuters.words()).common_contexts(["June", "April"]))
