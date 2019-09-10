"""
How can we identify a language? 
Idea I: Let us count the stop words.
Idea II: Let us use word bigrams.
"""

# Idea I
from nltk.tokenize import wordpunct_tokenize
sentence = "Zalando SE is a European e-commerce company based in Berlin, Germany. The company follows a platform approach, offering Fashion and Lifestyle products to customers in 17 European markets. Zalando was founded in Germany in 2008. Swedish company Kinnevik is the largest owner with 32%."
tokens = wordpunct_tokenize(sentence)
print(tokens)

# Explore stop word corpus
from nltk.corpus import stopwords
print(stopwords.readme().replace("\n", " "))

# German stop words
print(stopwords.raw("german").replace("\n", " "))

# How many stop words for english and german?
print(len(stopwords.words(["english", "german"])))

# Classify language by counting stop words
language_ratios = {}
test_words = [word.lower() for word in test_tokens]
test_words_set = set(test_words)

for language in stopwords.fileids():
    # For some languages it would be a wise idea to tokenize the stop words by punctuation too.
    stopwords_set = set(stopwords.words(language)) 
    common_elements = test_words_set.intersection(stopwords_set)
    language_ratios[language] = len(common_elements)
    
print(language_ratios)

# The sentence is in?
top_counted_language = max(language_ratios, key=language_ratios.get)
print(top_counted_language)

# Which words were counted?
print(test_words_set.intersection(set(stopwords.words(top_counted_language))))


# Idea II
from nltk import ngrams, word_tokenize, FreqDist
import operator
import pickle
import string

# As words
# Remove punctuation and digits
sentence = sentence.translate(str.maketrans("", "", string.punctuation + string.digits))
tokens_words = word_tokenize(sentence.lower())
print(tokens_words)

# As chars
tokens_chars = list(tokens_words[0])
print(tokens_chars)

# As unigrams
tokens_words_unigrams = list(ngrams(tokens_words, 1))
print(tokens_words_unigrams)

# As bigrams
tokens_words_bigrams = list(ngrams(tokens_words, 2, pad_left=True, pad_right=True, left_pad_symbol="_", right_pad_symbol="_"))
print(tokens_words_bigrams)

# Let us count
print(FreqDist(tokens_words_unigrams))
# alternatively
unigram_dict = dict()
for a, b in fdist.items():
        unigram_dict[" ".join(a)] = b
print(unigram_dict)

# Take a look at the ngram files
file = "ngram_langid_files/LangId.train.English.txt"
with open(file, encoding="utf8") as f:
        content = f.read().lower()
print(content.replace("\n", "")[:50])

with open("ngram_langid_files/English.unigram.pickle", "rb") as handle:
    unigram_english_dict = pickle.load(handle)
print(unigram_english_dict)

# How often is the word "so" in the text
print(bigram_english_dict.get("so"))

# Which words are used most frequently? Top3
english_unigram_freqs = sorted(unigram_english_dict.items(), key=operator.itemgetter(1), reverse=True)
print(english_unigram_freqs[:3])

# Generating unigram and bigram frequencies for English, French and Italian from training files
def get_ngram_count_dict(tokens, n):
    if n == 1:
        n_grams = ngrams(tokens, n)
    else:
        n_grams = ngrams(tokens, n, pad_left=True, pad_right=True, left_pad_symbol="_", right_pad_symbol="_") # Fun fact: If I remove padding here and later when testing, and also remove the "_" from the unigram dicts, the accuracy rises slightly. However, it"s not statistically significant due to the small size of the data.
    fdist = FreqDist(n_grams)
    ngram_dict = dict()
    for a,b in fdist.items():
        ngram_dict[" ".join(a)] = b
    return ngram_dict

# Calls get_ngram_count_dict to get a unigram and bigram dict from file.
def get_unigram_bigram_dicts(file):
    with open(file, encoding="utf8") as f:
        content = f.read()
    tokens = content.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = word_tokenize(tokens.lower())
    unigram_dict = get_ngram_count_dict(tokens, 1)     
    bigram_dict = get_ngram_count_dict(tokens, 2)     
    return (unigram_dict, bigram_dict)

# Dumps unigram and bigram dictionary of training data of given language to .pickle files.
def dump_pickle(language):
    file = "ngram_langid_files/LangId.train." + language + ".txt"
    unigram_dict, bigram_dict = get_unigram_bigram_dicts(file)
    with open("ngram_langid_files/" + language + ".unigram.pickle", "wb") as handle:
        # HIGHEST_PROTOCOL instructs pickle to use the highest protocol version available.
        pickle.dump(unigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open("ngram_langid_files/" + language + ".bigram.pickle", "wb") as handle:
        pickle.dump(bigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
dump_pickle("English")
dump_pickle("French")
dump_pickle("Italian")

with open("ngram_langid_files/LangId.train.English.txt", encoding="utf8") as f:
    for i, l in enumerate(f):
        pass
number_of_sents_en = i + 1
with open("ngram_langid_files/LangId.train.French.txt", encoding="utf8") as f:
    for i, l in enumerate(f):
        pass
number_of_sents_fr = i + 1
with open("ngram_langid_files/LangId.train.Italian.txt", encoding="utf8") as f:
    for i, l in enumerate(f):
        pass
number_of_sents_it = i + 1

print("number of sentences in training data")
print("english:", number_of_sents_en)
print("french:", number_of_sents_fr)
print("italian:", number_of_sents_it)

# dentifying language for each line of the test file using bigram probabilities

with open("ngram_langid_files/English.unigram.pickle", "rb") as handle:
    unigram_english_dict = pickle.load(handle)
    
with open("ngram_langid_files/English.bigram.pickle", "rb") as handle:
    bigram_english_dict = pickle.load(handle)
    
with open("ngram_langid_files/French.unigram.pickle", "rb") as handle:
    unigram_french_dict = pickle.load(handle)
    
with open("ngram_langid_files/French.bigram.pickle", "rb") as handle:
    bigram_french_dict = pickle.load(handle)
    
with open("ngram_langid_files/Italian.unigram.pickle", "rb") as handle:
    unigram_italian_dict = pickle.load(handle)
    
with open("ngram_langid_files/Italian.bigram.pickle", "rb") as handle:
    bigram_italian_dict = pickle.load(handle)
    
vocabulary_size = len(unigram_english_dict) + len(unigram_french_dict) + len(unigram_italian_dict)

# Get probability of given bigram belonging to the language which bigram_dict is in
# first_word is the first word of the word bigram.
def get_bigram_probability(bigram, first_word, bigram_dict, first_word_dict): 
    bigram_count = bigram_dict.get(bigram)
    if bigram_count is None:
        bigram_count = 0
    
    first_word_count = first_word_dict.get(first_word)
    if first_word_count is None:
        first_word_count = 0
    
    return (bigram_count + 1) / (first_word_count + vocabulary_size)

# Get probability that a given bigram list is of a language (specified by its bigram_dict)
def get_language_probability(bigram_list, first_words, bigram_dict, first_word_dict):
    result = 1.0
    index = 0
    for bigram in bigram_list:
        result *= get_bigram_probability(bigram, first_words[index], bigram_dict, first_word_dict)
        index += 1
    return result

# Load correct solutions
solution_dict = dict()
with open("ngram_langid_files/LangId.sol.txt") as f:
    for line in f:
       (key, val) = line.split()
       solution_dict[int(key)] = val
        
line_no = 1
result_dict = dict()
correct = 0
incorrect_line_numbers = []

# This needs to be done because I am using padding for bigrams so the unigram dicts in their raw forms can"t be used in get_bigram_probability():
unigram_english_dict["_"] = number_of_sents_en
unigram_french_dict["_"] = number_of_sents_fr
unigram_italian_dict["_"] = number_of_sents_it

with open("ngram_langid_files/LangId.test.txt", encoding="utf8") as f:
    for line in f:
        tokens = line.translate(str.maketrans("", "", string.punctuation + string.digits))
        tokens = word_tokenize(tokens.lower())
        bigrams = ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol="_", right_pad_symbol="_")
        # bigram_list will be exactly like bigrams. It is required because this is how bigrams are represented in the dictionary.
        bigram_list = [] 
        # The first words of each bigram. This is the similar to making a unigram_list. We use it because we don"t want something in the form [(this,), ...]. Also because we want this to include "_". We want it to include "_" because we"re not using the unigrams for classification but as part of a formula to judge bigram frequency based on the starting word.
        first_words = [] 
        for b in bigrams:
            bigram_list.append(" ".join(b))
            first_words.append(b[0])
        
        english_prob = get_language_probability(bigram_list, first_words, bigram_english_dict, unigram_english_dict)
        french_prob = get_language_probability(bigram_list, first_words, bigram_french_dict, unigram_french_dict)
        italian_prob = get_language_probability(bigram_list, first_words, bigram_italian_dict, unigram_italian_dict)
        
        max_prob = max(english_prob, french_prob, italian_prob)
        if max_prob == english_prob:
            result_dict[line_no] = "English"
        elif max_prob == french_prob:
            result_dict[line_no] = "French"
        else:
            result_dict[line_no] = "Italian"
        
        if solution_dict[line_no] == result_dict[line_no]:
            correct += 1
        else:
            incorrect_line_numbers.append(line_no)
            
        line_no += 1

# Storing results from result_dict to file:
with open("ngram_langid_files/LangId.result.txt", "w") as f:
    for (key, val) in result_dict.items():
        f.write(" ".join([str(key), val]) + "\n")
        
print("Accuracy: {:2.2f}%".format(correct * 100 / len(solution_dict)))

# Testing with our own sentence
sentence = "This is a short sentence."
tokens = sentence.translate(str.maketrans("", "", string.punctuation + string.digits))
tokens = word_tokenize(tokens.lower())
bigrams_pre = ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol="_", right_pad_symbol="_")
bigrams = []
bigrams_first_words = []
for b in bigrams_pre:
    bigrams.append(" ".join(b))
    bigrams_first_words.append(b[0])
print("Sentence bigrams:", bigrams, "with first word: ", bigrams_first_words)

# Calculate Probabilities
english_prob = get_language_probability(bigrams, bigrams_first_words, bigram_english_dict, unigram_english_dict)
french_prob = get_language_probability(bigrams, bigrams_first_words, bigram_french_dict, unigram_french_dict)
italian_prob = get_language_probability(bigrams, bigrams_first_words, bigram_italian_dict, unigram_italian_dict)

def get_normalized_probabilities(list_of_probabilities):
    sum_of_probabilities = sum(list_of_probabilities)
    result = []
    for probability in list_of_probabilities:
        result.append(probability / sum_of_probabilities)
    return result

probabilities = [english_prob, french_prob, italian_prob]
normalized_probabilities = get_normalized_probabilities(probabilities)

print("Relative Probabilities")
print("English: ", round(normalized_probabilities[0] * 100, 2), "%", sep="")
print("French: ", round(normalized_probabilities[1] * 100, 2), "%", sep="")
print("Italian: ", round(normalized_probabilities[2] * 100, 2), "%", sep="")
