import nltk

# Download all required NLP packages to support the agent's functionality

print('Downloading NLTK data...')

# Download  each package and handle exceptions with logging to terminal for troubleshooting. Required since NLTK package compatibility can be tricky.

# Use the punkt tokenizer for sentence splitting.
# punkt_tab: Older version of the Punkt tokenizer for sentence splitting.
try:
    nltk.download('punkt_tab', quiet=True)
    print('✅ punkt_tab downloaded')
except:
    print('⚠️ punkt_tab failed')

# punkt: Newer Tokenizer for splitting sentences and words.
try:
    nltk.download('punkt', quiet=True) 
    print('✅ punkt downloaded')
except:
    print('⚠️ punkt failed')

# Sentiment analysis lexicon for VADER.
try:
    nltk.download('vader_lexicon', quiet=True)
    print('✅ vader_lexicon downloaded') 
except:
    print('⚠️ vader_lexicon failed')

# Common stop words for filtering.
try:
    nltk.download('stopwords', quiet=True)
    print('✅ stopwords downloaded')
except:
    print('⚠️ stopwords failed')

# Part-of-speech (POS) tagger.
try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print('✅ pos tagger downloaded')
except:
    print('⚠️ pos tagger failed')
    
# Newer English POS tagger.
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    print('✅ Newer pos tagger downloaded')
except:
    print('⚠️ pos tagger failed')
    
# Named entity chunker for entity recognition.
try:
    nltk.download('maxent_ne_chunker', quiet=True)
    print('✅ Named entity chunker downloaded')
except:
    print('⚠️ Named entity chunker failed')

# English word list for various NLP tasks.
try:
    nltk.download('words', quiet=True)
    print('✅ Words corpus downloaded')
except:
    print('⚠️ Words corpus failed')

# Brown corpus for training and evaluation.
try:
    nltk.download('brown', quiet=True)
    print('✅ Brown corpus downloaded')
except:
    print('⚠️ Brown corpus failed')

# Lexical database for English, used for lemmatization and synonyms.
try:
    nltk.download('wordnet', quiet=True)
    print('✅ WordNet downloaded')
except:
    print('⚠️ WordNet failed')

print('NLTK setup complete!')