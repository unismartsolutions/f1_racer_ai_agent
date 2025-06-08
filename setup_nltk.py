import nltk

print('Downloading NLTK data...')
try:
    nltk.download('punkt_tab', quiet=True)
    print('✅ punkt_tab downloaded')
except:
    print('⚠️ punkt_tab failed')

try:
    nltk.download('punkt', quiet=True) 
    print('✅ punkt downloaded')
except:
    print('⚠️ punkt failed')

try:
    nltk.download('vader_lexicon', quiet=True)
    print('✅ vader_lexicon downloaded') 
except:
    print('⚠️ vader_lexicon failed')

try:
    nltk.download('stopwords', quiet=True)
    print('✅ stopwords downloaded')
except:
    print('⚠️ stopwords failed')

try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print('✅ pos tagger downloaded')
except:
    print('⚠️ pos tagger failed')
    
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    print('✅ Newer pos tagger downloaded')
except:
    print('⚠️ pos tagger failed')
    
try:
    nltk.download('maxent_ne_chunker', quiet=True)
    print('✅ Named entity chunker downloaded')
except:
    print('⚠️ Named entity chunker failed')

try:
    nltk.download('words', quiet=True)
    print('✅ Words corpus downloaded')
except:
    print('⚠️ Words corpus failed')

try:
    nltk.download('brown', quiet=True)
    print('✅ Brown corpus downloaded')
except:
    print('⚠️ Brown corpus failed')

try:
    nltk.download('wordnet', quiet=True)
    print('✅ WordNet downloaded')
except:
    print('⚠️ WordNet failed')

print('NLTK setup complete!')