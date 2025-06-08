import nltk
import sys
import os

def setup_nltk_data():
    """Download all required NLTK data with error handling"""
    
    print("🏁 F1 Racer AI Agent - NLTK Setup")
    print("=" * 50)
    print("📥 Downloading required NLTK data...")
    print()
    
    # Resources to download (with both old and new names for compatibility)
    resources = [
        # Tokenizers (both old and new versions)
        ('punkt', 'Punkt sentence tokenizer'),
        ('punkt_tab', 'Punkt sentence tokenizer (newer version)'),
        
        # Sentiment analysis
        ('vader_lexicon', 'VADER sentiment lexicon'),
        
        # Stopwords
        ('stopwords', 'Stopwords corpus'),
        
        # POS taggers (both old and new versions)
        ('averaged_perceptron_tagger', 'POS tagger'),
        ('averaged_perceptron_tagger_eng', 'POS tagger (newer version)'),
        
        # Named entity chunker
        ('maxent_ne_chunker', 'Named entity chunker'),
        
        # Word corpus
        ('words', 'Words corpus'),
        
        # Additional useful resources
        ('brown', 'Brown corpus'),
        ('wordnet', 'WordNet'),
    ]
    
    successful_downloads = 0
    failed_downloads = []
    
    for resource, description in resources:
        try:
            print(f"📦 Downloading {resource} ({description})...")
            nltk.download(resource, quiet=True)
            print(f"   ✅ Successfully downloaded {resource}")
            successful_downloads += 1
        except Exception as e:
            print(f"   ⚠️  Failed to download {resource}: {e}")
            failed_downloads.append((resource, str(e)))
        
        print()
    
    # Summary
    print("=" * 50)
    print("📊 Download Summary:")
    print(f"   ✅ Successful: {successful_downloads}")
    print(f"   ❌ Failed: {len(failed_downloads)}")
    
    if failed_downloads:
        print("\n⚠️  Failed downloads:")
        for resource, error in failed_downloads:
            print(f"   - {resource}: {error}")
        print("\n💡 Note: Some failures are normal due to version differences.")
        print("   The agent includes fallback mechanisms for missing resources.")
    
    # Test basic functionality
    print("\n🧪 Testing NLTK functionality...")
    test_nltk_functionality()
    
    print("\n✅ NLTK setup complete!")
    print("💡 You can now run the F1 agent with: python run_agent.py")

def test_nltk_functionality():
    """Test basic NLTK functionality"""
    
    test_text = "This is a great day for Formula 1 racing!"
    
    # Test tokenization
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        tokens = word_tokenize(test_text)
        sentences = sent_tokenize(test_text)
        print(f"   ✅ Tokenization working (tokens: {len(tokens)}, sentences: {len(sentences)})")
    except Exception as e:
        print(f"   ⚠️  Tokenization test failed: {e}")
    
    # Test sentiment analysis
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(test_text)
        print(f"   ✅ Sentiment analysis working (compound: {sentiment['compound']:.2f})")
    except Exception as e:
        print(f"   ⚠️  Sentiment analysis test failed: {e}")
    
    # Test POS tagging
    try:
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag
        tokens = word_tokenize(test_text)
        pos_tags = pos_tag(tokens)
        print(f"   ✅ POS tagging working ({len(pos_tags)} tags)")
    except Exception as e:
        print(f"   ⚠️  POS tagging test failed: {e}")
    
    # Test stopwords
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        print(f"   ✅ Stopwords working ({len(stop_words)} stopwords loaded)")
    except Exception as e:
        print(f"   ⚠️  Stopwords test failed: {e}")

def check_existing_data():
    """Check what NLTK data is already available"""
    
    print("🔍 Checking existing NLTK data...")
    
    # Check NLTK data path
    data_path = nltk.data.find('tokenizers')
    if data_path:
        print(f"   📁 NLTK data path: {data_path}")
    else:
        print("   📁 NLTK data path not found")
    
    # Try to check what's available
    try:
        import nltk.data
        print("   📦 Checking available resources...")
        
        resources_to_check = [
            'tokenizers/punkt',
            'tokenizers/punkt_tab',
            'vader_lexicon',
            'corpora/stopwords',
            'taggers/averaged_perceptron_tagger'
        ]
        
        for resource in resources_to_check:
            try:
                nltk.data.find(resource)
                print(f"      ✅ {resource} - Available")
            except LookupError:
                print(f"      ❌ {resource} - Missing")
    
    except Exception as e:
        print(f"   ⚠️  Could not check resources: {e}")
    
    print()

def main():
    """Main setup function"""
    
    try:
        # Check existing data first
        check_existing_data()
        
        # Download required data
        setup_nltk_data()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Make sure you have internet connection")
        print("   2. Try running as administrator/sudo if permission errors")
        print("   3. Check firewall/antivirus settings")
        print("   4. Try running: python -m nltk.downloader popular")
        sys.exit(1)

if __name__ == "__main__":
    main()
