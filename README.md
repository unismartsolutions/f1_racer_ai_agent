# F1 Racer AI Agent 🏎️

A sophisticated AI agent that simulates a Formula 1 racer's persona using advanced NLP techniques including **NLTK**, **spaCy**, and **Transformers**. The agent generates contextually-aware, natural responses that capture the authentic essence of a professional F1 driver's personality, emotions, and communication style.

## 🌟 Advanced NLP Features

### Core NLP Capabilities
- **NLTK Integration**: Sentiment analysis, tokenization, POS tagging, and named entity recognition
- **spaCy Processing**: Advanced linguistic analysis, dependency parsing, and entity extraction  
- **Transformer Models**: GPT-2 based text generation for creative, contextual responses
- **TextBlob Enhancement**: Linguistic corrections and additional text processing
- **Dynamic Content Generation**: Non-robotic responses using sophisticated NLP techniques

### AI-Powered Features
1. **Intelligent Sentiment Analysis**: NLTK's VADER for understanding fan comment emotions
2. **Contextual Entity Recognition**: spaCy for identifying people, places, and racing terms
3. **Advanced Text Generation**: Transformer-based creative content creation
4. **Linguistic Enhancement**: Real-time text improvement and F1-specific terminology integration
5. **Mood Intelligence**: NLP-driven emotional state analysis and adaptation

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ 
- Docker (recommended)
- At least 2GB RAM (for transformer models)

### Using Docker (Recommended - Includes All NLP Dependencies)

1. **Clone or create the repository**:
```bash
mkdir f1_racer_ai_agent
cd f1_racer_ai_agent
```

2. **Build the Docker container** (includes automatic NLP setup):
```bash
docker build -t f1-racer-agent .
```

3. **Run the interactive agent**:
```bash
docker run -it f1-racer-agent
```

### Manual Installation with NLP Libraries

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download required NLP models and data**:

   **Option A: Automated Setup (Recommended)**
   ```bash
   python setup_nltk.py
   python -m spacy download en_core_web_sm
   ```

   **Option B: Manual Setup**
   ```bash
   # Download spaCy model
   python -m spacy download en_core_web_sm
   
   # Download NLTK data manually
   python -c "
   import nltk
   resources = ['punkt', 'punkt_tab', 'vader_lexicon', 'stopwords', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'words']
   for resource in resources:
       try:
           nltk.download(resource, quiet=True)
           print(f'✅ Downloaded {resource}')
       except:
           print(f'⚠️  Could not download {resource}')
   "
   ```

3. **Run the agent**:
```bash
python run_agent.py  # Interactive mode
python f1_agent.py   # Demo with NLP features
```

### 🔧 Troubleshooting NLTK Issues

If you encounter NLTK errors like `punkt_tab not found`, try these solutions:

**Solution 1: Use the setup script**
```bash
python setup_nltk.py
```

**Solution 2: Manual NLTK download**
```bash
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import nltk; nltk.download('vader_lexicon')"
python -c "import nltk; nltk.download('stopwords')"
```

**Solution 3: Download all NLTK data**
```bash
python -c "import nltk; nltk.download('all')"
```

**Solution 4: Force download popular packages**
```bash
python -m nltk.downloader popular
```

**Note**: The agent includes comprehensive fallback mechanisms, so it will work even if some NLTK resources fail to download, though with reduced NLP capabilities.

## 📱 Usage Examples

### Interactive Mode
The `run_agent.py` script provides a user-friendly interface:

```
🏁 Welcome to the F1 Racer AI Agent!
Enter racer name (default: Alex Driver): Max Lightning
Enter team name (default: Racing Team): Thunder Racing

🏎️  F1 Agent Actions:
1. Generate Status Post
2. Reply to Fan Comment
3. Mention Teammate/Competitor
...
```

### Programmatic Usage
```python
from f1_agent import F1RacerAgent, RaceStage, RaceResult

# Create agent
agent = F1RacerAgent("Lewis Hamilton", "Mercedes AMG")

# Update context
agent.update_context(
    stage=RaceStage.POST_RACE,
    last_result=RaceResult.WIN,
    position=1,
    circuit_name="Silverstone",
    race_name="British Grand Prix"
)

# Generate content
post = agent.speak("win")
reply = agent.reply_to_comment("Amazing drive!")
thoughts = agent.think()
```

## 🎭 Enhanced Example Outputs

### After a Victory (NLP-Enhanced)
**Input Context**: Win at British Grand Prix, P1  
**NLP Analysis**: High positive sentiment, victory-related entities  
**Generated Post**:
```
YES! Absolutely buzzing! Massive thanks to the incredible crew for this 
phenomenal car. We were extracting pace all race long and it paid off! 🏆 
#BritishGP #Victory #Champions #F1
```

### After a Difficult Race (Sentiment-Aware)
**Input Context**: DNF at Monaco Grand Prix  
**NLP Analysis**: Negative sentiment with resilience keywords  
**Generated Post**:
```
Absolutely gutted. Not the result we wanted today. Gave everything out there, 
but the tire degradation wasn't quite there. The team will analyze and we'll 
bounce back stronger next time. Thanks for the support. 💪 #MonacoGP #NeverGiveUp #Resilience
```

### During Practice Session (Technical Focus)
**Input Context**: FP2 at Spa-Francorchamps  
**NLP Analysis**: Technical terminology, preparation sentiment  
**Generated Post**:
```
Productive FP2. Finding the limit step by step with the aerodynamics package. 
Engineering is giving me brilliant feedback on the differential settings. 
Let's keep extracting pace! ⚡ #BelgianGP #FP2 #TeamWork #Setup
```

### NLP-Enhanced Fan Interactions

**Fan Comment**: "Amazing overtake in turn 1! How did you see that gap?"  
**NLP Analysis**: Positive sentiment (0.89), racing-specific entities detected  
**Agent Reply**:
```
Great question! Always happy to connect with the fans. 😊 The gap opened up 
perfectly - it's all about timing and reading the other driver's line! 🏁
```

**Fan Comment**: "Tough luck today, but we believe in you and the team!"  
**NLP Analysis**: Supportive sentiment (0.34), encouragement keywords  
**Agent Reply**:
```
Thanks for sticking with us! With fans like you, we never give up. ❤️ 
That support means everything and keeps us pushing harder! 💪
```

### Contextual Mentions (Entity-Aware)
**Mentioning Teammate** (Positive context + NLP enhancement):
```
Great job @CarlosSpeed! Absolutely phenomenal to have such a strong teammate. 
Working together makes Thunder Racing unstoppable. The race craft was on point today! 🤝 
#TeamThunder #United #Partnership #F1
```

**Mentioning Competitor** (Competitive context + sentiment analysis):
```
Ready for the battle with @MaxVerstappen! Should be lightning-fast racing tomorrow. 
The aerodynamics package feels dialed in perfectly. May the best driver win! ⚔️ 
#F1 #Competition #BringItOn
```

## 🏗️ Architecture & Design

### Core Classes

#### `F1RacerAgent`
Main agent class with methods:
- `speak()`: Generate dynamic text content
- `reply_to_comment()`: Create contextual replies
- `mention_teammate_or_competitor()`: Generate mention posts
- `simulate_like_action()`: Simulate social media likes
- `think()`: Generate internal thoughts/analysis
- `update_context()`: Update race weekend awareness

#### `RaceContext`
Data structure maintaining:
- Current race stage (Practice, Qualifying, Race, Post-race)
- Session type (FP1, FP2, FP3, Q1, Q2, Q3, Race)
- Circuit and race information
- Recent results and current mood
- Racer and team details

### Dynamic Response System

The agent avoids robotic responses through:

1. **Template Variation**: Multiple templates for each context
2. **Vocabulary Pools**: Dynamic word selection based on emotional state
3. **Context Sensitivity**: Responses adapt to race stage and results
4. **Emotional Mapping**: Mood influences language choices
5. **Randomization**: Prevents repetitive outputs
6. **Recent Post Tracking**: Avoids immediate repetition

### Context Awareness Levels

1. **Race Weekend Stage**: Practice → Qualifying → Race → Post-race
2. **Session Granularity**: FP1, FP2, FP3, Q1, Q2, Q3, Race, Sprint
3. **Result Awareness**: Win, Podium, Points, DNF, Crash, Mechanical
4. **Emotional State**: Positive, Negative, Focused, Excited
5. **Team Integration**: References team members and collective effort

## 🛠️ Advanced Technical Implementation

### NLP Architecture

#### Core NLP Stack
- **NLTK Integration**: 
  - VADER sentiment analysis for emotional intelligence
  - POS tagging for linguistic structure analysis
  - Named entity recognition for contextual understanding
  - Tokenization and stopword filtering for text processing

- **spaCy Processing**:
  - Advanced linguistic analysis with en_core_web_sm model
  - Dependency parsing for sentence structure understanding
  - Entity extraction for person, place, and organization detection
  - Linguistic feature analysis for enhanced context awareness

- **Transformer Models**:
  - DistilGPT-2 for creative text generation
  - Temperature-controlled sampling for varied outputs
  - Context-aware prompt engineering for F1-specific content
  - Fallback mechanisms for robust performance

#### NLP-Enhanced Agent Components

```python
class NLPProcessor:
    """Advanced NLP processor using NLTK, spaCy, and Transformers"""
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]
    def extract_entities(self, text: str) -> List[Tuple[str, str]]
    def extract_keywords(self, text: str) -> List[str]
    def generate_creative_text(self, prompt: str, max_length: int) -> str
    def enhance_text_with_context(self, base_text: str, context_keywords: List[str]) -> str
```

### Dynamic Response Generation System

1. **Multi-Layer Content Creation**:
   - Template-based foundation with NLP enhancement
   - Sentiment-driven vocabulary selection
   - Transformer-assisted creative generation
   - Contextual linguistic refinement

2. **Intelligent Context Processing**:
   - NLP-based mood analysis using sentiment scoring
   - Entity-aware content personalization
   - Keyword-driven thematic consistency
   - Historical interaction learning

3. **Advanced Fan Interaction**:
   - Real-time sentiment analysis of fan comments
   - Entity extraction for personalized responses
   - Context-appropriate reply generation
   - Emotional intelligence in engagement

### Dependencies and Models
- **Core**: Python 3.7+ with comprehensive NLP stack
- **Required**: NLTK (3.8+), spaCy (3.4+), Transformers (4.21+), PyTorch
- **Models**: en_core_web_sm (spaCy), DistilGPT-2 (Transformers)
- **Memory**: ~2GB RAM recommended for optimal transformer performance

### Performance Optimization
- Efficient model loading with fallback mechanisms
- Caching strategies for repeated NLP operations
- Memory-optimized transformer inference
- Real-time processing suitable for interactive use

## 🐳 Docker Configuration

The application includes a complete Docker setup:

**Dockerfile Features**:
- Python 3.11 slim base image
- Minimal system dependencies
- Optimized for container deployment
- Interactive mode support
- Future web interface ready

**Build Commands**:
```bash
# Build the image
docker build -t f1-racer-agent .

# Run interactively
docker run -it f1-racer-agent

# Run with custom command
docker run -it f1-racer-agent python f1_agent.py
```

## 🧪 Comprehensive Testing with NLP Validation

The agent includes extensive testing for both traditional functionality and advanced NLP features:

### Core Functionality Tests
1. **Agent Initialization**: Basic setup and configuration validation
2. **Context Management**: Race weekend progression and state tracking
3. **Response Generation**: Dynamic content creation across all scenarios
4. **Social Media Simulation**: All interaction types and action simulation

### Advanced NLP Tests
5. **Sentiment Analysis Validation**: VADER sentiment scoring accuracy
6. **Entity Recognition Testing**: spaCy entity extraction verification  
7. **Keyword Extraction**: NLTK-based keyword identification
8. **Enhanced Reply Analysis**: NLP-driven fan interaction intelligence
9. **Mood Analysis with NLP**: Sentiment-based emotional state management
10. **Creative Text Generation**: Transformer model output quality

### Testing Commands
```bash
# Run comprehensive test suite (includes NLP tests)
python test_agent.py

# Docker-based testing
docker run f1-racer-agent python test_agent.py

# Test specific NLP features
python -c "from test_agent import F1AgentTester; 
           tester = F1AgentTester(); 
           tester.test_nlp_capabilities()"
```

### Demo Scenarios with NLP Analysis
1. **Practice Session at Monaco**: FP2 context with technical focus and NLP sentiment tracking
2. **Victory at Silverstone**: Post-race celebration with positive sentiment analysis
3. **DNF at Spa**: Disappointment with NLP-enhanced resilience messaging
4. **Podium at Monza**: Positive result with contextual entity recognition

### Interactive Testing Features
- **Real-time NLP Analysis**: Live sentiment and entity extraction during interactions
- **Context Configuration**: Manual race weekend setup with NLP mood analysis
- **Fan Comment Simulation**: Test various comment types with sentiment analysis
- **Complete Race Weekend**: Full progression with NLP-enhanced state transitions

## 🎯 Design Challenges & Solutions

### Challenge 1: Avoiding Robotic Responses
**Solution**: Multi-layered dynamic generation system
- Template variations with contextual substitution
- Emotional vocabulary pools
- Random selection with recent post tracking
- Context-sensitive phrase generation

### Challenge 2: Authentic F1 Persona
**Solution**: Comprehensive racing vocabulary and sentiment mapping
- Authentic racing terminology integration
- Emotional state modeling based on results
- Team-focused language patterns
- Professional athlete communication style

### Challenge 3: Contextual Awareness
**Solution**: Structured state management system
- Hierarchical context tracking (weekend → session → result)
- Mood adaptation based on recent performance
- Session-specific language patterns
- Historical context influence on current responses

### Challenge 4: Social Media Authenticity
**Solution**: Platform-appropriate response generation
- Hashtag generation based on context
- Emoji integration with emotional mapping
- Mention formatting and relationship context
- Engagement pattern simulation

## 🔮 Current NLP Implementation & Future Enhancements

### ✅ Currently Implemented NLP Features
1. **NLTK Integration**: Complete sentiment analysis, tokenization, POS tagging, and NE recognition
2. **spaCy Processing**: Advanced linguistic analysis with en_core_web_sm model
3. **Transformer Integration**: DistilGPT-2 for creative text generation with fallback mechanisms
4. **Intelligent Mood Analysis**: NLP-driven emotional state detection and adaptation
5. **Enhanced Fan Interaction**: Sentiment-aware reply generation with entity recognition
6. **Dynamic Vocabulary Selection**: Context-aware word choice using linguistic analysis
7. **Creative Content Generation**: Transformer-assisted dynamic response creation

### 🚀 Potential Advanced Enhancements
1. **Larger Language Models**: Integration with GPT-3.5/4 or other large models for more sophisticated generation
2. **Multi-language Support**: International F1 audience engagement with spaCy multilingual models
3. **Real-time F1 Data Integration**: Live race data feeds for enhanced contextual awareness
4. **Advanced Learning**: Response quality improvement through user feedback analysis
5. **Voice Synthesis**: Audio response generation using neural TTS models
6. **Image Analysis**: Visual content understanding for enhanced social media interaction
7. **Predictive Context**: Race calendar integration for anticipated context changes

### 🔬 Research-Level Features
- **Few-shot Learning**: Adaptation to specific driver personalities with minimal examples
- **Conversational Memory**: Long-term interaction history with persistent personality development
- **Emotion Recognition**: Advanced emotional intelligence with nuanced state modeling
- **Style Transfer**: Adaptation to different social media platforms and communication styles
- **Multimodal Integration**: Combined text, image, and audio processing for rich interactions

---

**Enhanced with ❤️ using NLTK, spaCy, and Transformers for the Formula 1 and AI community**

*"In racing, as in NLP, it's about finding the perfect balance between precision and creativity."* - F1 Racer AI Agent
