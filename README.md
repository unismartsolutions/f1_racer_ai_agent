# F1 Racer AI Agent

AI agent that simulates a Formula 1 racer's persona using NLP libraries like NLTK, spaCy, and Transformers. Generates contextual responses that capture the personality and communication style of a professional F1 driver.

## Features

- **NLTK Integration**: Sentiment analysis, tokenization, POS tagging, named entity recognition
- **spaCy Processing**: Linguistic analysis, dependency parsing, entity extraction  
- **Transformer Models**: GPT-2 based text generation for creative responses
- **Dynamic Content Generation**: Avoids repetitive responses using NLP techniques
- **Contextual Awareness**: Understands race weekend stages and adjusts responses accordingly

## Quick Start

### Using Docker (Recommended)

1. **Build the container**:
```bash
docker build -t f1-racer-agent .
```

2. **Run the agent**:
```bash
docker run -it f1-racer-agent
```

### Manual Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download NLP models**:
```bash
python setup_nltk.py
python -m spacy download en_core_web_sm
```

3. **Run the agent**:
```bash
python run_agent.py  # Interactive mode
python f1_agent.py   # Demo mode
```

### Troubleshooting NLTK

If you get NLTK errors:

```bash
python setup_nltk.py
```

Or try:
```bash
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

### Interactive Mode

Run `python run_agent.py` and you'll be prompted to:
1. Set up racer name and team
2. Configure initial context (race stage, circuit, etc.)
3. Choose from various actions like generating posts, replying to comments, etc.

Example session:
```
F1 Racer AI Agent
Enter racer name (default: Alex Driver): Lewis Hamilton
Enter team name (default: Racing Team): Mercedes

Initial Context Setup
Select current race weekend stage:
1. Practice
2. Qualifying
3. Race
4. Post Race
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

## Example Outputs

### After a Win
```
YES! Absolutely buzzing! Massive thanks to the incredible crew for this 
phenomenal car. We were extracting pace all race long and it paid off! 🏆 
#BritishGP #Victory #Champions #F1
```

### After a DNF
```
Absolutely gutted. Not the result we wanted today. Gave everything out there, 
but the tire degradation wasn't quite there. The team will analyze and we'll 
bounce back stronger next time. Thanks for the support. 💪 #MonacoGP #NeverGiveUp
```

### Fan Interaction
**Fan Comment**: "Amazing overtake in turn 1! How did you see that gap?"  
**Agent Reply**: "Great question! Always happy to connect with the fans. 😊 The gap opened up perfectly - it's all about timing and reading the other driver's line! 🏁"

## Architecture

### Core Classes

- **F1RacerAgent**: Main agent with `speak()`, `reply_to_comment()`, `mention_teammate_or_competitor()`, etc.
- **RaceContext**: Maintains race weekend state (stage, session, results, mood)
- **NLPProcessor**: Handles NLTK, spaCy, and Transformers processing

### Context Awareness

The agent tracks:
- Race weekend stage (Practice → Qualifying → Race → Post-race)
- Session details (FP1, FP2, Q1, Q2, Q3, Race, Sprint)
- Recent results and current mood
- Circuit and team information

### NLP Features

- **Sentiment Analysis**: VADER sentiment scoring for mood detection
- **Entity Recognition**: spaCy entity extraction for contextual understanding
- **Text Generation**: DistilGPT-2 for creative content with fallbacks
- **Keyword Extraction**: NLTK-based keyword identification

## Dependencies

- Python 3.7+
- NLTK (3.8+) 
- spaCy (3.4+) with en_core_web_sm model
- Transformers (4.21+)
- PyTorch
- TextBlob

Recommended: 2GB+ RAM for optimal transformer performance

## Testing

Run the test suite:
```bash
python test_agent.py
```

This tests all functionality including NLP features and generates a comprehensive report.

## Docker

The included Dockerfile sets up the complete environment:

```bash
# Build
docker build -t f1-racer-agent .

# Run interactively  
docker run -it f1-racer-agent

# Run demo
docker run -it f1-racer-agent python f1_agent.py
```

## Files

- `f1_agent.py` - Main agent implementation
- `run_agent.py` - Interactive interface
- `test_agent.py` - Comprehensive test suite
- `setup_nltk.py` - NLTK data download helper
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container setup

## Implementation Notes

The agent uses multiple layers of text generation:
1. Template-based foundation
2. NLP-enhanced vocabulary selection
3. Sentiment-driven mood adaptation
4. Contextual hashtag generation
5. Final linguistic processing

Response variety is achieved through:
- Multiple templates per context
- Dynamic vocabulary pools
- NLP-based enhancements
- Randomization with recent post tracking

The system includes comprehensive fallbacks so it works even if some NLP components fail to load.