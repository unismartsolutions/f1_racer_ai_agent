# Submission F1 Racer AI Agent
## Created by Tevin Richard

## Overview 
This is my basic attempt at creating the requested  Formula 1 racer simulation system that generates authentic social media content and fan interactions using advanced Natural Language Processing capabilities. The agent was designed and developed as per the assessment criteria which includes contextual awareness of race weekends, understands sentiment in fan comments, and most importantly the ability to produce dynamic responses that capture the personality of a professional F1 driver.

I have assumed relevant configuration settings for the agent based on common F1 racing concepts as you will see in your interactions with my agent.

**PS: I loved the Disney Cars movies growing up so I decided to work the famous Lightening McQueen into my agent!**

### Key Capabilities


- **Agent Context Configuration and real time adaptation**: Initially set the context of the adjust to configure it's persona which adjusts the personality and messaging based on current race weekend stage and recent performance

- **Contextual Content Generation:**  Creates authentic racing content that adapts to the race weekend situation. .

- **Intelligent Fan Interaction:** Understands the sentiment and intent behind fan comments to deliver appropriate responses. The agent can detect questions ("How did you manage that overtake?"), positive praise ("Amazing drive!"), criticism, or neutral observations, then select response templates that match the fan's emotional tone and topic. 

- **Dynamic Response System:** Prevents repetitive content by using a diverse response library with multiple templates for each situation. The agent tracks previously generated content and selects different phrasings, emojis, and hashtags to keep interactions fresh and authentic.

- **Multi-Modal Communication:** Handles various social media activities including:

   - Status updates for different race contexts (pre-race, post-qualifying, victory celebrations)
   - Personalized replies to fan comments with appropriate emotional tone
   - Mentions of teammates and competitors with context-appropriate messages
   - Simulated "liking" of content with sentiment analysis
   - Internal "thoughts" that reveal the driver's perspective
   - Real-time Context Adaptation: Shifts personality and messaging based on the current situation. T

- **Personality Consistency:** Maintains the authentic voice of a professional F1 driver across all interactions, balancing competitive spirit, technical knowledge, and fan appreciation while avoiding controversial topics or inappropriate responses.

- **Interactive Demo Capabilities:**  Includes pre-configured scenarios to demonstrate different racing situations, allowing users to experience the full range of agent responses without needing to manually set up each race context.


## Architecture & Modules

### Core Components

#### 1. **F1RacerAgent** (`f1_agent.py`)
The main agent class that orchestrates all functionality:
- Maintains race context and driver state
- Coordinates with NLP processor for text analysis
- Manages response generation and content libraries
- Tracks interaction history and content uniqueness

#### 2. **NLPProcessor** (`f1_agent.py`)
Handles all natural language processing operations:
- **NLTK Integration**: Sentiment analysis using VADER, tokenization, POS tagging
- **spaCy Processing**: Entity recognition, dependency parsing, linguistic analysis
- **Transformers Support**: GPT-2 based text generation with fallback mechanisms
- **Robust Error Handling**: Graceful degradation when NLP models are unavailable

#### 3. **RaceContext** (`f1_agent.py`)
State management system that tracks:
- Current race weekend stage (Practice → Qualifying → Race → Post-race)
- Session details (Free Practise (FP), Qualifiers (Q), Race)
- Recent results and performance data
- Circuit information and race metadata
- Driver mood and emotional state

#### 4. **Interactive Interface** (`run_agent.py`)
Command-line interface providing:
- Agent configuration and setup
- Context management and updates
- Interactive content generation
- Demo scenarios and race weekend simulation

### Supporting Systems

#### 5. **Response Libraries** (`f1_agent.py`)
Structured content templates for:
- Race result celebrations and disappointments. These have been configured so that they canb be edited, updated or ammended as required.
- Practice and qualifying session updates
- Fan interaction (like) reply (comments)  frameworks
- Mention templates for teammates and competitors

#### 6. **Testing Suite** (`test_agent.py`)
Comprehensive validation system covering:
- NLP functionality and accuracy
- Content generation quality and uniqueness
- Context awareness and state transitions
- Performance benchmarking and error handling
- This module helps perform a quick diagnostic check on the agent

## Simplified Application Flow

```
1. User Interaction
   ↓
2. Interface Layer (CLI/Docker)
   ↓
3. Agent Coordination Layer - get context from user via the CLI/Docker interface
   ↓
4. Context Analysis & State Management - Sets the context/persona of the agent
   ↓
5. NLP Processing (Sentiment, Keywords, Entities) - interactions with the agent
   ↓
6. Content Generation (Templates + NLP Enhancement) - outputs from the agent
   ↓
7. Response Delivery & History Tracking 
```

### Detailed Process Flow

1. **Context Setup**: Agent receives race weekend information, driver details, and current stage
2. **Content Request**: User requests specific content type or interaction (post, reply, mention, etc.)
3. **NLP Analysis**: If responding to comments, analyzes sentiment and extracts keywords
4. **Template Selection**: Chooses appropriate response templates based on context and mood
5. **Content Enhancement**: Applies NLP-driven vocabulary selection and personalization
6. **Hashtag Generation**: Creates contextually relevant hashtags for social media posts
7. **Response Delivery**: Returns polished content with appropriate formatting and emojis

### Diagramatic representation of Agent operation:
![alt text](image.png)

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- 2GB+ RAM recommended for optimal NLP model performance
- Internet connection for initial model downloads
- Docker installation (if running the Docker Container build method)

I have include two options for getting started with my F1 agent: 

1. Docker method - This is quick and can be set up with a few minutes with simple 2 bash commands (assuming you aldready have docker engine installed on your PC)

2. Manual setup method - This method involves setting up a virtual environment on your PC and configuring that environment for running my agent. I have made this simpler by adding the "setup.sh" bash script to my repo so that you can configure the environment and all required packages with one bash terminal prompt. 

NB: I developed my submission on a Windows PC.

### Quick Start with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/unismartsolutions/f1_racer_ai_agent.git
cd f1-racer-agent

# Build the docker container - if you get an error, please chekc that docker enginer is running
docker build -t f1-racer-agent .

# Run the container to start the application
docker run -it f1-racer-agent
```

### Manual Installation

```bash
# 1. Run the setup script - assuming you will be using a Bash Terminal. 
# This scrip will create the virtual environment, activate it, install requirements, run the python script to download all required NLTK packages and download the required SpaCy model
. setup.sh

# 2. Run the agent

# There are two scripts that may be run:

python run_agent.py  # Interactive mode - This mode allows the user to interact with the F1 agent using the CLI interface in the bash terminal

python f1_agent.py   # Demo mode - This mode which the mail script will simply print the demo scenarios to the user. THe user may cycle through the scenarios by pressing "Enter" until all 4 scenarios are complete.
```

### Troubleshooting Setup Issues

If you encounter NLTK download errors: 

You may or may not experience some issues but every now and then I get errors with specific NLTK models. I have provided a manual troublehsootin guide below incase you have any issues:

```bash
python setup_nltk.py
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import nltk; nltk.download('vader_lexicon')"
```

For spaCy model issues:
```bash
python -m spacy download en_core_web_sm --force-reinstall
```

## Usage Examples - How to use the F1 Agent

### Interactive Mode

```bash
python run_agent.py
```

1. Configure driver name and team - You can just press enter to use my preconfigured Defaults 
2. Set race weekend context (circuit, stage, session) - Same as above just press enter for fast config with defaults
3. Choose from available actions:
   - Generate status posts
   - Reply to fan comments
   - Create mentions
   - Simulate a Like Action
   - Update race context - If you want to reconfigure the agent and change persona/behabviour
   - View Agent Thoughts - Get a view of the agent thinking process to debug or understand the response generated
   - View agent infirmation - Lets you see the details of the agent based on previous interactions
   - Run Demo Scenarios - Invoke the sample scenarios to understand the agent behavior and output
   - Quick Race Weekend Simulation - Simulate a race weekend

### Programmatic Usage

```python
from f1_agent import F1RacerAgent, RaceStage, RaceResult

# Create and configure agent
agent = F1RacerAgent("Lewis Hamilton", "Mercedes AMG")

# Set race context
agent.update_context(
    stage=RaceStage.POST_RACE,
    last_result=RaceResult.WIN,
    position=1,
    circuit_name="Silverstone",
    race_name="British Grand Prix"
)

# Generate content
victory_post = agent.speak("win")
fan_reply = agent.reply_to_comment("Amazing drive today!")
internal_thoughts = agent.think()
```

### Sample Outputs

**Victory Post:**
```
YES! Absolutely buzzing! Massive thanks to the incredible crew for this 
phenomenal car. We were extracting pace all race long and it paid off! 🏆 
#BritishGP #Victory #P1 #F1 #TeamMercedes
```

**Fan Interaction:**
```
Fan: "How did you manage that incredible overtake in turn 1?"
Agent: "Great question! Always happy to connect with curious fans! 
The gap opened up perfectly - all about timing and reading the other 
driver's line! 🤔😊"
```

## Technical Implementation Details

### NLP Pipeline Architecture

1. **Text Preprocessing**: Tokenization, normalization, and cleaning
2. **Sentiment Analysis**: VADER lexicon-based scoring with contextual adjustments
3. **Entity Recognition**: spaCy NER for extracting racing-related entities
4. **Keyword Extraction**: POS-tag based filtering for relevant content terms
5. **Content Generation**: Template-based foundation with NLP-enhanced vocabulary selection

### Context Awareness System

The agent maintains sophisticated awareness through:
- **Temporal Context**: Understanding of race weekend progression
- **Performance Context**: Recent results and their emotional impact
- **Social Context**: History of interactions and content patterns
- **Environmental Context**: Circuit characteristics and race-specific factors

### Content Variation Mechanisms

- **Template Rotation**: Multiple response patterns per context type
- **NLP-Enhanced Selection**: Vocabulary pools selected based on sentiment and keywords
- **Dynamic Hashtag Generation**: Context-aware social media tagging
- **Recency Tracking**: Prevents repetitive content through history analysis

## Development & Testing

### Running Tests

```bash
python test_agent.py
```

The test suite validates:
- Core functionality and error handling
- NLP processing accuracy and performance
- Content quality and uniqueness
- Context transition behavior
- Integration between all components

### Performance Metrics

- Response generation: < 0.1s average
- Content uniqueness: > 80% for repeated requests
- Sentiment analysis accuracy: High correlation with manual evaluation
- Context retention: Maintains state across complex race weekend scenarios

## File Structure

```
f1-racer-agent/
├── f1_agent.py           # Core agent implementation
├── run_agent.py          # Interactive CLI interface
├── test_agent.py         # Comprehensive test suite
├── setup_nltk.py         # NLP model setup utility
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── setup.sh            # Automated setup script
├── .gitignore          # Version control exclusions
└── README.md           # Project documentation
```

## Dependencies & Requirements

### Core NLP Libraries
- **NLTK 3.8+**: Sentiment analysis, tokenization, POS tagging
- **spaCy 3.4+**: Entity recognition, linguistic processing
- **Transformers 4.21+**: Advanced text generation capabilities
- **PyTorch**: Neural network model execution

### Supporting Libraries
- **TextBlob**: Additional text processing utilities
- **NumPy**: Numerical computations for NLP operations

### Development Tools
- **pytest**: Test framework for validation
- **Docker**: Containerization for consistent deployment

## Future Enhancements

- Web-based interface for broader accessibility
- Integration with real F1 data feeds for live context updates
- Multi-language support for international fan engagement
- Advanced personality customization and driver-specific traits
- Real-time social media platform integration
- Enhanced emotional intelligence and context understanding

## Contributing

This project follows standard Python development practices with comprehensive testing requirements. All contributions should maintain the existing code quality standards and include appropriate test coverage for new functionality.

## Mention how this can be modified for a real world application