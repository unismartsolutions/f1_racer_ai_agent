import random
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# NLP Libraries as specified in requirements
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import spacy
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
from textblob import TextBlob
import numpy as np

# Download required NLTK data with fallback for different NLTK versions
def download_nltk_data():
    """Download NLTK data with version compatibility"""
    
    # Essential downloads with fallbacks for different NLTK versions
    nltk_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),  # Newer NLTK versions
        ('vader_lexicon', 'vader_lexicon'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),  # Newer versions
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words')
    ]
    
    for resource_path, resource_name in nltk_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                print(f"📥 Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not download {resource_name}: {e}")
                continue

# Download NLTK data
download_nltk_data()

class RaceStage(Enum):
    """Enum for different race weekend stages"""
    PRACTICE = "practice"
    QUALIFYING = "qualifying"
    RACE = "race"
    POST_RACE = "post_race"

class SessionType(Enum):
    """Enum for specific session types"""
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    RACE = "Race"
    SPRINT = "Sprint"

class RaceResult(Enum):
    """Enum for race results"""
    WIN = "win"
    PODIUM = "podium"
    POINTS = "points"
    DNF = "dnf"
    CRASH = "crash"
    MECHANICAL = "mechanical"
    DISAPPOINTING = "disappointing"

@dataclass
class RaceContext:
    """Data class to store race weekend context"""
    stage: RaceStage
    session_type: Optional[SessionType]
    circuit_name: str
    race_name: str
    last_result: Optional[RaceResult]
    position: Optional[int]
    team_name: str
    racer_name: str
    mood: str  # positive, neutral, negative, excited, focused

class NLPProcessor:
    """Advanced NLP processor using NLTK, spaCy, and Transformers"""
    
    def __init__(self):
        # Initialize with error handling for different NLTK versions
        self.nltk_ready = False
        self.spacy_ready = False
        self.transformers_ready = False
        
        # Initialize NLTK components with fallbacks
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
            self.nltk_ready = True
            print("✅ NLTK initialized successfully")
        except Exception as e:
            print(f"⚠️  NLTK initialization warning: {e}")
            print("🔄 Using fallback sentiment analysis...")
            self.sentiment_analyzer = None
            self.stop_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Initialize spaCy (with fallback to smaller model)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_ready = True
            print("✅ spaCy initialized successfully")
        except OSError:
            print("⚠️  Warning: spaCy model 'en_core_web_sm' not found. Installing...")
            try:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_ready = True
                print("✅ spaCy model installed and loaded")
            except Exception as e:
                print(f"⚠️  Warning: Using basic spaCy model: {e}")
                try:
                    self.nlp = spacy.blank("en")
                    self.spacy_ready = True
                except Exception:
                    print("⚠️  spaCy not available, using fallback text processing")
                    self.nlp = None
                    self.spacy_ready = False
        
        # Initialize text generation pipeline (with fallback)
        try:
            self.text_generator = pipeline(
                "text-generation", 
                model="distilgpt2",
                tokenizer="distilgpt2",
                max_length=100,
                do_sample=True,
                temperature=0.8,
                pad_token_id=50256
            )
            self.transformers_ready = True
            print("✅ Transformers initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load transformer model: {e}")
            print("🔄 Using template-based text generation...")
            self.text_generator = None
            self.transformers_ready = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using NLTK's VADER with fallback"""
        if self.nltk_ready and self.sentiment_analyzer:
            try:
                return self.sentiment_analyzer.polarity_scores(text)
            except Exception as e:
                print(f"⚠️  VADER error: {e}, using fallback")
        
        # Fallback sentiment analysis
        return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Simple fallback sentiment analysis"""
        positive_words = ['amazing', 'great', 'awesome', 'fantastic', 'brilliant', 'excellent', 'good', 'win', 'victory', 'success']
        negative_words = ['bad', 'terrible', 'awful', 'disappointing', 'frustrating', 'failed', 'loss', 'crash', 'dnf']
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower) / 10
        neg_score = sum(1 for word in negative_words if word in text_lower) / 10
        
        # Simple compound score calculation
        compound = pos_score - neg_score
        compound = max(-1, min(1, compound))  # Clamp to [-1, 1]
        
        return {
            'pos': min(1.0, pos_score),
            'neg': min(1.0, neg_score),
            'neu': max(0.0, 1.0 - pos_score - neg_score),
            'compound': compound
        }
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities using spaCy with fallback"""
        if self.spacy_ready and self.nlp:
            try:
                doc = self.nlp(text)
                return [(ent.text, ent.label_) for ent in doc.ents]
            except Exception as e:
                print(f"⚠️  spaCy entity extraction error: {e}")
        
        # Fallback entity extraction (simple pattern matching)
        return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> List[Tuple[str, str]]:
        """Simple fallback entity extraction"""
        entities = []
        
        # Simple patterns for F1-related entities
        f1_circuits = ['Monaco', 'Silverstone', 'Spa', 'Monza', 'Interlagos', 'Suzuka', 'Barcelona']
        f1_teams = ['Mercedes', 'Ferrari', 'Red Bull', 'McLaren', 'Alpine', 'Aston Martin']
        
        for circuit in f1_circuits:
            if circuit in text:
                entities.append((circuit, 'GPE'))  # Geo-political entity
        
        for team in f1_teams:
            if team in text:
                entities.append((team, 'ORG'))  # Organization
        
        return entities
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using NLTK POS tagging with fallback"""
        if self.nltk_ready:
            try:
                tokens = word_tokenize(text.lower())
                pos_tags = pos_tag(tokens)
                
                # Extract nouns, adjectives, and verbs (excluding stopwords)
                keywords = [
                    word for word, pos in pos_tags 
                    if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB')) 
                    and word not in self.stop_words 
                    and len(word) > 2
                ]
                return list(set(keywords))
            except Exception as e:
                print(f"⚠️  NLTK keyword extraction error: {e}")
        
        # Fallback keyword extraction
        return self._fallback_keyword_extraction(text)
    
    def _fallback_keyword_extraction(self, text: str) -> List[str]:
        """Simple fallback keyword extraction"""
        # Simple word extraction excluding common stop words
        words = text.lower().split()
        keywords = [
            word.strip('.,!?;:"()[]') for word in words 
            if len(word) > 3 and word not in self.stop_words
        ]
        return list(set(keywords))
    
    def _safe_sentence_tokenize(self, text: str) -> List[str]:
        """Safe sentence tokenization with fallback"""
        try:
            # Try newer NLTK first
            return sent_tokenize(text)
        except Exception:
            try:
                # Try with different tokenizer
                from nltk.tokenize.punkt import PunktSentenceTokenizer
                tokenizer = PunktSentenceTokenizer()
                return tokenizer.tokenize(text)
            except Exception:
                # Fallback to simple split
                import re
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if s.strip()]
    
    def enhance_text_with_context(self, base_text: str, context_keywords: List[str]) -> str:
        """Enhance text using contextual keywords"""
        # Use TextBlob for linguistic analysis
        blob = TextBlob(base_text)
        
        # Add contextual elements based on keywords
        enhanced_parts = []
        
        for sentence in blob.sentences:
            enhanced_sentence = str(sentence)
            
            # Add F1-specific context based on detected themes
            if any(keyword in context_keywords for keyword in ['speed', 'fast', 'quick']):
                enhanced_sentence = enhanced_sentence.replace('good', 'lightning-fast')
                enhanced_sentence = enhanced_sentence.replace('great', 'absolutely flying')
            
            if any(keyword in context_keywords for keyword in ['team', 'crew', 'mechanics']):
                enhanced_sentence = enhanced_sentence.replace('team', 'incredible crew')
                enhanced_sentence = enhanced_sentence.replace('work', 'dedication')
            
            enhanced_parts.append(enhanced_sentence)
        
        return ' '.join(enhanced_parts)
    
    def generate_creative_text(self, prompt: str, max_length: int = 50) -> str:
        """Generate creative text using transformers (if available)"""
        if not self.transformers_ready or self.text_generator is None:
            # Fallback to template-based generation
            return self._fallback_generation(prompt)
        
        try:
            # Clean and prepare prompt
            clean_prompt = prompt.strip()
            if len(clean_prompt) < 5:
                clean_prompt = "As an F1 driver, I want to say that"
            
            # Generate text
            results = self.text_generator(
                clean_prompt,
                max_length=min(max_length, 100),
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            generated = results[0]['generated_text']
            
            # Extract the new content (beyond the prompt)
            if len(generated) > len(clean_prompt):
                new_content = generated[len(clean_prompt):].strip()
                
                # Clean up the generated text using safe tokenization
                sentences = self._safe_sentence_tokenize(new_content)
                if sentences:
                    # Take the first complete sentence
                    first_sentence = sentences[0]
                    
                    # Ensure it's F1-appropriate
                    if self._is_appropriate_f1_content(first_sentence):
                        return first_sentence
            
            return self._fallback_generation(prompt)
            
        except Exception as e:
            print(f"⚠️  Text generation error: {e}")
            return self._fallback_generation(prompt)
    
    def _is_appropriate_f1_content(self, text: str) -> bool:
        """Check if generated content is appropriate for F1 context"""
        inappropriate_terms = ['violence', 'hate', 'offensive', 'inappropriate']
        text_lower = text.lower()
        return not any(term in text_lower for term in inappropriate_terms)
    
    def _fallback_generation(self, prompt: str) -> str:
        """Fallback text generation method"""
        fallback_continuations = [
            "the team has been working incredibly hard.",
            "we're fully focused on maximizing our performance.",
            "every session is an opportunity to improve.",
            "the support from fans means everything to us.",
            "we'll keep pushing to extract every tenth."
        ]
        return random.choice(fallback_continuations)

class F1RacerAgent:
    """
    Advanced AI Agent that simulates a Formula 1 racer's persona using
    sophisticated NLP techniques including NLTK, spaCy, and Transformers
    """
    
    def __init__(self, racer_name: str = "Alex Driver", team_name: str = "Racing Team"):
        self.racer_name = racer_name
        self.team_name = team_name
        self.context = RaceContext(
            stage=RaceStage.PRACTICE,
            session_type=SessionType.FP1,
            circuit_name="Circuit",
            race_name="Grand Prix",
            last_result=None,
            position=None,
            team_name=team_name,
            racer_name=racer_name,
            mood="focused"
        )
        
        # Initialize NLP processor
        self.nlp_processor = NLPProcessor()
        
        # Initialize vocabulary and templates
        self._init_vocabulary()
        self._init_templates()
        
        # Track recent posts and interactions
        self.recent_posts = []
        self.interaction_history = []
        self.max_recent_posts = 10
    
    def _init_vocabulary(self):
        """Initialize F1-specific vocabulary pools with NLP enhancement"""
        
        self.f1_vocabulary = {
            "performance_terms": {
                "positive": ["pace", "grip", "balance", "downforce", "aerodynamics", "setup", 
                           "traction", "braking", "cornering", "straight-line speed"],
                "technical": ["differential", "suspension", "tire degradation", "fuel strategy",
                            "pit window", "undercut", "overcut", "DRS zones"],
                "competitive": ["qualifying", "pole position", "fastest lap", "sector times",
                              "gap management", "tire strategy", "race craft"]
            },
            
            "emotional_expressions": {
                "positive": {
                    "high": ["absolutely buzzing", "over the moon", "incredible feeling", 
                           "pure joy", "ecstatic", "phenomenal"],
                    "medium": ["really pleased", "happy with that", "solid result", 
                             "good progress", "positive vibes"],
                    "low": ["satisfied", "content", "decent", "okay with that"]
                },
                "negative": {
                    "high": ["absolutely gutted", "devastated", "heartbroken", "crushed"],
                    "medium": ["disappointed", "frustrated", "not ideal", "tough to take"],
                    "low": ["not quite there", "room for improvement", "we'll learn"]
                }
            },
            
            "team_references": ["the boys", "my crew", "the garage", "engineering", 
                              "mechanics", "pit crew", "strategists", "everyone back at the factory",
                              "the whole team", "my engineers", "the technical team"],
            
            "racing_actions": ["pushing hard", "on the limit", "extracting pace", 
                             "finding speed", "maximizing performance", "getting everything out of the car",
                             "attacking", "defending", "managing the gap", "controlling the race"]
        }
    
    def _init_templates(self):
        """Initialize response templates with NLP-enhanced structures"""
        
        self.response_templates = {
            "win": [
                "YES! {emotion_high}! Massive thanks to {team_ref} for this {adjective} car. We {racing_action} and it paid off!",
                "{emotion_high} What a race! The {performance_term} was perfect today. {team_ref} absolutely nailed the setup!",
                "P1! {emotion_high} Can't describe this feeling. {team_ref} worked so hard for this moment. We did it!",
                "VICTORY! {emotion_high} Pure {emotion_medium} right now. The {technical_term} was spot on today!"
            ],
            
            "podium": [
                "P{position}! {emotion_medium} Great result for {team_ref}. We maximized our {performance_term} today.",
                "{emotion_medium} race! P{position} feels good. The {technical_term} was working well for us.",
                "On the podium! {emotion_medium} We {racing_action} all session long. More to come!",
                "P{position}! {emotion_medium} The {performance_term} was strong. Thanks to {team_ref} for the hard work!"
            ],
            
            "disappointing": [
                "{emotion_negative} Not the result we wanted. Gave everything but the {technical_term} wasn't quite there.",
                "Tough day. {emotion_negative} Sometimes racing breaks your heart. We'll analyze and bounce back stronger.",
                "{emotion_negative} The {performance_term} didn't come together today. {team_ref} will learn from this.",
                "Not our day. {emotion_negative} We'll keep {racing_action} and come back fighting next time."
            ],
            
            "practice": [
                "Good work in {session} today. Getting comfortable with the {performance_term}. {team_ref} is working hard on the setup!",
                "{session} complete! {emotion_medium} feeling with the car. The {technical_term} is coming together nicely.",
                "Productive {session}. {racing_action} step by step. {team_ref} is giving me great feedback.",
                "{emotion_medium} session today. Finding the limit with the {performance_term}. Looking forward to tomorrow!"
            ],
            
            "qualifying": [
                "Quali day! Time to find those extra tenths. The {performance_term} feels good. Let's see what we can extract!",
                "{emotion_medium} feeling going into qualifying. {team_ref} has done great work with the {technical_term}.",
                "Q-day! {emotion_medium} Ready to extract every bit of pace. The {performance_term} is dialed in.",
                "Qualifying mode activated! {emotion_medium} preparation from {team_ref}. Time to put it all together!"
            ]
        }
    
    def update_context(self, stage: RaceStage, session_type: Optional[SessionType] = None,
                      circuit_name: str = None, race_name: str = None,
                      last_result: Optional[RaceResult] = None, position: Optional[int] = None):
        """Update the agent's contextual awareness with NLP analysis"""
        
        # Store previous context for comparison
        previous_mood = self.context.mood
        
        # Update context
        if circuit_name:
            self.context.circuit_name = circuit_name
        if race_name:
            self.context.race_name = race_name
        
        self.context.stage = stage
        self.context.session_type = session_type
        self.context.last_result = last_result
        self.context.position = position
        
        # Advanced mood analysis using NLP
        self._analyze_and_update_mood()
        
        # Log context change for learning
        context_change = {
            "timestamp": datetime.now(),
            "stage": stage.value,
            "mood_change": previous_mood != self.context.mood,
            "new_mood": self.context.mood
        }
        self.interaction_history.append(context_change)
        
        print(f"🏁 Context updated: {stage.value} at {self.context.circuit_name} (Mood: {self.context.mood})")
    
    def _analyze_and_update_mood(self):
        """Advanced mood analysis using NLP techniques"""
        
        mood_factors = []
        
        # Analyze race result impact
        if self.context.last_result:
            if self.context.last_result == RaceResult.WIN:
                mood_factors.extend(["victory", "success", "achievement", "joy"])
            elif self.context.last_result in [RaceResult.PODIUM, RaceResult.POINTS]:
                mood_factors.extend(["satisfied", "progress", "positive", "good"])
            elif self.context.last_result in [RaceResult.DNF, RaceResult.CRASH]:
                mood_factors.extend(["disappointed", "frustrated", "setback", "challenging"])
        
        # Analyze stage context
        if self.context.stage == RaceStage.PRACTICE:
            mood_factors.extend(["focused", "preparation", "development", "learning"])
        elif self.context.stage == RaceStage.QUALIFYING:
            mood_factors.extend(["intensity", "precision", "pressure", "concentrated"])
        elif self.context.stage == RaceStage.RACE:
            mood_factors.extend(["adrenaline", "competition", "battle", "determination"])
        
        # Use NLP sentiment analysis on mood factors
        mood_text = " ".join(mood_factors)
        sentiment = self.nlp_processor.analyze_sentiment(mood_text)
        
        # Determine mood based on sentiment analysis
        if sentiment['compound'] >= 0.5:
            self.context.mood = "positive"
        elif sentiment['compound'] <= -0.3:
            self.context.mood = "negative"
        elif sentiment['neu'] > 0.7:
            self.context.mood = "focused"
        else:
            self.context.mood = "neutral"
    
    def _select_contextual_vocabulary(self, category: str, intensity: str = "medium") -> str:
        """Select vocabulary using NLP-based contextual analysis"""
        
        try:
            if category in self.f1_vocabulary:
                if isinstance(self.f1_vocabulary[category], dict):
                    if intensity in self.f1_vocabulary[category]:
                        options = self.f1_vocabulary[category][intensity]
                    else:
                        # Fallback to any available subcategory
                        options = list(self.f1_vocabulary[category].values())[0]
                        if isinstance(options, dict):
                            options = list(options.values())[0]
                else:
                    options = self.f1_vocabulary[category]
                
                if isinstance(options, list) and options:
                    return random.choice(options)
            
            # Fallback
            return category.replace('_', ' ')
            
        except Exception as e:
            print(f"⚠️  Vocabulary selection error: {e}")
            return category.replace('_', ' ')
    
    def speak(self, context_type: str = "general") -> str:
        """
        Generate dynamic text using advanced NLP techniques
        """
        
        # Determine context type from current state if not specified
        if context_type == "general":
            context_type = self._determine_context_type()
        
        # Get base template
        templates = self.response_templates.get(context_type, self.response_templates["practice"])
        base_template = random.choice(templates)
        
        # Prepare context variables with NLP enhancement
        context_vars = self._prepare_context_variables()
        
        # Generate enhanced content using NLP
        enhanced_template = self._enhance_template_with_nlp(base_template, context_vars)
        
        # Apply vocabulary substitutions
        message = self._apply_vocabulary_substitutions(enhanced_template, context_vars)
        
        # Add contextual hashtags
        hashtags = self._generate_contextual_hashtags()
        
        # Apply final NLP processing
        final_message = self._apply_final_nlp_processing(message, hashtags)
        
        # Track for learning
        self._track_generated_content(final_message, context_type)
        
        return final_message
    
    def _determine_context_type(self) -> str:
        """Determine context type using NLP analysis of current state"""
        
        if self.context.last_result == RaceResult.WIN:
            return "win"
        elif self.context.last_result == RaceResult.PODIUM:
            return "podium"
        elif self.context.last_result in [RaceResult.DNF, RaceResult.CRASH, RaceResult.DISAPPOINTING]:
            return "disappointing"
        elif self.context.stage == RaceStage.PRACTICE:
            return "practice"
        elif self.context.stage == RaceStage.QUALIFYING:
            return "qualifying"
        else:
            return "practice"
    
    def _prepare_context_variables(self) -> Dict[str, str]:
        """Prepare context variables with NLP-enhanced selection"""
        
        # Determine emotional intensity based on current mood and context
        if self.context.mood == "positive":
            emotion_intensity = "high" if self.context.last_result == RaceResult.WIN else "medium"
        elif self.context.mood == "negative":
            emotion_intensity = "high" if self.context.last_result in [RaceResult.DNF, RaceResult.CRASH] else "medium"
        else:
            emotion_intensity = "medium"
        
        return {
            "racer_name": self.racer_name,
            "team_name": self.team_name,
            "circuit": self.context.circuit_name,
            "race_name": self.context.race_name,
            "session": self.context.session_type.value if self.context.session_type else "session",
            "position": str(self.context.position) if self.context.position else "3",
            "emotion_high": self._select_contextual_vocabulary("emotional_expressions", "high"),
            "emotion_medium": self._select_contextual_vocabulary("emotional_expressions", "medium"),
            "emotion_negative": self._select_contextual_vocabulary("emotional_expressions", "medium"),
            "team_ref": self._select_contextual_vocabulary("team_references"),
            "performance_term": self._select_contextual_vocabulary("performance_terms"),
            "technical_term": self._select_contextual_vocabulary("performance_terms"),
            "racing_action": self._select_contextual_vocabulary("racing_actions"),
            "adjective": self._select_contextual_vocabulary("emotional_expressions", emotion_intensity)
        }
    
    def _enhance_template_with_nlp(self, template: str, context_vars: Dict[str, str]) -> str:
        """Enhance template using NLP techniques"""
        
        # Use spaCy for linguistic analysis
        doc = self.nlp_processor.nlp(template)
        
        # Identify and enhance specific linguistic patterns
        enhanced_template = template
        
        # Add context-specific enhancements based on POS tags and dependencies
        for token in doc:
            if token.pos_ == "ADJ" and token.text in template:
                # Enhance adjectives with more specific F1 terms
                if self.context.stage == RaceStage.RACE:
                    enhanced_template = enhanced_template.replace(token.text, "race-winning")
                elif self.context.stage == RaceStage.QUALIFYING:
                    enhanced_template = enhanced_template.replace(token.text, "pole-position")
        
        return enhanced_template
    
    def _apply_vocabulary_substitutions(self, template: str, context_vars: Dict[str, str]) -> str:
        """Apply vocabulary substitutions with NLP validation"""
        
        message = template
        
        # Apply basic substitutions
        for key, value in context_vars.items():
            placeholder = "{" + key + "}"
            if placeholder in message:
                message = message.replace(placeholder, value)
        
        # Apply advanced NLP-based substitutions
        doc = self.nlp_processor.nlp(message)
        
        # Enhance based on named entities and context
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.text == self.racer_name:
                # Could add personalization here
                pass
        
        return message
    
    def _generate_contextual_hashtags(self) -> str:
        """Generate hashtags using NLP keyword extraction"""
        
        tags = []
        
        # Extract context-based keywords
        context_text = f"{self.context.stage.value} {self.context.race_name} {self.context.circuit_name}"
        keywords = self.nlp_processor.extract_keywords(context_text)
        
        # Always include race-specific tag
        if self.context.race_name != "Grand Prix":
            race_tag = f"#{self.context.race_name.replace(' ', '').replace('Grand', '').replace('Prix', 'GP')}"
            tags.append(race_tag)
        
        # Add session-specific tags
        if self.context.session_type:
            tags.append(f"#{self.context.session_type.value}")
        
        # Add mood-based tags using sentiment analysis
        mood_tags = {
            "positive": ["#Victory", "#Champions", "#Success", "#Winning"],
            "negative": ["#NeverGiveUp", "#ComeBackStronger", "#Learning", "#Resilience"],
            "focused": ["#RaceMode", "#Concentrated", "#Preparation", "#Focus"],
            "neutral": ["#F1", "#Racing", "#TeamWork", "#Motorsport"]
        }
        
        if self.context.mood in mood_tags:
            tags.append(random.choice(mood_tags[self.context.mood]))
        
        # Add general F1 tags
        general_tags = ["#F1", "#Racing", "#Speed", "#Motorsport", "#TeamWork"]
        tags.append(random.choice(general_tags))
        
        return " ".join(tags[:4])  # Limit to 4 hashtags
    
    def _apply_final_nlp_processing(self, message: str, hashtags: str) -> str:
        """Apply final NLP processing and validation"""
        
        # Combine message and hashtags
        full_message = f"{message} {hashtags}"
        
        # Use TextBlob for final linguistic corrections
        blob = TextBlob(full_message)
        
        # Apply any necessary corrections (basic)
        corrected = str(blob.correct()) if hasattr(blob, 'correct') else full_message
        
        # Ensure appropriate length and structure
        if len(corrected) > 280:  # Twitter-like limit
            sentences = self.nlp_processor.nlp(corrected).sents
            if len(list(sentences)) > 1:
                # Keep first sentence and hashtags
                first_sentence = list(sentences)[0].text
                corrected = f"{first_sentence} {hashtags}"
        
        return corrected
    
    def _track_generated_content(self, content: str, context_type: str):
        """Track generated content for learning and avoiding repetition"""
        
        entry = {
            "timestamp": datetime.now(),
            "content": content,
            "context_type": context_type,
            "mood": self.context.mood,
            "stage": self.context.stage.value
        }
        
        self.recent_posts.append(entry)
        
        # Maintain size limit
        if len(self.recent_posts) > self.max_recent_posts:
            self.recent_posts.pop(0)
    
    def reply_to_comment(self, original_comment: str) -> str:
        """Generate contextual reply using advanced NLP analysis"""
        
        # Analyze comment sentiment and content using NLP
        sentiment = self.nlp_processor.analyze_sentiment(original_comment)
        entities = self.nlp_processor.extract_entities(original_comment)
        keywords = self.nlp_processor.extract_keywords(original_comment)
        
        # Determine reply strategy based on NLP analysis
        reply_type = self._determine_reply_type(sentiment, keywords, entities)
        
        # Generate reply using NLP-enhanced templates
        reply_templates = self._get_reply_templates(reply_type)
        base_reply = random.choice(reply_templates)
        
        # Enhance reply with context
        enhanced_reply = self._enhance_reply_with_context(base_reply, keywords, sentiment)
        
        return enhanced_reply
    
    def _determine_reply_type(self, sentiment: Dict, keywords: List[str], entities: List) -> str:
        """Determine reply type using NLP analysis"""
        
        # Positive sentiment
        if sentiment['compound'] > 0.3:
            if any(word in keywords for word in ['great', 'amazing', 'awesome', 'fantastic', 'brilliant']):
                return "grateful_positive"
            else:
                return "positive_general"
        
        # Negative or supportive sentiment
        elif sentiment['compound'] < -0.1 or any(word in keywords for word in ['unlucky', 'better', 'support']):
            return "supportive_response"
        
        # Question detection
        elif any(word in keywords for word in ['what', 'how', 'when', 'where', 'why']):
            return "question_response"
        
        else:
            return "neutral_response"
    
    def _get_reply_templates(self, reply_type: str) -> List[str]:
        """Get reply templates based on type"""
        
        templates = {
            "grateful_positive": [
                "Thanks for the support! Every cheer makes a difference. 🙏",
                "Really appreciate it! The fans are what make this sport special. ❤️",
                "Thank you! Your energy gives us extra motivation. 💪",
                "Grateful for supporters like you! See you at the track! 🏁"
            ],
            "supportive_response": [
                "Thanks for sticking with us! We'll come back stronger. 💪",
                "Appreciate the support! That's what keeps us going. 🙏",
                "Thank you! With fans like you, we never give up. ❤️",
                "Your support means everything! Next one's for you. 🏁"
            ],
            "question_response": [
                "Great question! Always happy to connect with the fans. 😊",
                "Thanks for asking! Love hearing from you all. 🙏",
                "Interesting question! The fans always keep us thinking. 🤔",
                "Good point! Always learning from your perspectives. 📚"
            ],
            "neutral_response": [
                "Thanks for the message! Always great to hear from fans. 😊",
                "Appreciate you taking the time to comment! 🙏",
                "Thank you! The fan support is incredible. ❤️",
                "Thanks! Messages like this make the hard work worth it. 💪"
            ]
        }
        
        return templates.get(reply_type, templates["neutral_response"])
    
    def _enhance_reply_with_context(self, base_reply: str, keywords: List[str], sentiment: Dict) -> str:
        """Enhance reply with contextual elements"""
        
        enhanced_reply = base_reply
        
        # Add context-specific elements based on current state
        if self.context.stage == RaceStage.RACE and "race" in keywords:
            enhanced_reply += " Race day energy is unmatched!"
        elif self.context.stage == RaceStage.QUALIFYING and "qualifying" in keywords:
            enhanced_reply += " Quali day is always special!"
        
        # Add emotional context based on recent results
        if self.context.last_result == RaceResult.WIN and sentiment['compound'] > 0.5:
            enhanced_reply += " Still buzzing from that result!"
        
        return enhanced_reply
    
    def mention_teammate_or_competitor(self, person_name: str, context: str = "positive") -> str:
        """Generate mention post using NLP-enhanced content"""
        
        # Analyze the person's name for any contextual clues
        name_doc = self.nlp_processor.nlp(person_name)
        
        # Select appropriate template based on context and NLP analysis
        mention_templates = self._get_mention_templates(context)
        base_mention = random.choice(mention_templates)
        
        # Prepare context variables
        context_vars = {
            "person": person_name,
            "team_name": self.team_name,
            "adjective": self._select_contextual_vocabulary("emotional_expressions", "medium"),
            "emotion": self._select_contextual_vocabulary("emotional_expressions", "medium"),
            "performance_term": self._select_contextual_vocabulary("performance_terms")
        }
        
        # Apply substitutions
        mention = base_mention.format(**context_vars)
        
        # Add hashtags
        hashtags = self._generate_contextual_hashtags()
        
        return f"{mention} {hashtags}"
    
    def _get_mention_templates(self, context: str) -> List[str]:
        """Get mention templates based on context"""
        
        templates = {
            "positive": [
                "Great work by @{person}! {adjective} to see the level everyone brings. This is what F1 is all about!",
                "Respect to @{person} for that {performance_term}! Always pushes us to be better!",
                "Hat off to @{person}! {emotion} The competition makes us all stronger.",
                "Props to @{person}! {adjective} driving today. Love racing against the best!"
            ],
            "teammate": [
                "Team effort with @{person} today! {adjective} to have such a strong teammate.",
                "Great job @{person}! {emotion} Working together makes {team_name} stronger!",
                "@{person} bringing the {performance_term}! Together we're unstoppable.",
                "Solid work @{person}! {emotion} Team {team_name} united!"
            ],
            "competitive": [
                "Ready for the battle with @{person}! {emotion} Should be {adjective} racing.",
                "Looking forward to racing @{person} tomorrow! {adjective} competition ahead.",
                "@{person} bringing the heat! {emotion} Love these battles.",
                "Game on @{person}! {adjective} preparation done. Time to see who wants it more!"
            ]
        }
        
        return templates.get(context, templates["positive"])
    
    def simulate_like_action(self, post_content: str) -> str:
        """Simulate liking a post with NLP analysis"""
        
        # Analyze the post content
        sentiment = self.nlp_processor.analyze_sentiment(post_content)
        keywords = self.nlp_processor.extract_keywords(post_content)
        
        # Choose reaction based on content analysis
        if sentiment['compound'] > 0.5:
            reactions = ["❤️ Loved", "🏁 Absolutely loved", "💪 Fully supported", "🔥 This is fire"]
        elif sentiment['compound'] > 0.1:
            reactions = ["👍 Liked", "🙌 Supported", "💯 This", "✨ Quality content"]
        else:
            reactions = ["👍 Acknowledged", "🤝 Respect", "💙 Seen", "🏁 Noted"]
        
        action = random.choice(reactions)
        preview = post_content[:50] + "..." if len(post_content) > 50 else post_content
        
        return f"{action}: '{preview}'"
    
    def think(self) -> str:
        """Generate internal thoughts using NLP-enhanced analysis"""
        
        thoughts = []
        
        # Analyze current context using NLP
        context_keywords = [
            self.context.stage.value,
            self.context.session_type.value if self.context.session_type else "",
            self.context.mood,
            self.context.last_result.value if self.context.last_result else ""
        ]
        
        context_text = " ".join(filter(None, context_keywords))
        context_analysis = self.nlp_processor.analyze_sentiment(context_text)
        
        # Generate thoughts based on NLP analysis
        if self.context.stage == RaceStage.PRACTICE:
            thoughts.append(f"Analyzing the {self._select_contextual_vocabulary('performance_terms')} from {self.context.session_type.value if self.context.session_type else 'practice'}.")
            thoughts.append("Working with engineering to find those extra tenths.")
            
        elif self.context.stage == RaceStage.QUALIFYING:
            thoughts.append("Every tenth counts in qualifying. Mental preparation is key.")
            thoughts.append("Building confidence through each session - need to peak at the right moment.")
            
        elif self.context.stage == RaceStage.RACE:
            thoughts.append("Race day mentality: stay calm, execute the plan, capitalize on opportunities.")
            thoughts.append(f"Managing {self._select_contextual_vocabulary('performance_terms')} will be crucial today.")
        
        # Add emotional context based on sentiment analysis
        if context_analysis['compound'] > 0.3:
            thoughts.append("Feeling positive about our direction and the team's hard work.")
        elif context_analysis['compound'] < -0.3:
            thoughts.append("Challenging moments build character. We'll come back stronger.")
        
        # Add recent result reflection
        if self.context.last_result:
            if self.context.last_result == RaceResult.WIN:
                thoughts.append("That victory shows what's possible when everything clicks.")
            elif self.context.last_result in [RaceResult.DNF, RaceResult.CRASH]:
                thoughts.append("Racing taught us a lesson, but that's how we grow.")
        
        # Add motivational element using NLP generation
        motivational_prompt = f"As an F1 driver, I believe that"
        motivational_thought = self.nlp_processor.generate_creative_text(motivational_prompt, 30)
        thoughts.append(motivational_thought)
        
        return f"💭 Internal analysis: {' '.join(thoughts)}"
    
    def get_agent_info(self) -> Dict:
        """Return comprehensive agent state with NLP insights"""
        
        base_info = {
            "racer_name": self.racer_name,
            "team_name": self.team_name,
            "current_stage": self.context.stage.value,
            "session_type": self.context.session_type.value if self.context.session_type else None,
            "circuit": self.context.circuit_name,
            "race": self.context.race_name,
            "last_result": self.context.last_result.value if self.context.last_result else None,
            "position": self.context.position,
            "mood": self.context.mood,
            "recent_posts_count": len(self.recent_posts),
            "interaction_history_count": len(self.interaction_history)
        }
        
        # Add NLP insights
        if self.recent_posts:
            recent_content = " ".join([post["content"] for post in self.recent_posts[-3:]])
            sentiment_analysis = self.nlp_processor.analyze_sentiment(recent_content)
            
            base_info.update({
                "recent_sentiment_compound": round(sentiment_analysis['compound'], 3),
                "recent_sentiment_positive": round(sentiment_analysis['pos'], 3),
                "recent_sentiment_negative": round(sentiment_analysis['neg'], 3),
            })
        
        return base_info

# Enhanced demo with NLP features
if __name__ == "__main__":
    print("🏁 F1 Racer AI Agent with Advanced NLP")
    print("=" * 60)
    print("🔬 Using NLTK, spaCy, and Transformers for enhanced text generation")
    print()
    
    # Create agent instance
    agent = F1RacerAgent("Max Lightning", "Thunder Racing")
    
    # Demonstrate NLP capabilities
    scenarios = [
        {
            "name": "🏃 Practice Session with NLP Analysis",
            "stage": RaceStage.PRACTICE,
            "session": SessionType.FP2,
            "circuit": "Monaco",
            "race": "Monaco Grand Prix"
        },
        {
            "name": "🏆 Victory with Sentiment Analysis",
            "stage": RaceStage.POST_RACE,
            "result": RaceResult.WIN,
            "position": 1,
            "circuit": "Silverstone",
            "race": "British Grand Prix"
        },
        {
            "name": "💔 Setback with Emotional Intelligence",
            "stage": RaceStage.POST_RACE,
            "result": RaceResult.DNF,
            "circuit": "Spa-Francorchamps",
            "race": "Belgian Grand Prix"
        }
    ]
    
    for scenario in scenarios:
        print(f"📍 {scenario['name']}")
        print("-" * 50)
        
        # Update context
        agent.update_context(
            stage=scenario["stage"],
            session_type=scenario.get("session"),
            circuit_name=scenario["circuit"],
            race_name=scenario["race"],
            last_result=scenario.get("result"),
            position=scenario.get("position")
        )
        
        # Demonstrate NLP-enhanced outputs
        print("🗣️  Status Post:", agent.speak())
        print("💭 NLP Thoughts:", agent.think())
        
        # Demonstrate advanced reply analysis
        test_comment = "Amazing driving today! Keep pushing!"
        print(f"💬 Analyzing comment: '{test_comment}'")
        sentiment = agent.nlp_processor.analyze_sentiment(test_comment)
        print(f"   Sentiment: {sentiment['compound']:.2f} (pos: {sentiment['pos']:.2f})")
        print("📤 Reply:", agent.reply_to_comment(test_comment))
        
        print("📢 Mention:", agent.mention_teammate_or_competitor("Carlos Speed", "teammate"))
        print()
        
        # Show NLP insights
        info = agent.get_agent_info()
        if "recent_sentiment_compound" in info:
            print(f"🔬 NLP Insights - Recent sentiment: {info['recent_sentiment_compound']}")
        
        input("Press Enter to continue...")
    
    print("\n✅ NLP-Enhanced F1 Agent Demo Complete!")
    print("🔬 Features demonstrated:")
    print("  - NLTK sentiment analysis")
    print("  - spaCy entity recognition") 
    print("  - Transformer-based text generation")
    print("  - Advanced contextual awareness")
    print("  - Linguistic analysis and enhancement")
