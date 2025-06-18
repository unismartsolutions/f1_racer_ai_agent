import random
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# NLP Libraries
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

class RaceStage(Enum):
    """Represents the main stages of a Formula 1 race weekend."""
    PRACTICE = "practice"        # Practice sessions where teams test setups and gather data
    QUALIFYING = "qualifying"    # Qualifying session to determine starting grid positions
    RACE = "race"                # The main race event
    POST_RACE = "post_race"      # Activities and analysis after the race has finished

class SessionType(Enum):
    """Enumerates the specific session types during a race weekend."""
    FP1 = "FP1"                  # Free Practice 1
    FP2 = "FP2"                  # Free Practice 2
    FP3 = "FP3"                  # Free Practice 3
    Q1 = "Q1"                    # Qualifying session 1
    Q2 = "Q2"                    # Qualifying session 2
    Q3 = "Q3"                    # Qualifying session 3
    RACE = "Race"                # The main race session

class RaceResult(Enum):
    """Possible outcomes or results for a driver in a race."""
    WIN = "win"                  # Finished in first place
    PODIUM = "podium"            # Finished in the top three
    POINTS = "points"            # Finished in a points-scoring position
    DNF = "dnf"                  # Did not finish the race
    CRASH = "crash"              # Retired due to a crash
    MECHANICAL = "mechanical"    # Retired due to a mechanical issue
    DISAPPOINTING = "disappointing"  # Finished with a disappointing result

@dataclass
class RaceContext:
    """Race weekend context data"""
    stage: RaceStage
    session_type: Optional[SessionType]
    circuit_name: str
    race_name: str
    last_result: Optional[RaceResult]
    position: Optional[int]
    team_name: str
    racer_name: str
    mood: str

class NLPProcessor:
    """NLP processor using NLTK, spacy, and transformers"""
    
    def __init__(self):
        self.nltk_ready = False
        self.spacy_ready = False
        
        # Initialize NLTK
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
            self.nltk_ready = True
            print("NLTK loaded successfully")
        except Exception:
            self.sentiment_analyzer = None
            self.stop_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            print("NLTK fallback mode")
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_ready = True
            print("spaCy loaded successfully")
        except OSError:
            try:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_ready = True
                print("spaCy model downloaded and loaded")
            except Exception:
                self.nlp = None
                self.spacy_ready = False
                print("spaCy fallback mode")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using NLTK's VADER with fallback"""
        if self.nltk_ready and self.sentiment_analyzer:
            try:
                return self.sentiment_analyzer.polarity_scores(text)
            except Exception:
                pass
        
        return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Simple fallback sentiment analysis"""
        positive_words = ['amazing', 'great', 'awesome', 'fantastic', 'brilliant', 'excellent', 'good', 'win', 'victory', 'success']
        negative_words = ['bad', 'terrible', 'awful', 'disappointing', 'frustrating', 'failed', 'loss', 'crash', 'dnf']
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower) / 10
        neg_score = sum(1 for word in negative_words if word in text_lower) / 10
        
        compound = pos_score - neg_score
        compound = max(-1, min(1, compound))
        
        return {
            'pos': min(1.0, pos_score),
            'neg': min(1.0, neg_score),
            'neu': max(0.0, 1.0 - pos_score - neg_score),
            'compound': compound
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using NLTK POS tagging with fallback"""
        if self.nltk_ready:
            try:
                tokens = word_tokenize(text.lower())
                pos_tags = pos_tag(tokens)
                
                keywords = [
                    word for word, pos in pos_tags 
                    if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB')) 
                    and word not in self.stop_words 
                    and len(word) > 2
                ]
                return list(set(keywords))
            except Exception:
                pass
        
        return self._fallback_keyword_extraction(text)
    
    def _fallback_keyword_extraction(self, text: str) -> List[str]:
        """Simple fallback keyword extraction"""
        words = text.lower().split()
        keywords = [
            word.strip('.,!?;:"()[]') for word in words 
            if len(word) > 3 and word not in self.stop_words
        ]
        return list(set(keywords))

class F1RacerAgent:
    """
    AI Agent that simulates a Formula 1 racer's persona with reliable text generation
    """
    
    def __init__(self, racer_name: str = "Lightening McQueen", team_name: str = "Rusteez"):
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
        
        self.nlp_processor = NLPProcessor()
        
        # Initialize response libraries
        self._init_response_library()
        
        self.recent_posts = []
        self.interaction_history = []
        self.max_recent_posts = 10
    
    def _init_response_library(self):
        """Initialize comprehensive response library"""
        
        self.response_library = {
            "win": [
                "YES! What an incredible race! Can't believe we pulled that off!",
                "VICTORY! Absolutely buzzing right now! Massive thanks to the entire team!",
                "P1! What a feeling! The car was absolutely perfect today!",
                "WE DID IT! Incredible team effort! This one's for everyone back at the factory!",
                "Victory feels amazing! So grateful for the team's hard work!",
                "YES! First place! The setup was spot on today!",
                "What a race! Couldn't have done it without this incredible team!",
                "P1 baby! Every lap was pure perfection today!"
            ],
            
            "podium": [
                "Great result today! Really happy with the progress we're making!",
                f"P{self.context.position if self.context.position else '2'}! Solid points in the bag! Team did an amazing job!",
                "Happy with that result! We maximized what we had today!",
                "Good day at the office! The car felt strong out there!",
                "Podium finish! Love seeing the team's hard work pay off!",
                "Strong result for the team! Building momentum for the next race!",
                "Quality points today! The setup was working well!",
                "Happy to be on the podium! Great team effort!"
            ],
            
            "disappointing": [
                "Tough day but these things happen in racing. We'll bounce back stronger!",
                "Not our day today but the team never gives up. On to the next one!",
                "Disappointed but that's racing. Already looking ahead to next weekend!",
                "Sometimes racing breaks your heart. But we'll keep fighting!",
                "Not the result we wanted. Gave everything but it wasn't enough today!",
                "Gutted with today's result but we learn and move forward!",
                "Racing can be cruel sometimes. We'll analyze and come back stronger!",
                "Tough to take but the team spirit remains strong!"
            ],
            
            "practice": [
                "Good session today! Learning more about the car with every lap!",
                "Productive practice session! Getting the setup dialed in nicely!",
                "Solid work in practice today! The car is feeling better and better!",
                "Another step forward today! Team is doing great work in the garage!",
                "Feeling comfortable with the car! Good progress in practice!",
                "Making steady improvements! Each session teaches us something new!",
                "Car balance is coming together! Positive signs for tomorrow!",
                "Building confidence lap by lap! Team is working hard on the setup!"
            ],
            
            "qualifying": [
                "Qualifying day! Time to find those extra tenths! Car feels good!",
                "Ready for quali! The setup feels solid! Let's see what we can do!",
                "Quali time! Going to give it everything we've got today!",
                "Q-day! Feeling confident about our pace! Time to put it together!",
                "Qualifying mode activated! Time to extract every bit of performance!",
                "Ready to attack! The car has been feeling strong all weekend!",
                "Time to find the limit! Excited to see what we can achieve!",
                "Quali day! This is where the real fun begins!"
            ],
            
            "general": [
                "Always giving 100% for the team and the fans!",
                "Another day at the office! Love what I do!",
                "Working hard with the team to extract every bit of performance!",
                "Grateful for all the support from everyone!",
                "Living the dream! Racing at the highest level!",
                "Team effort makes the dream work!",
                "Focused and ready for whatever comes next!",
                "Blessed to be part of this incredible sport!"
            ]
        }
        
        # Response patterns for replies
        self.reply_patterns = {
            "positive": [
                "Thanks for the support! Every cheer makes a difference!",
                "Really appreciate it! The fans are what make this sport special!",
                "Thank you! Your energy gives us extra motivation!",
                "Grateful for supporters like you! See you at the track!",
                "Thanks! Messages like this keep us going!",
                "Appreciate the love! Fan support means everything!"
            ],
            
            "supportive": [
                "Thanks for sticking with us! We'll come back stronger!",
                "Appreciate the support! That's what keeps us going!",
                "Thank you! With fans like you, we never give up!",
                "Your support means everything! Next one's for you!",
                "Thanks for believing in us! We won't let you down!",
                "Loyalty like yours is incredible! Thank you!"
            ],
            
            "question": [
                "Great question! Always happy to connect with the fans!",
                "Thanks for asking! Love hearing from you all!",
                "Interesting question! The fans always keep us thinking!",
                "Good point! Always learning from your perspectives!",
                "Love the engagement! Thanks for following our journey!",
                "Thanks for the interest! Fan questions are the best!"
            ]
        }
    
    def update_context(self, stage: RaceStage, session_type: Optional[SessionType] = None,
                      circuit_name: str = None, race_name: str = None,
                      last_result: Optional[RaceResult] = None, position: Optional[int] = None,
                      mood: str = None):
        """Update the agent's contextual awareness"""
        
        previous_mood = self.context.mood
        
        if circuit_name:
            self.context.circuit_name = circuit_name
        if race_name:
            self.context.race_name = race_name
        
        self.context.stage = stage
        self.context.session_type = session_type
        self.context.last_result = last_result
        self.context.position = position
        
        # Only analyze mood if not explicitly provided
        if mood:
            self.context.mood = mood
        else:
            self._analyze_and_update_mood()
        
        context_change = {
            "timestamp": datetime.now(),
            "stage": stage.value,
            "mood_change": previous_mood != self.context.mood,
            "new_mood": self.context.mood
        }
        self.interaction_history.append(context_change)
        
        print(f"Context updated: {stage.value} at {self.context.circuit_name} (Mood: {self.context.mood})")
    
    def _analyze_and_update_mood(self):
        """Analyze and update mood based on context"""
        
        if self.context.last_result:
            if self.context.last_result == RaceResult.WIN:
                self.context.mood = "ecstatic"
            elif self.context.last_result in [RaceResult.PODIUM, RaceResult.POINTS]:
                self.context.mood = "positive"
            elif self.context.last_result in [RaceResult.DNF, RaceResult.CRASH, RaceResult.DISAPPOINTING]:
                self.context.mood = "disappointed"
            else:
                self.context.mood = "neutral"
        else:
            if self.context.stage == RaceStage.PRACTICE:
                self.context.mood = "focused"
            elif self.context.stage == RaceStage.QUALIFYING:
                self.context.mood = "intense"
            elif self.context.stage == RaceStage.RACE:
                self.context.mood = "determined"
            else:
                self.context.mood = "neutral"
    
    def speak(self, context_type: str = "general") -> str:
        """
        Generate contextual F1 racer posts based on user selection
        """
        
        print(f"Generating post for context: {context_type}")
        
        # Map user selection to appropriate response type
        if context_type == "general":
            context_type = self._determine_context_from_state()
        
        # Get base message from response library
        if context_type in self.response_library:
            base_messages = self.response_library[context_type]
            base_message = random.choice(base_messages)
        else:
            # Fallback to general if type not found
            base_message = random.choice(self.response_library["general"])
        
        # Add contextual elements
        contextual_message = self._add_contextual_elements(base_message, context_type)
        
        # Generate and add hashtags
        hashtags = self._generate_hashtags(context_type)
        
        # Combine final message
        final_message = f"{contextual_message} {hashtags}"
        
        # Track the post
        self._track_generated_content(final_message, context_type)
        
        return final_message
    
    def _determine_context_from_state(self) -> str:
        """Determine context type from current agent state"""
        
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
            return "general"
    
    def _add_contextual_elements(self, base_message: str, context_type: str) -> str:
        """Add contextual elements to the base message"""
        
        # Add position reference for podium finishes
        if context_type == "podium" and self.context.position:
            if "P2" in base_message or "P3" in base_message:
                base_message = base_message.replace("P2", f"P{self.context.position}")
                base_message = base_message.replace("P3", f"P{self.context.position}")
        
        # Add session reference for practice/qualifying
        if context_type == "practice" and self.context.session_type:
            session_refs = {
                SessionType.FP1: "FP1",
                SessionType.FP2: "FP2", 
                SessionType.FP3: "FP3"
            }
            if self.context.session_type in session_refs:
                session_name = session_refs[self.context.session_type]
                if "practice" in base_message and session_name not in base_message:
                    base_message = base_message.replace("practice", f"{session_name}")
        
        return base_message
    
    def _generate_hashtags(self, context_type: str) -> str:
        """Generate contextual hashtags"""
        
        hashtags = []
        
        # Race-specific hashtag
        if self.context.race_name and "Grand Prix" in self.context.race_name:
            race_hashtag = self.context.race_name.replace(" Grand Prix", "GP").replace(" ", "")
            hashtags.append(f"#{race_hashtag}")
        
        # Context-specific hashtags
        if context_type == "win":
            hashtags.extend(["#Victory", "#P1", "#Champions"])
        elif context_type == "podium":
            hashtags.extend(["#Podium", "#Points"])
        elif context_type == "disappointing":
            hashtags.extend(["#NeverGiveUp", "#ComeBackStronger"])
        elif context_type == "practice":
            if self.context.session_type:
                hashtags.append(f"#{self.context.session_type.value}")
            hashtags.append("#Practice")
        elif context_type == "qualifying":
            hashtags.extend(["#Qualifying", "#Quali"])
        
        # General F1 hashtags
        general_hashtags = ["#F1", "#Racing", "#TeamWork", "#Motorsport", "#Speed"]
        hashtags.extend(random.sample(general_hashtags, min(2, len(general_hashtags))))
        
        # Team hashtag
        if self.team_name != "Racing Team":
            team_hashtag = f"#Team{self.team_name.replace(' ', '')}"
            hashtags.append(team_hashtag)
        
        # Limit to 4-5 hashtags to avoid clutter
        return " ".join(hashtags[:5])
    
    def reply_to_comment(self, original_comment: str) -> str:
        """Generate contextual reply to fan comments with comprehensive analysis"""
        
        print(f"Analyzing comment: '{original_comment}'")
        
        # Analyze comment sentiment and extract keywords
        sentiment = self.nlp_processor.analyze_sentiment(original_comment)
        keywords = self.nlp_processor.extract_keywords(original_comment)
        comment_lower = original_comment.lower()
        
        print(f"Sentiment: {sentiment['compound']:.2f} (pos: {sentiment['pos']:.2f}, neg: {sentiment['neg']:.2f})")
        
        # Classify comment type first
        comment_type = self._classify_comment_type(comment_lower, sentiment, keywords)
        print(f"Comment type: {comment_type}")
        
        # Generate appropriate response based on type
        if comment_type == "question":
            response = self._handle_question(comment_lower, keywords)
        elif comment_type == "negative":
            response = self._handle_negative_comment(comment_lower, sentiment, keywords)
        elif comment_type == "positive":
            response = self._handle_positive_comment(comment_lower, sentiment, keywords)
        else:  # neutral
            response = self._handle_neutral_comment(comment_lower, sentiment, keywords)
        
        # Add appropriate emoji
        response = self._add_reply_emoji(response, comment_type, sentiment)
        
        return response
    
    def _classify_comment_type(self, comment_lower: str, sentiment: Dict, keywords: List[str]) -> str:
        """Classify comment as question, positive, negative, or neutral"""
        
        # Question detection - check for question words and question marks
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can you', 'do you', 'will you', 'are you', 'have you']
        
        if '?' in comment_lower or any(indicator in comment_lower for indicator in question_indicators):
            return "question"
        
        # Negative comment detection
        negative_indicators = [
            'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'bad', 'pathetic', 
            'useless', 'slow', 'overrated', 'lucky', 'should retire', 'give up', 
            'not good enough', 'embarrassing', 'shameful', 'failure', 'trash', 'rubbish',
            'hate', 'dislike', 'annoying', 'boring', 'stupid', 'idiot', 'loser'
        ]
        
        if sentiment['compound'] <= -0.3 or any(indicator in comment_lower for indicator in negative_indicators):
            return "negative"
        
        # Positive comment detection
        positive_indicators = [
            'amazing', 'incredible', 'fantastic', 'brilliant', 'awesome', 'great', 'excellent',
            'outstanding', 'phenomenal', 'perfect', 'love', 'best', 'legend', 'hero', 'champion',
            'congratulations', 'well done', 'proud', 'inspiring', 'respect'
        ]
        
        if sentiment['compound'] >= 0.3 or any(indicator in comment_lower for indicator in positive_indicators):
            return "positive"
        
        # Everything else is neutral
        return "neutral"
    
    def _handle_question(self, comment_lower: str, keywords: List[str]) -> str:
        """Handle question-type comments"""
        
        # Technical/Car questions
        if any(word in comment_lower for word in ['car', 'setup', 'balance', 'downforce', 'tires', 'tire', 'engine', 'gearbox', 'brakes']):
            responses = [
                "Great question! The technical side is so complex but fascinating!",
                "Good question! Working with the engineers to find the perfect setup!",
                "Thanks for asking! The car setup is constantly evolving!",
                "Interesting question! Every little detail makes a difference!",
                "Love technical questions! The engineering side amazes me daily!"
            ]
        
        # Circuit/Track questions
        elif any(word in comment_lower for word in ['circuit', 'track', 'corner', 'turn', 'chicane', 'straight']):
            responses = [
                f"Good question! {self.context.circuit_name} has its own unique challenges!",
                "Great question! Every track teaches you something different!",
                "Thanks for asking! Each circuit has its own personality!",
                "Love track questions! The variety keeps us on our toes!",
                "Good point! Circuit knowledge is crucial in F1!"
            ]
        
        # Strategy questions
        elif any(word in comment_lower for word in ['strategy', 'pit', 'fuel', 'tactics', 'decision', 'timing']):
            responses = [
                "Great strategy question! So many variables to consider!",
                "Good question! The strategists work around the clock!",
                "Thanks for asking! Strategy can make or break a race!",
                "Love strategy questions! It's like chess at 300km/h!",
                "Interesting question! Split-second decisions matter so much!"
            ]
        
        # Racing/Competition questions
        elif any(word in comment_lower for word in ['race', 'racing', 'compete', 'battle', 'fight', 'overtake', 'defend']):
            responses = [
                "Great racing question! This is what we live for!",
                "Good question! Racing at this level is pure adrenaline!",
                "Thanks for asking! The competition makes us all better!",
                "Love racing questions! Every battle is a learning experience!",
                "Interesting question! Wheel-to-wheel racing is the best part!"
            ]
        
        # Personal/Career questions
        elif any(word in comment_lower for word in ['you', 'your', 'feel', 'think', 'favorite', 'best', 'worst', 'experience']):
            responses = [
                "Thanks for the personal question! Love connecting with fans!",
                "Great question! Always happy to share with racing enthusiasts!",
                "Good question! Fan curiosity keeps us grounded!",
                "Thanks for asking! These interactions mean a lot!",
                "Love personal questions! The fan connection is special!"
            ]
        
        # Future/Career questions
        elif any(word in comment_lower for word in ['future', 'next', 'season', 'championship', 'goals', 'plans']):
            responses = [
                "Great question! Always focused on the next challenge!",
                "Good question! Taking it one race at a time!",
                "Thanks for asking! The future looks exciting!",
                "Love future questions! So much to look forward to!",
                "Good question! Never stop pushing for more!"
            ]
        
        # Default question response
        else:
            responses = [
                "Great question! Always happy to connect with curious fans!",
                "Good question! Love the engagement from supporters!",
                "Thanks for asking! Fan questions are always interesting!",
                "Interesting question! The fans ask the best questions!",
                "Good question! Always learning from fan perspectives!"
            ]
        
        return random.choice(responses)
    
    def _handle_negative_comment(self, comment_lower: str, sentiment: Dict, keywords: List[str]) -> str:
        """Handle negative comments professionally and graciously"""
        
        # Harsh criticism
        if any(word in comment_lower for word in ['terrible', 'awful', 'horrible', 'worst', 'pathetic', 'useless', 'trash', 'rubbish']):
            responses = [
                "I understand the frustration. We're always working to improve!",
                "Thanks for the honest feedback. That's what drives us to be better!",
                "I hear you. Days like this motivate us to come back stronger!",
                "Appreciate the passion! We'll keep pushing to earn your support!",
                "Fair criticism. We know we have more to give and we will!"
            ]
        
        # Performance criticism
        elif any(word in comment_lower for word in ['slow', 'disappointing', 'not good enough', 'overrated', 'lucky']):
            responses = [
                "Thanks for keeping it real! We know there's more pace to find!",
                "I appreciate honest feedback. Pushes us to extract more performance!",
                "You're right to expect more. We're working hard to deliver!",
                "Fair point! Every criticism helps us focus on improvement!",
                "Thanks for the feedback. We'll prove the doubters wrong!"
            ]
        
        # Suggestions to retire/give up
        elif any(phrase in comment_lower for phrase in ['should retire', 'give up', 'quit', 'stop racing']):
            responses = [
                "I respect your opinion, but giving up isn't in my DNA!",
                "Thanks for the feedback! Retirement isn't on my mind right now!",
                "I hear you, but this sport is my passion and I'll keep fighting!",
                "Appreciate the honesty! Competition is what makes this sport great!",
                "Fair opinion! But I still have more to give to this sport!"
            ]
        
        # Team criticism
        elif any(word in comment_lower for word in ['team', 'strategy', 'pit crew', 'engineers']):
            responses = [
                "I'll always back my team. We win and lose together!",
                "The team works incredibly hard. We're all learning and improving!",
                "Thanks for the feedback. We'll use it to get better as a unit!",
                "I believe in this team completely. Better days ahead!",
                "Criticism noted. The team and I are committed to improvement!"
            ]
        
        # Default negative response
        else:
            responses = [
                "Thanks for the honest feedback! Always room for improvement!",
                "I appreciate all perspectives from fans. Keeps us motivated!",
                "Fair criticism! We'll use this energy to come back stronger!",
                "Thanks for caring enough to comment! Passion drives this sport!",
                "I hear you! Every opinion matters in this journey!"
            ]
        
        return random.choice(responses)
    
    def _handle_positive_comment(self, comment_lower: str, sentiment: Dict, keywords: List[str]) -> str:
        """Handle positive comments with enthusiasm"""
        
        # High praise/amazement
        if any(word in comment_lower for word in ['amazing', 'incredible', 'fantastic', 'phenomenal', 'outstanding', 'perfect']):
            responses = [
                "Thank you so much! Comments like this make all the hard work worth it!",
                "Really appreciate it! The team and I are buzzing from support like this!",
                "Thanks! Your enthusiasm gives us incredible motivation!",
                "Means the world! Fan energy like this is what drives us!",
                "Thank you! Love connecting with passionate fans like you!"
            ]
        
        # Congratulations/celebrations
        elif any(word in comment_lower for word in ['congratulations', 'congrats', 'well done', 'winner', 'champion', 'victory']):
            responses = [
                "Thank you! Incredible feeling to share this moment with fans!",
                "Thanks! The whole team deserves this celebration!",
                "Appreciate it! Still buzzing from that result!",
                "Thank you! Moments like this are what we race for!",
                "Thanks! Your support made this victory even sweeter!"
            ]
        
        # Performance praise
        elif any(word in comment_lower for word in ['drive', 'driving', 'performance', 'skill', 'talent', 'racing']):
            responses = [
                "Thanks! Always trying to extract every tenth for the team!",
                "Appreciate it! Been working hard on that with the engineers!",
                "Thank you! The car felt amazing out there!",
                "Thanks! Love pushing the limits of what's possible!",
                "Appreciate the support! This is what we train for!"
            ]
        
        # Inspiration/respect comments
        elif any(word in comment_lower for word in ['inspiring', 'respect', 'hero', 'legend', 'role model', 'proud']):
            responses = [
                "Thank you! That means more than you know!",
                "Really appreciate it! Just trying to give my best every day!",
                "Thanks! Honored to represent the sport we all love!",
                "Means everything! The fan support keeps us motivated!",
                "Thank you! Racing fans are the most passionate in the world!"
            ]
        
        # Love/support comments
        elif any(word in comment_lower for word in ['love', 'support', 'fan', 'follow', 'believe']):
            responses = [
                "Thank you! Fan support like yours is incredible!",
                "Appreciate it! Love connecting with dedicated supporters!",
                "Thanks! Your loyalty means everything to us!",
                "Thank you! Fans are what make this sport so special!",
                "Appreciate the love! This community is amazing!"
            ]
        
        # Default positive response
        else:
            responses = [
                "Thank you! Really appreciate the positive energy!",
                "Thanks! Messages like this make our day!",
                "Appreciate it! Love hearing from enthusiastic fans!",
                "Thank you! Your support means the world!",
                "Thanks! Fan positivity keeps us going!"
            ]
        
        return random.choice(responses)
    
    def _handle_neutral_comment(self, comment_lower: str, sentiment: Dict, keywords: List[str]) -> str:
        """Handle neutral/observational comments"""
        
        # Observations about racing/results
        if any(word in comment_lower for word in ['race', 'result', 'weekend', 'session', 'today']):
            responses = [
                "Thanks for following! Always appreciate fan engagement!",
                "Appreciate the comment! Love connecting with the racing community!",
                "Thanks for watching! Fan support means everything!",
                "Thanks for the message! Great to hear from followers!",
                "Appreciate it! The fan connection makes this sport special!"
            ]
        
        # General observations
        elif any(word in comment_lower for word in ['interesting', 'nice', 'cool', 'good', 'okay']):
            responses = [
                "Thanks for the comment! Always great to hear from fans!",
                "Appreciate the message! Love the fan interaction!",
                "Thanks! Connecting with supporters is the best part!",
                "Thanks for taking the time to comment!",
                "Appreciate it! Fan engagement keeps us motivated!"
            ]
        
        # Future/upcoming events
        elif any(word in comment_lower for word in ['next', 'looking forward', 'excited', 'hope']):
            responses = [
                "Thanks! Excited for what's coming next too!",
                "Appreciate it! The journey continues!",
                "Thanks! Always looking ahead to the next challenge!",
                "Thanks for the support! More to come!",
                "Appreciate it! The best is yet to come!"
            ]
        
        # Default neutral response
        else:
            responses = [
                "Thanks for the message! Always appreciate fan interaction!",
                "Thanks for commenting! Love connecting with supporters!",
                "Appreciate the message! Fan engagement means a lot!",
                "Thanks! Great to hear from the racing community!",
                "Thanks for taking the time to comment!"
            ]
        
        return random.choice(responses)
    
    def _add_reply_emoji(self, response: str, comment_type: str, sentiment: Dict) -> str:
        """Add contextually appropriate emojis based on comment type"""
        
        if comment_type == "question":
            return response + " 🤔😊"
        elif comment_type == "negative":
            if sentiment['compound'] <= -0.6:
                return response + " 💪🙏"  # Show strength and respect
            else:
                return response + " 👍🙏"  # Acknowledge and thank
        elif comment_type == "positive":
            if sentiment['compound'] >= 0.6:
                return response + " 🙏❤️"  # High gratitude
            else:
                return response + " 😊🙏"  # Positive acknowledgment
        else:  # neutral
            return response + " 🙏"
        
        return response
    
    def mention_teammate_or_competitor(self, person_name: str, context: str = "positive") -> str:
        """Generate mention posts for teammates or competitors"""
        
        mention_templates = {
            "positive": [
                f"Great work by @{person_name}! Love the level of competition in F1!",
                f"Respect to @{person_name} for that performance! This is what racing is all about!",
                f"Hat off to @{person_name}! The competition makes us all stronger!",
                f"Props to @{person_name}! Amazing driving today!"
            ],
            
            "teammate": [
                f"Team effort with @{person_name} today! Great to have such a strong teammate!",
                f"Solid work @{person_name}! Together we make {self.team_name} stronger!",
                f"@{person_name} bringing the speed! Teamwork makes the dream work!",
                f"Great job @{person_name}! {self.team_name} united!"
            ],
            
            "competitive": [
                f"Ready for the battle with @{person_name}! Should be great racing tomorrow!",
                f"Looking forward to racing @{person_name}! Competition at its finest!",
                f"@{person_name} bringing the heat! Love these battles!",
                f"Game on @{person_name}! May the best driver win!"
            ]
        }
        
        # Select appropriate template
        templates = mention_templates.get(context, mention_templates["positive"])
        mention_text = random.choice(templates)
        
        # Add hashtags
        hashtags = self._generate_hashtags("general")
        
        return f"{mention_text} {hashtags}"
    
    def simulate_like_action(self, post_content: str) -> str:
        """Simulate liking a post"""
        
        sentiment = self.nlp_processor.analyze_sentiment(post_content)
        
        if sentiment['compound'] > 0.5:
            reactions = ["❤️ Loved", "❤️❤️❤️ Absolutely loved", "💪 Fully supported", "🔥 This is fire"]
        elif sentiment['compound'] > 0.1:
            reactions = ["👍 Liked", "🙌 Supported", "💯 This", "✨ Quality content"]
        else:
            reactions = ["👍 Acknowledged", "🤝 Respect", "💙 Seen", "👍 Noted"]
        
        action = random.choice(reactions)
        preview = post_content[:50] + "..." if len(post_content) > 50 else post_content
        
        return f"{action}: '{preview}'"
    
    def think(self) -> str:
        """Generate internal thoughts based on current context"""
        
        thoughts = []
        
        if self.context.stage == RaceStage.PRACTICE:
            thoughts.extend([
                f"Focusing on the setup for {self.context.circuit_name}.",
                "Every lap teaches us something new about the car.",
                "Working closely with the engineers to find the right balance.",
                "Building confidence step by step."
            ])
        elif self.context.stage == RaceStage.QUALIFYING:
            thoughts.extend([
                "Every tenth counts in qualifying. Mental preparation is key.",
                "Need to find that perfect lap when it matters most.",
                "The car setup needs to be spot on for tomorrow.",
                "Qualifying separates the contenders from the pretenders."
            ])
        elif self.context.stage == RaceStage.RACE:
            thoughts.extend([
                "Race day is what we live for. Time to execute the plan.",
                "Managing tires and fuel will be crucial today.",
                "Stay calm, hit your marks, capitalize on opportunities.",
                "Every decision on track could change the championship."
            ])
        else:
            thoughts.extend([
                "Reflecting on the weekend and looking ahead.",
                "Always learning, always improving.",
                "The team's dedication never ceases to amaze me.",
                "Each race weekend brings new challenges and opportunities."
            ])
        
        # Add mood-specific thoughts
        if self.context.mood == "ecstatic":
            thoughts.append("That win shows what's possible when everything clicks!")
        elif self.context.mood == "disappointed":
            thoughts.append("Setbacks like this only make us more determined.")
        elif self.context.mood == "focused":
            thoughts.append("Complete focus on the task at hand.")
        
        selected_thoughts = random.sample(thoughts, min(2, len(thoughts)))
        return f"💭 Internal thoughts: {' '.join(selected_thoughts)}"
    
    def _track_generated_content(self, content: str, context_type: str):
        """Track generated content for analysis"""
        
        entry = {
            "timestamp": datetime.now(),
            "content": content,
            "context_type": context_type,
            "mood": self.context.mood,
            "stage": self.context.stage.value
        }
        
        self.recent_posts.append(entry)
        
        if len(self.recent_posts) > self.max_recent_posts:
            self.recent_posts.pop(0)
    
    def get_agent_info(self) -> Dict:
        """Return comprehensive agent state"""
        
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
        
        if self.recent_posts:
            recent_content = " ".join([post["content"] for post in self.recent_posts[-3:]])
            sentiment_analysis = self.nlp_processor.analyze_sentiment(recent_content)
            
            base_info.update({
                "recent_sentiment_compound": round(sentiment_analysis['compound'], 3),
                "recent_sentiment_positive": round(sentiment_analysis['pos'], 3),
                "recent_sentiment_negative": round(sentiment_analysis['neg'], 3),
            })
        
        return base_info

# Demo and testing
if __name__ == "__main__":
    print("F1 Racer AI Agent - Reliable Implementation")
    print("=" * 55)
    
    agent = F1RacerAgent("Max Lightning", "Thunder Racing")
    
    # Test all context types that users can select
    test_scenarios = [
        {
            "name": "WIN Test",
            "context_type": "win",
            "stage": RaceStage.POST_RACE,
            "result": RaceResult.WIN,
            "position": 1,
            "circuit": "Silverstone",
            "race": "British Grand Prix"
        },
        {
            "name": "DISAPPOINTING Test",
            "context_type": "disappointing",
            "stage": RaceStage.POST_RACE,
            "result": RaceResult.DNF,
            "circuit": "Spa-Francorchamps",
            "race": "Belgian Grand Prix"
        },
        {
            "name": "PRACTICE Test",
            "context_type": "practice",
            "stage": RaceStage.PRACTICE,
            "session": SessionType.FP2,
            "circuit": "Monza",
            "race": "Italian Grand Prix"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 40)
        
        # Update context
        agent.update_context(
            stage=scenario["stage"],
            session_type=scenario.get("session"),
            circuit_name=scenario["circuit"],
            race_name=scenario["race"],
            last_result=scenario.get("result"),
            position=scenario.get("position")
        )
        
        # Test the specific context type
        print(f"User Selection: {scenario['context_type']}")
        post = agent.speak(scenario['context_type'])
        print(f"Generated Post: {post}")
        print(f"Agent Thoughts: {agent.think()}")
        
        print()
        input("Press Enter to continue...")
    
    # Test improved reply functionality with all comment types
    print("\n" + "=" * 60)
    print("TESTING COMPREHENSIVE REPLY SYSTEM")
    print("Testing: POSITIVE, NEGATIVE, NEUTRAL, and QUESTIONS")
    print("=" * 60)
    
    # Set context for reply testing
    agent.update_context(
        stage=RaceStage.POST_RACE,
        last_result=RaceResult.WIN,
        position=1,
        circuit_name="Silverstone",
        race_name="British Grand Prix"
    )
    
    print("\n1. POSITIVE COMMENTS:")
    print("-" * 30)
    positive_comments = [
        "Amazing drive today! You were absolutely flying out there!",
        "Incredible performance! That was some brilliant racing!",
        "Congratulations on the victory! Well deserved!",
        "You're such an inspiration! Keep it up champion!",
        "Outstanding work! That overtake was pure genius!",
        "Love watching you race! You're my favorite driver!"
    ]
    
    for comment in positive_comments:
        print(f"Fan: \"{comment}\"")
        reply = agent.reply_to_comment(comment)
        print(f"Agent: {reply}")
        print()
    
    print("\n2. NEGATIVE COMMENTS:")
    print("-" * 30)
    negative_comments = [
        "That was terrible driving! You should give up racing!",
        "Worst performance I've ever seen. Completely disappointing.",
        "You're so overrated. Just got lucky today.",
        "The team strategy was awful. Fire the whole crew!",
        "Pathetic result. Not good enough for F1.",
        "You're too slow. Time to retire and let someone better race."
    ]
    
    for comment in negative_comments:
        print(f"Fan: \"{comment}\"")
        reply = agent.reply_to_comment(comment)
        print(f"Agent: {reply}")
        print()
    
    print("\n3. NEUTRAL COMMENTS:")
    print("-" * 30)
    neutral_comments = [
        "Interesting race today. Good to see the competition.",
        "Nice work in qualifying yesterday.",
        "Looking forward to the next race weekend.",
        "The car seemed okay during practice.",
        "Hope the weather is better next time.",
        "Thanks for signing my hat at the track!"
    ]
    
    for comment in neutral_comments:
        print(f"Fan: \"{comment}\"")
        reply = agent.reply_to_comment(comment)
        print(f"Agent: {reply}")
        print()
    
    print("\n4. QUESTIONS (Technical):")
    print("-" * 30)
    technical_questions = [
        "How does the car balance feel at high speed corners?",
        "What's the most challenging part about tire strategy?",
        "Why did you pit when you did during the race?",
        "How important is aerodynamic downforce at this circuit?",
        "What's your favorite aspect of the car setup?",
        "How do you work with engineers to find the right balance?"
    ]
    
    for question in technical_questions:
        print(f"Fan: \"{question}\"")
        reply = agent.reply_to_comment(question)
        print(f"Agent: {reply}")
        print()
    
    print("\n5. QUESTIONS (Personal/Racing):")
    print("-" * 30)
    personal_questions = [
        "What's it like racing wheel to wheel at 300km/h?",
        "How do you stay so calm during intense battles?",
        "What's your favorite circuit to race on?",
        "How do you prepare mentally for qualifying?",
        "What advice would you give to young drivers?",
        "Who has been your biggest inspiration in racing?"
    ]
    
    for question in personal_questions:
        print(f"Fan: \"{question}\"")
        reply = agent.reply_to_comment(question)
        print(f"Agent: {reply}")
        print()
    
    # Test with disappointment context for support comments
    print("Testing negative context (DNF) with supportive comments:")
        
    agent.update_context(
        stage=RaceStage.POST_RACE,
        last_result=RaceResult.DNF,
        circuit_name="Monaco",
        race_name="Monaco Grand Prix"
    )
    
    support_comments = [
        "Tough luck today but we still believe in you!",
        "DNF is heartbreaking but you'll come back stronger!",
        "Unlucky mate! Next race is yours!",
        "Keep your head up! These things happen in racing!",
        "Don't let the haters get you down. You're still amazing!",
        "Bad day at the office but we know what you're capable of!"
    ]
    
    for comment in support_comments:
        print(f"Fan: \"{comment}\"")
        reply = agent.reply_to_comment(comment)
        print(f"Agent: {reply}")
        print()
