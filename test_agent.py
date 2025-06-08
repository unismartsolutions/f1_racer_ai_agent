#!/usr/bin/env python3
"""
Comprehensive testing suite for F1 Racer AI Agent with NLP capabilities
Tests all core functionality and validates NLP-enhanced output quality
"""

import sys
import time
import warnings
from typing import List, Dict, Any

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from f1_agent import F1RacerAgent, RaceStage, SessionType, RaceResult
except ImportError as e:
    print(f"❌ Error importing F1 agent: {e}")
    print("💡 Make sure all NLP dependencies are installed")
    sys.exit(1)

class F1AgentTester:
    """Comprehensive testing suite for F1 Racer AI Agent"""
    
    def __init__(self):
        self.agent = F1RacerAgent("Test Driver", "Test Team")
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = {
            "test": test_name,
            "status": status,
            "passed": passed,
            "details": details
        }
        self.test_results.append(result)
        print(f"{status}: {test_name}")
        if details:
            print(f"    {details}")
    
    def test_agent_initialization(self):
        """Test agent initialization and basic properties"""
        print("\n🧪 Testing Agent Initialization...")
        
        # Test default initialization
        try:
            agent = F1RacerAgent()
            self.log_test("Default agent creation", True)
        except Exception as e:
            self.log_test("Default agent creation", False, str(e))
        
        # Test custom initialization
        try:
            agent = F1RacerAgent("Lewis Hamilton", "Mercedes")
            self.log_test("Custom agent creation", 
                         agent.racer_name == "Lewis Hamilton" and agent.team_name == "Mercedes")
        except Exception as e:
            self.log_test("Custom agent creation", False, str(e))
        
        # Test context initialization
        try:
            self.log_test("Context initialization", 
                         self.agent.context.racer_name == "Test Driver",
                         f"Racer: {self.agent.context.racer_name}")
        except Exception as e:
            self.log_test("Context initialization", False, str(e))
    
    def test_context_updates(self):
        """Test context update functionality"""
        print("\n🧪 Testing Context Updates...")
        
        # Test stage update
        try:
            self.agent.update_context(RaceStage.QUALIFYING)
            self.log_test("Stage update", 
                         self.agent.context.stage == RaceStage.QUALIFYING)
        except Exception as e:
            self.log_test("Stage update", False, str(e))
        
        # Test full context update
        try:
            self.agent.update_context(
                stage=RaceStage.POST_RACE,
                session_type=SessionType.RACE,
                circuit_name="Monaco",
                race_name="Monaco Grand Prix",
                last_result=RaceResult.WIN,
                position=1
            )
            
            success = (
                self.agent.context.stage == RaceStage.POST_RACE and
                self.agent.context.circuit_name == "Monaco" and
                self.agent.context.last_result == RaceResult.WIN and
                self.agent.context.position == 1
            )
            self.log_test("Full context update", success)
        except Exception as e:
            self.log_test("Full context update", False, str(e))
        
        # Test mood update
        try:
            previous_mood = self.agent.context.mood
            self.agent.update_context(RaceStage.POST_RACE, last_result=RaceResult.WIN)
            mood_updated = self.agent.context.mood != previous_mood
            self.log_test("Mood update", mood_updated, 
                         f"Mood: {self.agent.context.mood}")
        except Exception as e:
            self.log_test("Mood update", False, str(e))
    
    def test_text_generation(self):
        """Test various text generation scenarios"""
        print("\n🧪 Testing Text Generation...")
        
        test_scenarios = [
            {"context": "general", "expected_length": 50},
            {"context": "win", "expected_length": 50},
            {"context": "practice", "expected_length": 50},
            {"context": "qualifying", "expected_length": 50},
            {"context": "disappointing", "expected_length": 50}
        ]
        
        for scenario in test_scenarios:
            try:
                # Set appropriate context for the test
                if scenario["context"] == "win":
                    self.agent.update_context(RaceStage.POST_RACE, last_result=RaceResult.WIN)
                elif scenario["context"] == "practice":
                    self.agent.update_context(RaceStage.PRACTICE, session_type=SessionType.FP2)
                elif scenario["context"] == "qualifying":
                    self.agent.update_context(RaceStage.QUALIFYING, session_type=SessionType.Q3)
                elif scenario["context"] == "disappointing":
                    self.agent.update_context(RaceStage.POST_RACE, last_result=RaceResult.DNF)
                
                text = self.agent.speak(scenario["context"])
                
                success = (
                    len(text) >= scenario["expected_length"] and
                    isinstance(text, str) and
                    text.strip() != ""
                )
                
                self.log_test(f"Text generation - {scenario['context']}", success,
                             f"Length: {len(text)}, Preview: '{text[:50]}...'")
                
            except Exception as e:
                self.log_test(f"Text generation - {scenario['context']}", False, str(e))
    
    def test_reply_generation(self):
        """Test reply generation for various comment types"""
        print("\n🧪 Testing Reply Generation...")
        
        test_comments = [
            "Great drive today!",
            "Unlucky with the result, better next time",
            "What's your favorite circuit?",
            "Amazing performance!",
            "Keep pushing, we believe in you!",
            "That was an incredible overtake!"
        ]
        
        for comment in test_comments:
            try:
                reply = self.agent.reply_to_comment(comment)
                
                success = (
                    isinstance(reply, str) and
                    len(reply) > 10 and
                    reply.strip() != ""
                )
                
                self.log_test(f"Reply to: '{comment[:30]}...'", success,
                             f"Reply: '{reply[:50]}...'")
                
            except Exception as e:
                self.log_test(f"Reply to: '{comment[:30]}...'", False, str(e))
    
    def test_mention_generation(self):
        """Test mention generation functionality"""
        print("\n🧪 Testing Mention Generation...")
        
        test_scenarios = [
            {"person": "Carlos Sainz", "context": "positive"},
            {"person": "Max Verstappen", "context": "competitive"},
            {"person": "George Russell", "context": "teammate"}
        ]
        
        for scenario in test_scenarios:
            try:
                mention = self.agent.mention_teammate_or_competitor(
                    scenario["person"], scenario["context"]
                )
                
                success = (
                    isinstance(mention, str) and
                    scenario["person"] in mention and
                    len(mention) > 20
                )
                
                self.log_test(f"Mention - {scenario['context']}", success,
                             f"Preview: '{mention[:60]}...'")
                
            except Exception as e:
                self.log_test(f"Mention - {scenario['context']}", False, str(e))
    
    def test_like_simulation(self):
        """Test like action simulation"""
        print("\n🧪 Testing Like Simulation...")
        
        test_posts = [
            "Great race today everyone!",
            "Looking forward to next weekend",
            "Team work makes the dream work"
        ]
        
        for post in test_posts:
            try:
                like_action = self.agent.simulate_like_action(post)
                
                success = (
                    isinstance(like_action, str) and
                    len(like_action) > 10 and
                    any(emoji in like_action for emoji in ["❤️", "👍", "🏁", "💪"])
                )
                
                self.log_test(f"Like simulation", success,
                             f"Action: '{like_action}'")
                
            except Exception as e:
                self.log_test(f"Like simulation", False, str(e))
    
    def test_thinking_capability(self):
        """Test the thinking/internal analysis capability"""
        print("\n🧪 Testing Thinking Capability...")
        
        test_contexts = [
            RaceStage.PRACTICE,
            RaceStage.QUALIFYING,
            RaceStage.RACE,
            RaceStage.POST_RACE
        ]
        
        for context in test_contexts:
            try:
                self.agent.update_context(context)
                thoughts = self.agent.think()
                
                success = (
                    isinstance(thoughts, str) and
                    len(thoughts) > 20 and
                    "💭" in thoughts
                )
                
                self.log_test(f"Thinking - {context.value}", success,
                             f"Preview: '{thoughts[:60]}...'")
                
            except Exception as e:
                self.log_test(f"Thinking - {context.value}", False, str(e))
    
    def test_hashtag_generation(self):
        """Test contextual hashtag generation"""
        print("\n🧪 Testing Hashtag Generation...")
        
        # Test different contexts
        contexts = [
            (RaceStage.PRACTICE, SessionType.FP1),
            (RaceStage.QUALIFYING, SessionType.Q3),
            (RaceStage.POST_RACE, None)
        ]
        
        for stage, session in contexts:
            try:
                self.agent.update_context(stage, session_type=session, race_name="Test Grand Prix")
                post = self.agent.speak()
                
                has_hashtags = "#" in post
                has_race_tag = "GP" in post or "F1" in post
                
                success = has_hashtags and has_race_tag
                
                self.log_test(f"Hashtag generation - {stage.value}", success,
                             f"Found hashtags in: '{post[-50:]}'")
                
            except Exception as e:
                self.log_test(f"Hashtag generation - {stage.value}", False, str(e))
    
    def test_nlp_capabilities(self):
        """Test NLP functionality including sentiment analysis and entity extraction"""
        print("\n🧪 Testing NLP Capabilities...")
        
        # Test sentiment analysis
        try:
            test_texts = [
                "This is amazing! What a fantastic result!",
                "Really disappointed with today's outcome.",
                "Looking forward to the next race weekend."
            ]
            
            for text in test_texts:
                sentiment = self.agent.nlp_processor.analyze_sentiment(text)
                
                success = (
                    isinstance(sentiment, dict) and
                    'compound' in sentiment and
                    'pos' in sentiment and
                    'neg' in sentiment and
                    -1 <= sentiment['compound'] <= 1
                )
                
                if not success:
                    self.log_test("NLP sentiment analysis", False, f"Invalid sentiment for: '{text[:30]}...'")
                    return
                    
            self.log_test("NLP sentiment analysis", True, "All sentiment analyses successful")
            
        except Exception as e:
            self.log_test("NLP sentiment analysis", False, str(e))
        
        # Test entity extraction
        try:
            test_text = "Lewis Hamilton is racing for Mercedes at Silverstone this weekend."
            entities = self.agent.nlp_processor.extract_entities(test_text)
            
            success = isinstance(entities, list)
            self.log_test("NLP entity extraction", success, 
                         f"Extracted {len(entities)} entities")
            
        except Exception as e:
            self.log_test("NLP entity extraction", False, str(e))
        
        # Test keyword extraction
        try:
            test_text = "The car feels great today with excellent grip and perfect balance."
            keywords = self.agent.nlp_processor.extract_keywords(test_text)
            
            success = (
                isinstance(keywords, list) and
                len(keywords) > 0 and
                all(isinstance(word, str) for word in keywords)
            )
            
            self.log_test("NLP keyword extraction", success,
                         f"Extracted keywords: {keywords[:3]}")
            
        except Exception as e:
            self.log_test("NLP keyword extraction", False, str(e))
    
    def test_enhanced_reply_analysis(self):
        """Test NLP-enhanced reply generation"""
        print("\n🧪 Testing Enhanced Reply Analysis...")
        
        test_comments = [
            ("Amazing drive today! You were flying out there!", "positive"),
            ("Tough luck with the DNF, but we know you'll bounce back", "supportive"), 
            ("What's your favorite part about racing at Monaco?", "question"),
            ("Keep pushing! The team is behind you 100%", "motivational")
        ]
        
        for comment, expected_type in test_comments:
            try:
                # Analyze comment using NLP
                sentiment = self.agent.nlp_processor.analyze_sentiment(comment)
                keywords = self.agent.nlp_processor.extract_keywords(comment)
                
                # Generate reply
                reply = self.agent.reply_to_comment(comment)
                
                success = (
                    isinstance(reply, str) and
                    len(reply) > 10 and
                    isinstance(sentiment, dict) and
                    isinstance(keywords, list)
                )
                
                self.log_test(f"Enhanced reply - {expected_type}", success,
                             f"Sentiment: {sentiment.get('compound', 0):.2f}, Reply: '{reply[:40]}...'")
                
            except Exception as e:
                self.log_test(f"Enhanced reply - {expected_type}", False, str(e))
    
    def test_mood_analysis_with_nlp(self):
        """Test NLP-enhanced mood analysis"""
        print("\n🧪 Testing NLP Mood Analysis...")
        
        mood_scenarios = [
            (RaceResult.WIN, "positive"),
            (RaceResult.DNF, "negative"),
            (RaceResult.PODIUM, "positive"),
            (None, "focused")
        ]
        
        for result, expected_mood_category in mood_scenarios:
            try:
                self.agent.update_context(
                    RaceStage.POST_RACE if result else RaceStage.PRACTICE,
                    last_result=result
                )
                
                # Check if mood was analyzed correctly
                mood = self.agent.context.mood
                
                # Verify mood makes sense for the result
                mood_match = False
                if expected_mood_category == "positive" and mood in ["positive"]:
                    mood_match = True
                elif expected_mood_category == "negative" and mood in ["negative"]:
                    mood_match = True
                elif expected_mood_category == "focused" and mood in ["focused", "neutral"]:
                    mood_match = True
                
                self.log_test(f"NLP mood analysis - {result.value if result else 'neutral'}", 
                             mood_match, f"Expected: {expected_mood_category}, Got: {mood}")
                
            except Exception as e:
                self.log_test(f"NLP mood analysis - {result.value if result else 'neutral'}", 
                             False, str(e))
    
    def test_dynamic_responses(self):
        """Test that responses are dynamic and not repetitive"""
        print("\n🧪 Testing Dynamic Responses...")
        
        # Generate multiple responses with same context
        self.agent.update_context(RaceStage.PRACTICE, session_type=SessionType.FP2)
        
        responses = []
        for i in range(5):
            try:
                response = self.agent.speak("practice")
                responses.append(response)
            except Exception as e:
                self.log_test("Dynamic response generation", False, str(e))
                return
        
        # Check for uniqueness
        unique_responses = len(set(responses))
        total_responses = len(responses)
        
        # At least 80% should be unique (NLP enhancement should increase variety)
        uniqueness_ratio = unique_responses / total_responses
        success = uniqueness_ratio >= 0.8
        
        self.log_test("Response uniqueness", success,
                     f"Unique: {unique_responses}/{total_responses} ({uniqueness_ratio:.1%})")
        """Test that responses are dynamic and not repetitive"""
        print("\n🧪 Testing Dynamic Responses...")
        
        # Generate multiple responses with same context
        self.agent.update_context(RaceStage.PRACTICE, session_type=SessionType.FP2)
        
        responses = []
        for i in range(5):
            try:
                response = self.agent.speak("practice")
                responses.append(response)
            except Exception as e:
                self.log_test("Dynamic response generation", False, str(e))
                return
        
        # Check for uniqueness
        unique_responses = len(set(responses))
        total_responses = len(responses)
        
        # At least 80% should be unique
        uniqueness_ratio = unique_responses / total_responses
        success = uniqueness_ratio >= 0.8
        
        self.log_test("Response uniqueness", success,
                     f"Unique: {unique_responses}/{total_responses} ({uniqueness_ratio:.1%})")
    
    def test_agent_info(self):
        """Test agent information retrieval"""
        print("\n🧪 Testing Agent Info...")
        
        try:
            info = self.agent.get_agent_info()
            
            required_fields = ["racer_name", "team_name", "current_stage", "mood"]
            has_required_fields = all(field in info for field in required_fields)
            
            self.log_test("Agent info structure", has_required_fields,
                         f"Fields: {list(info.keys())}")
            
        except Exception as e:
            self.log_test("Agent info structure", False, str(e))
    
    def test_race_weekend_simulation(self):
        """Test complete race weekend progression"""
        print("\n🧪 Testing Race Weekend Simulation...")
        
        weekend_progression = [
            (RaceStage.PRACTICE, SessionType.FP1),
            (RaceStage.PRACTICE, SessionType.FP2),
            (RaceStage.QUALIFYING, SessionType.Q3),
            (RaceStage.RACE, SessionType.RACE),
            (RaceStage.POST_RACE, None)
        ]
        
        try:
            for stage, session in weekend_progression:
                self.agent.update_context(
                    stage=stage,
                    session_type=session,
                    circuit_name="Test Circuit",
                    race_name="Test Grand Prix"
                )
                
                # Generate content for each stage
                post = self.agent.speak()
                thoughts = self.agent.think()
                
                stage_success = (
                    len(post) > 20 and
                    len(thoughts) > 20 and
                    self.agent.context.stage == stage
                )
                
                if not stage_success:
                    self.log_test("Race weekend simulation", False,
                                 f"Failed at {stage.value}")
                    return
            
            self.log_test("Race weekend simulation", True,
                         "Successfully progressed through all stages")
            
        except Exception as e:
            self.log_test("Race weekend simulation", False, str(e))
    
    def run_performance_test(self):
        """Test response generation performance"""
        print("\n🧪 Testing Performance...")
        
        try:
            import time
            
            start_time = time.time()
            
            # Generate 10 responses
            for i in range(10):
                self.agent.speak("general")
            
            end_time = time.time()
            elapsed = end_time - start_time
            avg_time = elapsed / 10
            
            # Should generate responses quickly (under 0.1s each)
            success = avg_time < 0.1
            
            self.log_test("Response generation performance", success,
                         f"Average: {avg_time:.3f}s per response")
            
        except Exception as e:
            self.log_test("Response generation performance", False, str(e))
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🏁 F1 Racer AI Agent - Comprehensive Test Suite")
        print("=" * 60)
        
        # Run all test methods
        test_methods = [
            self.test_agent_initialization,
            self.test_context_updates,
            self.test_nlp_capabilities,
            self.test_text_generation,
            self.test_reply_generation,
            self.test_enhanced_reply_analysis,
            self.test_mention_generation,
            self.test_like_simulation,
            self.test_thinking_capability,
            self.test_hashtag_generation,
            self.test_dynamic_responses,
            self.test_mood_analysis_with_nlp,
            self.test_agent_info,
            self.test_race_weekend_simulation,
            self.run_performance_test
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ Test suite error in {test_method.__name__}: {e}")
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 90:
            status = "🟢 EXCELLENT"
        elif pass_rate >= 75:
            status = "🟡 GOOD"
        elif pass_rate >= 50:
            status = "🟠 NEEDS IMPROVEMENT"
        else:
            status = "🔴 CRITICAL ISSUES"
        
        print(f"Overall Status: {status}")
        
        # Show failed tests
        failed_tests = [r for r in self.test_results if not r["passed"]]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  - {test['test']}")
                if test["details"]:
                    print(f"    Details: {test['details']}")
        
        print("\n" + "=" * 60)
        
        return pass_rate >= 75  # Return True if tests pass acceptable threshold

def main():
    """Main test execution"""
    tester = F1AgentTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
