#!/usr/bin/env python3
"""
Interactive F1 Racer AI Agent Runner with NLP Features
Allows users to configure and interact with the F1 agent in various scenarios
"""

import os
import sys
import warnings

# Suppress transformer warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from f1_agent import F1RacerAgent, RaceStage, SessionType, RaceResult
except ImportError as e:
    print(f"❌ Error importing F1 agent: {e}")
    print("💡 Make sure all NLP dependencies are installed:")
    print("   pip install nltk spacy transformers torch textblob")
    print("   python -m spacy download en_core_web_sm")
    sys.exit(1)

class F1AgentInterface:
    """Interactive interface for the F1 Racer AI Agent"""
    
    def __init__(self):
        self.agent = None
        self.setup_agent()
    
    def setup_agent(self):
        """Setup agent with user configuration"""
        print("🏁 Welcome to the F1 Racer AI Agent!")
        print("=" * 50)
        
        # Get agent configuration
        racer_name = input("Enter racer name (default: Alex Driver): ").strip()
        if not racer_name:
            racer_name = "Alex Driver"
            
        team_name = input("Enter team name (default: Racing Team): ").strip()
        if not team_name:
            team_name = "Racing Team"
        
        self.agent = F1RacerAgent(racer_name, team_name)
        print(f"\n✅ Agent created: {racer_name} from {team_name}")
    
    def display_menu(self):
        """Display the main menu options"""
        print("\n🏎️  F1 Agent Actions:")
        print("1. Generate Status Post")
        print("2. Reply to Fan Comment")
        print("3. Mention Teammate/Competitor")
        print("4. Simulate Like Action")
        print("5. Update Race Context")
        print("6. View Agent Thoughts")
        print("7. View Agent Info")
        print("8. Run Demo Scenarios")
        print("9. Quick Race Weekend Simulation")
        print("0. Exit")
    
    def get_user_choice(self) -> str:
        """Get user menu choice"""
        return input("\nEnter your choice (0-9): ").strip()
    
    def generate_status_post(self):
        """Generate a status post"""
        print("\n📱 Generating Status Post...")
        
        context_types = {
            "1": "general",
            "2": "win", 
            "3": "podium",
            "4": "disappointing",
            "5": "practice",
            "6": "qualifying"
        }
        
        print("Select post type:")
        for key, value in context_types.items():
            print(f"{key}. {value.title()}")
        
        choice = input("Enter choice (1-6, default: general): ").strip()
        context_type = context_types.get(choice, "general")
        
        post = self.agent.speak(context_type)
        print(f"\n📤 Generated Post:\n{post}")
    
    def reply_to_comment(self):
        """Generate reply to a fan comment"""
        print("\n💬 Reply to Fan Comment")
        comment = input("Enter fan comment: ").strip()
        
        if comment:
            reply = self.agent.reply_to_comment(comment)
            print(f"\n📤 Reply:\n{reply}")
        else:
            print("❌ No comment provided")
    
    def mention_person(self):
        """Generate mention post"""
        print("\n🏷️  Mention Teammate/Competitor")
        person_name = input("Enter person's name: ").strip()
        
        if person_name:
            context_types = {"1": "positive", "2": "teammate", "3": "competitive"}
            print("Select mention context:")
            for key, value in context_types.items():
                print(f"{key}. {value.title()}")
            
            choice = input("Enter choice (1-3, default: positive): ").strip()
            context = context_types.get(choice, "positive")
            
            mention = self.agent.mention_teammate_or_competitor(person_name, context)
            print(f"\n📤 Mention Post:\n{mention}")
        else:
            print("❌ No name provided")
    
    def simulate_like(self):
        """Simulate liking a post"""
        print("\n👍 Simulate Like Action")
        post_content = input("Enter post content to like: ").strip()
        
        if post_content:
            like_action = self.agent.simulate_like_action(post_content)
            print(f"\n✅ Action: {like_action}")
        else:
            print("❌ No post content provided")
    
    def update_context(self):
        """Update race context"""
        print("\n🏁 Update Race Context")
        
        # Stage selection
        stages = {
            "1": RaceStage.PRACTICE,
            "2": RaceStage.QUALIFYING, 
            "3": RaceStage.RACE,
            "4": RaceStage.POST_RACE
        }
        
        print("Select race stage:")
        for key, stage in stages.items():
            print(f"{key}. {stage.value.title()}")
        
        stage_choice = input("Enter choice (1-4): ").strip()
        stage = stages.get(stage_choice, RaceStage.PRACTICE)
        
        # Session type (if applicable)
        session_type = None
        if stage in [RaceStage.PRACTICE, RaceStage.QUALIFYING]:
            sessions = {
                "1": SessionType.FP1, "2": SessionType.FP2, "3": SessionType.FP3,
                "4": SessionType.Q1, "5": SessionType.Q2, "6": SessionType.Q3,
                "7": SessionType.RACE, "8": SessionType.SPRINT
            }
            
            print("Select session type:")
            for key, session in sessions.items():
                print(f"{key}. {session.value}")
            
            session_choice = input("Enter choice (1-8, optional): ").strip()
            session_type = sessions.get(session_choice)
        
        # Circuit and race name
        circuit = input("Enter circuit name (e.g., Monaco, Silverstone): ").strip()
        race_name = input("Enter race name (e.g., Monaco Grand Prix): ").strip()
        
        # Last result (if post-race)
        last_result = None
        position = None
        if stage == RaceStage.POST_RACE:
            results = {
                "1": RaceResult.WIN, "2": RaceResult.PODIUM, "3": RaceResult.POINTS,
                "4": RaceResult.DNF, "5": RaceResult.CRASH, "6": RaceResult.DISAPPOINTING
            }
            
            print("Select last result:")
            for key, result in results.items():
                print(f"{key}. {result.value.title()}")
            
            result_choice = input("Enter choice (1-6): ").strip()
            last_result = results.get(result_choice)
            
            if last_result in [RaceResult.WIN, RaceResult.PODIUM, RaceResult.POINTS]:
                try:
                    position = int(input("Enter finishing position (1-20): "))
                except ValueError:
                    position = None
        
        # Update agent context
        self.agent.update_context(
            stage=stage,
            session_type=session_type,
            circuit_name=circuit or "Circuit",
            race_name=race_name or "Grand Prix",
            last_result=last_result,
            position=position
        )
        
        print("✅ Context updated successfully!")
    
    def view_thoughts(self):
        """Display agent's internal thoughts"""
        print("\n💭 Agent Thoughts:")
        thoughts = self.agent.think()
        print(thoughts)
    
    def view_agent_info(self):
        """Display current agent information"""
        print("\n📊 Agent Information:")
        info = self.agent.get_agent_info()
        
        for key, value in info.items():
            if value is not None:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    def run_demo_scenarios(self):
        """Run predefined demo scenarios"""
        print("\n🎬 Running Demo Scenarios...")
        
        scenarios = [
            {
                "name": "🏃 Practice Session at Monaco",
                "stage": RaceStage.PRACTICE,
                "session": SessionType.FP2,
                "circuit": "Monaco",
                "race": "Monaco Grand Prix"
            },
            {
                "name": "🏆 Victory at Silverstone",
                "stage": RaceStage.POST_RACE,
                "result": RaceResult.WIN,
                "position": 1,
                "circuit": "Silverstone",
                "race": "British Grand Prix"
            },
            {
                "name": "💔 DNF at Spa",
                "stage": RaceStage.POST_RACE,
                "result": RaceResult.DNF,
                "circuit": "Spa-Francorchamps",
                "race": "Belgian Grand Prix"
            },
            {
                "name": "🥉 Podium at Monza",
                "stage": RaceStage.POST_RACE,
                "result": RaceResult.PODIUM,
                "position": 3,
                "circuit": "Monza",
                "race": "Italian Grand Prix"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print("-" * 40)
            
            # Update context
            self.agent.update_context(
                stage=scenario["stage"],
                session_type=scenario.get("session"),
                circuit_name=scenario["circuit"],
                race_name=scenario["race"],
                last_result=scenario.get("result"),
                position=scenario.get("position")
            )
            
            # Generate outputs
            print("📱 Status:", self.agent.speak())
            print("💭 Thoughts:", self.agent.think())
            print("💬 Reply to 'Amazing drive!':", self.agent.reply_to_comment("Amazing drive today!"))
            
            input("\nPress Enter to continue to next scenario...")
    
    def race_weekend_simulation(self):
        """Simulate an entire race weekend"""
        print("\n🏁 Quick Race Weekend Simulation")
        
        circuit = input("Enter circuit name (default: Interlagos): ").strip() or "Interlagos"
        race_name = input("Enter race name (default: Brazilian Grand Prix): ").strip() or "Brazilian Grand Prix"
        
        weekend_stages = [
            {"name": "Friday Practice", "stage": RaceStage.PRACTICE, "session": SessionType.FP1},
            {"name": "Saturday Qualifying", "stage": RaceStage.QUALIFYING, "session": SessionType.Q3},
            {"name": "Sunday Race - Good Result", "stage": RaceStage.POST_RACE, "result": RaceResult.PODIUM, "position": 2},
        ]
        
        for stage_info in weekend_stages:
            print(f"\n📅 {stage_info['name']}")
            print("-" * 30)
            
            self.agent.update_context(
                stage=stage_info["stage"],
                session_type=stage_info.get("session"),
                circuit_name=circuit,
                race_name=race_name,
                last_result=stage_info.get("result"),
                position=stage_info.get("position")
            )
            
            print("📱", self.agent.speak())
            print("💭", self.agent.think())
            
            input("Press Enter for next stage...")
    
    def run(self):
        """Main interface loop"""
        while True:
            try:
                self.display_menu()
                choice = self.get_user_choice()
                
                if choice == "0":
                    print("\n👋 Thanks for using F1 Racer AI Agent!")
                    break
                elif choice == "1":
                    self.generate_status_post()
                elif choice == "2":
                    self.reply_to_comment()
                elif choice == "3":
                    self.mention_person()
                elif choice == "4":
                    self.simulate_like()
                elif choice == "5":
                    self.update_context()
                elif choice == "6":
                    self.view_thoughts()
                elif choice == "7":
                    self.view_agent_info()
                elif choice == "8":
                    self.run_demo_scenarios()
                elif choice == "9":
                    self.race_weekend_simulation()
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Exiting F1 Agent. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

def main():
    """Main entry point"""
    try:
        interface = F1AgentInterface()
        interface.run()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
