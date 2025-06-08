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
    """Interactive interface for the F1 Racer AI Agent with mandatory setup"""
    
    def __init__(self):
        self.agent = None
        self.setup_complete = False
        self.context_configured = False
    
    def initial_setup(self):
        """Mandatory initial setup before accessing main features"""
        print("🏁 Welcome to the F1 Racer AI Agent!")
        print("=" * 60)
        print("📋 MANDATORY SETUP - Please configure your agent first")
        print()
        
        # Step 1: Agent Configuration
        print("🔧 STEP 1: Agent Configuration")
        print("-" * 30)
        
        while True:
            racer_name = input("Enter racer name (e.g., Lewis Hamilton): ").strip()
            if racer_name:
                break
            print("❌ Racer name is required. Please try again.")
        
        while True:
            team_name = input("Enter team name (e.g., Mercedes AMG): ").strip()
            if team_name:
                break
            print("❌ Team name is required. Please try again.")
        
        self.agent = F1RacerAgent(racer_name, team_name)
        print(f"\n✅ Agent created: {racer_name} from {team_name}")
        self.setup_complete = True
        
        # Step 2: Initial Context Setup
        print(f"\n🏎️  STEP 2: Initial Race Context Setup")
        print("-" * 30)
        print("💡 Setting up race context ensures realistic and appropriate responses")
        print()
        
        self.mandatory_context_setup()
        
        print("\n" + "=" * 60)
        print("✅ SETUP COMPLETE! Your F1 Agent is ready for action!")
        print(f"👤 Driver: {racer_name}")
        print(f"🏁 Team: {team_name}")
        print(f"📍 Current Context: {self.agent.context.stage.value.title()} at {self.agent.context.circuit_name}")
        print("=" * 60)
        input("\nPress Enter to continue to main menu...")
    
    def mandatory_context_setup(self):
        """Mandatory context setup - simplified but required"""
        
        # Stage selection (required)
        print("🏁 Select current race stage:")
        stages = {
            "1": (RaceStage.PRACTICE, "Practice Session"),
            "2": (RaceStage.QUALIFYING, "Qualifying"),
            "3": (RaceStage.RACE, "Race Day"),
            "4": (RaceStage.POST_RACE, "Post-Race")
        }
        
        for key, (stage, description) in stages.items():
            print(f"{key}. {description}")
        
        while True:
            stage_choice = input("Choose stage (1-4): ").strip()
            if stage_choice in stages:
                selected_stage, stage_desc = stages[stage_choice]
                break
            print("❌ Please select a valid option (1-4)")
        
        # Circuit and race name (required)
        while True:
            circuit = input("Enter circuit name (e.g., Monaco, Silverstone): ").strip()
            if circuit:
                break
            print("❌ Circuit name is required.")
        
        while True:
            race_name = input("Enter race name (e.g., Monaco Grand Prix): ").strip()
            if race_name:
                break
            print("❌ Race name is required.")
        
        # Optional session type for practice/qualifying
        session_type = None
        if selected_stage in [RaceStage.PRACTICE, RaceStage.QUALIFYING]:
            print(f"\n📋 Select session type for {stage_desc}:")
            if selected_stage == RaceStage.PRACTICE:
                sessions = {"1": SessionType.FP1, "2": SessionType.FP2, "3": SessionType.FP3}
            else:
                sessions = {"1": SessionType.Q1, "2": SessionType.Q2, "3": SessionType.Q3}
            
            for key, session in sessions.items():
                print(f"{key}. {session.value}")
            
            session_choice = input("Choose session (or press Enter to skip): ").strip()
            if session_choice in sessions:
                session_type = sessions[session_choice]
        
        # Optional recent result for post-race
        last_result = None
        position = None
        if selected_stage == RaceStage.POST_RACE:
            print(f"\n📊 What was the result of the recent race?")
            results = {
                "1": (RaceResult.WIN, "Victory (P1)"),
                "2": (RaceResult.PODIUM, "Podium (P2-P3)"),
                "3": (RaceResult.POINTS, "Points finish (P4-P10)"),
                "4": (RaceResult.DNF, "Did Not Finish (DNF)"),
                "5": (RaceResult.CRASH, "Crash/Accident"),
                "6": (RaceResult.DISAPPOINTING, "Poor result but finished")
            }
            
            for key, (result, description) in results.items():
                print(f"{key}. {description}")
            
            result_choice = input("Choose result (or press Enter to skip): ").strip()
            if result_choice in results:
                last_result, result_desc = results[result_choice]
                
                if last_result in [RaceResult.WIN, RaceResult.PODIUM, RaceResult.POINTS]:
                    while True:
                        try:
                            pos_input = input(f"Enter finishing position for {result_desc}: ").strip()
                            if pos_input:
                                position = int(pos_input)
                                if 1 <= position <= 20:
                                    break
                                else:
                                    print("❌ Position must be between 1 and 20")
                            else:
                                break
                        except ValueError:
                            print("❌ Please enter a valid number")
        
        # Update agent context
        self.agent.update_context(
            stage=selected_stage,
            session_type=session_type,
            circuit_name=circuit,
            race_name=race_name,
            last_result=last_result,
            position=position
        )
        
        self.context_configured = True
        print(f"\n✅ Context configured: {stage_desc} at {circuit}")
        if last_result:
            print(f"📊 Recent result: {last_result.value.title()}" + (f" (P{position})" if position else ""))
    
    def display_menu(self):
        """Display the main menu options (only after setup)"""
        if not self.setup_complete or not self.context_configured:
            print("❌ Setup must be completed first!")
            return
            
        print(f"\n🏎️  F1 Agent Menu - {self.agent.racer_name} ({self.agent.team_name})")
        print(f"📍 Current: {self.agent.context.stage.value.title()} at {self.agent.context.circuit_name}")
        print("-" * 60)
        print("1. Generate Status Post")
        print("2. Reply to Fan Comment")
        print("3. Mention Teammate/Competitor") 
        print("4. Simulate Like Action")
        print("5. Update Race Context")
        print("6. View Agent Thoughts")
        print("7. View Agent Info")
        print("8. Run Demo Scenarios")
        print("9. Quick Race Weekend Simulation")
        print("10. Debug Content Generation")
        print("11. Reconfigure Agent")  # New option
        print("0. Exit")
    
    def reconfigure_agent(self):
        """Allow user to reconfigure agent and context"""
        print("\n🔧 Agent Reconfiguration")
        print("=" * 40)
        
        choice = input("What would you like to reconfigure?\n1. Agent details (name, team)\n2. Race context only\nChoice (1-2): ").strip()
        
        if choice == "1":
            print("\n👤 Reconfiguring agent details...")
            racer_name = input(f"Enter new racer name (current: {self.agent.racer_name}): ").strip()
            team_name = input(f"Enter new team name (current: {self.agent.team_name}): ").strip()
            
            if racer_name or team_name:
                self.agent = F1RacerAgent(
                    racer_name or self.agent.racer_name,
                    team_name or self.agent.team_name
                )
                print("✅ Agent details updated!")
                
                # Must reconfigure context for new agent
                print("\n🏁 Setting up context for reconfigured agent...")
                self.mandatory_context_setup()
            else:
                print("❌ No changes made.")
        
        elif choice == "2":
            print("\n🏁 Reconfiguring race context...")
            self.mandatory_context_setup()
        
        else:
            print("❌ Invalid choice.")
    
    def check_setup_status(self) -> bool:
        """Check if setup is complete before allowing actions"""
        if not self.setup_complete or not self.context_configured:
            print("\n❌ Setup is not complete!")
            print("💡 Please restart the application to complete setup.")
            return False
        return True
    
    def get_user_choice(self) -> str:
        """Get user menu choice"""
        return input("\nEnter your choice (0-10): ").strip()
    
    def generate_status_post(self):
        """Generate a status post"""
        print("\n📱 Generating Status Post...")
        
        context_types = {
            "1": "general",
            "2": "win", 
            "3": "podium",
            "4": "loss",  # Changed from disappointing
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
    
    def debug_content_generation(self):
        """Debug content generation with detailed output"""
        print("\n🔧 Debug Content Generation")
        
        context_types = {
            "1": "win",
            "2": "podium", 
            "3": "loss",  # Changed from disappointing
            "4": "practice",
            "5": "qualifying",
            "6": "general"
        }
        
        print("Select context type to debug:")
        for key, value in context_types.items():
            print(f"{key}. {value.title()}")
        
        choice = input("Enter choice (1-6, default: win): ").strip()
        context_type = context_types.get(choice, "win")
        
        try:
            debug_info = self.agent.debug_generation(context_type)
            print("\n" + "="*50)
            print("🎯 Debug completed! Check output above for details.")
        except Exception as e:
            print(f"❌ Debug failed: {e}")
            print("💡 Try updating the context first (option 5)")
    
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
                elif choice == "10":
                    self.debug_content_generation()
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