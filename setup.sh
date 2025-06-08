
# 1. Create a virtual environment
py -m venv .venv
# Use the below if you have not aliased python on you system PATH
#python -m venv .venv

# 2. Add the VENV to .gignore to avoid committing it
echo "/.venv" >> .gitignore

# 3. Activate the virtual environment
source .venv/Scripts/activate  # I ran this on windows hence this Bash terminal command
# If you are on Linux or macOS, use the following command instead, just uncomment it:
# source .venv/bin/activate

# 4. Install the required packages
pip install -r requirements.txt

# 5. Run the setup script to download NLTK data and configure the models
python setup_nltk.py

# 6. Donwnload the required Spacey NLP models
python -m spacy download en_core_web_sm