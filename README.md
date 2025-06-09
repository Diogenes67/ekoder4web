EKoder-4o Enhanced
A tool that helps emergency doctors find the right medical codes for patient diagnoses.
What You Need Before Starting
1A Mac computer (Windows instructions coming soon)
2An OpenAI API key - Get one at platform.openai.com
3About 30 minutes for first-time setup
Step-by-Step Setup Instructions
Step 1: Open Terminal
•Press Command + Space on your keyboard
•Type "Terminal"
•Press Enter
•A black/white window will open - this is Terminal
Step 2: Install Anaconda (if you don't have it)
1Go to anaconda.com/download
2Download Anaconda for Mac
3Double-click the downloaded file
4Follow the installer (keep clicking "Continue" and "Agree")
Step 3: Download EKoder
In Terminal, copy and paste each line below, pressing Enter after each:
cd ~/Desktop
git clone https://github.com/Diogenes67/ekoder4o-enhanced.git
cd ekoder4o-enhanced
Step 4: Create the Environment
Copy and paste this line in Terminal and press Enter:
conda create -n ekoder4o python=3.11 -y
Wait for it to finish (might take 2-3 minutes).
Step 5: Activate the Environment
Copy and paste this line and press Enter:
conda activate ekoder4o
Step 6: Install Required Programs
Copy and paste this line and press Enter:
pip install streamlit pandas numpy scikit-learn openai openpyxl
Wait for everything to install (might take 5 minutes).
Step 7: Set Up Your OpenAI Key
Replace your-key-here with your actual OpenAI API key:
export OPENAI_API_KEY='your-key-here'
⚠️ Important: Your key should look something like: sk-proj-abcd1234...
Step 8: Run EKoder
Copy and paste this line and press Enter:
streamlit run ekoder_4o_enhanced.py
🎉 Success! Your web browser should open with EKoder running!
How to Use EKoder Daily
Starting EKoder Each Time
1Open Terminal
2Copy and paste these lines, pressing Enter after each:
cd ~/Desktop/ekoder4o-enhanced
conda activate ekoder4o
export OPENAI_API_KEY='your-key-here'
streamlit run ekoder_4o_enhanced.py
Using the Program
1Paste a clinical note in the text box
2Click "🚀 Process Note"
3Review the suggested diagnosis codes
4The program will show:
◦The most likely diagnosis
◦Other possible diagnoses to consider
◦All codes are ranked by relevance
Stopping the Program
•Go back to Terminal
•Press Control + C on your keyboard
Troubleshooting
"Command not found" error
•Make sure you installed Anaconda (Step 2)
•Try closing Terminal and opening it again
"ModuleNotFoundError"
•Run this command: pip install [missing-module-name]
•For example: pip install pandas
Program won't start
•Make sure you're in the right folder: cd ~/Desktop/ekoder4o-enhanced
•Make sure environment is active: conda activate ekoder4o
"Invalid API key"
•Check your OpenAI key is correct
•Make sure you included the quotes: export OPENAI_API_KEY='sk-proj-...'
Getting Help
If you're stuck:
1Take a screenshot of the error
2Note which step you're on
3Contact: [your-email@example.com]
Making It Even Easier
Want to start EKoder with just one click?
1Open Terminal
2Copy and paste:
echo '#!/bin/bash
cd ~/Desktop/ekoder4o-enhanced
conda activate ekoder4o
export OPENAI_API_KEY="your-key-here"
streamlit run ekoder_4o_enhanced.py' > ~/Desktop/Start_EKoder.command
3Press Enter
4Copy and paste:
chmod +x ~/Desktop/Start_EKoder.command
5Press Enter
Now you'll have a "Start_EKoder" icon on your Desktop - just double-click it to start!

Remember to replace:
•Diogenes67 with the actual GitHub username
•your-key-here with the actual OpenAI API key
•[your-email@example.com] with actual contact info

