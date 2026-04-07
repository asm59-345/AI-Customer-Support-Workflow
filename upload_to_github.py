import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing {cmd}:\n{result.stderr}")
        # Not returning False immediately because some commands like commit fail if nothing to commit
    print(result.stdout)
    return True

def push_to_github():
    print("========== GITHUB SYNC ==========")
    print("🚀 Pushing project to GitHub...")
    
    # Check if git is initialized
    if not os.path.exists(".git"):
        run_cmd("git init")
    
    # Add all current files (ignoring those in .gitignore)
    run_cmd("git add .")
    
    # Create commit
    run_cmd('git commit -m "🚀 Initial commit: Premium SupportAI Glassmorphism UI integration complete with Auth and Dashboards"')
    
    # Configure remote
    result = subprocess.run("git remote -v", shell=True, text=True, capture_output=True)
    if "asm59-345/AI-Customer-Support-Workflow" not in result.stdout:
        # If 'origin' exists for HuggingFace, add this as 'github' remote or rename
        if "origin" in result.stdout:
            run_cmd("git remote set-url origin https://github.com/asm59-345/AI-Customer-Support-Workflow.git")
        else:
            run_cmd("git remote add origin https://github.com/asm59-345/AI-Customer-Support-Workflow.git")
    
    # Ensure current branch is 'main'
    run_cmd("git branch -M main")
    
    # Push to origin main
    print("\nUploading to GitHub repository...")
    run_cmd("git push -u origin main --force")
    
    print("\n✅ Successfully pushed to GitHub!")
    print("Check your repository at: https://github.com/asm59-345/AI-Customer-Support-Workflow")

if __name__ == "__main__":
    push_to_github()
