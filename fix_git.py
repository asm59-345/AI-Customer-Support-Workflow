import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing {cmd}:\n{result.stderr}")
    print(result.stdout)
    return result.returncode == 0

print("Fixing GitHub Push Protection Error...")
run_cmd("git add upload_to_hf.py")
run_cmd("git commit --amend --no-edit")
print("\nNow attempting to push to GitHub again...")
run_cmd("git push -u origin main --force")
print("\nDone! If you still see auth errors, run 'git push -u origin main --force' manually.")
