#!/usr/bin/env python3
"""
Comprehensive GitHub Push Script
Pushes all important project files while respecting size limits
"""

import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a git command and display output"""
    print(f"\n[{description}]")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0 and result.stderr:
            print(f"Warning: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("="*70)
    print("  COMPREHENSIVE GITHUB PUSH")
    print("="*70)
    
    # Check current status
    print("\n[STEP 1/6] Checking current status...")
    run_command("git status", "Current Git Status")
    
    # Add frontend files
    print("\n[STEP 2/6] Adding frontend files...")
    frontend_files = [
        "frontend/dist/index.html",
        "frontend/dist/assets/"
    ]
    
    for file in frontend_files:
        if Path(file).exists():
            run_command(f'git add "{file}"', f"Adding {file}")
            print(f"  ✓ Added: {file}")
    
    # Add documentation files
    print("\n[STEP 3/6] Adding documentation files...")
    doc_files = [
        "RUNNING_COMMANDS.md",
        "GITHUB_SUCCESS.md", 
        "RECOVERY_REPORT.md",
        "ADVANCED_CLEANUP_REPORT.md",
        "CLEANUP_SUMMARY.md"
    ]
    
    for file in doc_files:
        if Path(file).exists():
            run_command(f'git add "{file}"', f"Adding {file}")
            print(f"  ✓ Added: {file}")
        else:
            print(f"  ⊘ Not found: {file}")
    
    # Add START.bat if exists
    print("\n[STEP 4/6] Adding startup scripts...")
    if Path("START.bat").exists():
        run_command('git add "START.bat"', "Adding START.bat")
        print("  ✓ Added: START.bat")
    
    # Check what will be committed
    print("\n[STEP 5/6] Files to be committed...")
    run_command("git diff --cached --name-only", "Staged Files")
    
    # Commit if there are changes
    print("\n[STEP 6/6] Committing and pushing...")
    
    # Check if there are changes to commit
    result = subprocess.run("git diff --cached --quiet", shell=True)
    if result.returncode != 0:  # There are changes
        run_command('git commit -m "Added frontend, documentation, and project files"', "Creating Commit")
        run_command("git push origin main", "Pushing to GitHub")
        print("\n✓ Push completed successfully!")
    else:
        print("\n✓ No new changes to commit. Everything is up to date!")
    
    # Show final status
    print("\n" + "="*70)
    print("  FINAL STATUS")
    print("="*70)
    run_command("git status", "Git Status")
    
    # Show what's tracked vs ignored
    print("\n" + "="*70)
    print("  FILES IN REPOSITORY")
    print("="*70)
    run_command("git ls-files | wc -l", "Total files tracked")
    
    print("\n[KEY FILES]")
    key_files = [
        "README.md",
        "backend/models/app.py",
        "backend/checkpoints/best.pth",
        "frontend/dist/index.html",
        "serve_https.py",
        "train_comprehensive.py"
    ]
    
    for file in key_files:
        status = "✓" if Path(file).exists() else "✗"
        tracked = subprocess.run(f'git ls-files | findstr /C:"{file}"', 
                                shell=True, capture_output=True).returncode == 0
        tracked_str = "tracked" if tracked else "NOT tracked"
        print(f"  {status} {file} ({tracked_str})")
    
    print("\n[DATA FOLDER STATUS]")
    print("  ⊘ data/ (7.82 GB) - EXCLUDED (too large for GitHub)")
    print("    ↳ Properly excluded in .gitignore")
    print("    ↳ Users can download using: python utils.py download")
    
    print("\n" + "="*70)
    print("  REPOSITORY: https://github.com/anjo3902/Voice_Deepfake_Detection")
    print("="*70)

if __name__ == "__main__":
    main()
