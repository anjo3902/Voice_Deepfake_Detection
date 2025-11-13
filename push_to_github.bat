@echo off
echo ============================================================
echo    GitHub Repository Setup and Push Automation
echo ============================================================
echo.

REM Configuration
set REPO_URL=https://github.com/anjo3902/Voice_Deepfake_Detection.git
set BRANCH=main
set COMMIT_MESSAGE=Initial commit: Voice Deepfake Detection with AASIST

echo [STEP 1/8] Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found! Please install Git first.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK] Git found
echo.

echo [STEP 2/8] Creating .gitignore...
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo *.egg-info/
echo dist/
echo build/
echo *.egg
echo.
echo # Virtual Environment
echo .venv/
echo venv/
echo ENV/
echo env/
echo.
echo # PyTorch Models
echo *.pth
echo *.pt
echo *.ckpt
echo !backend/checkpoints/zero_download_model.pth
echo !backend/checkpoints/best.pth
echo.
echo # Data - too large for GitHub
echo data/ASVspoof2019/
echo data/downloaded_dataset/
echo data/modern_tts_samples/
echo data/your_voice_samples/
echo.
echo # Logs
echo *.log
echo logs/
echo *.out
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
echo desktop.ini
echo.
echo # IDE
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo.
echo # Certificates
echo certificates/*.pem
echo certificates/*.pfx
echo !certificates/.gitkeep
echo.
echo # Uploads
echo backend/uploads/*
echo !backend/uploads/.gitkeep
echo.
echo # Jupyter
echo .ipynb_checkpoints/
echo *.ipynb
echo.
echo # Temporary
echo *.tmp
echo *.bak
echo tmp/
echo temp/
echo.
echo # Node
echo node_modules/
echo package-lock.json
) > .gitignore
echo [OK] .gitignore created
echo.

echo [STEP 3/8] Initializing Git repository...
if exist .git (
    echo [INFO] Git repository already initialized
) else (
    git init
    echo [OK] Git repository initialized
)
echo.

echo [STEP 4/8] Setting up branch...
git branch -M %BRANCH%
echo [OK] Branch set to '%BRANCH%'
echo.

echo [STEP 5/8] Adding remote origin...
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    git remote add origin %REPO_URL%
    echo [OK] Remote added: %REPO_URL%
) else (
    git remote set-url origin %REPO_URL%
    echo [OK] Remote updated: %REPO_URL%
)
echo.

echo [STEP 6/8] Staging files...
echo Adding all files excluding .gitignore patterns...
git add .
echo [OK] Files staged
echo.
echo Key files being committed:
echo   - README.md
echo   - backend/models/ (API + AASIST)
echo   - frontend/dist/ (React build)
echo   - backend/checkpoints/ (2 models, ~42 MB)
echo   - Python scripts (utils.py, train_comprehensive.py, etc.)
echo.

echo [STEP 7/8] Creating commit...
git commit -m "%COMMIT_MESSAGE%"
if errorlevel 1 (
    echo [WARNING] Commit may have failed (or nothing to commit)
) else (
    echo [OK] Commit created
)
echo.

echo [STEP 8/8] Pushing to GitHub...
echo This may take a few minutes (uploading ~42 MB models)...
echo.
git push -u origin %BRANCH%
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed!
    echo.
    echo Troubleshooting:
    echo   1. Check GitHub credentials
    echo   2. Verify repository exists: %REPO_URL%
    echo   3. Check internet connection
    echo   4. Try manual push: git push -u origin main
    echo.
    pause
    exit /b 1
)
echo.

echo ============================================================
echo               PUSH COMPLETE!
echo ============================================================
echo.
echo [REPOSITORY INFO]
echo   URL: %REPO_URL%
echo   Branch: %BRANCH%
echo   Commit: %COMMIT_MESSAGE%
echo.
echo [VIEW YOUR REPOSITORY]
echo   https://github.com/anjo3902/Voice_Deepfake_Detection
echo.
echo [NEXT STEPS]
echo   1. Visit your repository on GitHub
echo   2. Add a license (MIT recommended)
echo   3. Add topics: python, pytorch, deepfake, aasist
echo   4. Star your own repo!
echo.
echo [GIT COMMANDS REFERENCE]
echo   Check status:    git status
echo   View log:        git log --oneline
echo   Pull changes:    git pull origin main
echo   Future commits:  git add . ^&^& git commit -m "message" ^&^& git push
echo.
pause
