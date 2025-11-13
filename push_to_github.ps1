# GitHub Repository Setup & Push Script
# Automated script to initialize and push Voice Deepfake Detection project

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║" -ForegroundColor Cyan -NoNewline
Write-Host "        GitHub Repository Setup & Push Automation         " -ForegroundColor White -NoNewline
Write-Host "║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# Configuration
$repoUrl = "https://github.com/anjo3902/Voice_Deepfake_Detection.git"
$branch = "main"
$commitMessage = "Initial commit: Voice Deepfake Detection system with AASIST architecture"

Write-Host "`n[STEP 1/8] Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "  ✅ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Git not found! Please install Git first." -ForegroundColor Red
    Write-Host "  Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[STEP 2/8] Creating .gitignore..." -ForegroundColor Yellow
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# PyTorch & Models
*.pth
*.pt
*.ckpt
!backend/checkpoints/zero_download_model.pth
!backend/checkpoints/best.pth

# Data & Datasets (too large for GitHub)
data/ASVspoof2019/
data/downloaded_dataset/
data/modern_tts_samples/
data/your_voice_samples/

# Logs
*.log
logs/
*.out

# OS
.DS_Store
Thumbs.db
desktop.ini

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Certificates (sensitive)
certificates/*.pem
certificates/*.pfx
!certificates/.gitkeep

# Uploads
backend/uploads/*
!backend/uploads/.gitkeep

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Temporary files
*.tmp
*.bak
*.swp
tmp/
temp/

# Node modules (if any)
node_modules/
package-lock.json
"@

Set-Content -Path ".gitignore" -Value $gitignoreContent -Encoding UTF8
Write-Host "  ✅ .gitignore created" -ForegroundColor Green

Write-Host "`n[STEP 3/8] Initializing Git repository..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "  ⚠️  Git repository already initialized" -ForegroundColor Yellow
} else {
    git init
    Write-Host "  ✅ Git repository initialized" -ForegroundColor Green
}

Write-Host "`n[STEP 4/8] Setting up branch..." -ForegroundColor Yellow
try {
    git branch -M $branch
    Write-Host "  ✅ Branch set to '$branch'" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Branch setup: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n[STEP 5/8] Adding remote origin..." -ForegroundColor Yellow
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "  ⚠️  Remote already exists: $existingRemote" -ForegroundColor Yellow
    Write-Host "  Updating remote URL..." -ForegroundColor Yellow
    git remote set-url origin $repoUrl
} else {
    git remote add origin $repoUrl
}
Write-Host "  ✅ Remote set to: $repoUrl" -ForegroundColor Green

Write-Host "`n[STEP 6/8] Staging files..." -ForegroundColor Yellow
Write-Host "  Adding all files (excluding .gitignore patterns)..." -ForegroundColor Gray

# Add all files
git add .

# Get status
$stagedFiles = git diff --cached --name-only
$fileCount = ($stagedFiles | Measure-Object).Count

Write-Host "  ✅ Staged $fileCount files" -ForegroundColor Green
Write-Host "`n  Key files being committed:" -ForegroundColor Cyan
Write-Host "    - README.md" -ForegroundColor White
Write-Host "    - backend/models/ (API + AASIST architecture)" -ForegroundColor White
Write-Host "    - frontend/dist/ (React production build)" -ForegroundColor White
Write-Host "    - backend/checkpoints/ (models: 2 files, ~42 MB)" -ForegroundColor White
Write-Host "    - serve_https.py, utils.py, train_comprehensive.py" -ForegroundColor White
Write-Host "    - certificates/ (SSL certs)" -ForegroundColor White

Write-Host "`n[STEP 7/8] Creating commit..." -ForegroundColor Yellow
try {
    git commit -m $commitMessage
    Write-Host "  ✅ Commit created successfully" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Commit: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n[STEP 8/8] Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes (uploading ~42 MB models)..." -ForegroundColor Gray

try {
    # Push to GitHub
    git push -u origin $branch 2>&1 | ForEach-Object {
        if ($_ -match "Username") {
            Write-Host "  ℹ️  Enter your GitHub credentials..." -ForegroundColor Cyan
        }
        Write-Host "  $_" -ForegroundColor Gray
    }
    
    Write-Host "`n  ✅ Successfully pushed to GitHub!" -ForegroundColor Green
    
} catch {
    Write-Host "`n  ❌ Push failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`n  Troubleshooting:" -ForegroundColor Yellow
    Write-Host "    1. Check GitHub credentials" -ForegroundColor White
    Write-Host "    2. Verify repository exists: $repoUrl" -ForegroundColor White
    Write-Host "    3. Check internet connection" -ForegroundColor White
    Write-Host "    4. Try manual push: git push -u origin main" -ForegroundColor White
    exit 1
}

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║" -ForegroundColor Green -NoNewline
Write-Host "                 ✅ PUSH COMPLETE! ✅                      " -ForegroundColor White -NoNewline
Write-Host "║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`n[REPOSITORY INFO]" -ForegroundColor Cyan
Write-Host "  URL: $repoUrl" -ForegroundColor White
Write-Host "  Branch: $branch" -ForegroundColor White
Write-Host "  Commit: $commitMessage" -ForegroundColor White

Write-Host "`n[VIEW YOUR REPOSITORY]" -ForegroundColor Cyan
Write-Host "  https://github.com/anjo3902/Voice_Deepfake_Detection" -ForegroundColor Green

Write-Host "`n[NEXT STEPS]" -ForegroundColor Yellow
Write-Host "  1. Visit your repository on GitHub" -ForegroundColor White
Write-Host "  2. Add a license (MIT recommended)" -ForegroundColor White
Write-Host "  3. Enable GitHub Pages (if needed)" -ForegroundColor White
Write-Host "  4. Add topics/tags: python, pytorch, deepfake, aasist" -ForegroundColor White
Write-Host "  5. Consider adding: CODE_OF_CONDUCT.md, CONTRIBUTING.md" -ForegroundColor White

Write-Host "`n[GIT COMMANDS REFERENCE]" -ForegroundColor Cyan
Write-Host "  Check status:    git status" -ForegroundColor Gray
Write-Host "  View log:        git log --oneline" -ForegroundColor Gray
Write-Host "  Pull changes:    git pull origin main" -ForegroundColor Gray
Write-Host "  Future commits:  git add . && git commit -m 'message' && git push" -ForegroundColor Gray

Write-Host ""
