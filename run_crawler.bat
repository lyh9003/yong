@echo off
echo [1/5] Loading env...

if not exist .env (
    echo ERROR: .env file not found.
    pause
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if "%%a"=="OPENAI_API_KEY" set OPENAI_API_KEY=%%b
)

if "%OPENAI_API_KEY%"=="" (
    echo ERROR: OPENAI_API_KEY not set.
    pause
    exit /b 1
)

echo [2/5] Checking dependencies...
python -c "import bs4, requests, tqdm, pandas, openai, rapidfuzz" 2>nul
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt -q
    if %errorlevel% neq 0 (
        echo ERROR: pip install failed.
        pause
        exit /b 1
    )
)

echo [3/5] Git pull...
git pull origin main
if %errorlevel% neq 0 (
    echo ERROR: git pull failed.
    pause
    exit /b 1
)

echo [4/5] Running crawler...
python naver_news.py
if %errorlevel% neq 0 (
    echo ERROR: crawler failed.
    pause
    exit /b 1
)

echo [5/5] Committing...
git add Total_Filtered_No_Comment.csv archive/
git diff --cached --quiet
if %errorlevel% equ 0 (
    echo No new articles collected.
    pause
    exit /b 0
)
git commit -m "Update Total_Filtered_No_Comment.csv [skip ci]"

echo [6/6] Pushing...
git pull --rebase origin main
git push origin main
if %errorlevel% neq 0 (
    echo ERROR: git push failed.
    pause
    exit /b 1
)

echo Done!
pause
