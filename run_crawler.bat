@echo off
chcp 65001 > nul
echo [1/5] 환경변수 로드 중...

if not exist .env (
    echo .env 파일이 없습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요.
    pause
    exit /b 1
)

for /f "tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="OPENAI_API_KEY" set OPENAI_API_KEY=%%b
)

if "%OPENAI_API_KEY%"=="" (
    echo OPENAI_API_KEY가 설정되지 않았습니다.
    pause
    exit /b 1
)

echo [2/5] 최신 데이터 가져오는 중...
git pull origin main
if %errorlevel% neq 0 (
    echo git pull 실패
    pause
    exit /b 1
)

echo [3/5] 크롤러 실행 중...
python naver_news.py
if %errorlevel% neq 0 (
    echo 크롤러 실행 실패
    pause
    exit /b 1
)

echo [4/5] 변경사항 커밋 중...
git add Total_Filtered_No_Comment.csv
git diff --cached --quiet
if %errorlevel% equ 0 (
    echo 새로 수집된 기사가 없습니다.
    pause
    exit /b 0
)
git commit -m "Update Total_Filtered_No_Comment.csv [skip ci]"

echo [5/5] GitHub에 업로드 중...
git push origin main
if %errorlevel% neq 0 (
    echo git push 실패
    pause
    exit /b 1
)

echo 완료!
pause
