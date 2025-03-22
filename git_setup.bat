@echo off
echo DeepSeek Medical Fine-tuning Git Kurulum Scripti
echo ----------------------------------------

REM Git repo'yu başlat
git init

REM Dosyaları ekle
git add .

REM İlk commit'i oluştur
git commit -m "Initial commit: DeepSeek medical model fine-tuning setup"

REM GitHub repo bilgilerini al
set /p GITHUB_USER=GitHub kullanıcı adınız: 
set /p REPO_NAME=Repo adı (örn: deepseek-medical-ft): 

REM Remote'u ekle
git remote add origin https://github.com/%GITHUB_USER%/%REPO_NAME%.git

REM Main branch'i oluştur ve push et
git branch -M main
git push -u origin main

echo.
echo Kurulum tamamlandı!
echo GitHub reponuz: https://github.com/%GITHUB_USER%/%REPO_NAME%
echo.
pause 