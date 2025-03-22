@echo off
echo DeepSeek Fine-tuning Kurulum ve Başlatma Scripti
echo ----------------------------------------

REM SSH key kontrolü
if not exist "%USERPROFILE%\.ssh\id_rsa" (
    echo SSH key bulunamadı!
    echo SSH key oluşturmak için:
    echo ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    pause
    exit /b 1
)

REM Sunucu bilgilerini al
set /p SERVER_IP=Sunucu IP adresi: 
set /p SERVER_USER=Sunucu kullanıcı adı: 
set /p SERVER_DIR=Sunucuda çalışma dizini (örn: /home/user/deepseek_ft): 

REM SSH config dosyasını güncelle
powershell -Command "(Get-Content ssh_config) -replace 'your_server_ip', '%SERVER_IP%' | Set-Content ssh_config"
powershell -Command "(Get-Content ssh_config) -replace 'your_username', '%SERVER_USER%' | Set-Content ssh_config"

REM SSH config dosyasını kopyala
if not exist "%USERPROFILE%\.ssh" mkdir "%USERPROFILE%\.ssh"
copy ssh_config "%USERPROFILE%\.ssh\config"

REM SSH key'i sunucuya kopyala
echo SSH key sunucuya kopyalanıyor...
ssh-copy-id %SERVER_USER%@%SERVER_IP%

REM Sunucuda dizin oluştur
echo Sunucuda çalışma dizini oluşturuluyor...
ssh %SERVER_USER%@%SERVER_IP% "mkdir -p %SERVER_DIR%"

REM Dosyaları sunucuya kopyala
echo Dosyalar sunucuya kopyalanıyor...
scp -r ./* %SERVER_USER%@%SERVER_IP%:%SERVER_DIR%/

REM Sunucuda kurulumu başlat
echo Sunucuda kurulum başlatılıyor...
ssh %SERVER_USER%@%SERVER_IP% "cd %SERVER_DIR% && chmod +x setup.sh run_training.sh && ./setup.sh"

echo Kurulum tamamlandı!
echo Eğitimi başlatmak için:
echo ssh %SERVER_USER%@%SERVER_IP%
echo cd %SERVER_DIR%
echo ./run_training.sh

pause 