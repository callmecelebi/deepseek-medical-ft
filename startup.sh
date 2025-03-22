#!/bin/bash

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Hata kontrolü
set -e

echo -e "${YELLOW}DeepSeek Fine-tuning Kurulum ve Başlatma Scripti${NC}"
echo "----------------------------------------"

# SSH key kontrolü
if [ ! -f ~/.ssh/id_rsa ]; then
    echo -e "${RED}SSH key bulunamadı!${NC}"
    echo "SSH key oluşturmak için:"
    echo "ssh-keygen -t rsa -b 4096 -C \"your_email@example.com\""
    exit 1
fi

# Sunucu bilgilerini al
read -p "Sunucu IP adresi: " SERVER_IP
read -p "Sunucu kullanıcı adı: " SERVER_USER
read -p "Sunucuda çalışma dizini (örn: /home/user/deepseek_ft): " SERVER_DIR

# SSH config dosyasını güncelle
sed -i "s/your_server_ip/$SERVER_IP/" ssh_config
sed -i "s/your_username/$SERVER_USER/" ssh_config

# SSH config dosyasını kopyala
mkdir -p ~/.ssh
cp ssh_config ~/.ssh/config
chmod 600 ~/.ssh/config

# SSH key'i sunucuya kopyala
echo -e "${YELLOW}SSH key sunucuya kopyalanıyor...${NC}"
ssh-copy-id $SERVER_USER@$SERVER_IP

# Sunucuda dizin oluştur
echo -e "${YELLOW}Sunucuda çalışma dizini oluşturuluyor...${NC}"
ssh $SERVER_USER@$SERVER_IP "mkdir -p $SERVER_DIR"

# Dosyaları sunucuya kopyala
echo -e "${YELLOW}Dosyalar sunucuya kopyalanıyor...${NC}"
scp -r ./* $SERVER_USER@$SERVER_IP:$SERVER_DIR/

# Sunucuda kurulumu başlat
echo -e "${YELLOW}Sunucuda kurulum başlatılıyor...${NC}"
ssh $SERVER_USER@$SERVER_IP "cd $SERVER_DIR && chmod +x setup.sh run_training.sh && ./setup.sh"

echo -e "${GREEN}Kurulum tamamlandı!${NC}"
echo -e "Eğitimi başlatmak için:"
echo -e "ssh $SERVER_USER@$SERVER_IP"
echo -e "cd $SERVER_DIR"
echo -e "./run_training.sh" 