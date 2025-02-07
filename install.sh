#!/bin/bash

# Exit on any error
set -e

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing NGINX..."
sudo apt install nginx -y

echo "Starting and enabling NGINX..."
sudo systemctl start nginx
sudo systemctl enable nginx

echo "Cloning website from GitHub..."
GIT_REPO_URL="https://github.com/naeimrezaeian/rudn3.git"
TARGET_DIR="/var/www/assistant"
sudo rm -rf $TARGET_DIR
sudo git clone $GIT_REPO_URL $TARGET_DIR

echo "Setting permissions for web directory..."
sudo chown -R www-data:www-data $TARGET_DIR
sudo chmod -R 755 $TARGET_DIR

echo "Configuring NGINX..."
NGINX_CONFIG="/etc/nginx/sites-available/assistant"
sudo tee $NGINX_CONFIG > /dev/null <<EOF
server {
    listen 80;
    server_name assistant.local;

    root $TARGET_DIR;
    index index.html index.htm;

    location / {
        try_files \$uri \$uri/ =404;
    }
}
EOF

echo "Enabling NGINX configuration..."
sudo ln -sf $NGINX_CONFIG /etc/nginx/sites-enabled/
sudo unlink /etc/nginx/sites-enabled/default

echo "Testing NGINX configuration..."
sudo nginx -t

echo "Restarting NGINX..."
sudo systemctl restart nginx

echo "Installation and setup complete!"
echo "Don't forget to update your /etc/hosts file with:"
echo "127.0.0.1 assistant.local"
