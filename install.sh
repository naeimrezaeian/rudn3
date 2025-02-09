#!/bin/bash

# Exit on any error
set -e

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y



echo "Allowing traffic on port 80..."
sudo ufw allow 80/tcp

if [ $? -eq 0 ]; then
  echo "Port 80 opened successfully."
else
  echo "Error: Failed to open port 80."
fi
echo "Restarting UFW..."
sudo ufw disable
sudo ufw enable

echo "Installing NGINX..."
sudo apt install nginx -y

echo "Starting and enabling NGINX..."
sudo systemctl start nginx
sudo systemctl enable nginx


echo "Cloning website from GitHub..."
GIT_REPO_URL="https://github.com/naeimrezaeian/rudn3.git"
TARGET_DIR="/var/www/assistant"
TEMP_DIR="/tmp/rudn3_clone"

sudo git clone $GIT_REPO_URL $TEMP_DIR


echo "Copying files from website folder to $TARGET_DIR"
sudo mkdir -p "$TARGET_DIR"
sudo cp -r "$TEMP_DIR/web/"* "$TARGET_DIR"

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

echo "Cleaning up temporary directory..."
sudo rm -rf "$TEMP_DIR"

echo "Installation and setup complete!"
echo "Don't forget to update your /etc/hosts file with:"
echo "127.0.0.1 assistant.local"
