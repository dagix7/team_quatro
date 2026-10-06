# Deployment Guide - AWS Lightsail

## Server Info
- **IP:** http://3.126.82.53/
- **SSH Key:** `C:\Users\Kendie\Downloads\LightsailDefaultKey-eu-central-1 (2).pem`
- **Docker:** Pre-installed

## Step-by-Step Deployment

### 1. Connect to Server

From Windows PowerShell:

```powershell
# Set correct permissions for PEM file (if needed)
icacls "C:\Users\Kendie\Downloads\LightsailDefaultKey-eu-central-1 (2).pem" /inheritance:r
icacls "C:\Users\Kendie\Downloads\LightsailDefaultKey-eu-central-1 (2).pem" /grant:r "%USERNAME%:R"

# Connect via SSH
ssh -i "C:\Users\Kendie\Downloads\LightsailDefaultKey-eu-central-1 (2).pem" ubuntu@3.126.82.53
```

### 2. Clean Up Existing Application

```bash
# Check what's running
docker ps -a

# Stop and remove all containers
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true

# Remove old images (optional)
docker system prune -af
```

### 3. Check System Resources

```bash
# Check swap memory
free -h
swapon --show

# If no swap, create 2GB swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Check disk space
df -h

# Check memory
free -h
```

### 4. Clone/Update Repository

```bash
# If first time
cd ~
git clone https://github.com/dagix7/team_quatro.git
cd team_quatro

# If already exists
cd ~/team_quatro
git fetch origin
git checkout main
git pull origin main
```

### 5. Prepare Assets

```bash
# Make sure you're on main branch with latest code
git status

# Prepare demo assets
cd app
python3 prepare_assets.py
cd ..
```

### 6. Deploy with Docker

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### 7. Verify Deployment

```bash
# Check container status
docker ps

# Check logs
docker-compose logs -f

# Test locally
curl http://localhost:8501/_stcore/health

# Test externally
curl http://3.126.82.53/
```

### 8. Access the App

Open in browser: **http://3.126.82.53/**

## Quick Commands

```bash
# View logs
docker-compose logs -f streamlit-app

# Restart app
docker-compose restart

# Stop app
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Check resource usage
docker stats addis-ride-forecast
```

## Troubleshooting

### App won't start
```bash
# Check logs
docker-compose logs

# Check if port 8501 is in use
sudo netstat -tlnp | grep 8501

# Try manual run to see errors
docker-compose up
```

### Out of memory
```bash
# Check memory
free -h

# Add/increase swap
sudo fallocate -l 4G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Port 80 already in use
```bash
# Find what's using port 80
sudo netstat -tlnp | grep :80

# Stop Apache/Nginx if running
sudo systemctl stop apache2
sudo systemctl stop nginx

# Or change docker-compose.yml to use different port
# ports:
#   - "8080:8501"  # Then access at http://3.126.82.53:8080
```

### Can't pull from GitHub
```bash
# Set up SSH key or use HTTPS with token
git config --global credential.helper store
git pull
```

## Alternative: Manual Deployment (No Docker)

If Docker has issues:

```bash
# Install Python
sudo apt update
sudo apt install python3.11 python3-pip python3-venv -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r app/requirements.txt

# Prepare assets
cd app && python prepare_assets.py && cd ..

# Run with nohup
nohup streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

# Check if running
ps aux | grep streamlit

# Stop
pkill -f streamlit
```

## Security Notes

- The app is running on port 80 (HTTP only)
- For production, consider adding HTTPS with Let's Encrypt
- Keep the SSH key secure
- Regularly update packages: `docker-compose pull && docker-compose up -d`
