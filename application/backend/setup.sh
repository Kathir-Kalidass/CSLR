#!/bin/bash
# ====================================
# CSLR Backend Setup Script
# ====================================

set -e  # Exit on error

echo "=================================================="
echo "CSLR Backend Setup"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/6] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if [[ $(echo "$python_version 3.10" | awk '{print ($1 >= $2)}') != "1" ]]; then
    echo -e "${RED}Error: Python 3.10+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"

# Create virtual environment
echo -e "\n${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}[3/6] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo -e "\n${YELLOW}[4/6] Installing dependencies...${NC}"
echo "This may take several minutes..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Install PyTorch with CUDA (if needed)
echo -e "\n${YELLOW}[5/6] Checking PyTorch installation...${NC}"
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}✓ PyTorch with CUDA already installed${NC}"
else
    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo -e "${GREEN}✓ PyTorch installed${NC}"
fi

# Create directories
echo -e "\n${YELLOW}[6/6] Creating required directories...${NC}"
mkdir -p checkpoints logs data uploads
echo -e "${GREEN}✓ Directories created${NC}"

# Copy environment file
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ Please edit .env file with your configuration${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Make scripts executable
chmod +x health_check.sh
chmod +x scripts/*.py 2>/dev/null || true

# Summary
echo ""
echo "=================================================="
echo -e "${GREEN}Setup Complete! 🎉${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv:     source venv/bin/activate"
echo "  2. Edit config:       nano .env"
echo "  3. Run health check:  ./health_check.sh"
echo "  4. Start server:      uvicorn app.main:app --reload"
echo ""
echo "Optional:"
echo "  - Generate API key:   python3 scripts/generate_api_key.py"
echo "  - Test API:          python3 scripts/test_api.py"
echo "  - Run with Docker:    docker build -t cslr-backend ."
echo ""
echo "=================================================="
