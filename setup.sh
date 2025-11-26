#!/bin/bash

# =============================================================================
# GenAI Learning Environment Setup Script
# =============================================================================

set -e  # Exit on error

echo "🏥 GenAI Learning Environment Setup"
echo "===================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}📍 Working directory: $SCRIPT_DIR${NC}"
echo ""

# =============================================================================
# Step 1: Check Python version
# =============================================================================

echo -e "${BLUE}Step 1: Checking Python version...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# =============================================================================
# Step 2: Create virtual environment
# =============================================================================

echo -e "${BLUE}Step 2: Setting up virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    else
        echo -e "${YELLOW}→ Using existing virtual environment${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# =============================================================================
# Step 3: Activate virtual environment
# =============================================================================

echo -e "${BLUE}Step 3: Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# =============================================================================
# Step 4: Upgrade pip
# =============================================================================

echo -e "${BLUE}Step 4: Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# =============================================================================
# Step 5: Install dependencies
# =============================================================================

echo -e "${BLUE}Step 5: Installing dependencies...${NC}"
echo "This may take 2-3 minutes..."

if pip install -r requirements.txt --quiet; then
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Some packages may have failed to install${NC}"
    echo "Try running manually: pip install -r requirements.txt"
fi
echo ""

# =============================================================================
# Step 6: Set up environment file
# =============================================================================

echo -e "${BLUE}Step 6: Setting up environment file...${NC}"

if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp .env.example .env
        echo -e "${GREEN}✓ .env file created from template${NC}"
    else
        echo -e "${YELLOW}→ Keeping existing .env file${NC}"
    fi
else
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created from template${NC}"
fi
echo ""

# =============================================================================
# Step 7: Check for API keys
# =============================================================================

echo -e "${BLUE}Step 7: Checking API keys...${NC}"

if grep -q "sk-proj" .env || grep -q "sk-[a-zA-Z0-9]" .env; then
    echo -e "${GREEN}✓ API keys appear to be configured${NC}"
else
    echo -e "${YELLOW}⚠ No API keys detected in .env file${NC}"
    echo ""
    echo "You need to add your API keys to the .env file:"
    echo "  1. Open .env in a text editor"
    echo "  2. Add your OpenAI key: OPENAI_API_KEY=sk-..."
    echo "  3. Add your Anthropic key: ANTHROPIC_API_KEY=sk-ant-..."
    echo ""
    echo "Get API keys from:"
    echo "  • OpenAI: https://platform.openai.com/api-keys"
    echo "  • Anthropic: https://console.anthropic.com/"
    echo ""
fi
echo ""

# =============================================================================
# Step 8: Test installation
# =============================================================================

echo -e "${BLUE}Step 8: Testing installation...${NC}"

# Test if packages are importable
python3 << EOF
import sys
try:
    import openai
    import anthropic
    import langchain
    import numpy
    import pandas
    print("✓ All core packages are importable")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Installation test passed${NC}"
else
    echo -e "${RED}❌ Installation test failed${NC}"
    echo "Try running: pip install -r requirements.txt"
fi
echo ""

# =============================================================================
# Setup Complete
# =============================================================================

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Add your API keys to the .env file:"
echo "   ${YELLOW}nano .env${NC}  # or use your preferred editor"
echo ""
echo "2. Run your first example:"
echo "   ${YELLOW}cd level_1_fundamentals/01_basic_chat${NC}"
echo "   ${YELLOW}python main.py${NC}"
echo ""
echo "3. Read the learning roadmap:"
echo "   ${YELLOW}cat README.md${NC}"
echo ""
echo "📚 For detailed instructions, see: ${BLUE}QUICKSTART.md${NC}"
echo ""
echo "Remember to activate the virtual environment in new terminal sessions:"
echo "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "${BLUE}Happy learning! 🚀${NC}"
echo ""
