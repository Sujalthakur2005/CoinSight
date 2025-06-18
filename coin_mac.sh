#!/bin/bash

echo "Checking for Python..."

# Check if python3 is available and preferred
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
    echo "Using python3."
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
    echo "Using default python."
else
    echo ""
    echo "Python is not found. Please install Python 3.x. You can download it from:"
    echo "https://www.python.org/downloads/mac-os-x/"
    echo "Alternatively, you can install it via Homebrew: brew install python"
    echo ""
    echo "After installing, please run this script again."
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "Installing required Python packages..."

# Ensure pip is installed for the selected Python command
if ! "$PYTHON_CMD" -m pip --version &>/dev/null; then
    echo "pip is not found. Attempting to install pip..."
    "$PYTHON_CMD" -m ensurepip --default-pip
    if [ $? -ne 0 ]; then
        echo "Error: Could not install pip. Please install pip manually or check your Python installation."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "pip installed."
fi

# Upgrade pip to ensure it's up-to-date
"$PYTHON_CMD" -m pip install --upgrade pip

# Install all necessary packages
"$PYTHON_CMD" -m pip install requests numpy scikit-learn pycoingecko art

if [ $? -ne 0 ]; then
    echo ""
    echo "An error occurred during package installation."
    echo "Please check the error messages above. Common issues include:"
    echo "1. No internet connection."
    echo "2. Outdated pip (try running 'pip install --upgrade pip')."
    echo "3. Permissions issues (you might need to use 'sudo' before the pip install command, but it's generally not recommended for user packages)."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
else
    echo ""
    echo "All packages installed successfully!"
    echo "You can now run your Python script using: $PYTHON_CMD your_script_name.py"
    echo ""
fi

read -p "Press Enter to exit..."
