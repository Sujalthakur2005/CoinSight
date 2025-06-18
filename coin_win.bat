@echo off
color 0A
title Crypto Tools Installer

echo Checking for Python...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Python is not found. Please install Python 3.x from the official website:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During Python installation, make sure to check the box that says:
    echo "Add Python to PATH" or "Add Python 3.x to PATH".
    echo.
    echo After installing, please run this script again.
    pause
    exit /b
) else (
    echo Python found.
)

echo.
echo Installing required Python packages...

:: Check if pip is installed
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Pip is not found. Attempting to install pip...
    python -m ensurepip --default-pip
    if %errorlevel% neq 0 (
        echo Error: Could not install pip. Please install pip manually or check your Python installation.
        pause
        exit /b
    )
    echo Pip installed.
)

:: Upgrade pip to ensure it's up-to-date
python -m pip install --upgrade pip

:: Install all necessary packages
pip install requests numpy scikit-learn pycoingecko art

if %errorlevel% neq 0 (
    echo.
    echo An error occurred during package installation.
    echo Please check the error messages above. Common issues include:
    echo 1. No internet connection.
    echo 2. Outdated pip.
    echo 3. Permissions issues (try running as Administrator).
    echo.
    pause
    exit /b
) else (
    echo.
    echo All packages installed successfully!
    echo You can now run your Python script.
    echo.
)

pause
