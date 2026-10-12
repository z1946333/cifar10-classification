@echo off
cd /d "%~dp0"
set "CIFAR_PYTHON=D:\conda_envs\yahui_torch\python.exe"
if not exist "%CIFAR_PYTHON%" (
  echo Python not found. Edit CIFAR_PYTHON in this launcher for your computer.
  pause
  exit /b 1
)
"%CIFAR_PYTHON%" -m tools.launch
pause
