@echo off
cd /d "%~dp0"
"D:\conda_envs\yahui_torch\python.exe" -u -m scripts.train --smoke --device cpu
pause
