@echo off
title 卷积核实验室 - 卷出无限效果
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% -c "import sys" >nul 2>nul
if errorlevel 1 goto NOPY

%PY% -c "import streamlit, numpy, matplotlib, scipy" >nul 2>nul
if errorlevel 1 goto INSTALL

goto RUN

:INSTALL
echo.
echo  首次运行，正在安装依赖，请稍候（约 1~3 分钟）...
echo.
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto PIPFAIL

:RUN
set "PORT=8501"
for /f %%p in ('%PY% "%~dp0find_port.py"') do set "PORT=%%p"
echo.
echo  正在启动，浏览器会自动打开 http://localhost:%PORT%
echo  关闭本窗口或按 Ctrl+C 即可停止。
echo.
%PY% -m streamlit run app.py --server.port %PORT% --server.headless false
goto END

:NOPY
echo.
echo  [错误] 没有找到 Python
echo  请先安装 Python 3.9 或更高版本，安装时务必勾选 Add Python to PATH
echo  下载地址：https://www.python.org/downloads/
goto END

:PIPFAIL
echo.
echo  依赖安装失败，可以试试国内镜像：
echo    %PY% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
goto END

:END
echo.
echo  服务已停止。按任意键关闭窗口。
pause >nul
