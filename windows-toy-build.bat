@echo off
call buildconf.bat
if errorlevel 1 exit /b 1
call configure.bat --disable-all --enable-cli --enable-toy=shared
if errorlevel 1 exit /b 1
nmake
if errorlevel 1 exit /b 1
exit /b 0
