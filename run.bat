@echo off
call "D:\UTOKNOI-ForeCast\env\Scripts\activate.bat"
python D:\UTOKNOI-ForeCast\main.py
echo Close in 5 Seconds
TIMEOUT /T 5 /NOBREAK
exit