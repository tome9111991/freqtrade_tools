@echo off
:: Erzwinge UTF-8 Codierung (Damit Emojis und Umlaute im Windows Terminal funktionieren)
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo Starte Bitcoin Bottom Analyzer...
echo =================================

:: Aktiviere die virtuelle Python-Umgebung
call .\.venv\Scripts\activate.bat

:: Starte das Skript
python btc_analyzer.py

:: Halte das Fenster offen, damit man die Ergebnisse lesen kann
echo.
pause
