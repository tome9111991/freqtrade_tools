@echo off
:: Erzwinge UTF-8 Codierung
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo Starte Bitcoin Bottom Analyzer...
echo =================================

:: Prüfe ob die virtuelle Umgebung existiert
if not exist ".venv\" (
    echo Keine virtuelle Umgebung gefunden. Erstelle .venv...
    py -m venv .venv
    if errorlevel 1 (
        echo FEHLER: Python py konnte nicht gefunden werden. Bitte installiere Python.
        pause
        exit /b
    )
    
    echo Installiere Abhaengigkeiten aus requirements.txt...
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\pip install -r requirements.txt
    if errorlevel 1 (
        echo FEHLER: Installation der Pakete fehlgeschlagen.
        pause
        exit /b
    )
    echo Setup abgeschlossen!
    echo.
)

:: Aktiviere die virtuelle Python-Umgebung
call .\.venv\Scripts\activate.bat

:: Starte das Skript
python btc_analyzer.py

:: Halte das Fenster offen, damit man die Ergebnisse lesen kann
echo.
pause