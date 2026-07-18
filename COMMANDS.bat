@echo off
echo ============================================================
echo  TheBrain v13 - SIMPLE TRAINING
echo ============================================================
echo.
echo Open TheBrain, then paste ONE line:
echo.
echo   easytrain
echo.
echo That is all. It trains on the data\ folder automatically:
echo   - English Q&A  (data\conv\*.conv)
echo   - Arabic       (data\conv\ar\)
echo   - French       (data\conv\fr\)
echo   - Text notes   (data\text\*.txt)
echo.
echo Optional:
echo   easytrain data 50     (50 epochs instead of default 30)
echo.
echo After training, just type questions:
echo   hello
echo   How does ransomware work?
echo   trainstatus
echo.
pause
