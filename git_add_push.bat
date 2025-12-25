@echo off
cd /d "C:\Users\ofek1\OneDrive\שולחן העבודה\Master's\Courses\Semester A\NLP\HW\HW2\NLP_HW2"

if not exist .git (
    git init
)

git add .

git status --porcelain >nul
if %errorlevel% equ 0 (
    git commit -m "Add all new files"
    
    git remote -v >nul 2>&1
    if %errorlevel% equ 0 (
        git push
    ) else (
        echo No remote repository configured. Files have been committed locally.
        echo To push, first add a remote: git remote add origin ^<url^>
    )
) else (
    echo No changes to commit.
)

