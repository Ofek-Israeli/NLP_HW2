$workspacePath = "C:\Users\ofek1\OneDrive\שולחן העבודה\Master's\Courses\Semester A\NLP\HW\HW2\NLP_HW2"
Set-Location $workspacePath

# Initialize git if needed
if (-not (Test-Path .git)) {
    git init
}

# Add all files
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    git commit -m "Add all new files"
    
    # Check if remote exists, if not ask user or skip push
    $remote = git remote -v
    if ($remote) {
        git push
    } else {
        Write-Host "No remote repository configured. Files have been committed locally."
        Write-Host "To push, first add a remote: git remote add origin <url>"
    }
} else {
    Write-Host "No changes to commit."
}

