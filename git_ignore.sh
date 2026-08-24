# This is a Config file not  bash/executable


cat <<EOT > .gitignore
# Exclude virtual environment
thesis/
venv/
ENV/

# Exclude Python byte code & cache
__pycache__/
*.pyc

# Exclude temporary OS / IDE files
.DS_Store
.vscode/
.idea/
EOT
