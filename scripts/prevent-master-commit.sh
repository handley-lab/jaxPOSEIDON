#!/bin/bash
# Prevent commits directly to main branch

current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")

if [ "$current_branch" = "main" ] || [ "$current_branch" = "main" ]; then
    echo "❌ Direct commits to '$current_branch' branch are not allowed!"
    echo "Please create a feature branch instead:"
    echo "  git checkout -b feature/your-feature-name"
    echo "  # make your changes"
    echo "  # git add && git commit"
    echo "  # then merge via PR"
    exit 1
fi

exit 0
