rm -rf _build
make html
cd _build/html
touch .nojekyll
git init
git add .
git commit -m "gh-pages"
git remote add origin git@github.com-bpampel:bpampel/bdld.git
git push --force origin main:gh-pages
