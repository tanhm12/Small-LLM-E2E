branch=$(git branch --show-current)
commit_head=$(git log -1 --format="%H")
echo $branch > repo_info.txt
echo $commit_head >> repo_info.txt

set -e


IMAGE_VERSION=$(date "+%Y%m%d")
commit_head=$(echo $commit_head | sed -e "s/^\(.\{8\}\).*/\1/" )
docker build -t Zayt/pythia-engbot:${IMAGE_VERSION}-${commit_head} -f Dockerfile \
    --build-arg CACHEBUST=$(date +%s) \
    --build-arg commit_head=$commit_head --build-arg branch=$branch \
    . 2>&1 