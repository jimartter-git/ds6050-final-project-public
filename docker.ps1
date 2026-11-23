# Run a command in the dev container, using the GPU when Docker can reach one.
# Usage: .\docker.ps1 [command...]   (no command opens a shell)
Set-Location $PSScriptRoot
$image = 'ds6050-penn-action'

if (-not (Get-ChildItem 'data/Penn_Action/labels' -Filter '*.mat' -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    Write-Host 'Penn Action labels not found; real-data runs need: .\docker.ps1 python scripts/download_data.py'
}

docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) {
    docker compose -f docker/docker-compose.yml build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$files = @('-f', 'docker/docker-compose.yml')
docker run --rm --gpus all $image python -c 'import sys, torch; sys.exit(not torch.cuda.is_available())' *> $null
if ($LASTEXITCODE -eq 0) {
    $files += @('-f', 'docker/docker-compose.gpu.yml')
    Write-Host 'GPU detected: container will use CUDA.'
} else {
    Write-Host 'No usable GPU: container will use CPU.'
}

docker compose @files run --rm dev @args
exit $LASTEXITCODE
