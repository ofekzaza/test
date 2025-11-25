# Requires: Docker Desktop
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Helpers: Docker availability checks ---
function Test-DockerCli {
    try { docker --version | Out-Null; return $true } catch { return $false }
}

function Test-DockerEngine {
    try { docker version --format '{{.Server.Version}}' | Out-Null; return $true } catch { return $false }
}

function Ensure-DockerDesktop {
    if (-not (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue)) {
        $dockerDesktopExe = Join-Path $Env:ProgramFiles "Docker\\Docker\\Docker Desktop.exe"
        if (Test-Path $dockerDesktopExe) {
            Start-Process -FilePath $dockerDesktopExe | Out-Null
        }
    }
}

# ===== Basic constants =====
# Change these as needed
$ImageName = "test-fastapi"
$ImageTag = "latest"
$ContainerName = "$ImageName-container"
$HostPort = 8000          # Host port to expose
$ContainerPort = 8000      # Container port (matches uvicorn in Dockerfile)
$Dockerfile = "Dockerfile" # Path to Dockerfile
$BuildContext = "."       # Build context directory

# Optional constants (leave empty to skip)
$Platform = ""            # e.g. "linux/amd64" or "linux/arm64"
$Network = ""             # e.g. "bridge" or a custom network name

Write-Host "[0/4] Checking Docker Desktop / Engine readiness" -ForegroundColor Cyan
if (-not (Test-DockerCli)) {
    Write-Error "Docker CLI not found. Please install Docker Desktop and reopen PowerShell."
}
Ensure-DockerDesktop

$timeoutSec = 180
$startTime = Get-Date
while (-not (Test-DockerEngine)) {
    if ((New-TimeSpan -Start $startTime -End (Get-Date)).TotalSeconds -gt $timeoutSec) {
        Write-Error "Docker engine not ready after $timeoutSec seconds."
        Write-Host "Tips:" -ForegroundColor Yellow
        Write-Host "- Open Docker Desktop and wait until it shows 'Docker is running'" -ForegroundColor Yellow
        Write-Host "- Ensure WSL2 backend is enabled in Docker Desktop settings (General)" -ForegroundColor Yellow
        Write-Host "- If using corporate proxy/VPN, ensure Docker is allowed" -ForegroundColor Yellow
        exit 1
    }
    Start-Sleep -Seconds 3
}

# Ensure Linux container mode/context
try {
    $currentContext = (docker context show).Trim()
} catch {
    $currentContext = ""
}

if ($currentContext -match "windows") {
    Write-Error "Docker Desktop is in Windows containers mode. Switch to Linux containers from the Docker Desktop tray icon (Switch to Linux containers)."
    exit 1
}

# Try to switch to desktop-linux context if available and not already selected
try {
    $contexts = docker context ls --format '{{.Name}}'
    if ($contexts -match 'desktop-linux' -and $currentContext -ne 'desktop-linux') {
        docker context use desktop-linux | Out-Null
    }
} catch {
    # Non-fatal; continue with current context
}

Write-Host "[1/4] Building image: $ImageName:$ImageTag" -ForegroundColor Cyan
$buildArgs = @("build", "-t", "$ImageName:$ImageTag", "-f", $Dockerfile)
if ($Platform -ne "") { $buildArgs += @("--platform", $Platform) }
$buildArgs += $BuildContext
docker @buildArgs

Write-Host "[2/4] Removing existing container if present: $ContainerName" -ForegroundColor Cyan
$existingId = docker ps -a --filter "name=^$ContainerName$" -q
if ($existingId) {
    docker rm -f $existingId | Out-Null
}

Write-Host "[3/4] Running container: $ContainerName (http://localhost:$HostPort)" -ForegroundColor Cyan
$runArgs = @("run", "-d", "--name", $ContainerName, "-p", "$HostPort:$ContainerPort")
if ($Network -ne "") { $runArgs += @("--network", $Network) }
$runArgs += "$ImageName:$ImageTag"
docker @runArgs

Write-Host "[4/4] Done. Open: http://localhost:$HostPort/docs (Swagger) | http://localhost:$HostPort/redoc (ReDoc)" -ForegroundColor Green
Write-Host "To view logs: docker logs -f $ContainerName"
Write-Host "To stop/remove: docker rm -f $ContainerName"


