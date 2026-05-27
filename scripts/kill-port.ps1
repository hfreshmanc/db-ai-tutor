param(
    [Parameter(Mandatory = $true)]
    [int]$Port
)

$connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $pids) {
        if ($processId -le 0) { continue }
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Write-Host "Killed PID $processId on port $Port"
        } catch {
            Write-Warning "Could not kill PID ${processId}: $_"
        }
    }
}

# uvicorn --reload 父进程退出后，子 worker 可能仍占用端口
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'uvicorn main:app|multiprocessing\.spawn' } |
    ForEach-Object {
        try {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
            Write-Host "Killed orphan python PID $($_.ProcessId)"
        } catch {
            Write-Warning "Could not kill orphan PID $($_.ProcessId): $_"
        }
    }

Start-Sleep -Seconds 1
$still = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($still) {
    $anyAlive = $false
    foreach ($conn in $still) {
        $op = [int]$conn.OwningProcess
        if ($op -gt 0 -and (Get-Process -Id $op -ErrorAction SilentlyContinue)) {
            $anyAlive = $true
            break
        }
    }
    if (-not $anyAlive) {
        Write-Warning "Port $Port still shows LISTENING but owning PID is gone (stale Windows state). Continuing; if uvicorn cannot bind, reboot once or close WSL/Hyper-V port forwards."
        exit 0
    }
    Write-Warning "Port $Port still in use. Close old terminals or end python.exe in Task Manager."
    exit 1
}

if (-not $connections) {
    Write-Host "Port $Port is free."
} else {
    Write-Host "Port $Port cleared."
}
