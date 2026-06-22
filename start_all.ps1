# ============================================================
#  start_all.ps1 — เปิดทั้งระบบทีเดียว (Dashboard server + Trade bot)
# ============================================================
#  วิธีใช้:
#    1. เปิด MetaTrader 5 (Exness) + login บัญชี demo ค้างไว้ก่อน
#    2. คลิกขวาไฟล์นี้ > Run with PowerShell   หรือรันใน terminal:
#         powershell -ExecutionPolicy Bypass -File .\start_all.ps1
#
#  ตัวเลือก:
#    -Port 8001      เปลี่ยนพอร์ต dashboard (default 8001)
#    -NoBot          เปิดเฉพาะ server (ไม่เปิดบอทเทรด)
#    -NoServer       เปิดเฉพาะบอทเทรด (ไม่เปิด server)
# ============================================================
param(
    [int]$Port = 8001,
    [switch]$NoBot,
    [switch]$NoServer
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

# หา python ของ venv (fallback เป็น python ระบบ)
$py = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Warning "ไม่พบ venv — ใช้ python ของระบบแทน (แนะนำสร้าง venv: python -m venv venv)"
    $py = "python"
}

Write-Host "==================================================" -ForegroundColor DarkYellow
Write-Host "  Gold Trading System — start_all" -ForegroundColor Yellow
Write-Host "  python : $py" -ForegroundColor DarkGray
Write-Host "  port   : $Port" -ForegroundColor DarkGray
Write-Host "==================================================" -ForegroundColor DarkYellow

if (-not $NoServer) {
    Write-Host "[1] เปิด Dashboard server (พอร์ต $Port)..." -ForegroundColor Cyan
    Start-Process -FilePath $py `
        -ArgumentList "-m", "uvicorn", "gold_server:app", "--port", "$Port" `
        -WorkingDirectory $root
    Start-Sleep -Seconds 3
    Write-Host "    → http://localhost:$Port  (dashboard)" -ForegroundColor Green
    Write-Host "    → http://localhost:$Port/trades  (bot trades)" -ForegroundColor Green
}

if (-not $NoBot) {
    Write-Host "[2] เปิดบอทเทรด (gold_trader.py)..." -ForegroundColor Cyan
    Start-Process -FilePath $py `
        -ArgumentList "gold_trader.py" `
        -WorkingDirectory $root
    Write-Host "    → บอทเริ่มทำงาน (ดู log ในหน้าต่างใหม่)" -ForegroundColor Green
}

Write-Host ""
Write-Host "เปิดครบแล้ว — ปิดได้โดยปิดหน้าต่าง process หรือ Ctrl+C ในแต่ละหน้าต่าง" -ForegroundColor Yellow
