# Đăng ký 2 tác vụ chạy nền tự động trên Windows Task Scheduler:
#   1. VN_EPB_Live_Alert: Chạy mỗi 30 phút (cập nhật dự báo Live & gửi Telegram nếu vượt ngưỡng VÀNG/ĐỎ)
#   2. VN_EPB_Verify_Daily: Chạy mỗi sáng 09:00 LT (đối chiếu forecast_log.parquet với COSMIC-2 thật)
#
# Cách chạy (mở PowerShell tại thư mục gốc dự án):
#   powershell -ExecutionPolicy Bypass -File .\setup_scheduler.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Không tìm thấy $PythonExe. Hãy kiểm tra lại môi trường .venv."
}

$AlertAction = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m src.send_alert" `
    -WorkingDirectory $ProjectRoot

$AlertTrigger = New-ScheduledTaskTrigger `
    -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Days 365)

Register-ScheduledTask `
    -TaskName "VN_EPB_Live_Alert" `
    -Action $AlertAction `
    -Trigger $AlertTrigger `
    -Description "Dự báo EPB/RTK-GNSS Việt Nam mỗi 30 phút và gửi cảnh báo Telegram khi vượt ngưỡng." `
    -Force

$VerifyAction = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m src.verify_forecasts" `
    -WorkingDirectory $ProjectRoot

$VerifyTrigger = New-ScheduledTaskTrigger -Daily -At "09:00"

Register-ScheduledTask `
    -TaskName "VN_EPB_Verify_Daily" `
    -Action $VerifyAction `
    -Trigger $VerifyTrigger `
    -Description "Đối chiếu tự động nhật ký dự báo EPB với vệ tinh COSMIC-2 mỗi sáng." `
    -Force

Write-Host "Đã đăng ký thành công 2 tác vụ trên Windows Task Scheduler:" -ForegroundColor Green
Write-Host "  - VN_EPB_Live_Alert (mỗi 30 phút)"
Write-Host "  - VN_EPB_Verify_Daily (09:00 sáng hàng ngày)"
