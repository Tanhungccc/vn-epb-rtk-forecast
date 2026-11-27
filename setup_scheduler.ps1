# Đăng ký 4 tác vụ chạy nền tự động trên Windows Task Scheduler:
#   1. VN_EPB_Live_Alert: Chạy mỗi 30 phút (cập nhật dự báo Live & gửi Telegram nếu vượt ngưỡng VÀNG/ĐỎ)
#   2. VN_EPB_Verify_Daily: Chạy mỗi sáng 09:00 LT (đối chiếu forecast_log.parquet với COSMIC-2 thật)
#   3. VN_EPB_Retrain_Monthly: Chạy định kỳ 30 ngày (Champion vs. Challenger trên dữ liệu tích lũy)
#   4. VN_EPB_Telegram_Bot: Tự bật chatbot lắng nghe lệnh (/dubao, /swpc, /web) khi đăng nhập Windows
#
# Cách chạy:
#   powershell -ExecutionPolicy Bypass -File .\setup_scheduler.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonwExe = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $PythonwExe)) { $PythonwExe = $PythonExe }

if (-not (Test-Path $PythonExe)) {
    Write-Error "Không tìm thấy $PythonExe. Hãy kiểm tra lại môi trường .venv."
}

# 1. Tự động dự báo & cảnh báo Telegram mỗi 30 phút
$AlertAction = New-ScheduledTaskAction -Execute $PythonwExe -Argument "-m src.send_alert" -WorkingDirectory $ProjectRoot
$AlertTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "VN_EPB_Live_Alert" -Action $AlertAction -Trigger $AlertTrigger -Description "Dự báo EPB/RTK-GNSS mỗi 30 phút và gửi Telegram khi vượt ngưỡng." -Force

# 2. Tự động đối chiếu COSMIC-2 & tích lũy dữ liệu mỗi sáng 09:00 LT
$VerifyAction = New-ScheduledTaskAction -Execute $PythonwExe -Argument "-m src.verify_forecasts" -WorkingDirectory $ProjectRoot
$VerifyTrigger = New-ScheduledTaskTrigger -Daily -At "09:00"
Register-ScheduledTask -TaskName "VN_EPB_Verify_Daily" -Action $VerifyAction -Trigger $VerifyTrigger -Description "Đối chiếu tự động nhật ký dự báo EPB với vệ tinh COSMIC-2 lúc 09:00 hàng ngày." -Force

# 3. Tự động đối đầu Champion vs Challenger mỗi 30 ngày (09:30 LT)
$RetrainAction = New-ScheduledTaskAction -Execute $PythonwExe -Argument "-m src.retrain_pipeline" -WorkingDirectory $ProjectRoot
$RetrainTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddDays(30).AddHours(9.5) -RepetitionInterval (New-TimeSpan -Days 30) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "VN_EPB_Retrain_Monthly" -Action $RetrainAction -Trigger $RetrainTrigger -Description "Tái huấn luyện Champion vs Challenger định kỳ 30 ngày." -Force

# 4. Tự bật Telegram Chatbot (/dubao, /swpc, /web) khi đăng nhập Windows
$BotAction = New-ScheduledTaskAction -Execute $PythonwExe -Argument "-m src.telegram_bot" -WorkingDirectory $ProjectRoot
$BotTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
Register-ScheduledTask -TaskName "VN_EPB_Telegram_Bot" -Action $BotAction -Trigger $BotTrigger -Description "Chatbot Telegram tương tác cho hệ thống dự báo EPB Việt Nam." -Force

Write-Host "Đã đăng ký thành công 4 tác vụ tự động trên Windows Task Scheduler:" -ForegroundColor Green
Get-ScheduledTask -TaskName "VN_EPB_*" | Select-Object TaskName, State | Format-Table -AutoSize
