param(
    [ValidateSet("easy_attention_sgu", "self_attention", "lstm")]
    [string]$Model = "easy_attention_sgu",
    [ValidateSet(16, 32)]
    [int]$Ds = 16,
    [int]$SimTime = 10000,
    [ValidateSet(100, 120)]
    [int]$PredLen = 100,
    [int]$SeqLen = 128,
    [ValidateRange(1, 3600)]
    [int]$RefreshSeconds = 5
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$progressName = "{0}_ds{1}_t{2}_s{3}_p{4}_progress.json" -f $Model, $Ds, $SimTime, $SeqLen, $PredLen
$progressPath = Join-Path $projectRoot (Join-Path "results\logs" $progressName)
$escapedRoot = [Regex]::Escape($projectRoot)

while ($true) {
    Clear-Host
    Write-Host "KS Three-model Training Progress" -ForegroundColor Cyan
    Write-Host ("=" * 70)
    Write-Host ("Model           : {0}" -f $Model)
    Write-Host ("Configuration   : ds={0}, t={1}, s={2}, p={3}" -f $Ds, $SimTime, $SeqLen, $PredLen)
    Write-Host ("Progress file   : {0}" -f $progressPath)

    $trainingProcess = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match $escapedRoot -and
            $_.CommandLine -match "main\.py" -and
            $_.CommandLine -match ([Regex]::Escape($Model))
        } |
        Select-Object -First 1

    if ($trainingProcess) {
        Write-Host ("Process status  : RUNNING (PID {0})" -f $trainingProcess.ProcessId) -ForegroundColor Green
    }
    else {
        Write-Host "Process status  : NOT DETECTED" -ForegroundColor Yellow
    }

    if (-not (Test-Path -LiteralPath $progressPath)) {
        Write-Host "Training status : waiting for the first completed epoch"
        Write-Host ""
        Write-Host ("Refresh every {0}s; Ctrl+C stops this viewer only." -f $RefreshSeconds) -ForegroundColor DarkGray
        Start-Sleep -Seconds $RefreshSeconds
        continue
    }

    try {
        $progress = Get-Content -Raw -LiteralPath $progressPath | ConvertFrom-Json
        $percent = 100.0 * [double]$progress.epoch / [double]$progress.num_epochs
        Write-Host ("Training status : {0}" -f $progress.status)
        Write-Host ("Completed       : {0}/{1} ({2:N1}%)" -f $progress.epoch, $progress.num_epochs, $percent)
        Write-Host ("Train MSE       : {0:E8}" -f [double]$progress.train_loss)
        Write-Host ("Validation MSE  : {0:E8}" -f [double]$progress.validation_loss)
        Write-Host ("Best val MSE    : {0:E8}" -f [double]$progress.best_validation_loss)
        Write-Host ("Patience count  : {0}" -f $progress.patience_counter)
        Write-Host ("Last update     : {0}" -f (Get-Item -LiteralPath $progressPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))

        if ($null -ne $progress.rollout_selection_metrics) {
            $rollout = $progress.rollout_selection_metrics
            $crossing = $rollout.first_crossing_1_0
            $crossingText = if ($crossing.reached) { [string]$crossing.crossing_step } else { "not reached" }
            Write-Host ""
            Write-Host "Latest rollout validation" -ForegroundColor Cyan
            Write-Host ("  Collapse flag     : {0}" -f $rollout.collapse_flag)
            Write-Host ("  Late std ratio    : {0:N6}" -f [double]$rollout.late_std_ratio)
            Write-Host ("  First NRMSE >= 1  : {0}" -f $crossingText)
            Write-Host ("  1200-step AUC     : {0:N6}" -f [double]$rollout.nrmse_auc_over_first_1200)
        }
    }
    catch {
        Write-Host "Progress file is being replaced atomically; retrying." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host ("Refresh every {0}s; Ctrl+C stops this viewer only." -f $RefreshSeconds) -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
