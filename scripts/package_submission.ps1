param(
    [string]$Experiment = "01_robust_automl"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$experimentDir = Join-Path $root "submissions\$Experiment"
$agentDir = Join-Path $experimentDir "agent"
$zipPath = Join-Path $experimentDir "submission.zip"
if (-not (Test-Path -LiteralPath (Join-Path $agentDir "agent.yaml"))) {
    throw "agent.yaml not found at $agentDir"
}
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath
}
Compress-Archive -Path (Join-Path $agentDir "*") -DestinationPath $zipPath -CompressionLevel Optimal
Write-Output $zipPath
