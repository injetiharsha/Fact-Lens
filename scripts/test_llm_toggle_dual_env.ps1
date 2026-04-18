param(
    [string]$Claim = "The Earth revolves around the Sun.",
    [string]$Language = "en",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $OutDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutDir = "tests/llm_toggle/dual_env_$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$envPath = ".env"
if (-not (Test-Path $envPath)) {
    throw "Missing .env at repo root."
}

function Get-EnvValue {
    param([string]$Path, [string]$Key)
    $line = (Get-Content $Path | Where-Object { $_ -match "^$Key=" } | Select-Object -First 1)
    if (-not $line) { return $null }
    return ($line -replace "^$Key=", "")
}

function Set-EnvValue {
    param([string]$Path, [string]$Key, [string]$Value)
    $lines = Get-Content $Path
    $updated = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^$Key=") {
            $lines[$i] = "$Key=$Value"
            $updated = $true
            break
        }
    }
    if (-not $updated) {
        $lines += "$Key=$Value"
    }
    Set-Content -Path $Path -Value $lines -Encoding UTF8
}

function Resolve-PythonEnvs {
    $out = @()
    if (Test-Path ".\.venv\Scripts\python.exe") {
        $out += @{ name = "venv"; py = ".\.venv\Scripts\python.exe" }
    }
    if (Test-Path ".\.venv-gpu\Scripts\python.exe") {
        $out += @{ name = "venv_gpu"; py = ".\.venv-gpu\Scripts\python.exe" }
    }
    if ($out.Count -eq 0) {
        throw "No virtual env python executables found."
    }
    return $out
}

function Run-Mode {
    param(
        [string]$EnvName,
        [string]$PyPath,
        [string]$ModeName,
        [string]$FlagValue,
        [string]$ClaimText,
        [string]$Lang,
        [string]$OutputJson
    )

    Set-EnvValue -Path $envPath -Key "ENABLE_LLM_VERIFIER" -Value $FlagValue

    $script = @'
import json, os, sys
repo_root = os.environ.get("RFCS_REPO_ROOT", "")
if repo_root:
    sys.path.insert(0, repo_root)
from api.routes.claim import _pipeline_config
from pipeline.claim_pipeline import ClaimPipeline

claim = sys.argv[1]
language = sys.argv[2]
mode = sys.argv[3]
env_name = sys.argv[4]
out_json = sys.argv[5]

p = ClaimPipeline(_pipeline_config(language))
r = p.analyze(claim=claim, language=language)
out = {
    "env": env_name,
    "mode": mode,
    "claim": claim,
    "language": language,
    "verdict": r.verdict,
    "confidence": float(r.confidence),
    "reasoning": r.reasoning,
    "details": r.details,
    "evidence_count": len(r.evidence),
    "evidence_top5": r.evidence[:5],
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps({
    "env": env_name,
    "mode": mode,
    "verdict": out["verdict"],
    "confidence": out["confidence"],
    "evidence_count": out["evidence_count"]
}, ensure_ascii=False))
'@

    $tmp = Join-Path $OutDir "tmp_run_${EnvName}_${ModeName}.py"
    Set-Content -Path $tmp -Value $script -Encoding UTF8
    $env:RFCS_REPO_ROOT = (Get-Location).Path
    & $PyPath $tmp $ClaimText $Lang $ModeName $EnvName $OutputJson
    if ($LASTEXITCODE -ne 0) {
        throw "Python run failed for env=$EnvName mode=$ModeName"
    }
    Remove-Item -LiteralPath $tmp -Force
}

$orig = Get-EnvValue -Path $envPath -Key "ENABLE_LLM_VERIFIER"

try {
    $envs = Resolve-PythonEnvs
    $summary = [ordered]@{
        claim = $Claim
        language = $Language
        generated_at = (Get-Date).ToString("s")
        results = @()
    }

    foreach ($e in $envs) {
        $offJson = Join-Path $OutDir "$($e.name)_llm_off.json"
        $onJson = Join-Path $OutDir "$($e.name)_llm_on.json"

        Run-Mode -EnvName $e.name -PyPath $e.py -ModeName "llm_off" -FlagValue "0" -ClaimText $Claim -Lang $Language -OutputJson $offJson
        Run-Mode -EnvName $e.name -PyPath $e.py -ModeName "llm_on"  -FlagValue "1" -ClaimText $Claim -Lang $Language -OutputJson $onJson

        $off = Get-Content $offJson | ConvertFrom-Json
        $on = Get-Content $onJson | ConvertFrom-Json

        $summary.results += [ordered]@{
            env = $e.name
            llm_off = [ordered]@{
                verdict = $off.verdict
                confidence = $off.confidence
                evidence_count = $off.evidence_count
                file = $offJson
            }
            llm_on = [ordered]@{
                verdict = $on.verdict
                confidence = $on.confidence
                evidence_count = $on.evidence_count
                file = $onJson
            }
            changed = [ordered]@{
                verdict_changed = ($off.verdict -ne $on.verdict)
                confidence_delta = ([double]$on.confidence - [double]$off.confidence)
                evidence_count_delta = ([int]$on.evidence_count - [int]$off.evidence_count)
            }
        }
    }

    $cmpPath = Join-Path $OutDir "comparison_dual_env.json"
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Path $cmpPath -Encoding UTF8
    Write-Output "Wrote: $cmpPath"
    $summary | ConvertTo-Json -Depth 8
}
finally {
    if ($null -eq $orig) {
        $lines = Get-Content $envPath | Where-Object { $_ -notmatch "^ENABLE_LLM_VERIFIER=" }
        Set-Content -Path $envPath -Value $lines -Encoding UTF8
    } else {
        Set-EnvValue -Path $envPath -Key "ENABLE_LLM_VERIFIER" -Value $orig
    }
}
