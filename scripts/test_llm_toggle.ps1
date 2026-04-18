param(
    [string]$Claim = "The Earth revolves around the Sun.",
    [string]$Language = "en",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $OutDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutDir = "tests/llm_toggle/$stamp"
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

function Run-Mode {
    param(
        [string]$ModeName,
        [string]$FlagValue,
        [string]$ClaimText,
        [string]$Lang,
        [string]$OutputJson
    )

    Set-EnvValue -Path $envPath -Key "ENABLE_LLM_VERIFIER" -Value $FlagValue

    $py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
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
out_json = sys.argv[4]

p = ClaimPipeline(_pipeline_config(language))
r = p.analyze(claim=claim, language=language)
out = {
    "mode": mode,
    "claim": claim,
    "language": language,
    "verdict": r.verdict,
    "confidence": float(r.confidence),
    "reasoning": r.reasoning,
    "details": r.details,
    "evidence_count": len(r.evidence),
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps({
    "mode": mode,
    "verdict": out["verdict"],
    "confidence": out["confidence"],
    "evidence_count": out["evidence_count"]
}, ensure_ascii=False))
'@
    $tmp = Join-Path $OutDir "tmp_run_$ModeName.py"
    Set-Content -Path $tmp -Value $script -Encoding UTF8
    $env:RFCS_REPO_ROOT = (Get-Location).Path
    & $py $tmp $ClaimText $Lang $ModeName $OutputJson
    if ($LASTEXITCODE -ne 0) {
        throw "Python run failed for mode=$ModeName"
    }
    Remove-Item -LiteralPath $tmp -Force
}

$orig = Get-EnvValue -Path $envPath -Key "ENABLE_LLM_VERIFIER"

try {
    $offJson = Join-Path $OutDir "test_llm_off.json"
    $onJson = Join-Path $OutDir "test_llm_on.json"

    Run-Mode -ModeName "llm_off" -FlagValue "0" -ClaimText $Claim -Lang $Language -OutputJson $offJson
    Run-Mode -ModeName "llm_on"  -FlagValue "1" -ClaimText $Claim -Lang $Language -OutputJson $onJson

    $off = Get-Content $offJson | ConvertFrom-Json
    $on = Get-Content $onJson | ConvertFrom-Json

    $summary = [ordered]@{
        claim = $Claim
        language = $Language
        llm_off = [ordered]@{
            verdict = $off.verdict
            confidence = $off.confidence
            evidence_count = $off.evidence_count
        }
        llm_on = [ordered]@{
            verdict = $on.verdict
            confidence = $on.confidence
            evidence_count = $on.evidence_count
        }
        changed = [ordered]@{
            verdict_changed = ($off.verdict -ne $on.verdict)
            confidence_delta = ([double]$on.confidence - [double]$off.confidence)
            evidence_count_delta = ([int]$on.evidence_count - [int]$off.evidence_count)
        }
        files = [ordered]@{
            off = $offJson
            on = $onJson
        }
    }

    $cmpPath = Join-Path $OutDir "comparison.json"
    $summary | ConvertTo-Json -Depth 6 | Set-Content -Path $cmpPath -Encoding UTF8
    Write-Output "Wrote: $cmpPath"
    $summary | ConvertTo-Json -Depth 6
}
finally {
    if ($null -eq $orig) {
        # If key did not exist before, remove it.
        $lines = Get-Content $envPath | Where-Object { $_ -notmatch "^ENABLE_LLM_VERIFIER=" }
        Set-Content -Path $envPath -Value $lines -Encoding UTF8
    } else {
        Set-EnvValue -Path $envPath -Key "ENABLE_LLM_VERIFIER" -Value $orig
    }
}
