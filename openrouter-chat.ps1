# openrouter-chat.ps1
param(
    [string]$Prompt = "Say hello world",
    [string]$Model = "mistralai/mistral-7b-instruct"
)

$apiKey = "" # <-- Replace with your actual OpenRouter API key

$body = @{
    model = $Model
    messages = @(@{ role = "user"; content = $Prompt })
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "https://openrouter.ai/api/v1/chat/completions" `
    -Headers @{ "Authorization" = "Bearer $apiKey"; "Content-Type" = "application/json" } `
    -Method Post -Body $body

$response.choices[0].message.content