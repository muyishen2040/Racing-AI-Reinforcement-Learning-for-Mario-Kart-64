param(
    [Parameter(Mandatory = $true)]
    [string]$InstanceId,
    [string]$Profile = "default",
    [string]$Region = "us-east-2",
    [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$Viewer = "$env:LOCALAPPDATA\mk64-tools\vncviewer.exe"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Viewer)) { throw "VNC viewer not found: $Viewer" }
if (-not (Test-Path -LiteralPath $Key)) { throw "SSH key not found: $Key" }
$address = aws --profile $Profile --region $Region ec2 describe-instances --instance-ids $InstanceId --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
if ($LASTEXITCODE -ne 0 -or -not $address -or $address -eq "None") { throw "Instance has no public IP. Start it first and wait until it is running." }

# Start a localhost-only display if this instance was restarted.
$target = "ubuntu@$address"
ssh -i $Key -o BatchMode=yes -o StrictHostKeyChecking=accept-new $target 'ss -ltn | grep -q 127.0.0.1:5901 || vncserver :1 -localhost yes -SecurityTypes None -geometry 1024x768 -xstartup /usr/bin/openbox'
if ($LASTEXITCODE -ne 0) { throw "Could not start or reach the remote VNC display. Check SSH access and the security group." }

if (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 5901 -State Listen -ErrorAction SilentlyContinue) { throw "Local port 5901 is already in use; close the existing tunnel first." }
$quotedKey = '"' + $Key + '"'
$tunnel = Start-Process -FilePath ssh.exe -ArgumentList @('-i', $quotedKey, '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes', '-N', '-L', '127.0.0.1:5901:127.0.0.1:5901', $target) -WindowStyle Hidden -PassThru
try {
    Start-Sleep -Seconds 2
    $tunnel.Refresh()
    if ($tunnel.HasExited) { throw "Could not establish the SSH tunnel." }
    Write-Host "Connected. Close the viewer to stop the SSH tunnel."
    $viewerProcess = Start-Process -FilePath $Viewer -ArgumentList 'localhost::5901' -PassThru
    $viewerProcess.WaitForExit()
} finally {
    $tunnel.Refresh()
    if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id }
}
