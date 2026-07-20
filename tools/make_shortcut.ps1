$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$path = Join-Path $desktop '3Photon.lnk'
$lnk = $ws.CreateShortcut($path)
$lnk.TargetPath = 'D:\3Photon\run.bat'
$lnk.WorkingDirectory = 'D:\3Photon'
$lnk.Description = '3Photon point cloud visualizer'
$lnk.WindowStyle = 7  # minimized console window
$lnk.Save()
Write-Host "created: $path"
