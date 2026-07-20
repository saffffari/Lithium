$ws = New-Object -ComObject WScript.Shell
$lnkPath = Join-Path $env:USERPROFILE "Desktop\3Photon 0.1.lnk"
$s = $ws.CreateShortcut($lnkPath)
$s.TargetPath       = "D:\3Photon\dist\3Photon.exe"
$s.WorkingDirectory = "D:\3Photon\dist"
$s.IconLocation     = "D:\3Photon\assets\3photon.ico,0"
$s.Description      = "3Photon 0.1 - point cloud annotation platform"
$s.Save()

# Verify
$v = $ws.CreateShortcut($lnkPath)
Write-Output "Created: $lnkPath"
Write-Output "Target:  $($v.TargetPath)"
Write-Output "WorkDir: $($v.WorkingDirectory)"
Write-Output "Icon:    $($v.IconLocation)"
