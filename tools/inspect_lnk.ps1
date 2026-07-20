$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$env:USERPROFILE\Desktop\3Photon.lnk")
Write-Output "Target:  $($s.TargetPath)"
Write-Output "Args:    $($s.Arguments)"
Write-Output "WorkDir: $($s.WorkingDirectory)"
Write-Output "Icon:    $($s.IconLocation)"
