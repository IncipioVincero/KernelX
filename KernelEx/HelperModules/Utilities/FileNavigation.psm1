function Find-DirectoryByName {
    param (
        [Parameter(Mandatory = $true)]
        [string]$DirName,

        [string]$SearchRoot = "C:\"
    )

    $verifiedMatches = Get-ChildItem -Path $SearchRoot -Recurse -Directory -ErrorAction SilentlyContinue |
               Where-Object { $_.Name -eq $DirName }

    if ($verifiedMatches.Count -eq 0) {
        Write-Host "No directories named '$DirName' were found under $SearchRoot."
        return $null
    }

    if ($verifiedMatches.Count -gt 1) {
        Write-Host "🔍 Multiple directories found:"
        for ($i = 0; $i -lt $verifiedMatches.Count; $i++) {
            Write-Host "$i $($verifiedMatches[$i].FullName)"
        }

        $choice = Read-Host "Enter the number of the correct directory"
        if ($choice -match '^\d+$' -and $choice -lt $verifiedMatches.Count) {
            return $verifiedMatches[$choice].FullName
        } else {
            Write-Host "Invalid selection."
            return $null
        }
    } else {
        return $verifiedMatches[0].FullName
    }
}

Export-ModuleMember -Function Find-DirectoryByName