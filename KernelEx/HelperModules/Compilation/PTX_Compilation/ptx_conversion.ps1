param (
    [Parameter(Mandatory = $true)]
    [string]$Arch,

    [string]$Directory = ".",  # Default to current dir if not provided

    [string[]]$FileList
)

Import-Module -Name "C:\Users\kw2175\Documents\HPML_Final\KernelEx\HelperModules\Utilities\FileNavigation.psm1"

try {
    if ($FileList) {
        # Case: Specific list of files provided
        try {
            foreach ($file in $FileList) {
                $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file)
                Write-Host "Compiling $file for $Arch..."
                
                nvcc -arch=$Arch -ptx $file -o "$baseName.ptx"              
            }
        }
        catch
        {
            Write-Host "ERROR: PTX conversion for file list: $($FileList -join ', ') failed for the following reason:"
            Write-Host $_.Exception.Message
        }

        Write-Host "PTX files generated for architecture [$Arch] from file list: $($FileList -join ', ')"
    }
    else {
        try {

            $Directory = Find-DirectoryByName -DirName "Draft_Kernels"
            # Case: Process all .cu files in directory
            $cuFiles = Get-ChildItem -Path $Directory -Filter *.cu

            foreach ($cuFile in $cuFiles) {
                $baseName = [System.IO.Path]::GetFileNameWithoutExtension($cuFile.Name)
                Write-Host "Compiling Details:"
                Write-Host "Arch: $Arch"
                Write-Host "File: $cuFile"
                Write-Host "Base name: $baseName"
     
                nvcc -arch="$Arch" -ptx "$cuFile" -o "$baseName.ptx"
                
            }

        }
        catch
        {
            Write-Host "ERROR: PTX conversion for files in the directory [$Directory] failed for the following reason."
            Write-Host $_.Exception.Message
        }
        
    }
}
catch {
    Write-Host "ERROR: Conversion process unsuccessful for the following reason:"
    Write-Host $_.Exception.Message
}