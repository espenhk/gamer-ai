<#
.SYNOPSIS
    Setup and launch script for tmnf-ai on Windows.

.DESCRIPTION
    Two modes:
      Fresh machine  — installs Python 3.11+, Poetry, Git, Trackmania Nations
                       Forever, TMInterface 1.4, and Python dependencies, then
                       launches everything.
      Existing setup — detects what is already present, skips install steps, and
                       goes straight to launching TMInterface + the requested command.

    Idempotent: safe to re-run at any time.

.PARAMETER Command
    The Python command to run after setup, e.g.:
        "python main.py my_experiment"
        "python grid_search.py config/gs_cmaes.yaml"

.PARAMETER DryRun
    Print what would be done without executing anything.

.EXAMPLE
    .\setup_and_run.ps1 "python main.py my_experiment"
    .\setup_and_run.ps1 "python grid_search.py config/gs_cmaes.yaml" -DryRun

.NOTES
    Trackmania Nations Forever:
        Official installer from Nadeo's CDN.  Silent install is supported via
        the NSIS `/S` flag; the game installs to
        C:\Program Files (x86)\TmNationsForever by default.  TMInterface
        requires this to be installed before it is launched.

    TMInterface 1.4:
        Official installer: https://donadigo.com/tminterface
        Only version 1.4.x is compatible with the tminterface Python package
        used by this project.  Later versions changed the Python API.

    Python package dependencies (tminterface, pygbx, etc.):
        All dependencies are managed by Poetry and pinned in poetry.lock.
        Running `poetry install` (performed automatically by this script)
        is sufficient — no manual source installation is required.

    winget:
        This script uses winget to install Python and Git on Windows 10/11.
        winget ships with Windows 10 1809+ and Windows 11 via the
        "App Installer" package.  If winget is not available, install Python
        and Git manually before running this script.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "",

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Step([string]$msg) {
    Write-Host "[SETUP] $msg" -ForegroundColor Cyan
}

function Write-Skip([string]$msg) {
    Write-Host "[SKIP]  $msg" -ForegroundColor DarkGray
}

function Write-Ok([string]$msg) {
    Write-Host "[OK]    $msg" -ForegroundColor Green
}

function Write-Err([string]$msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

function Invoke-Step([string]$description, [scriptblock]$action) {
    Write-Step $description
    if ($DryRun) {
        Write-Host "        (dry-run — skipped)" -ForegroundColor Yellow
        return
    }
    & $action
}

function Test-CommandExists([string]$name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Assert-Success([string]$context) {
    if ($LASTEXITCODE -ne 0) {
        Write-Err "$context failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
}

function Assert-Winget {
    if (-not (Test-CommandExists "winget")) {
        Write-Err "winget is not available on this machine."
        Write-Err "Install the Windows Package Manager (App Installer) from the Microsoft Store,"
        Write-Err "or install the missing tool manually, then re-run this script."
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

$ScriptDir  = $PSScriptRoot

# Trackmania Nations Forever (official Nadeo CDN).  NSIS installer — supports
# `/S` for silent install and `/D=<path>` for a custom directory.
$TMNFInstallDir      = Join-Path ${env:ProgramFiles(x86)} "TmNationsForever"
$TMNFExe             = Join-Path $TMNFInstallDir "TmForever.exe"
$TMNFInstallerUrl    = "https://nadeo-download.cdn.ubi.com/trackmaniaforever/tmnationsforever_setup.exe"
# Set to the known SHA-256 of the installer above to enable integrity verification.
# Leave empty to skip verification (not recommended for production use).
$TMNFInstallerSha256 = ""

# TMInterface 1.4
$TMIfaceDir = Join-Path $env:USERPROFILE "TMInterface"
$TMIfaceExe = Join-Path $TMIfaceDir "TMInterface.exe"
# TMInterface 1.4 installer (official release page; update URL + SHA-256 when a new 1.4.x drops)
$TMIfaceInstallerUrl    = "https://github.com/donadigo/tminterface/releases/download/1.4.3/TMInterface_1.4.3_Setup.exe"
# Set to the known SHA-256 of the installer above to enable integrity verification.
# Leave empty to skip verification (not recommended for production use).
$TMIfaceInstallerSha256 = ""

# ---------------------------------------------------------------------------
# 1. Python 3.11+
# ---------------------------------------------------------------------------

function Install-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) {
        $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -and [version]$ver -ge [version]"3.11") {
            Write-Skip "Python $ver already in PATH."
            return
        }
    }

    Invoke-Step "Installing Python 3.11 via winget" {
        Assert-Winget
        winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements
        Assert-Success "Python install"
        # Refresh PATH for the rest of this process
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")

        # Verify the newly installed interpreter is actually on PATH at 3.11+
        $pyNew = Get-Command python -ErrorAction SilentlyContinue
        $verNew = $null
        if ($pyNew) {
            $verNew = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        }
        if (-not $verNew -or [version]$verNew -lt [version]"3.11") {
            Write-Err "Python 3.11 was installed but 'python' on PATH still resolves to an older version ($verNew)."
            Write-Err "Ensure the Python 3.11 install directory appears before older versions in your PATH, then re-run."
            exit 1
        }
        Write-Ok "Python $verNew available."
    }
}

# ---------------------------------------------------------------------------
# 2. Git
# ---------------------------------------------------------------------------

function Install-Git {
    if (Test-CommandExists "git") {
        Write-Skip "Git already in PATH."
        return
    }

    Invoke-Step "Installing Git via winget" {
        Assert-Winget
        winget install --id Git.Git --source winget --silent --accept-package-agreements --accept-source-agreements
        Assert-Success "Git install"
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
    }
}

# ---------------------------------------------------------------------------
# 3. Poetry
# ---------------------------------------------------------------------------

function Install-Poetry {
    if (Test-CommandExists "poetry") {
        Write-Skip "Poetry already in PATH."
        return
    }

    Invoke-Step "Installing Poetry" {
        $installer = Join-Path $env:TEMP "install-poetry.py"
        # Official installer from https://install.python-poetry.org
        Invoke-WebRequest -Uri "https://install.python-poetry.org" -OutFile $installer -UseBasicParsing
        python $installer
        Assert-Success "Poetry install"
        # Add Poetry bin dir to PATH for this session
        $poetryBin = Join-Path $env:APPDATA "Python\Scripts"
        if (Test-Path $poetryBin) {
            $env:Path = "$poetryBin;$env:Path"
        }
    }
}

# ---------------------------------------------------------------------------
# 4. Trackmania Nations Forever
# ---------------------------------------------------------------------------

function Install-TMNF {
    if (Test-Path $TMNFExe) {
        Write-Skip "Trackmania Nations Forever already installed at $TMNFExe."
        return
    }

    Invoke-Step "Downloading Trackmania Nations Forever installer" {
        $installer = Join-Path $env:TEMP "tmnationsforever_setup.exe"
        Write-Step "  Fetching $TMNFInstallerUrl"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $TMNFInstallerUrl -OutFile $installer -UseBasicParsing
        Assert-Success "TMNF download"

        if ($TMNFInstallerSha256) {
            $actualHash = (Get-FileHash -Path $installer -Algorithm SHA256).Hash
            if ($actualHash.ToUpperInvariant() -ne $TMNFInstallerSha256.ToUpperInvariant()) {
                Write-Err "TMNF installer SHA-256 mismatch!"
                Write-Err "  Expected: $TMNFInstallerSha256"
                Write-Err "  Actual:   $actualHash"
                Write-Err "Aborting to avoid executing an unexpected binary."
                exit 1
            }
            Write-Ok "Installer checksum verified."
        } else {
            Write-Host "[WARN]  TMNF installer SHA-256 not configured — skipping integrity check." -ForegroundColor Yellow
        }

        Write-Step "  Running TMNF installer (silent)"
        # NSIS: /S = silent (case-sensitive), /D=<path> must be last and unquoted.
        $proc = Start-Process -FilePath $installer -ArgumentList "/S", "/D=$TMNFInstallDir" -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-Err "TMNF installation failed (exit code $($proc.ExitCode))."
            Write-Err "Please install Trackmania Nations Forever manually from: $TMNFInstallerUrl"
            exit $proc.ExitCode
        }

        if (-not (Test-Path $TMNFExe)) {
            Write-Err "TMNF executable not found after install at $TMNFExe."
            Write-Err "Please install Trackmania Nations Forever manually from: $TMNFInstallerUrl"
            exit 1
        }
        Write-Ok "Trackmania Nations Forever installed at $TMNFInstallDir."
    }
}

# ---------------------------------------------------------------------------
# 5. TMInterface 1.4
# ---------------------------------------------------------------------------

function Install-TMInterface {
    if (Test-Path $TMIfaceExe) {
        Write-Skip "TMInterface already found at $TMIfaceExe."
        return
    }

    Invoke-Step "Downloading TMInterface 1.4 installer" {
        $installer = Join-Path $env:TEMP "TMInterface_Setup.exe"
        Write-Step "  Fetching $TMIfaceInstallerUrl"
        Invoke-WebRequest -Uri $TMIfaceInstallerUrl -OutFile $installer -UseBasicParsing
        Assert-Success "TMInterface download"

        if ($TMIfaceInstallerSha256) {
            $actualHash = (Get-FileHash -Path $installer -Algorithm SHA256).Hash
            if ($actualHash.ToUpperInvariant() -ne $TMIfaceInstallerSha256.ToUpperInvariant()) {
                Write-Err "TMInterface installer SHA-256 mismatch!"
                Write-Err "  Expected: $TMIfaceInstallerSha256"
                Write-Err "  Actual:   $actualHash"
                Write-Err "Aborting to avoid executing an unexpected binary."
                exit 1
            }
            Write-Ok "Installer checksum verified."
        } else {
            Write-Host "[WARN]  TMInterface installer SHA-256 not configured — skipping integrity check." -ForegroundColor Yellow
        }

        Write-Step "  Running TMInterface installer (silent)"
        # /SILENT — run without wizard UI; /DIR — install location
        $proc = Start-Process -FilePath $installer -ArgumentList "/SILENT", "/DIR=`"$TMIfaceDir`"" -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-Err "TMInterface installation failed (exit code $($proc.ExitCode))."
            Write-Err "Please install TMInterface 1.4 manually from: https://donadigo.com/tminterface"
            exit $proc.ExitCode
        }

        if (-not (Test-Path $TMIfaceExe)) {
            Write-Err "TMInterface executable not found after install at $TMIfaceExe."
            Write-Err "Please install TMInterface 1.4 manually from: https://donadigo.com/tminterface"
            exit 1
        }
    }
}

# ---------------------------------------------------------------------------
# 6. Poetry dependencies
# ---------------------------------------------------------------------------

function Install-PoetryDeps {
    Invoke-Step "Running 'poetry install' (skips if lock is satisfied)" {
        Push-Location $ScriptDir
        try {
            poetry install --with tmnf
            Assert-Success "poetry install"
        } finally {
            Pop-Location
        }
    }
}

# ---------------------------------------------------------------------------
# 7. Launch TMInterface
# ---------------------------------------------------------------------------

function Start-TMInterface {
    $procs = Get-Process -Name "TMInterface" -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Skip "TMInterface process already running (PID $($procs[0].Id))."
        return
    }

    if (-not (Test-Path $TMIfaceExe)) {
        Write-Err "Cannot launch TMInterface — executable not found at $TMIfaceExe."
        exit 1
    }

    Invoke-Step "Launching TMInterface" {
        Start-Process -FilePath $TMIfaceExe
        Write-Step "Waiting for TMInterface to initialise (up to 30 s)…"
        $deadline = (Get-Date).AddSeconds(30)
        $ready    = $false
        while ((Get-Date) -lt $deadline) {
            $proc = Get-Process -Name "TMInterface" -ErrorAction SilentlyContinue
            if ($proc) { $ready = $true; break }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) {
            Write-Err "TMInterface did not start within 30 seconds."
            exit 1
        }
        # Extra settle time for the plugin to register its TCP listener
        Start-Sleep -Seconds 3
        Write-Ok "TMInterface is running."
    }
}

# ---------------------------------------------------------------------------
# 8. Run user command
# ---------------------------------------------------------------------------

function Invoke-UserCommand {
    if (-not $Command) {
        Write-Step "No command supplied — setup complete.  Pass a command as the first argument to launch training."
        return
    }

    Write-Step "Running: $Command"
    if ($DryRun) {
        Write-Host "        (dry-run — skipped)" -ForegroundColor Yellow
        return
    }

    Push-Location $ScriptDir
    try {
        # Tokenize safely using PSParser — respects quoted arguments and avoids
        # Invoke-Expression evaluation of arbitrary PowerShell syntax.
        $parseErrors = $null
        $tokens = [System.Management.Automation.PSParser]::Tokenize($Command, [ref]$parseErrors) |
            Where-Object { $_.Type -notin @("NewLine", "Comment") }

        if ($parseErrors -and $parseErrors.Count -gt 0) {
            Write-Err "Unable to parse command safely: $Command"
            exit 1
        }

        $unsupported = $tokens | Where-Object { $_.Type -notin @("Command", "CommandArgument", "String", "Number") }
        if ($unsupported) {
            Write-Err "Command contains unsupported PowerShell syntax.  Provide an executable and plain arguments only: $Command"
            exit 1
        }

        $cmdParts = @($tokens | ForEach-Object { $_.Content })
        if ($cmdParts.Count -eq 0) {
            Write-Err "No executable found in command: $Command"
            exit 1
        }

        $exe     = $cmdParts[0]
        $cmdArgs = if ($cmdParts.Count -gt 1) { $cmdParts[1..($cmdParts.Count - 1)] } else { @() }

        if ($exe -eq "python") {
            # Run inside the Poetry venv
            & poetry run python @cmdArgs
        } else {
            & $exe @cmdArgs
        }
        Assert-Success "User command"
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if ($DryRun) {
    Write-Host "=== DRY RUN — no changes will be made ===" -ForegroundColor Yellow
}

Install-Python
Install-Git
Install-Poetry
Install-TMNF
Install-TMInterface
Install-PoetryDeps
Start-TMInterface
Invoke-UserCommand

Write-Ok "Done."
