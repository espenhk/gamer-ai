<#
.SYNOPSIS
    Setup and launch script for all supported games on Windows.

.DESCRIPTION
    Two modes:
      Fresh machine  - installs Python 3.11+, Poetry, Git, and ALL supported
                       game binaries (Trackmania Nations Forever + TMInterface,
                       StarCraft II, TORCS, BeamNG Python deps, Car Racing deps),
                       then launches runtime services for the selected game and
                       runs the user command.
      Existing setup - detects what is already present, skips install steps, and
                       goes straight to starting the selected game services and
                       running the requested command.

    Supported games: tmnf, sc2, torcs, beamng, car_racing.

    Idempotent: safe to re-run at any time.

.PARAMETER Command
    The Python command to run after setup, e.g.:
        "python main.py my_experiment"
        "python grid_search.py config/gs_cmaes.yaml"
        "python -m distributed.worker --coordinator http://192.168.1.10:5555 --token mytoken --no-interrupt"

.PARAMETER Game
    The game whose runtime services to launch before running Command.
    One of: tmnf, sc2, torcs, beamng, car_racing.  Default: tmnf.

    For distributed multi-node runs, pass the same game you are training on so
    the correct background services (TMInterface for tmnf, TORCS SCR server for
    torcs, etc.) are started before the worker process connects.

.PARAMETER DryRun
    Print what would be done without executing anything.

.EXAMPLE
    # TMNF single experiment
    .\setup_and_run.ps1 -Game tmnf "python main.py my_tmnf_run"

    # SC2 distributed worker
    .\setup_and_run.ps1 -Game sc2 "python -m distributed.worker --coordinator http://10.0.0.5:5555 --token s3cr3t --game sc2 --no-interrupt"

    # Dry run to inspect what would be installed
    .\setup_and_run.ps1 -Game torcs -DryRun

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

    StarCraft II:
        The SC2 binary for Windows is distributed via the Battle.net launcher.
        This script downloads the Battle.net installer automatically.  You must
        log in to Battle.net once and install StarCraft II via the launcher
        before SC2 training can run.  PySC2 minigame maps are downloaded and
        placed automatically.

    TORCS:
        This script downloads the TORCS 1.3.7 Windows installer from
        SourceForge.  gym_torcs is installed from PyPI / source.  Before
        training, start TORCS manually in SCR-server mode:
            torcs -nofuel -nodamage -nolaptime
        then start a Practice race in the TORCS menu.

    BeamNG.drive:
        BeamNG.drive is a commercial game and cannot be installed automatically.
        Purchase it from https://www.beamng.com/ and install it manually.
        This script only installs the beamng-gym Python package.

    Car Racing:
        Uses the Gymnasium Box2D car_racing environment.  This script installs
        SWIG (required to compile Box2D) via winget and then installs
        gymnasium[box2d] into the Poetry virtual environment.

    Python package dependencies:
        All core dependencies are managed by Poetry and pinned in poetry.lock.
        Running `poetry install` (performed automatically by this script)
        is sufficient - no manual source installation is required.

    winget:
        This script uses winget to install Python, Git, SWIG, etc. on Windows
        10/11.  winget ships with Windows 10 1809+ and Windows 11 via the
        "App Installer" package.  If winget is not available, install the
        missing tools manually before running this script.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "",

    [Parameter()]
    [ValidateSet("tmnf", "sc2", "torcs", "beamng", "car_racing")]
    [string]$Game = "tmnf",

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
        Write-Host "        (dry-run - skipped)" -ForegroundColor Yellow
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

# Trackmania Nations Forever (official Nadeo CDN).  NSIS installer - supports
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

# StarCraft II — Battle.net launcher (SC2 itself must be installed via Battle.net after login).
# PySC2 maps are downloaded separately and placed in the SC2 Maps/ directory.
$BattleNetInstallerUrl = "https://www.battle.net/setup/Battle.net-Setup.exe"
$SC2DefaultDir         = Join-Path $env:USERPROFILE "StarCraftII"
$SC2MapsDir            = Join-Path $SC2DefaultDir "Maps"
# PySC2 mini-game map pack (official Blizzard release).
$SC2MiniGameMapsUrl    = "https://github.com/Blizzard/s2client-proto/releases/download/v4.10/mini_games.zip"
# Set to the known SHA-256 of the zip above to enable integrity verification.
$SC2MiniGameMapsSha256 = ""

# TORCS 1.3.7 Windows installer (SourceForge).
# Update URL + SHA-256 when upgrading to a newer release.
$TORCSInstallerUrl    = "https://sourceforge.net/projects/torcs/files/torcs/1.3.7/torcs-1.3.7-win32.exe/download"
$TORCSInstallDir      = Join-Path ${env:ProgramFiles} "TORCS"
$TORCSExe             = Join-Path $TORCSInstallDir "wtorcs.exe"
$TORCSInstallerSha256 = ""

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
            Write-Host "[WARN]  TMNF installer SHA-256 not configured - skipping integrity check." -ForegroundColor Yellow
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
            Write-Host "[WARN]  TMInterface installer SHA-256 not configured - skipping integrity check." -ForegroundColor Yellow
        }

        Write-Step "  Running TMInterface installer (silent)"
        # /SILENT - run without wizard UI; /DIR - install location
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
# 5b. StarCraft II (Battle.net launcher + PySC2 maps)
# ---------------------------------------------------------------------------

function Install-SC2 {
    # Check whether the Battle.net launcher is already present.
    $battleNetExe = Join-Path ${env:ProgramFiles(x86)} "Battle.net\Battle.net.exe"
    if (-not (Test-Path $battleNetExe)) {
        Invoke-Step "Downloading Battle.net launcher" {
            $installer = Join-Path $env:TEMP "Battle.net-Setup.exe"
            Write-Step "  Fetching $BattleNetInstallerUrl"
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $BattleNetInstallerUrl -OutFile $installer -UseBasicParsing
            Assert-Success "Battle.net download"

            Write-Step "  Running Battle.net installer (silent)"
            $proc = Start-Process -FilePath $installer -ArgumentList "--lang=enUS", "--installpath=`"${env:ProgramFiles(x86)}\Battle.net`"" -Wait -PassThru
            if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 1) {
                # Exit code 1 is common for Battle.net when it restarts itself;
                # treat it as a soft warning rather than a hard failure.
                Write-Host "[WARN]  Battle.net installer exited with code $($proc.ExitCode) - this may be normal." -ForegroundColor Yellow
            }
            Write-Host "[WARN]  Battle.net is installed. To use SC2 you must:" -ForegroundColor Yellow
            Write-Host "        1. Log in to Battle.net and install StarCraft II." -ForegroundColor Yellow
            Write-Host "        2. Then re-run this script or start training manually." -ForegroundColor Yellow
        }
    } else {
        Write-Skip "Battle.net launcher already present at $battleNetExe."
        Write-Host "[INFO]  Ensure StarCraft II is installed via Battle.net before running sc2 training." -ForegroundColor Cyan
    }

    # Download and place PySC2 mini-game maps.
    if (Test-Path $SC2MapsDir) {
        Write-Skip "SC2 Maps directory already exists at $SC2MapsDir."
    } else {
        Invoke-Step "Downloading PySC2 mini-game maps" {
            $mapsZip = Join-Path $env:TEMP "sc2_mini_games.zip"
            Write-Step "  Fetching $SC2MiniGameMapsUrl"
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $SC2MiniGameMapsUrl -OutFile $mapsZip -UseBasicParsing
            Assert-Success "SC2 maps download"

            if ($SC2MiniGameMapsSha256) {
                $actualHash = (Get-FileHash -Path $mapsZip -Algorithm SHA256).Hash
                if ($actualHash.ToUpperInvariant() -ne $SC2MiniGameMapsSha256.ToUpperInvariant()) {
                    Write-Err "SC2 mini-game maps zip SHA-256 mismatch!"
                    Write-Err "  Expected: $SC2MiniGameMapsSha256"
                    Write-Err "  Actual:   $actualHash"
                    exit 1
                }
                Write-Ok "Maps zip checksum verified."
            } else {
                Write-Host "[WARN]  SC2 maps zip SHA-256 not configured - skipping integrity check." -ForegroundColor Yellow
            }

            New-Item -ItemType Directory -Path $SC2MapsDir -Force | Out-Null
            Write-Step "  Expanding maps zip to $SC2MapsDir"
            Expand-Archive -Path $mapsZip -DestinationPath $SC2MapsDir -Force
            Assert-Success "SC2 maps extract"
            Write-Ok "SC2 mini-game maps extracted to $SC2MapsDir."
        }
    }
}

# ---------------------------------------------------------------------------
# 5c. TORCS
# ---------------------------------------------------------------------------

function Install-TORCS {
    if (Test-Path $TORCSExe) {
        Write-Skip "TORCS already installed at $TORCSExe."
        return
    }

    Invoke-Step "Downloading TORCS 1.3.7 Windows installer" {
        $installer = Join-Path $env:TEMP "torcs-setup.exe"
        Write-Step "  Fetching $TORCSInstallerUrl"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $TORCSInstallerUrl -OutFile $installer -UseBasicParsing
        Assert-Success "TORCS download"

        if ($TORCSInstallerSha256) {
            $actualHash = (Get-FileHash -Path $installer -Algorithm SHA256).Hash
            if ($actualHash.ToUpperInvariant() -ne $TORCSInstallerSha256.ToUpperInvariant()) {
                Write-Err "TORCS installer SHA-256 mismatch!"
                Write-Err "  Expected: $TORCSInstallerSha256"
                Write-Err "  Actual:   $actualHash"
                exit 1
            }
            Write-Ok "Installer checksum verified."
        } else {
            Write-Host "[WARN]  TORCS installer SHA-256 not configured - skipping integrity check." -ForegroundColor Yellow
        }

        Write-Step "  Running TORCS installer (silent)"
        # NSIS: /S = silent (case-sensitive), /D=<path> must be last and unquoted.
        $proc = Start-Process -FilePath $installer -ArgumentList "/S", "/D=$TORCSInstallDir" -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-Err "TORCS installation failed (exit code $($proc.ExitCode))."
            Write-Err "Please install TORCS 1.3.7 manually from http://torcs.sourceforge.net/"
            exit $proc.ExitCode
        }

        if (-not (Test-Path $TORCSExe)) {
            Write-Err "TORCS executable not found after install at $TORCSExe."
            Write-Err "Please install TORCS manually from http://torcs.sourceforge.net/"
            exit 1
        }
        Write-Ok "TORCS installed at $TORCSInstallDir."
    }
}

# ---------------------------------------------------------------------------
# 5d. SWIG (required by gymnasium[box2d] for Car Racing)
# ---------------------------------------------------------------------------

function Install-SWIG {
    if (Test-CommandExists "swig") {
        Write-Skip "SWIG already in PATH."
        return
    }

    Invoke-Step "Installing SWIG via winget (required for gymnasium[box2d])" {
        Assert-Winget
        winget install --id SWIG.SWIG --source winget --silent --accept-package-agreements --accept-source-agreements
        Assert-Success "SWIG install"
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
    }
}

# ---------------------------------------------------------------------------
# 5e. BeamNG — Python package only (binary is commercial, must be installed manually)
# ---------------------------------------------------------------------------

function Install-BeamNG {
    Write-Host "[INFO]  BeamNG.drive is a commercial game and cannot be installed automatically." -ForegroundColor Cyan
    Write-Host "[INFO]  Purchase and install it from https://www.beamng.com/ before running beamng training." -ForegroundColor Cyan
    Write-Host "[INFO]  The beamng-gym Python package will be installed by Install-PoetryDeps." -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 6. Poetry dependencies (all games)
# ---------------------------------------------------------------------------

function Install-PoetryDeps {
    Invoke-Step "Running 'poetry install' for all available dependency groups" {
        Push-Location $ScriptDir
        try {
            # Install all declared poetry groups so every game is ready to run.
            poetry install --with tmnf --with sc2
            Assert-Success "poetry install"
        } finally {
            Pop-Location
        }
    }

    # gym_torcs is not on PyPI — install from source via pip into the Poetry venv.
    $gymTorcsMarker = Join-Path $env:TEMP "gym_torcs_installed.flag"
    if (Test-Path $gymTorcsMarker) {
        Write-Skip "gym_torcs already installed (marker found)."
    } else {
        Invoke-Step "Installing gym_torcs from GitHub into Poetry venv" {
            Push-Location $ScriptDir
            try {
                poetry run pip install "git+https://github.com/ugo-nama-kun/gym_torcs.git" --quiet
                Assert-Success "gym_torcs install"
                New-Item -ItemType File -Path $gymTorcsMarker -Force | Out-Null
            } finally {
                Pop-Location
            }
        }
    }

    # beamng-gym is a separate package not included in poetry groups.
    $beamngMarker = Join-Path $env:TEMP "beamng_gym_installed.flag"
    if (Test-Path $beamngMarker) {
        Write-Skip "beamng-gym already installed (marker found)."
    } else {
        Invoke-Step "Installing beamng-gym into Poetry venv" {
            Push-Location $ScriptDir
            try {
                poetry run pip install beamng-gym --quiet
                Assert-Success "beamng-gym install"
                New-Item -ItemType File -Path $beamngMarker -Force | Out-Null
            } finally {
                Pop-Location
            }
        }
    }

    # gymnasium[box2d] requires SWIG for Box2D compilation (Car Racing).
    Invoke-Step "Installing gymnasium[box2d] for Car Racing into Poetry venv" {
        Push-Location $ScriptDir
        try {
            poetry run pip install "gymnasium[box2d]" --quiet
            Assert-Success "gymnasium[box2d] install"
        } finally {
            Pop-Location
        }
    }
}

# ---------------------------------------------------------------------------
# 7. Start game-specific runtime services
# ---------------------------------------------------------------------------

function Start-TMInterface {
    $procs = Get-Process -Name "TMInterface" -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Skip "TMInterface process already running (PID $($procs[0].Id))."
        return
    }

    if (-not (Test-Path $TMIfaceExe)) {
        Write-Err "Cannot launch TMInterface - executable not found at $TMIfaceExe."
        exit 1
    }

    Invoke-Step "Launching TMInterface" {
        Start-Process -FilePath $TMIfaceExe
        Write-Step "Waiting for TMInterface to initialise (up to 30 s)..."
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

function Start-TORCSServer {
    $procs = Get-Process -Name "wtorcs", "torcs" -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Skip "TORCS process already running (PID $($procs[0].Id))."
        return
    }

    if (-not (Test-Path $TORCSExe)) {
        Write-Err "Cannot launch TORCS - executable not found at $TORCSExe."
        Write-Err "Install TORCS from http://torcs.sourceforge.net/ or re-run this script."
        exit 1
    }

    Invoke-Step "Launching TORCS in SCR server mode" {
        # -nofuel -nodamage -nolaptime: standard flags for SCR training
        Start-Process -FilePath $TORCSExe -ArgumentList "-nofuel", "-nodamage", "-nolaptime"
        Write-Host "[WARN]  TORCS is starting. In the TORCS menu, start a race in Practice mode" -ForegroundColor Yellow
        Write-Host "        using the SCR server robot before running training commands." -ForegroundColor Yellow
        # Give TORCS a few seconds to open its window.
        Start-Sleep -Seconds 5
        $proc = Get-Process -Name "wtorcs", "torcs" -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Ok "TORCS is running (PID $($proc[0].Id))."
        } else {
            Write-Host "[WARN]  TORCS process not detected - it may have failed to start." -ForegroundColor Yellow
        }
    }
}

function Start-GameServices {
    Write-Step "Starting runtime services for game: $Game"
    switch ($Game) {
        "tmnf" {
            Start-TMInterface
        }
        "torcs" {
            Start-TORCSServer
        }
        "sc2" {
            # PySC2 manages the SC2 process automatically - no manual launch needed.
            Write-Skip "SC2: PySC2 launches and manages the SC2 process automatically."
        }
        "beamng" {
            Write-Host "[WARN]  BeamNG.drive must be running with a scenario loaded before training starts." -ForegroundColor Yellow
            Write-Host "        Launch BeamNG.drive manually, then run your training command." -ForegroundColor Yellow
        }
        "car_racing" {
            # Gymnasium car_racing runs entirely in-process - no external service needed.
            Write-Skip "car_racing: no external game process required."
        }
        default {
            Write-Err "Unknown game: $Game.  Valid values: tmnf, sc2, torcs, beamng, car_racing."
            exit 1
        }
    }
}

# ---------------------------------------------------------------------------
# 8. Run user command
# ---------------------------------------------------------------------------

function Invoke-UserCommand {
    if (-not $Command) {
        Write-Step "No command supplied - setup complete.  Pass a command as the first argument to launch training."
        return
    }

    Write-Step "Running: $Command"
    if ($DryRun) {
        Write-Host "        (dry-run - skipped)" -ForegroundColor Yellow
        return
    }

    Push-Location $ScriptDir
    try {
        # Tokenize safely using PSParser - respects quoted arguments and avoids
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
    Write-Host "=== DRY RUN - no changes will be made ===" -ForegroundColor Yellow
}

Write-Step "Selected game: $Game"

Install-Python
Install-Git
Install-Poetry

# Install ALL game binaries so the VM can run any game on demand.
Install-TMNF
Install-TMInterface
Install-SC2
Install-TORCS
Install-SWIG
Install-BeamNG

# Install Python dependencies for all games.
Install-PoetryDeps

# Start runtime services for the selected game only.
Start-GameServices

Invoke-UserCommand

Write-Ok "Done."
