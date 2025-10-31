# backup.ps1
# Create a dated Git backup branch, tag, zip archive,
# and keep only the N most recent backups (default: 3).

param(
  [int]$Keep = 3,                # number of backup branches to keep
  [switch]$IncludeRemote = $true # also prune on remote
)

# --- Helpers ---------------------------------------------------------------

function Run-Git {
  param([string[]]$GitArgs)

  # Call git and capture both output and exit code
  $out = & git @GitArgs 2>&1
  $code = $LASTEXITCODE

  if ($code -ne 0) {
    throw ("Git command failed (exit {0}): git {1}`n{2}" -f $code, ($GitArgs -join ' '), $out)
  }
  return $out
}

function Git-Ref-Exists {
  param([string]$Ref)
  & git show-ref --verify --quiet $Ref 2>$null
  return ($LASTEXITCODE -eq 0)
}

# --- Preconditions ---------------------------------------------------------

# Ensure git is available
try {
  $gitVer = Run-Git @("--version")
} catch {
  Write-Error "Git is not available on PATH. Please install Git or open Git Bash/PowerShell with Git."
  exit 1
}

# Ensure we are inside a git repo
try {
  Run-Git @("rev-parse","--git-dir") | Out-Null
} catch {
  Write-Error "This folder is not a Git repository. cd into your repo first."
  exit 1
}

# --- Start -----------------------------------------------------------------

$STAMP = Get-Date -Format "yyyy-MM-dd"
Write-Host "Starting backup for $STAMP ..."

# Detect current branch automatically
$currentBranch = (Run-Git @("rev-parse","--abbrev-ref","HEAD")).Trim()
Write-Host "Current branch: $currentBranch"

# Update current branch
Run-Git @("pull","origin",$currentBranch) | Out-Null

# Names for branch and tag
$backupBranch = "backup/$STAMP"
$tagName      = "snapshot-$STAMP"

# Create/push backup branch (idempotent)
$headSha = (Run-Git @("rev-parse","HEAD")).Trim()

if (Git-Ref-Exists "refs/heads/$backupBranch") {
  Write-Host "Backup branch already exists: $backupBranch (updating to $headSha)"
  # Move the branch to current HEAD (fast-forward or reset) and push
  Run-Git @("branch","-f",$backupBranch,$headSha) | Out-Null
} else {
  Run-Git @("branch",$backupBranch) | Out-Null
}
Run-Git @("push","-u","origin",$backupBranch) | Out-Null

# Create/push tag (idempotent)
if (Git-Ref-Exists "refs/tags/$tagName") {
  Write-Host "Tag already exists: $tagName (updating to $headSha)"
  Run-Git @("tag","-f",$tagName,$headSha) | Out-Null
} else {
  Run-Git @("tag","-a",$tagName,"-m","Snapshot before GPT modifications",$headSha) | Out-Null
}
Run-Git @("push","--force","origin",$tagName) | Out-Null

# Local zip archive
$RepoName  = Split-Path -Leaf (Get-Location)
$CommitSha = $headSha.Substring(0,7)
$BackupFile = "$HOME\${RepoName}_backup_${STAMP}_${CommitSha}.zip"
try {
  Compress-Archive -Path * -DestinationPath $BackupFile -Force
  Write-Host "Created local zip archive: $BackupFile"
} catch {
  Write-Warning "Zip creation failed (non-fatal): $($_.Exception.Message)"
}

# --- Cleanup old backups (keep most recent $Keep) --------------------------

Write-Host "`nCleaning up old backups (keeping last $Keep)..."
Run-Git @("fetch","--all","--prune") | Out-Null

# Local backup branches sorted by committer date (newest first)
$localRefs = Run-Git @("for-each-ref","refs/heads/backup","--format=%(refname:short)|%(committerdate:iso8601)","--sort=-committerdate")

if ($localRefs) {
  $localObjs = $localRefs -split "`n" | Where-Object { $_ -ne "" } | ForEach-Object {
    $parts = $_ -split "\|",2
    [pscustomobject]@{ Name=$parts[0]; Date=$parts[1] }
  }

  $toKeep   = $localObjs | Select-Object -First $Keep
  $toDelete = $localObjs | Select-Object -Skip $Keep

  if ($toDelete.Count -gt 0) {
    Write-Host "Deleting old local backups..."
    foreach ($b in $toDelete) {
      if ($b.Name -eq $currentBranch) {
        Write-Warning "Skipping currently checked-out branch: $($b.Name)"
        continue
      }
      Write-Host "  Removing local branch: $($b.Name)"
      Run-Git @("branch","-D",$b.Name) | Out-Null
    }
  } else {
    Write-Host "No old local backups to delete."
  }

  if ($IncludeRemote) {
    Write-Host "Deleting old remote backups..."
    $remoteRefs = Run-Git @("for-each-ref","refs/remotes/origin/backup","--format=%(refname:short)|%(committerdate:iso8601)","--sort=-committerdate")
    if ($remoteRefs) {
      $remoteObjs = $remoteRefs -split "`n" | Where-Object { $_ -ne "" } | ForEach-Object {
        $p = $_ -split "\|",2
        $short = $p[0] -replace "^origin/",""
        [pscustomobject]@{ Name=$short; Date=$p[1] }
      }
      $keepNames = $toKeep.Name
      foreach ($rb in $remoteObjs) {
        if ($keepNames -contains $rb.Name) { continue }
        Write-Host "  Removing remote branch: origin/$($rb.Name)"
        Run-Git @("push","origin","--delete",$rb.Name) | Out-Null
      }
    }
  }
}

Write-Host "`n✅ Backup complete!"
Write-Host "   -> Branch: $backupBranch"
Write-Host "   -> Tag:    $tagName"
Write-Host "   -> Local:  $BackupFile"
