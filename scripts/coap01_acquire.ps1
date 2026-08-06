$ErrorActionPreference = 'Stop'
$root = 'C:\MIMII_VERIFIED_ACQUISITION'
$old = 'E:\MIMII'
$archives = Join-Path $root 'archives'
$logs = Join-Path $root 'logs'
$projectLog = 'research_validation/provenance/CONTROLLED_DOWNLOAD_LOG.csv'
$manifest = Import-Csv 'research_validation/provenance/OFFICIAL_RELEASE_MANIFEST.csv'
$run = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')

if (-not (Test-Path -LiteralPath $archives)) { throw 'Archive destination missing' }
$resolvedRoot = [IO.Path]::GetFullPath($root).TrimEnd('\')
$resolvedOld = [IO.Path]::GetFullPath($old).TrimEnd('\')
if ($resolvedRoot -eq $resolvedOld -or $resolvedRoot.StartsWith($resolvedOld + '\') -or $resolvedOld.StartsWith($resolvedRoot + '\')) { throw 'Destination overlaps historical dataset' }
$space = (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace
if ($space -lt 300000000000) { throw "Insufficient storage before download: $space bytes" }

$logPath = Join-Path $logs "coap01-$run.log"
"run=$run root=$root old=$old free_space=$space" | Out-File -LiteralPath $logPath -Encoding utf8
if (-not (Test-Path -LiteralPath $projectLog)) { "archive_name,source,local_path,download_started,download_completed,size_bytes,status,retry_count,notes" | Set-Content -LiteralPath $projectLog -Encoding utf8 }

foreach ($row in $manifest) {
    $name = $row.archive_name
    $final = Join-Path $archives $name
    if (Test-Path -LiteralPath $final) { throw "Refusing to overwrite existing archive: $final" }
    $part = "$final.part.$run"
    $url = "https://zenodo.org/api/records/3384388/files/$([uri]::EscapeDataString($name))/content"
    $start = (Get-Date).ToUniversalTime()
    $status = 'FAILED'; $note = ''
    try {
        "START $name $url" | Add-Content -LiteralPath $logPath
        & curl.exe --fail --location --retry 3 --retry-all-errors --retry-delay 60 --output $part $url
        if ($LASTEXITCODE -ne 0) { throw "curl exit code $LASTEXITCODE" }
        $item = Get-Item -LiteralPath $part
        $md5 = (Get-FileHash -LiteralPath $part -Algorithm MD5).Hash.ToLower()
        $sha = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLower()
        if ($md5 -ne $row.official_md5.ToLower()) {
            $bad = "$final.mismatch.$run"
            Move-Item -LiteralPath $part -Destination $bad
            $status = 'MISMATCH'; $note = "expected=$($row.official_md5) actual=$md5 sha256=$sha preserved=$bad"
            throw "MD5 mismatch for $name"
        }
        Move-Item -LiteralPath $part -Destination $final
        $status = 'MATCH'; $note = "md5=$md5 sha256=$sha"
        "MATCH $name size=$($item.Length) $note" | Add-Content -LiteralPath $logPath
    } catch {
        if (Test-Path -LiteralPath $part) { $failed = "$final.failed.$run"; Move-Item -LiteralPath $part -Destination $failed; $note = "$note preserved=$failed" }
        "STOP $name $($_.Exception.Message)" | Add-Content -LiteralPath $logPath
    }
    $end = (Get-Date).ToUniversalTime()
    $size = ''
    if (Test-Path -LiteralPath $final) { $size = (Get-Item -LiteralPath $final).Length }
    "$name,$url,$final,$($start.ToString('o')),$($end.ToString('o')),$size,$status,0,$($note -replace ',',';')" | Add-Content -LiteralPath $projectLog
    if ($status -ne 'MATCH') { throw "COAP-01 stopped at $name with status $status" }
}
"ALL_ARCHIVES_MATCH" | Add-Content -LiteralPath $logPath
