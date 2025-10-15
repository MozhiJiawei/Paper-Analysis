#!/usr/bin/env pwsh
<#
.SYNOPSIS
    ArXiv论文分析自动化脚本
.PARAMETER ParseDir
    解析目录路径，例如: "2025-09/09-01"
.PARAMETER ParseMonth
    解析月份路径，例如: "2025-09"，将分析该月份下所有未分析的日期目录
.EXAMPLE
    .\arxiv_analysis.ps1 -ParseDir "2025-09/09-01"
.EXAMPLE
    .\arxiv_analysis.ps1 -ParseMonth "2025-09"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [string]$ParseDir,
    
    [Parameter(Mandatory=$false)]
    [string]$ParseMonth
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 参数验证
if (-not $ParseDir -and -not $ParseMonth) {
    throw "必须提供 ParseDir 或 ParseMonth 参数之一"
}

if ($ParseDir -and $ParseMonth) {
    throw "不能同时提供 ParseDir 和 ParseMonth 参数，请只选择其中一个"
}

# 通用Python脚本执行函数
function Invoke-PythonScript {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments,
        [string]$TaskName
    )
    
    Write-Host "执行$TaskName..." -ForegroundColor Yellow
    Write-Host "执行命令: python `"$ScriptPath`" $($Arguments -join ' ')" -ForegroundColor Cyan
    
    $process = Start-Process -FilePath "python" -ArgumentList (@("`"$ScriptPath`"") + $Arguments) -PassThru -Wait -NoNewWindow
    
    if ($process.ExitCode -ne 0) {
        Write-Host ("=" * 80) -ForegroundColor Red
        Write-Host "$TaskName 执行失败，退出码: $($process.ExitCode)" -ForegroundColor Red
        Write-Host ("=" * 80) -ForegroundColor Red
        throw "$TaskName 执行失败"
    }
    
    Write-Host "$TaskName 执行完成" -ForegroundColor Green
}

# 更新README文件的函数
function Update-ReadmeWithReport {
    param(
        [string]$ParseDir,
        [string]$ReadmePath
    )
    
    Write-Host "更新README文件..." -ForegroundColor Yellow
    
    try {
        # 解析日期信息
        $dateParts = $ParseDir -split '[\\/]'
        if ($dateParts.Length -ne 2) {
            throw "日期格式不正确，期望格式: YYYY-MM/MM-DD，实际: $ParseDir"
        }
        
        $yearMonth = $dateParts[0]  # 例如: 2025-09
        $dayPart = $dateParts[1]   # 例如: 09-01
        
        # 构建显示日期 (YYYY-MM-DD格式)
        # 将 09-01 格式转换为 01
        $dayNumber = $dayPart -replace '^\d{2}-', ''
        $displayDate = "$yearMonth-$dayNumber"
        
        # 查找对应的报告文件
        $targetDir = Join-Path $ScriptDir $ParseDir
        $reportFiles = Get-ChildItem -Path $targetDir -Filter "ai_inference_report_*.md" -File
        
        if ($reportFiles.Count -eq 0) {
            Write-Host "警告: 未找到报告文件，跳过README更新" -ForegroundColor Yellow
            return
        }
        
        # 使用最新的报告文件
        $latestReport = $reportFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        # 统一使用正斜杠路径分隔符
        $reportRelativePath = "$($ParseDir -replace '\\', '/')/$($latestReport.Name)"
        
        # 读取现有README内容
        if (-not (Test-Path $ReadmePath)) {
            throw "README文件不存在: $ReadmePath"
        }
        
        $readmeContent = Get-Content $ReadmePath -Encoding UTF8
        $yearMonthClean = $yearMonth -replace '-', '_'
        $startMarker = "<!-- REPORTS_START_$yearMonthClean -->"
        $endMarker = "<!-- REPORTS_END_$yearMonthClean -->"
        
        # 查找标记位置
        $startIndex = -1
        $endIndex = -1
        
        for ($i = 0; $i -lt $readmeContent.Length; $i++) {
            if ($readmeContent[$i] -eq $startMarker) {
                $startIndex = $i
            }
            if ($readmeContent[$i] -eq $endMarker) {
                $endIndex = $i
                break
            }
        }
        
        if ($startIndex -eq -1 -or $endIndex -eq -1) {
            Write-Host "警告: 未找到对应年月的标记区域 ($startMarker, $endMarker)，跳过README更新" -ForegroundColor Yellow
            return
        }
        
        # 提取现有报告列表
        $existingReports = @()
        for ($i = $startIndex + 1; $i -lt $endIndex; $i++) {
            $line = $readmeContent[$i].Trim()
            if ($line -match '^\s*-\s*\[(.+?)\]') {
                $existingReports += $matches[1]
            }
        }
        
        # 检查是否已存在该日期的报告
        if ($existingReports -contains $displayDate) {
            Write-Host "报告 $displayDate 已存在于README中，更新路径..." -ForegroundColor Cyan
            
            # 更新现有条目
            $newContent = @()
            $newContent += $readmeContent[0..$startIndex]
            
            for ($i = $startIndex + 1; $i -lt $endIndex; $i++) {
                $line = $readmeContent[$i]
                if ($line -match '^\s*-\s*\[' + [regex]::Escape($displayDate) + '\]') {
                    $newContent += "- [$displayDate]($reportRelativePath)"
                } else {
                    $newContent += $line
                }
            }
            
            $newContent += $readmeContent[$endIndex..($readmeContent.Length - 1)]
            
        } else {
            Write-Host "添加新报告 $displayDate 到README..." -ForegroundColor Cyan
            
            # 添加新条目（按日期倒序）
            $allReports = @()
            $allReports += @{Date=$displayDate; Path=$reportRelativePath}
            
            # 添加现有报告
            for ($i = $startIndex + 1; $i -lt $endIndex; $i++) {
                $line = $readmeContent[$i].Trim()
                if ($line -match '^\s*-\s*\[(.+?)\]\((.+?)\)') {
                    $allReports += @{Date=$matches[1]; Path=$matches[2]}
                }
            }
            
            # 按日期排序（倒序）- 转换为DateTime对象进行正确排序
            $sortedReports = $allReports | Sort-Object { [DateTime]::ParseExact($_.Date, 'yyyy-MM-dd', $null) } -Descending
            
            # 构建新内容
            $newContent = @()
            $newContent += $readmeContent[0..$startIndex]
            
            foreach ($report in $sortedReports) {
                $newContent += "- [$($report.Date)]($($report.Path))"
            }
            
            $newContent += ""  # 空行
            $newContent += $readmeContent[$endIndex..($readmeContent.Length - 1)]
        }
        
        # 写入更新后的内容
        $newContent | Set-Content $ReadmePath -Encoding UTF8
        Write-Host "README更新完成" -ForegroundColor Green
        
    } catch {
        Write-Host "README更新失败: $_" -ForegroundColor Red
        throw
    }
}

# 扫描月份下所有需要分析的目录
function Get-DirectoriesToAnalyze {
    param(
        [string]$Month
    )
    
    Write-Host "扫描月份目录: $Month" -ForegroundColor Yellow
    
    $monthDir = Join-Path $ScriptDir $Month
    if (-not (Test-Path $monthDir)) {
        throw "月份目录不存在: $monthDir"
    }
    
    # 获取月份下所有日期目录
    $dayDirs = Get-ChildItem -Path $monthDir -Directory | Where-Object { $_.Name -match '^\d{2}-\d{2}$' }
    
    if ($dayDirs.Count -eq 0) {
        Write-Host "未找到任何日期目录" -ForegroundColor Yellow
        return @()
    }
    
    $directoriesToAnalyze = @()
    
    foreach ($dayDir in $dayDirs) {
        $fullParseDir = "$Month/$($dayDir.Name)"
        $targetDir = Join-Path $ScriptDir $fullParseDir
        
        # 检查是否存在arxiv_daily.txt文件
        $arxivDaily = Join-Path $targetDir "arxiv_daily.txt"
        if (-not (Test-Path $arxivDaily)) {
            Write-Host "跳过目录 $fullParseDir (缺少arxiv_daily.txt)" -ForegroundColor Gray
            continue
        }
        
        # 检查是否已经生成过技术简报
        $reportFiles = Get-ChildItem -Path $targetDir -Filter "ai_inference_report_*.md" -File
        if ($reportFiles.Count -gt 0) {
            Write-Host "跳过目录 $fullParseDir (已存在技术简报: $($reportFiles.Count)个)" -ForegroundColor Gray
            continue
        }
        
        Write-Host "发现待分析目录: $fullParseDir" -ForegroundColor Cyan
        $directoriesToAnalyze += $fullParseDir
    }
    
    return $directoriesToAnalyze
}

# 单目录分析函数（从原主流程中提取）
function Invoke-SingleDirectoryAnalysis {
    param(
        [string]$SingleParseDir
    )
    
    Write-Host "`n开始分析目录: $SingleParseDir" -ForegroundColor Green
    
    try {
        # 1. 检查必要文件
        $targetDir = Join-Path $ScriptDir $SingleParseDir
        $arxivDaily = Join-Path $targetDir "arxiv_daily.txt"
        
        if (-not (Test-Path $arxivDaily)) {
            throw "找不到arxiv_daily.txt文件: $arxivDaily"
        }
        
        # 2. 执行parser.py
        $parserScript = Join-Path $ScriptDir "parser.py"
        Invoke-PythonScript -ScriptPath $parserScript -Arguments @("`"$SingleParseDir`"") -TaskName "AI加速论文解析"

        # 4. 执行deep_analysis.py生成简报
        $deepAnalysisScript = Join-Path $ScriptDir "deep_analysis.py"
        Invoke-PythonScript -ScriptPath $deepAnalysisScript -Arguments @("`"$SingleParseDir`"") -TaskName "技术简报生成"
        
        # 5. 更新README文件
        $readmePath = Join-Path $ScriptDir "README.md"
        Update-ReadmeWithReport -ParseDir $SingleParseDir -ReadmePath $readmePath
        
        Write-Host "目录 $SingleParseDir 分析完成" -ForegroundColor Green
        
    } catch {
        Write-Host "目录 $SingleParseDir 分析失败: $_" -ForegroundColor Red
        throw
    }
}

try {
    # 确定分析模式
    if ($ParseMonth) {
        Write-Host "开始ArXiv论文批量分析流程 - 月份: $ParseMonth" -ForegroundColor Green
        
        # 获取需要分析的目录列表
        $directoriesToAnalyze = Get-DirectoriesToAnalyze -Month $ParseMonth
        
        if ($directoriesToAnalyze.Count -eq 0) {
            Write-Host "没有找到需要分析的目录" -ForegroundColor Yellow
            exit 0
        }
        
        Write-Host "找到 $($directoriesToAnalyze.Count) 个目录需要分析" -ForegroundColor Cyan
        
        # 批量分析所有目录
        $successCount = 0
        $failCount = 0
        
        foreach ($dir in $directoriesToAnalyze) {
            try {
                Invoke-SingleDirectoryAnalysis -SingleParseDir $dir
                $successCount++
            } catch {
                Write-Host "目录 $dir 分析失败，继续处理下一个..." -ForegroundColor Red
                $failCount++
            }
        }
        
        Write-Host "`n批量分析完成!" -ForegroundColor Green
        Write-Host "成功: $successCount 个目录" -ForegroundColor Green
        Write-Host "失败: $failCount 个目录" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })
        
    } else {
        Write-Host "开始ArXiv论文单目录分析流程 - $ParseDir" -ForegroundColor Green
        Invoke-SingleDirectoryAnalysis -SingleParseDir $ParseDir
    }
    
    # 6. 提交到Git
    Write-Host "`n提交到Git..." -ForegroundColor Yellow
    
    # 确定要处理的目录列表
    $dirsToCommit = if ($ParseMonth) { $directoriesToAnalyze } else { @($ParseDir) }
    
    # 处理每个目录的文件
    foreach ($relativePath in $dirsToCommit) {
        $targetDir = Join-Path $ScriptDir $relativePath
        
        Write-Host "处理目录: $relativePath" -ForegroundColor Cyan
        
        # 检查并添加arxiv_daily.txt文件
        $arxivDailyFile = Join-Path $targetDir "arxiv_daily.txt"
        if (Test-Path $arxivDailyFile) {
            # 使用--force参数强制添加被.gitignore忽略的文件
            git add --force $arxivDailyFile 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  已强制添加arxiv_daily.txt到Git (覆盖.gitignore规则)" -ForegroundColor Cyan
            } else {
                Write-Host "  警告: 添加arxiv_daily.txt失败" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  警告: 未找到arxiv_daily.txt文件: $arxivDailyFile" -ForegroundColor Yellow
        }
        
        # 查找并添加生成的技术简报文件
        $reportFiles = Get-ChildItem -Path $targetDir -Filter "ai_inference_report_*.md" -File
        if ($reportFiles.Count -gt 0) {
            foreach ($reportFile in $reportFiles) {
                git add $reportFile.FullName 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  已添加技术简报文件到Git: $($reportFile.Name)" -ForegroundColor Cyan
                } else {
                    Write-Host "  警告: 添加技术简报文件失败: $($reportFile.Name)" -ForegroundColor Yellow
                }
            }
        } else {
            Write-Host "  警告: 未找到技术简报文件 (ai_inference_report_*.md)" -ForegroundColor Yellow
        }
    }
    
    # 添加更新后的README文件
    $readmePath = Join-Path $ScriptDir "README.md"
    if (Test-Path $readmePath) {
        git add $readmePath 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "已添加README文件到Git" -ForegroundColor Cyan
        } else {
            Write-Host "警告: 添加README文件失败" -ForegroundColor Yellow
        }
    } else {
        Write-Host "警告: 未找到README文件: $readmePath" -ForegroundColor Yellow
    }
    
    # 检查是否有文件需要提交
    $statusAfterAdd = git status --porcelain 2>&1
    if ($statusAfterAdd) {
        if ($ParseMonth) {
            $commitMessage = "自动提交ArXiv论文批量分析结果 - 月份: $ParseMonth (共$($dirsToCommit.Count)个目录) ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
        } else {
            $commitMessage = "自动提交ArXiv论文分析结果 - $ParseDir ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
        }
        Write-Host "执行Git commit..." -ForegroundColor Cyan
        git commit -m $commitMessage 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Git提交失败"
        }
        Write-Host "已提交到本地Git仓库" -ForegroundColor Green
        
        # 推送到远端
        Write-Host "推送到远端..." -ForegroundColor Yellow
        $pushOutput = git push 2>&1
        $pushExitCode = $LASTEXITCODE
        
        # 检查推送结果
        if ($pushExitCode -eq 0) {
            Write-Host "已成功推送到远端仓库" -ForegroundColor Green
        } else {
            # 检查输出中是否包含"Bypassed rule violations"，这通常表示推送成功但有警告
            $outputString = $pushOutput -join "`n"
            if ($outputString -match "Bypassed rule violations") {
                Write-Host "推送成功（绕过了远程仓库的保护规则）" -ForegroundColor Yellow
                Write-Host "远程仓库返回信息: $outputString" -ForegroundColor Cyan
            } else {
                Write-Host "警告: Git push失败" -ForegroundColor Yellow
                Write-Host "错误信息: $outputString" -ForegroundColor Red
                # 不抛出异常，让脚本继续执行
            }
        }
    } else {
        Write-Host "没有新文件需要提交" -ForegroundColor Yellow
    }
    
    Write-Host "`nArXiv论文分析流程完成!" -ForegroundColor Green
    if ($ParseMonth) {
        Write-Host "批量分析月份: $ParseMonth" -ForegroundColor Cyan
        Write-Host "处理的目录数量: $($dirsToCommit.Count)" -ForegroundColor Cyan
    } else {
        Write-Host "解析结果保存在: $(Join-Path $ScriptDir $ParseDir)" -ForegroundColor Cyan
    }
    
} catch {
    Write-Host ("=" * 60) -ForegroundColor Red
    Write-Host "脚本执行失败!" -ForegroundColor Red
    Write-Host "错误信息: $_" -ForegroundColor Red
    Write-Host "错误位置: 第 $($_.InvocationInfo.ScriptLineNumber) 行" -ForegroundColor Red
    if ($_.Exception.InnerException) {
        Write-Host "内部异常: $($_.Exception.InnerException.Message)" -ForegroundColor Red
    }
    Write-Host ("=" * 60) -ForegroundColor Red
    exit 1
}
