#!/usr/bin/env pwsh
<#
.SYNOPSIS
    ArXiv论文分析自动化脚本
.PARAMETER ParseDir
    解析目录路径，例如: "2025-09/09-01"
.EXAMPLE
    .\arxiv_analysis.ps1 -ParseDir "2025-09/09-01"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$ParseDir
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

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

try {
    Write-Host "开始ArXiv论文分析流程 - $ParseDir" -ForegroundColor Green
    
    # 1. 检查必要文件
    $targetDir = Join-Path $ScriptDir $ParseDir
    $arxivDaily = Join-Path $targetDir "arxiv_daily.txt"
    
    if (-not (Test-Path $arxivDaily)) {
        throw "找不到arxiv_daily.txt文件: $arxivDaily"
    }
    
    # 2. 执行parser.py
#     $parserScript = Join-Path $ScriptDir "parser.py"
#     Invoke-PythonScript -ScriptPath $parserScript -Arguments @("`"$ParseDir`"") -TaskName "AI加速论文解析"

    # 4. 执行deep_analysis.py生成简报
    # $deepAnalysisScript = Join-Path $ScriptDir "deep_analysis.py"
    # Invoke-PythonScript -ScriptPath $deepAnalysisScript -Arguments @("`"$ParseDir`"") -TaskName "技术简报生成"
    
    # 5. 提交到Git
    Write-Host "提交到Git..." -ForegroundColor Yellow
    $relativePath = $ParseDir -replace '\\', '/'
    
    # 检查并添加arxiv_daily.txt文件
    $arxivDailyFile = Join-Path $targetDir "arxiv_daily.txt"
    if (Test-Path $arxivDailyFile) {
        # 使用--force参数强制添加被.gitignore忽略的文件
        git add --force $arxivDailyFile 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "已强制添加arxiv_daily.txt到Git (覆盖.gitignore规则)" -ForegroundColor Cyan
        } else {
            Write-Host "警告: 添加arxiv_daily.txt失败" -ForegroundColor Yellow
        }
    } else {
        Write-Host "警告: 未找到arxiv_daily.txt文件: $arxivDailyFile" -ForegroundColor Yellow
    }
    
    # 查找并添加生成的技术简报文件
    $reportFiles = Get-ChildItem -Path $targetDir -Filter "ai_inference_report_*.md" -File
    if ($reportFiles.Count -gt 0) {
        foreach ($reportFile in $reportFiles) {
            git add $reportFile.FullName 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "已添加技术简报文件到Git: $($reportFile.Name)" -ForegroundColor Cyan
            } else {
                Write-Host "警告: 添加技术简报文件失败: $($reportFile.Name)" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "警告: 未找到技术简报文件 (ai_inference_report_*.md)" -ForegroundColor Yellow
    }
    
    # 检查是否有文件需要提交
    $statusAfterAdd = git status --porcelain 2>&1
    if ($statusAfterAdd) {
        $commitMessage = "自动提交ArXiv论文分析结果 - $ParseDir ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
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
    
    Write-Host "ArXiv论文分析流程完成!" -ForegroundColor Green
    Write-Host "解析结果保存在: $targetDir" -ForegroundColor Cyan
    
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
