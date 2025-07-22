"""
LLM判别器模块
"""

import logging
from typing import Optional
from .config import LLM_PROMPTS
from .exceptions import LLMJudgmentError
from utils.doubao_api import call_doubao

# 配置日志
logger = logging.getLogger(__name__)


class LLMJudge:
    """大模型判别器，使用豆包API进行论文相关性判断"""
    
    def __init__(self):
        """初始化LLM判别器"""
        self.prompts = LLM_PROMPTS
    
    def get_summary(self, title: str, abstract: str) -> str:
        """
        获取论文的一句话总结
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            总结内容
        """
        try:
            full_prompt = self._build_prompt(title, abstract, "summary_task")
            
            messages = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
            
            logger.info("正在调用豆包API生成论文总结...")
            result = call_doubao(messages)
            if result.get("success"):
                logger.info("论文总结生成成功")
                return result.get("content", "").strip()
            else:
                logger.error(f"论文总结生成失败: {result.get('error', '未知错误')}")
                return f"总结生成失败: {result.get('error', '未知错误')}"
                
        except Exception as e:
            logger.error(f"论文总结生成出错: {str(e)}")
            raise LLMJudgmentError(f"总结生成出错: {str(e)}") from e
    
    def judge_relevance(self, title: str, abstract: str) -> str:
        """
        判断论文是否与AI推理加速相关
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            相关性判断结果
        """
        try:
            full_prompt = self._build_prompt(title, abstract, "relevance_task")
            
            messages = [
                {
                    "role": "user", 
                    "content": full_prompt
                }
            ]
            
            logger.info("正在调用豆包API判断论文相关性...")
            result = call_doubao(messages)
            if result.get("success"):
                logger.info("论文相关性判断成功")
                return result.get("content", "").strip()
            else:
                logger.error(f"论文相关性判断失败: {result.get('error', '未知错误')}")
                return f"相关性判断失败: {result.get('error', '未知错误')}"
                
        except Exception as e:
            logger.error(f"论文相关性判断出错: {str(e)}")
            raise LLMJudgmentError(f"相关性判断出错: {str(e)}") from e
    
    def translate_abstract(self, title: str, abstract: str) -> str:
        """
        将英文摘要翻译成中文
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            中文翻译
        """
        try:
            full_prompt = self._build_prompt(title, abstract, "translation_task")
            
            messages = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
            
            logger.info("正在调用豆包API翻译论文摘要...")
            result = call_doubao(messages)
            if result.get("success"):
                logger.info("论文摘要翻译成功")
                return result.get("content", "").strip()
            else:
                logger.error(f"论文摘要翻译失败: {result.get('error', '未知错误')}")
                return f"翻译失败: {result.get('error', '未知错误')}"
                
        except Exception as e:
            logger.error(f"论文摘要翻译出错: {str(e)}")
            raise LLMJudgmentError(f"翻译出错: {str(e)}") from e
    
    def _build_prompt(self, title: str, abstract: str, task_key: str) -> str:
        """
        构建完整的提示词
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            task_key: 任务类型键
            
        Returns:
            完整的提示词
        """
        logger.debug(f"构建提示词，任务类型: {task_key}")
        common_prefix = self.prompts["common_prefix"].format(
            title=title, abstract=abstract
        )
        task_prompt = self.prompts[task_key]
        
        return common_prefix + task_prompt
    
    def process_paper_judgment(self, title: str, abstract: str) -> dict:
        """
        处理论文的完整LLM判别
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            包含所有LLM判别结果的字典
        """
        result = {
            "summary": "",
            "relevance": "",
            "translation": ""
        }
        
        try:
            # 并行处理所有任务（这里简化为串行）
            result["summary"] = self.get_summary(title, abstract)
            result["relevance"] = self.judge_relevance(title, abstract)
            result["translation"] = self.translate_abstract(title, abstract)
            
            logger.info("论文完整LLM判别处理完成")
            
        except Exception as e:
            raise LLMJudgmentError(f"LLM判别处理失败: {str(e)}") from e
        
        return result 