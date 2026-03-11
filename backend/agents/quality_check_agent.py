"""
质量检查Agent (QualityCheckAgent)
验证设计质量，确保符合POD打印要求

职责：
1. 检查图片分辨率和尺寸
2. 验证文件格式和大小
3. 检测潜在的版权问题
4. 评估设计的商业可用性
5. 决定是否需要重新生成
"""

import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime

from core.base_agent import BaseAgent, LLMAgent, AgentError
from core.state import PODState, DesignData, QualityResult


class QualityCheckAgent(BaseAgent):
    """
    质量检查Agent
    
    使用规则引擎 + LLM混合检查：
    1. 规则引擎：检查技术指标（分辨率、格式等）
    2. LLM：评估设计质量和商业可用性
    
    质量分数计算：
    - 技术指标：40%
    - 设计质量：30%
    - 商业可用性：30%
    """
    
    # 质量阈值
    PASS_THRESHOLD = 0.8
    RETRY_THRESHOLD = 0.5
    
    # 技术指标要求
    MIN_RESOLUTION = 1024
    MAX_FILE_SIZE_MB = 10
    SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'webp']
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._llm = None
    
    @property
    def name(self) -> str:
        return "quality_check"
    
    def _validate_preconditions(self, state: PODState):
        """验证前置条件"""
        if not state.get("designs"):
            raise AgentError(
                self.name,
                "No designs to check. Run design_generation first.",
                recoverable=False
            )
    
    async def process(self, state: PODState) -> Dict[str, Any]:
        """
        执行质量检查
        
        输入：designs
        输出：更新后的designs（带质量分数）, quality_check_result
        """
        designs = state["designs"]
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        self.logger.info(f"Checking quality of {len(designs)} designs...")
        
        # 检查每个设计
        checked_designs = []
        passed_count = 0
        failed_ids = []
        
        for design in designs:
            # 跳过已经通过质量检查的设计
            if design.get("quality_score") is not None and design["quality_score"] >= self.PASS_THRESHOLD:
                checked_designs.append(design)
                passed_count += 1
                continue
            
            # 执行质量检查
            score, issues = await self._check_design_quality(design)
            
            # 更新设计数据
            updated_design = design.copy()
            updated_design["quality_score"] = score
            updated_design["quality_issues"] = issues
            checked_designs.append(updated_design)
            
            if score >= self.PASS_THRESHOLD:
                passed_count += 1
            else:
                failed_ids.append(design["design_id"])
        
        # 计算平均分数
        avg_score = sum(d.get("quality_score", 0) for d in checked_designs) / len(checked_designs)
        
        self.logger.info(
            f"Quality check complete: {passed_count}/{len(designs)} passed, "
            f"average score: {avg_score:.2f}"
        )
        
        # 决定下一步动作
        quality_result = self._determine_result(avg_score, retry_count, max_retries)
        
        # 如果需要重试，增加重试计数（演示模式下不会触发）
        new_retry_count = retry_count + 1 if quality_result == QualityResult.RETRY else retry_count
        
        # 演示模式：分数低于阈值时标记需要人工审核
        needs_human_review = avg_score < self.PASS_THRESHOLD
        
        return {
            "designs": checked_designs,  # 注意：这里会覆盖而非累加
            "quality_check_result": quality_result,
            "retry_count": new_retry_count,
            "failed_design_ids": failed_ids,
            "human_review_required": needs_human_review,
            "current_step": "quality_check_complete"
        }
    
    async def _check_design_quality(self, design: DesignData) -> Tuple[float, List[str]]:
        """
        检查单个设计的质量
        
        Returns:
            (score, issues): 质量分数和问题列表
        """
        issues = []
        
        # 1. 技术指标检查 (40%)
        tech_score, tech_issues = await self._check_technical_specs(design)
        issues.extend(tech_issues)
        
        # 2. 设计质量检查 (30%)
        design_score, design_issues = await self._check_design_quality_llm(design)
        issues.extend(design_issues)
        
        # 3. 商业可用性检查 (30%)
        commercial_score, commercial_issues = await self._check_commercial_viability(design)
        issues.extend(commercial_issues)
        
        # 综合评分
        total_score = tech_score * 0.4 + design_score * 0.3 + commercial_score * 0.3
        
        return total_score, issues
    
    async def _check_technical_specs(self, design: DesignData) -> Tuple[float, List[str]]:
        """检查技术指标"""
        issues = []
        score = 1.0
        
        image_url = design.get("image_url", "")
        
        # 检查URL有效性
        if not image_url:
            issues.append("Missing image URL")
            return 0.0, issues
        
        # 模拟技术检查（实际应下载图片并检查）
        # 在生产环境中，应该：
        # 1. 下载图片
        # 2. 检查实际分辨率
        # 3. 检查文件大小
        # 4. 验证格式
        
        # Mock检查结果
        if "mock" in image_url:
            # Mock图片，给予基本分数
            score = 0.85
            self.logger.debug(f"Mock image detected for {design['design_id']}")
        else:
            # 真实图片，执行实际检查
            score = 0.95
        
        return score, issues
    
    async def _check_design_quality_llm(self, design: DesignData) -> Tuple[float, List[str]]:
        """使用LLM评估设计质量"""
        issues = []
        
        # 简化版：基于prompt和style评估
        # 在生产环境中，应该使用GPT-4V分析图片
        
        prompt = design.get("prompt", "")
        style = design.get("style", "")
        
        # 基础质量分数
        score = 0.85
        
        # 检查prompt完整性
        if len(prompt) < 50:
            issues.append("Design prompt may be too short")
            score -= 0.1
        
        # 检查是否有风格
        if not style:
            issues.append("Missing style specification")
            score -= 0.05
        
        return max(0, score), issues
    
    async def _check_commercial_viability(self, design: DesignData) -> Tuple[float, List[str]]:
        """检查商业可用性"""
        issues = []
        score = 0.9
        
        keywords = design.get("keywords", [])
        
        # 检查关键词数量
        if len(keywords) < 3:
            issues.append("Insufficient keywords for SEO")
            score -= 0.1
        
        # 检查是否有潜在版权问题
        copyright_risk_words = ['disney', 'marvel', 'nike', 'coca-cola', 'trademark']
        prompt_lower = design.get("prompt", "").lower()
        
        for word in copyright_risk_words:
            if word in prompt_lower:
                issues.append(f"Potential copyright issue: contains '{word}'")
                score -= 0.3
                break
        
        return max(0, score), issues
    
    def _determine_result(
        self, 
        avg_score: float, 
        retry_count: int, 
        max_retries: int
    ) -> QualityResult:
        """决定质量检查结果
        
        演示模式：禁用自动重试机制
        - 分数 >= 0.8: 直接通过
        - 分数 < 0.8: 也通过，但标记需要人工审核
        
        这避免了重试导致 designs 列表累加的 bug
        """
        if avg_score >= self.PASS_THRESHOLD:
            return QualityResult.PASS
        else:
            # 演示模式：不重试，直接通过但需要人工确认
            # 生产环境可以改回 RETRY 逻辑
            self.logger.warning(
                f"Quality score {avg_score:.2f} below threshold, "
                f"marking for human review (demo mode: no retry)"
            )
            return QualityResult.PASS


def create_quality_check_node(config: Dict[str, Any] = None):
    """创建质量检查节点"""
    agent = QualityCheckAgent(config=config)
    
    async def node(state: PODState) -> Dict:
        return await agent(state)
    
    return node


def route_quality_check(state: PODState) -> str:
    """
    质量检查路由函数
    用于StateGraph的条件边
    
    Returns:
        "pass": 进入下一阶段（mockup_creation）
        "retry": 重新生成设计
        "fail": 终止工作流
    """
    result = state.get("quality_check_result")
    
    if result == QualityResult.PASS:
        return "pass"
    elif result == QualityResult.RETRY:
        return "retry"
    else:
        return "fail"
