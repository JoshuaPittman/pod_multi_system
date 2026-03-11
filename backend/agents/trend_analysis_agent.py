"""
趋势分析Agent (TrendAnalysisAgent)
使用Claude 3.5 Sonnet分析市场趋势、热门关键词、目标受众

职责：
1. 分析利基市场的当前趋势
2. 识别热门子话题和关键词
3. 分析目标受众画像
4. 评估竞争程度
5. 推荐设计风格
"""

import json
from typing import Dict, Any
from datetime import datetime

from core.base_agent import LLMAgent, AgentError
from core.state import PODState, TrendData


class TrendAnalysisAgent(LLMAgent):
    """
    趋势分析Agent
    
    使用Claude 3.5 Sonnet进行市场分析，因为：
    1. 强大的推理能力
    2. 丰富的市场知识
    3. 结构化输出能力
    """
    
    @property
    def name(self) -> str:
        return "trend_analysis"
    
    def _validate_preconditions(self, state: PODState):
        """验证必要的输入参数"""
        if not state.get("niche"):
            raise AgentError(self.name, "Missing required field: niche", recoverable=False)
        if not state.get("style"):
            raise AgentError(self.name, "Missing required field: style", recoverable=False)
    
    async def process(self, state: PODState) -> Dict[str, Any]:
        """
        分析市场趋势
        
        输入：niche, style
        输出：trend_data, design_prompts
        """
        niche = state["niche"]
        style = state["style"]
        num_designs = state.get("num_designs", 5)
        
        self.logger.info(f"Analyzing trends for niche: {niche}, style: {style}")
        
        # 构建分析提示词
        prompt = self._build_analysis_prompt(niche, style, num_designs)
        
        # 调用LLM
        response = await self.invoke_llm(prompt)
        
        # 解析响应
        trend_data, design_prompts = self._parse_response(response, niche, style)
        
        self.logger.info(
            f"Analysis complete: {len(trend_data['keywords'])} keywords, "
            f"{len(design_prompts)} design prompts (limiting to {num_designs})"
        )
        
        # 确保只返回用户请求的设计数量（LLM 可能返回更多）
        limited_prompts = design_prompts[:num_designs]
        
        cost_breakdown = state.get("cost_breakdown", {}).copy()
        cost_breakdown["anthropic"] = cost_breakdown.get("anthropic", 0) + self.llm_cost

        return {
            "trend_data": trend_data,
            "design_prompts": limited_prompts,
            "current_step": "trend_analysis_complete",
            "total_cost": state.get("total_cost", 0) + self.llm_cost,
            "cost_breakdown": cost_breakdown,
        }
    
    def _build_analysis_prompt(self, niche: str, style: str, num_designs: int) -> str:
        """构建分析提示词"""
        return f"""作为POD（Print-on-Demand）市场分析专家，深入分析"{niche}"利基市场的当前趋势。

目标风格：{style}
需要设计数量：{num_designs}

请完成以下任务：

1. **子话题识别**（3-5个）
   - 识别该利基市场中最热门的细分话题
   - 考虑季节性趋势和当前热点

2. **关键词分析**（10-15个）
   - 高搜索量、低竞争的长尾关键词
   - 适合用于产品标题和标签的关键词
   - 考虑购买意图强的关键词

3. **目标受众画像**
   - 年龄段、性别倾向
   - 兴趣爱好
   - 购买动机
   - 价格敏感度

4. **竞争分析**
   - 竞争程度评估（低/中/高）
   - 差异化机会

5. **设计建议**
   - 推荐{num_designs}个具体的设计主题
   - 每个主题包含：标题、元素描述、情感基调

请严格按照以下JSON格式返回结果：
{{
    "sub_topics": ["话题1", "话题2", "话题3"],
    "keywords": ["关键词1", "关键词2", ...],
    "audience": {{
        "age_range": "25-45",
        "gender": "偏女性/偏男性/均衡",
        "interests": ["兴趣1", "兴趣2"],
        "buying_motivation": "主要购买动机",
        "price_sensitivity": "低/中/高"
    }},
    "competition_level": "低/中/高",
    "seasonal_trends": ["趋势1", "趋势2"],
    "recommended_styles": ["风格1", "风格2"],
    "design_prompts": [
        {{
            "title": "设计标题1",
            "description": "详细的DALL-E提示词描述，包含风格、元素、颜色等",
            "mood": "情感基调",
            "keywords": ["相关关键词"]
        }},
        ...
    ]
}}

确保design_prompts包含{num_designs}个设计建议。
只返回JSON，不要添加任何其他文字。"""

    def _parse_response(self, response: str, niche: str, style: str) -> tuple:
        """解析LLM响应"""
        try:
            # 清理响应（移除可能的markdown代码块标记）
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()
            
            data = json.loads(clean_response)
            
            # 构建TrendData
            trend_data: TrendData = {
                "sub_topics": data.get("sub_topics", []),
                "keywords": data.get("keywords", []),
                "audience": data.get("audience", {}),
                "competition_level": data.get("competition_level", "medium"),
                "seasonal_trends": data.get("seasonal_trends", []),
                "recommended_styles": data.get("recommended_styles", [style]),
                "analyzed_at": datetime.now().isoformat()
            }
            
            # 提取设计提示词
            design_prompts = []
            for prompt_data in data.get("design_prompts", []):
                # 构建完整的DALL-E提示词
                full_prompt = self._build_dalle_prompt(prompt_data, style, niche)
                design_prompts.append(full_prompt)
            
            return trend_data, design_prompts
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # 返回默认值
            return self._get_default_trend_data(niche, style), self._get_default_prompts(niche, style)
    
    def _build_dalle_prompt(self, prompt_data: Dict, style: str, niche: str) -> str:
        """构建DALL-E提示词"""
        title = prompt_data.get("title", "")
        description = prompt_data.get("description", "")
        mood = prompt_data.get("mood", "")
        
        return f"""{title}: {description}. Style: {style}, {mood}. 
Perfect for print-on-demand products. High quality, clean design, 
suitable for {niche} audience. No text in the design."""
    
    def _get_default_trend_data(self, niche: str, style: str) -> TrendData:
        """返回默认趋势数据"""
        return {
            "sub_topics": [f"Popular {niche} themes"],
            "keywords": [niche, style, f"{niche} gift", f"{niche} lover"],
            "audience": {
                "age_range": "25-45",
                "gender": "均衡",
                "interests": [niche],
                "buying_motivation": "自用或送礼",
                "price_sensitivity": "中"
            },
            "competition_level": "medium",
            "seasonal_trends": [],
            "recommended_styles": [style],
            "analyzed_at": datetime.now().isoformat()
        }
    
    def _get_default_prompts(self, niche: str, style: str) -> list:
        """返回默认设计提示词"""
        return [
            f"A {style} illustration related to {niche}, clean design for print-on-demand"
        ]


# 创建节点函数的便捷方法
def create_trend_analysis_node(config: Dict[str, Any] = None):
    """创建趋势分析节点"""
    agent = TrendAnalysisAgent(config=config)
    
    async def node(state: PODState) -> Dict:
        return await agent(state)
    
    return node
