"""
优化建议Agent (OptimizationAgent)
分析销售数据并提供优化建议

职责：
1. 收集和分析销售数据
2. 识别表现最佳/最差的设计
3. 提供SEO和设计优化建议
4. 推荐新的设计方向
"""

import json
from typing import Dict, Any, List
from datetime import datetime

from core.base_agent import LLMAgent, AgentError
from core.state import PODState, SalesMetrics


class OptimizationAgent(LLMAgent):
    """
    优化建议Agent
    
    使用Claude分析数据并提供优化建议
    这个Agent通常在工作流完成后定期运行
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Haiku is sufficient for recommendations — ~20x cheaper than Sonnet
        super().__init__(config=config, model="claude-haiku-4-5-20251001")

    @property
    def name(self) -> str:
        return "optimization"
    
    def _validate_preconditions(self, state: PODState):
        """验证前置条件"""
        if not state.get("listings"):
            raise AgentError(
                self.name,
                "No listings available. Run platform_upload first.",
                recoverable=False
            )
    
    async def process(self, state: PODState) -> Dict[str, Any]:
        """
        分析数据并生成优化建议
        
        输入：listings, sales_data, designs, seo_content
        输出：optimization_recommendations, sales_data更新
        """
        listings = state["listings"]
        designs = state["designs"]
        seo_content = state.get("seo_content", [])
        niche = state["niche"]
        
        self.logger.info("Analyzing performance and generating recommendations...")
        
        # 模拟获取销售数据（实际应从平台API获取）
        sales_data = await self._fetch_sales_data(listings)
        
        # 分析数据
        analysis = self._analyze_performance(sales_data, designs, seo_content)
        
        # 生成优化建议
        recommendations = await self._generate_recommendations(
            analysis=analysis,
            niche=niche,
            designs=designs,
            seo_content=seo_content
        )
        
        llm_cost = self.llm_cost  # real token cost from invoke_llm
        cost_breakdown = state.get("cost_breakdown", {}).copy()
        cost_breakdown["anthropic"] = cost_breakdown.get("anthropic", 0) + llm_cost
        
        self.logger.info("Optimization analysis complete")
        
        return {
            "sales_data": sales_data,
            "optimization_recommendations": recommendations,
            "total_cost": state["total_cost"] + llm_cost,
            "cost_breakdown": cost_breakdown,
            "current_step": "optimization_complete",
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        }
    
    async def _fetch_sales_data(self, listings: List[Dict]) -> List[SalesMetrics]:
        """
        获取销售数据
        
        在生产环境中，应该从各平台API获取实际数据
        这里返回模拟数据
        """
        import random
        
        sales_data = []
        for listing in listings:
            # 模拟数据
            views = random.randint(10, 500)
            favorites = random.randint(0, int(views * 0.3))
            sales = random.randint(0, int(views * 0.05))
            revenue = sales * random.uniform(15, 35)
            
            sales_data.append(SalesMetrics(
                design_id=listing["design_id"],
                views=views,
                favorites=favorites,
                sales=sales,
                revenue=round(revenue, 2),
                conversion_rate=round(sales / views * 100 if views > 0 else 0, 2),
                updated_at=datetime.now().isoformat()
            ))
        
        return sales_data
    
    def _analyze_performance(
        self,
        sales_data: List[SalesMetrics],
        designs: List[Dict],
        seo_content: List[Dict]
    ) -> Dict[str, Any]:
        """分析性能数据"""
        if not sales_data:
            return {
                "total_views": 0,
                "total_sales": 0,
                "total_revenue": 0,
                "avg_conversion_rate": 0,
                "best_performers": [],
                "worst_performers": []
            }
        
        # 计算汇总指标
        total_views = sum(s["views"] for s in sales_data)
        total_sales = sum(s["sales"] for s in sales_data)
        total_revenue = sum(s["revenue"] for s in sales_data)
        avg_conversion = sum(s["conversion_rate"] for s in sales_data) / len(sales_data)
        
        # 排序找出最佳和最差表现
        sorted_by_revenue = sorted(sales_data, key=lambda x: x["revenue"], reverse=True)
        sorted_by_conversion = sorted(sales_data, key=lambda x: x["conversion_rate"], reverse=True)
        
        return {
            "total_views": total_views,
            "total_sales": total_sales,
            "total_revenue": round(total_revenue, 2),
            "avg_conversion_rate": round(avg_conversion, 2),
            "best_by_revenue": sorted_by_revenue[:3],
            "best_by_conversion": sorted_by_conversion[:3],
            "worst_performers": sorted_by_revenue[-3:] if len(sorted_by_revenue) >= 3 else sorted_by_revenue,
            "design_count": len(sales_data)
        }
    
    async def _generate_recommendations(
        self,
        analysis: Dict,
        niche: str,
        designs: List[Dict],
        seo_content: List[Dict]
    ) -> Dict[str, List[str]]:
        """使用LLM生成优化建议"""
        
        # 构建分析摘要
        summary = self._build_analysis_summary(analysis, designs, seo_content)
        
        prompt = f"""作为POD电商优化专家，基于以下性能数据为"{niche}"利基市场提供优化建议。

{summary}

请从以下方面提供具体、可操作的建议：

1. **设计优化**
   - 基于最佳表现设计的特征，建议新的设计方向
   - 针对表现差的设计提出改进建议

2. **SEO优化**
   - 标题和描述改进建议
   - 标签优化策略
   - 关键词建议

3. **定价策略**
   - 基于转化率的定价建议
   - 促销策略建议

4. **新产品建议**
   - 基于市场表现推荐的新设计主题
   - 建议扩展的产品类型

请按以下JSON格式返回：
{{
    "design_optimization": ["建议1", "建议2", ...],
    "seo_optimization": ["建议1", "建议2", ...],
    "pricing_strategy": ["建议1", "建议2", ...],
    "new_product_ideas": ["建议1", "建议2", ...],
    "priority_actions": ["最重要的行动1", "最重要的行动2", "最重要的行动3"]
}}

只返回JSON，不要添加任何其他文字。"""

        response = await self.invoke_llm(prompt)
        
        try:
            # 清理和解析响应
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()
            
            return json.loads(clean_response)
            
        except json.JSONDecodeError:
            # 返回默认建议
            return {
                "design_optimization": ["继续监控设计表现", "测试不同风格变体"],
                "seo_optimization": ["更新标签以反映季节趋势", "优化标题关键词"],
                "pricing_strategy": ["考虑进行价格测试"],
                "new_product_ideas": ["扩展到相关细分市场"],
                "priority_actions": ["分析最佳表现设计的共同特征"]
            }
    
    def _build_analysis_summary(
        self,
        analysis: Dict,
        designs: List[Dict],
        seo_content: List[Dict]
    ) -> str:
        """构建分析摘要文本"""
        summary_lines = [
            f"总设计数量: {analysis.get('design_count', 0)}",
            f"总浏览量: {analysis.get('total_views', 0)}",
            f"总销售量: {analysis.get('total_sales', 0)}",
            f"总收入: ${analysis.get('total_revenue', 0):.2f}",
            f"平均转化率: {analysis.get('avg_conversion_rate', 0):.2f}%",
            "",
            "最佳表现设计（按收入）:",
        ]
        
        for item in analysis.get("best_by_revenue", [])[:3]:
            summary_lines.append(
                f"  - {item['design_id']}: "
                f"${item['revenue']:.2f}收入, "
                f"{item['conversion_rate']:.2f}%转化率"
            )
        
        summary_lines.append("")
        summary_lines.append("表现较差设计:")
        
        for item in analysis.get("worst_performers", [])[:3]:
            summary_lines.append(
                f"  - {item['design_id']}: "
                f"${item['revenue']:.2f}收入, "
                f"{item['views']}浏览量"
            )
        
        return "\n".join(summary_lines)


def create_optimization_node(config: Dict[str, Any] = None):
    """创建优化建议节点"""
    agent = OptimizationAgent(config=config)
    
    async def node(state: PODState) -> Dict:
        return await agent(state)
    
    return node
