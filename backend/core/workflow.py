"""
POD多智能体系统 - 工作流编排
使用LangGraph StateGraph实现多Agent协作

核心特性：
1. StateGraph图结构定义工作流
2. 条件边实现智能路由
3. 质量检查循环（带retry_count防护）
4. Human-in-the-loop支持
5. Checkpoint持久化（支持断点续传）
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: langgraph not installed. Using mock implementation.")

from core.state import PODState, QualityResult, WorkflowStatus, create_initial_state

from agents import (
    create_trend_analysis_node,
    create_design_generation_node,
    create_quality_check_node,
    create_mockup_creation_node,
    create_seo_optimization_node,
    create_platform_upload_node,
    create_optimization_node,
    route_quality_check
)

logger = logging.getLogger(__name__)


class PODWorkflowBuilder:
    """
    POD工作流构建器
    
    负责：
    1. 创建和配置StateGraph
    2. 添加节点和边
    3. 配置Checkpoint持久化
    4. 编译生成可执行的工作流
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化工作流构建器
        
        Args:
            config: 配置字典，包含API密钥等
        """
        self.config = config or {}
        self.workflow = None
        self.app = None
        
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, workflow will use mock implementation")
    
    def build(
        self,
        include_optimization: bool = True,
        human_review_before_upload: bool = False,
        checkpointer: Optional[Any] = None
    ) -> "PODWorkflowBuilder":
        """
        构建工作流
        
        Args:
            include_optimization: 是否包含优化分析节点
            human_review_before_upload: 是否在上传前需要人工审核
            checkpointer: 自定义Checkpointer（默认使用MemorySaver）
        
        Returns:
            self，支持链式调用
        """
        if not LANGGRAPH_AVAILABLE:
            logger.warning("Building mock workflow")
            return self
        
        # 创建StateGraph
        self.workflow = StateGraph(PODState)
        
        # 添加节点
        self._add_nodes(include_optimization, human_review_before_upload)
        
        # 添加边
        self._add_edges(include_optimization, human_review_before_upload)
        
        # 编译工作流
        self._compile(checkpointer, human_review=human_review_before_upload)
        
        logger.info("Workflow built successfully")
        return self
    
    def _add_nodes(
        self, 
        include_optimization: bool,
        human_review: bool
    ):
        """添加工作流节点"""
        config = self.config
        
        # 1. 趋势分析节点
        self.workflow.add_node(
            "trend_analysis", 
            create_trend_analysis_node(config)
        )
        
        # 2. 设计生成节点
        self.workflow.add_node(
            "design_generation",
            create_design_generation_node(config)
        )
        
        # 3. 质量检查节点
        self.workflow.add_node(
            "quality_check",
            create_quality_check_node(config)
        )
        
        # 4. 产品合成节点
        self.workflow.add_node(
            "mockup_creation",
            create_mockup_creation_node(config)
        )
        
        # 5. SEO优化节点
        self.workflow.add_node(
            "seo_optimization",
            create_seo_optimization_node(config)
        )
        
        # 6. 人工审核节点（可选）
        if human_review:
            self.workflow.add_node(
                "human_review",
                self._create_human_review_node()
            )
        
        # 7. 平台上传节点
        self.workflow.add_node(
            "platform_upload",
            create_platform_upload_node(config)
        )
        
        # 8. 优化分析节点（可选）
        if include_optimization:
            self.workflow.add_node(
                "optimization",
                create_optimization_node(config)
            )
        
        logger.info(f"Added {7 if not include_optimization else 8} nodes to workflow")
    
    def _add_edges(
        self,
        include_optimization: bool,
        human_review: bool
    ):
        """添加工作流边"""
        # 起点 -> 趋势分析
        self.workflow.add_edge(START, "trend_analysis")
        
        # 趋势分析 -> 设计生成
        self.workflow.add_edge("trend_analysis", "design_generation")
        
        # 设计生成 -> 质量检查
        self.workflow.add_edge("design_generation", "quality_check")
        
        # 质量检查 -> 条件路由（核心：循环重试机制）
        self.workflow.add_conditional_edges(
            "quality_check",
            route_quality_check,
            {
                "pass": "mockup_creation",      # 通过 -> 产品合成
                "retry": "design_generation",   # 重试 -> 重新生成设计
                "fail": END                      # 失败 -> 结束
            }
        )
        
        # 产品合成 -> SEO优化
        self.workflow.add_edge("mockup_creation", "seo_optimization")
        
        # SEO优化后的路由
        if human_review:
            # SEO优化 -> 人工审核
            self.workflow.add_edge("seo_optimization", "human_review")
            # 人工审核 -> 条件路由
            self.workflow.add_conditional_edges(
                "human_review",
                self._route_human_review,
                {
                    "approved": "platform_upload",
                    "rejected": END
                }
            )
        else:
            # SEO优化 -> 直接上传
            self.workflow.add_edge("seo_optimization", "platform_upload")
        
        # 平台上传后的路由
        if include_optimization:
            self.workflow.add_edge("platform_upload", "optimization")
            self.workflow.add_edge("optimization", END)
        else:
            self.workflow.add_edge("platform_upload", END)
        
        logger.info("Added edges to workflow")

    def _compile(self, checkpointer: Optional[Any] = None, human_review: bool = False):
        """编译工作流"""
        if checkpointer is None:
            # 1. 尝试从配置中获取数据库 URL
            database_url = self.config.get("database_url")

            if database_url:
                try:
                    # 2. 引入 PostgresSaver (需确保安装了 langgraph-checkpoint-postgres)
                    from langgraph.checkpoint.postgres import PostgresSaver

                    # 3. 建立连接池
                    checkpointer = PostgresSaver.from_conn_string(database_url)

                    # 4. 关键步骤：初始化表结构 (首次运行时必须)
                    checkpointer.setup()

                    logger.info(f"Using PostgresSaver with DB: {database_url.split('@')[-1]}")  # 记录数据库名，隐去密码
                except Exception as e:
                    # 捕获所有异常（包括 ImportError 和连接错误）
                    logger.warning(f"Failed to initialize PostgresSaver: {e}. Falling back to MemorySaver.")
                    checkpointer = MemorySaver()
            else:
                checkpointer = MemorySaver()
                logger.info("No database_url found. Using MemorySaver (non-persistent).")

        # 编译 Graph - 只有在启用人工审核时才设置 interrupt_before
        compile_kwargs = {"checkpointer": checkpointer}
        if human_review:
            compile_kwargs["interrupt_before"] = ["human_review"]
        
        self.app = self.workflow.compile(**compile_kwargs)
        
        logger.info("Workflow compiled with checkpointer")
    
    def _create_human_review_node(self) -> Callable:
        """创建人工审核节点"""
        def human_review_node(state: PODState) -> Dict:
            """
            人工审核节点
            
            这个节点会在interrupt_before时暂停
            等待用户通过update_state更新审核结果
            """
            return {
                "human_review_required": True,
                "current_step": "awaiting_human_review",
                "updated_at": datetime.now().isoformat()
            }
        
        return human_review_node
    
    def _route_human_review(self, state: PODState) -> str:
        """人工审核路由函数"""
        if state.get("human_review_approved"):
            return "approved"
        return "rejected"
    
    def get_app(self):
        """获取编译后的应用"""
        return self.app
    
    def get_graph_visualization(self) -> str:
        """获取工作流图的可视化表示"""
        if self.app is None:
            return "Workflow not built yet"
        
        try:
            return self.app.get_graph().draw_mermaid()
        except Exception:
            return "Visualization not available"


class MockWorkflowRunner:
    """
    Mock工作流运行器（无LangGraph依赖）
    用于演示和测试
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def run(
        self,
        niche: str,
        style: str,
        num_designs: int = 5,
        target_platforms: Optional[list] = None,
        product_types: Optional[list] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """模拟运行工作流"""
        import uuid
        from datetime import datetime
        
        logger.info(f"[Mock] Running workflow for: {niche}")
        
        # 创建模拟状态
        state = create_initial_state(
            niche=niche,
            style=style,
            num_designs=num_designs,
            target_platforms=target_platforms,
            product_types=product_types,
            thread_id=thread_id
        )
        
        # 模拟各Agent执行
        steps = [
            ("trend_analysis", self._mock_trend_analysis),
            ("design_generation", self._mock_design_generation),
            ("quality_check", self._mock_quality_check),
            ("mockup_creation", self._mock_mockup_creation),
            ("seo_optimization", self._mock_seo_optimization),
            ("platform_upload", self._mock_platform_upload),
            ("optimization", self._mock_optimization),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"[Mock] Executing: {step_name}")
            updates = step_func(state)
            state.update(updates)
        
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        
        return state
    
    def _mock_trend_analysis(self, state: Dict) -> Dict:
        return {
            "trend_data": {
                "sub_topics": ["cute cats", "funny cats", "cat mom"],
                "keywords": ["cat lover", "cat mom", "kitty", "meow", "purrfect"],
                "audience": {"age_range": "25-45", "gender": "偏女性"},
                "competition_level": "medium",
                "seasonal_trends": [],
                "recommended_styles": [state["style"]],
                "analyzed_at": datetime.now().isoformat()
            },
            "design_prompts": [
                f"A {state['style']} cat illustration, cute and clean design",
                f"Cat silhouette in {state['style']} style, perfect for t-shirt",
                f"Funny cat face, {state['style']} art style",
            ][:state["num_designs"]],
            "current_step": "trend_analysis_complete"
        }
    
    def _mock_design_generation(self, state: Dict) -> Dict:
        import uuid
        designs = []
        for i, prompt in enumerate(state.get("design_prompts", [])):
            designs.append({
                "design_id": f"design_{uuid.uuid4().hex[:8]}",
                "prompt": prompt,
                "image_url": f"https://example.com/design_{i}.png",
                "style": state["style"],
                "keywords": ["cat", "cute", state["niche"]],
                "created_at": datetime.now().isoformat(),
                "quality_score": None,
                "quality_issues": None
            })
        return {
            "designs": designs,
            "total_cost": state["total_cost"] + len(designs) * 0.04,
            "current_step": "design_generation_complete"
        }
    
    def _mock_quality_check(self, state: Dict) -> Dict:
        import random
        designs = state.get("designs", [])
        for design in designs:
            design["quality_score"] = random.uniform(0.8, 0.98)
            design["quality_issues"] = []
        return {
            "designs": designs,
            "quality_check_result": "pass",
            "current_step": "quality_check_complete"
        }
    
    def _mock_mockup_creation(self, state: Dict) -> Dict:
        import uuid
        products = []
        for design in state.get("designs", []):
            for product_type in state.get("product_types", ["t-shirt"]):
                products.append({
                    "product_id": f"prod_{uuid.uuid4().hex[:8]}",
                    "design_id": design["design_id"],
                    "mockup_url": f"https://example.com/mockup_{product_type}.png",
                    "product_type": product_type,
                    "variant_ids": ["S", "M", "L"],
                    "printful_sync_id": None,
                    "created_at": datetime.now().isoformat()
                })
        return {
            "products": products,
            "current_step": "mockup_creation_complete"
        }
    
    def _mock_seo_optimization(self, state: Dict) -> Dict:
        seo_content = []
        for design in state.get("designs", []):
            seo_content.append({
                "design_id": design["design_id"],
                "title": f"Cute Cat Lover Gift - {state['style'].title()} Design T-Shirt",
                "description": "Perfect gift for cat lovers! High quality print.",
                "tags": ["cat lover", "cat mom", "cat gift", "funny cat"],
                "keywords": ["cat", "gift", "lover"],
                "optimized_at": datetime.now().isoformat()
            })
        return {
            "seo_content": seo_content,
            "total_cost": state["total_cost"] + 0.02,
            "current_step": "seo_optimization_complete"
        }
    
    def _mock_platform_upload(self, state: Dict) -> Dict:
        import uuid
        listings = []
        for seo in state.get("seo_content", []):
            for platform in state.get("target_platforms", ["etsy"]):
                listings.append({
                    "listing_id": f"list_{uuid.uuid4().hex[:8]}",
                    "design_id": seo["design_id"],
                    "platform": platform,
                    "listing_url": f"https://www.{platform}.com/listing/{uuid.uuid4().hex[:10]}",
                    "status": "active",
                    "listed_at": datetime.now().isoformat()
                })
        return {
            "listings": listings,
            "current_step": "platform_upload_complete"
        }
    
    def _mock_optimization(self, state: Dict) -> Dict:
        return {
            "sales_data": [],
            "optimization_recommendations": {
                "design_optimization": ["尝试更多颜色变体", "添加文字元素"],
                "seo_optimization": ["更新季节性关键词", "增加长尾关键词"],
                "pricing_strategy": ["考虑促销定价"],
                "new_product_ideas": ["扩展到手机壳产品"],
                "priority_actions": ["分析最佳表现设计", "优化标签策略", "测试新风格"]
            },
            "current_step": "optimization_complete"
        }
    
    def get_state(self, thread_id: str) -> Optional[Dict]:
        return None
    
    def resume(self, thread_id: str, updates: Dict = None) -> Dict:
        return {}


class PODWorkflowRunner:
    """
    POD工作流运行器
    
    负责：
    1. 运行工作流
    2. 处理中断和恢复
    3. 获取执行状态
    """
    
    def __init__(self, app: Any):
        """
        初始化运行器
        
        Args:
            app: 编译后的LangGraph应用
        """
        self.app = app
    
    async def run(
        self,
        niche: str,
        style: str,
        num_designs: int = 5,
        target_platforms: Optional[list] = None,
        product_types: Optional[list] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行工作流

        Args:
            niche: 利基市场
            style: 设计风格
            num_designs: 设计数量
            target_platforms: 目标平台
            product_types: 产品类型
            thread_id: 线程ID（用于恢复）

        Returns:
            最终状态
        """
        # 创建初始状态
        initial_state = create_initial_state(
            niche=niche,
            style=style,
            num_designs=num_designs,
            target_platforms=target_platforms,
            product_types=product_types,
            thread_id=thread_id
        )

        # 配置
        config = {"configurable": {"thread_id": initial_state["thread_id"]}}

        logger.info(f"Starting workflow for niche: {niche}, thread: {initial_state['thread_id']}")

        # 运行工作流
        try:
            final_state = None
            async for event in self.app.astream(initial_state, config):
                # 处理每个事件
                for node_name, node_output in event.items():
                    logger.info(f"Completed node: {node_name}")

            # 获取完整的累积状态（astream 只返回每个节点的增量更新）
            state_snapshot = self.app.get_state(config)
            if state_snapshot and state_snapshot.values:
                final_state = dict(state_snapshot.values)
                logger.info(f"Final state has {len(final_state.get('designs', []))} designs, "
                           f"{len(final_state.get('products', []))} products")

            return final_state

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def resume(
        self,
        thread_id: str,
        updates: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        恢复中断的工作流
        
        Args:
            thread_id: 线程ID
            updates: 状态更新（如人工审核结果）
        
        Returns:
            最终状态
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # 如果有状态更新，先应用
        if updates:
            self.app.update_state(config, updates)
            logger.info(f"Applied state updates: {list(updates.keys())}")
        
        # 继续执行
        logger.info(f"Resuming workflow for thread: {thread_id}")
        
        try:
            final_state = None
            for event in self.app.stream(None, config):
                for node_name, node_output in event.items():
                    logger.info(f"Completed node: {node_name}")
                    final_state = node_output
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow resume failed: {e}")
            raise
    
    def get_state(self, thread_id: str) -> Optional[PODState]:
        """获取工作流当前状态"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.app.get_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return None
    
    def get_history(self, thread_id: str) -> list:
        """获取工作流执行历史"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            history = list(self.app.get_state_history(config))
            return history
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []


def create_pod_workflow(
    config: Optional[Dict[str, Any]] = None,
    include_optimization: bool = True,
    human_review: bool = False,
    checkpointer: Optional[Any] = None
) -> PODWorkflowRunner:
    """
    创建POD工作流的便捷函数
    
    Args:
        config: 配置字典
        include_optimization: 是否包含优化节点
        human_review: 是否需要人工审核
        checkpointer: 自定义Checkpointer
    
    Returns:
        PODWorkflowRunner实例
    
    Usage:
        ```python
        runner = create_pod_workflow(config={
            "openai_api_key": "...",
            "anthropic_api_key": "...",
            "printful_api_key": "..."
        })
        
        result = runner.run(
            niche="cat lovers",
            style="minimalist",
            num_designs=5
        )
        ```
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available, using MockWorkflowRunner")
        return MockWorkflowRunner(config)
    
    builder = PODWorkflowBuilder(config)
    builder.build(
        include_optimization=include_optimization,
        human_review_before_upload=human_review,
        checkpointer=checkpointer
    )
    
    return PODWorkflowRunner(builder.get_app())


# 工作流图的Mermaid可视化
WORKFLOW_MERMAID = """
```mermaid
graph TD
    START((Start)) --> trend_analysis[趋势分析Agent]
    trend_analysis --> design_generation[设计生成Agent]
    design_generation --> quality_check{质量检查Agent}
    
    quality_check -->|Pass: score >= 0.8| mockup_creation[产品合成Agent]
    quality_check -->|Retry: score < 0.8 & retries < 3| design_generation
    quality_check -->|Fail: retries >= 3| END1((End))
    
    mockup_creation --> seo_optimization[SEO优化Agent]
    seo_optimization --> human_review{人工审核}
    
    human_review -->|Approved| platform_upload[平台上传Agent]
    human_review -->|Rejected| END2((End))
    
    platform_upload --> optimization[优化建议Agent]
    optimization --> END3((End))
    
    style quality_check fill:#ffcc00
    style human_review fill:#ff9999
    style START fill:#90EE90
    style END1 fill:#ffcccc
    style END2 fill:#ffcccc
    style END3 fill:#90EE90
```
"""
