"""
产品合成Agent (MockupCreationAgent)
使用Printful API创建产品Mockup

职责：
1. 调用Printful API创建产品Mockup
2. 为不同产品类型生成预览图
3. 管理产品变体（颜色、尺寸）
"""

import uuid
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from core.base_agent import ToolAgent, AgentError, with_retry
from core.state import PODState, DesignData, ProductData


class MockupCreationAgent(ToolAgent):
    """
    产品合成Agent
    
    使用Printful API创建产品Mockup
    支持多种产品类型：t-shirt, mug, poster等
    """
    
    PRINTFUL_API_BASE = "https://api.printful.com"
    MOCKUP_COST_PER_PRODUCT = 0.0  # Printful mockup免费
    
    # 产品模板ID映射（Printful产品ID）
    PRODUCT_TEMPLATES = {
        "t-shirt": {
            "id": 71,  # Unisex Staple T-Shirt
            "name": "Unisex Staple T-Shirt",
            "placement": "front"
        },
        "mug": {
            "id": 19,  # White Glossy Mug
            "name": "White Glossy Mug",
            "placement": "default"
        },
        "poster": {
            "id": 1,  # Enhanced Matte Paper Poster
            "name": "Enhanced Matte Paper Poster",
            "placement": "default"
        },
        "hoodie": {
            "id": 146,  # Unisex Heavy Blend Hoodie
            "name": "Unisex Heavy Blend Hoodie",
            "placement": "front"
        },
        "tote-bag": {
            "id": 83,  # Cotton Tote Bag
            "name": "Cotton Tote Bag",
            "placement": "front"
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            config=config,
            api_base_url=self.PRINTFUL_API_BASE,
            timeout=60.0
        )
        self.api_key = config.get("printful_api_key") if config else None
        self.demo_mode = config.get("demo_mode", False) if config else False
    
    @property
    def name(self) -> str:
        return "mockup_creation"
    
    def _get_headers(self) -> Dict[str, str]:
        """获取Printful API请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _validate_preconditions(self, state: PODState):
        """验证前置条件"""
        designs = state.get("designs", [])
        
        # 过滤出已通过质量检查的设计（quality_score 不为 None 且 >= 0.8）
        # 由于 operator.add 会导致重复，按 design_id 去重，只取最新的（有分数的版本）
        design_dict = {}
        for d in designs:
            design_id = d.get("design_id")
            if design_id and d.get("quality_score") is not None:
                design_dict[design_id] = d
        
        passed_designs = [d for d in design_dict.values() if d.get("quality_score", 0) >= 0.8]
        
        if not passed_designs:
            raise AgentError(
                self.name,
                "No designs passed quality check.",
                recoverable=False
            )
    
    async def process(self, state: PODState) -> Dict[str, Any]:
        """
        创建产品Mockup
        
        输入：designs（通过质量检查的）, product_types
        输出：products
        """
        designs = state["designs"]
        product_types = state.get("product_types", [])
        
        # 处理空的 product_types
        if not product_types or product_types == "" or (isinstance(product_types, list) and len(product_types) == 0):
            product_types = ["t-shirt"]  # 默认产品类型
        
        # 按 design_id 去重，只取有 quality_score 的版本（处理 operator.add 导致的重复）
        design_dict = {}
        for d in designs:
            design_id = d.get("design_id")
            if design_id and d.get("quality_score") is not None:
                design_dict[design_id] = d
        
        # 筛选通过质量检查的设计（quality_score >= 0.8）
        passed_designs = [d for d in design_dict.values() if d.get("quality_score", 0) >= 0.8]
        
        self.logger.info(
            f"Creating mockups for {len(passed_designs)} designs, "
            f"{len(product_types)} product types each"
        )
        
        # 为每个设计创建所有产品类型的Mockup
        products = []
        for design in passed_designs:
            for product_type in product_types:
                product = await self._create_mockup(design, product_type)
                if product:
                    products.append(product)
                if not self.demo_mode:
                    await asyncio.sleep(1.5)  # avoid Printful 429
        
        self.logger.info(f"Created {len(products)} product mockups")
        
        return {
            "products": products,
            "current_step": "mockup_creation_complete"
        }
    
    @with_retry(max_retries=3, delay=2.0, backoff=2.0)
    async def _create_mockup(
        self, 
        design: DesignData, 
        product_type: str
    ) -> ProductData:
        """为单个设计创建特定产品类型的Mockup"""
        product_id = f"prod_{uuid.uuid4().hex[:12]}"
        
        template = self.PRODUCT_TEMPLATES.get(product_type)
        if not template:
            self.logger.warning(f"Unknown product type: {product_type}")
            return None
        
        # 调用Printful Mockup Generator API
        mockup_url = await self._call_printful_mockup_api(
            design["image_url"],
            template
        )
        
        return ProductData(
            product_id=product_id,
            design_id=design["design_id"],
            mockup_url=mockup_url,
            product_type=product_type,
            variant_ids=self._get_variant_ids(product_type),
            printful_sync_id=None,  # 将在上传后填充
            created_at=datetime.now().isoformat()
        )
    
    async def _call_printful_mockup_api(
        self, 
        image_url: str, 
        template: Dict
    ) -> str:
        """
        调用Printful Mockup Generator API
        如果没有 API Key，则使用本地 Pillow 生成 Mockup
        """
        # Demo mode: skip Printful entirely, use design image as mockup
        if self.demo_mode:
            return image_url

        # 优先使用 Printful API（生产环境）
        if self.api_key:
            try:
                return await self._call_printful_api_internal(image_url, template)
            except Exception as e:
                self.logger.warning(f"Printful API failed, falling back to local: {e}")

        # 后备：使用本地 Pillow 生成 Mockup
        return await self._generate_local_mockup(image_url, template)
    
    async def _generate_local_mockup(self, image_url: str, template: Dict) -> str:
        """使用 Pillow 本地生成 Mockup"""
        try:
            from utils.local_mockup import LocalMockupGenerator
            
            # 根据 template 确定产品类型
            product_type = self._get_product_type_from_template(template)
            
            # 使用 /app/static/mockups 目录，这样可以通过 FastAPI 静态文件服务访问
            generator = LocalMockupGenerator(output_dir="/app/static/mockups")
            mockup_path = await generator.generate_mockup(
                design_image_path=image_url,
                product_type=product_type
            )
            
            # 返回本地文件路径（可通过 API 访问）
            # 在生产环境中应该上传到 CDN
            return f"/static/mockups/{Path(mockup_path).name}"
            
        except Exception as e:
            self.logger.error(f"Local mockup generation failed: {e}")
            # 最终后备：返回占位 URL
            return f"/static/placeholder_mockup_{template.get('id', 0)}.png"
    
    def _get_product_type_from_template(self, template: Dict) -> str:
        """从模板配置获取产品类型"""
        template_id = template.get("id")
        for product_type, config in self.PRODUCT_TEMPLATES.items():
            if config.get("id") == template_id:
                return product_type
        return "poster"  # 默认
    
    async def _call_printful_api_internal(
        self, 
        image_url: str, 
        template: Dict
    ) -> str:
        """Printful API 内部调用（保留用于生产环境）"""
        try:
            # Printful Mockup Generator API调用
            payload = {
                "variant_ids": [template["id"]],
                "format": "png",
                "files": [
                    {
                        "placement": template["placement"],
                        "image_url": image_url,
                        "position": {
                            "area_width": 1800,
                            "area_height": 2400,
                            "width": 1800,
                            "height": 1800,
                            "top": 300,
                            "left": 0
                        }
                    }
                ]
            }
            
            response = await self.api_request(
                "POST",
                "/mockup-generator/create-task/71",  # T-shirt模板
                json=payload
            )
            
            # 等待mockup生成完成
            task_key = response.get("result", {}).get("task_key")
            if task_key:
                mockup_result = await self._poll_mockup_task(task_key)
                return mockup_result
            
            return f"https://example.com/mockup_{uuid.uuid4().hex[:8]}.png"
            
        except Exception as e:
            self.logger.error(f"Printful API error: {e}")
            # 返回占位URL
            return f"https://example.com/mockup_{uuid.uuid4().hex[:8]}.png"
    
    async def _poll_mockup_task(self, task_key: str, max_attempts: int = 10) -> str:
        """轮询Mockup生成任务状态"""
        for _ in range(max_attempts):
            try:
                response = await self.api_request(
                    "GET",
                    f"/mockup-generator/task?task_key={task_key}"
                )
                
                status = response.get("result", {}).get("status")
                
                if status == "completed":
                    mockups = response.get("result", {}).get("mockups", [])
                    if mockups:
                        return mockups[0].get("mockup_url", "")
                elif status == "failed":
                    raise AgentError(self.name, "Mockup generation failed")
                
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error polling mockup task: {e}")
        
        raise AgentError(self.name, "Mockup generation timeout")
    
    def _get_variant_ids(self, product_type: str) -> List[str]:
        """获取产品变体ID（颜色、尺寸组合）"""
        # 简化版：返回默认变体
        # 实际应该根据产品类型返回完整的变体列表
        variants = {
            "t-shirt": ["S-Black", "M-Black", "L-Black", "S-White", "M-White", "L-White"],
            "mug": ["11oz", "15oz"],
            "poster": ["8x10", "12x18", "18x24"],
            "hoodie": ["S-Black", "M-Black", "L-Black"],
            "tote-bag": ["Default"]
        }
        return variants.get(product_type, ["Default"])


def create_mockup_creation_node(config: Dict[str, Any] = None):
    """创建产品合成节点"""
    agent = MockupCreationAgent(config=config)
    
    async def node(state: PODState) -> Dict:
        return await agent(state)
    
    return node
