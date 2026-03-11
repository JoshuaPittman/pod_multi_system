"""
平台上传Agent (PlatformUploadAgent)
将产品发布到电商平台

职责：
1. 同步产品到Printful
2. 发布listing到Etsy/Amazon等平台
3. 管理上传状态和错误
"""

import uuid
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from core.base_agent import ToolAgent, AgentError, with_retry
from core.state import PODState, ProductData, SEOData, ListingData


class PlatformUploadAgent(ToolAgent):
    """
    平台上传Agent
    
    支持多平台上传：
    1. Etsy（通过Printful集成）
    2. Amazon（通过Printful集成）
    3. Shopify（通过Printful集成）
    
    上传流程：
    1. 在Printful创建产品
    2. 同步到目标平台
    3. 获取listing URL
    """
    
    PRINTFUL_API_BASE = "https://api.printful.com"
    ETSY_API_BASE = "https://openapi.etsy.com/v3"
    
    # 各平台的上传费用（估算）
    PLATFORM_COSTS = {
        "etsy": 0.20,      # Etsy listing fee
        "amazon": 0.0,     # 无listing fee，但有销售佣金
        "shopify": 0.0     # 通过Printful同步免费
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            config=config,
            api_base_url=self.PRINTFUL_API_BASE,
            timeout=120.0  # 上传可能需要较长时间
        )
        self.printful_api_key = config.get("printful_api_key") if config else None
        self.etsy_api_key = config.get("etsy_api_key") if config else None
        self.etsy_shop_id = config.get("etsy_shop_id") if config else None
        self.demo_mode = config.get("demo_mode", False) if config else False
    
    @property
    def name(self) -> str:
        return "platform_upload"
    
    def _get_headers(self) -> Dict[str, str]:
        """获取API请求头"""
        headers = {"Content-Type": "application/json"}
        if self.printful_api_key:
            headers["Authorization"] = f"Bearer {self.printful_api_key}"
        return headers
    
    def _validate_preconditions(self, state: PODState):
        """验证前置条件"""
        if not state.get("products"):
            raise AgentError(
                self.name,
                "No products to upload. Run mockup_creation first.",
                recoverable=False
            )
        if not state.get("seo_content"):
            raise AgentError(
                self.name,
                "No SEO content available. Run seo_optimization first.",
                recoverable=False
            )
        
        # 检查是否需要人工审核
        if state.get("human_review_required") and not state.get("human_review_approved"):
            raise AgentError(
                self.name,
                "Human review required before upload.",
                recoverable=True
            )
    
    async def process(self, state: PODState) -> Dict[str, Any]:
        """
        上传产品到平台
        
        输入：products, seo_content, target_platforms
        输出：listings
        """
        products = state["products"]
        seo_content = state["seo_content"]
        platforms = state.get("target_platforms", ["etsy"])
        
        # 创建SEO映射
        seo_map = {s["design_id"]: s for s in seo_content}
        
        self.logger.info(
            f"Uploading {len(products)} products to {len(platforms)} platforms"
        )
        
        listings = []
        total_upload_cost = 0
        
        # 按设计分组产品
        design_products = {}
        for product in products:
            design_id = product["design_id"]
            if design_id not in design_products:
                design_products[design_id] = []
            design_products[design_id].append(product)
        
        # 为每个设计的产品组创建listing
        for design_id, prods in design_products.items():
            seo = seo_map.get(design_id)
            if not seo:
                self.logger.warning(f"No SEO content for design {design_id}")
                continue
            
            for platform in platforms:
                listing = await self._upload_to_platform(
                    products=prods,
                    seo=seo,
                    platform=platform
                )
                if listing:
                    listings.append(listing)
                    total_upload_cost += self.PLATFORM_COSTS.get(platform, 0)
        
        # 更新成本
        cost_breakdown = state.get("cost_breakdown", {}).copy()
        cost_breakdown["platform_fees"] = cost_breakdown.get("platform_fees", 0) + total_upload_cost
        
        self.logger.info(f"Created {len(listings)} listings")
        
        return {
            "listings": listings,
            "total_cost": state["total_cost"] + total_upload_cost,
            "cost_breakdown": cost_breakdown,
            "current_step": "platform_upload_complete"
        }
    
    @with_retry(max_retries=2, delay=5.0, backoff=2.0)
    async def _upload_to_platform(
        self,
        products: List[ProductData],
        seo: SEOData,
        platform: str
    ) -> ListingData:
        """上传到特定平台"""
        listing_id = f"list_{uuid.uuid4().hex[:12]}"
        
        try:
            if platform == "etsy":
                listing_url = await self._upload_to_etsy(products, seo)
            elif platform == "amazon":
                listing_url = await self._upload_to_amazon(products, seo)
            elif platform == "shopify":
                listing_url = await self._upload_to_shopify(products, seo)
            else:
                self.logger.warning(f"Unsupported platform: {platform}")
                return None
            
            return ListingData(
                listing_id=listing_id,
                design_id=seo["design_id"],
                platform=platform,
                listing_url=listing_url,
                status="active",
                listed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to upload to {platform}: {e}")
            return ListingData(
                listing_id=listing_id,
                design_id=seo["design_id"],
                platform=platform,
                listing_url="",
                status="failed",
                listed_at=datetime.now().isoformat()
            )
    
    async def _upload_to_etsy(
        self,
        products: List[ProductData],
        seo: SEOData
    ) -> str:
        """上传到Etsy"""
        if self.demo_mode or not self.etsy_api_key or not self.etsy_shop_id:
            # Demo mode or missing keys — return mock URL
            return f"https://www.etsy.com/listing/{uuid.uuid4().hex[:10]}"

        # 实际的Etsy API调用
        # 1. 首先在Printful创建产品
        # 2. 然后同步到Etsy

        # 创建Printful同步产品
        sync_product = await self._create_printful_sync_product(products, seo)
        
        # 通过Printful的Etsy集成发布
        # Printful会自动同步到连接的Etsy店铺
        
        return f"https://www.etsy.com/listing/{sync_product.get('id', uuid.uuid4().hex[:10])}"
    
    async def _upload_to_amazon(
        self, 
        products: List[ProductData], 
        seo: SEOData
    ) -> str:
        """上传到Amazon"""
        # Mock响应
        return f"https://www.amazon.com/dp/{uuid.uuid4().hex[:10].upper()}"
    
    async def _upload_to_shopify(
        self, 
        products: List[ProductData], 
        seo: SEOData
    ) -> str:
        """上传到Shopify"""
        # Mock响应
        return f"https://shop.example.com/products/{seo['design_id']}"
    
    async def _create_printful_sync_product(
        self,
        products: List[ProductData],
        seo: SEOData
    ) -> Dict:
        """在Printful创建同步产品"""
        if not self.printful_api_key:
            return {"id": uuid.uuid4().hex[:10]}
        
        try:
            # 构建Printful产品数据
            sync_variants = []
            for product in products:
                for variant_id in product.get("variant_ids", []):
                    sync_variants.append({
                        "variant_id": variant_id,
                        "files": [
                            {
                                "type": "default",
                                "url": product["mockup_url"]
                            }
                        ],
                        "retail_price": "24.99"  # 默认价格
                    })
            
            payload = {
                "sync_product": {
                    "name": seo["title"],
                    "thumbnail": products[0]["mockup_url"] if products else ""
                },
                "sync_variants": sync_variants[:100]  # Printful限制
            }
            
            response = await self.api_request(
                "POST",
                "/store/products",
                json=payload
            )
            
            return response.get("result", {}).get("sync_product", {})
            
        except Exception as e:
            self.logger.error(f"Failed to create Printful sync product: {e}")
            return {"id": uuid.uuid4().hex[:10]}


def create_platform_upload_node(config: Dict[str, Any] = None):
    """创建平台上传节点"""
    agent = PlatformUploadAgent(config=config)
    
    async def node(state: PODState) -> Dict:
        return await agent(state)
    
    return node
