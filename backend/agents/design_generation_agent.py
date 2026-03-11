"""
设计生成Agent (DesignGenerationAgent)
使用DALL-E 3生成POD产品设计图

职责：
1. 根据设计提示词生成高质量图片
2. 确保图片适合POD产品打印
3. 管理生成成本和质量
"""

import uuid
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from core.base_agent import BaseAgent, AgentError, with_retry
from core.state import PODState, DesignData


class DesignGenerationAgent(BaseAgent):
    """
    设计生成Agent
    
    使用DALL-E 3生成设计图，特点：
    1. 高质量图像生成
    2. 支持批量生成
    3. 成本追踪
    4. 并发控制
    """
    
    DALLE_COST_PER_IMAGE = 0.04  # DALL-E 3 1024x1024 价格
    MAX_CONCURRENT_REQUESTS = 3  # 并发限制
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._openai_client = None
        self.demo_mode = config.get("demo_mode", False) if config else False
    
    @property
    def name(self) -> str:
        return "design_generation"
    
    @property
    def openai_client(self):
        """延迟初始化OpenAI客户端（使用yunwu.ai中转）"""
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI
                
                # 优先使用yunwu API
                yunwu_key = self.config.get("yunwu_api_key")
                yunwu_base = self.config.get("yunwu_api_base", "https://yunwu.ai/v1")
                openai_key = self.config.get("openai_api_key")
                
                api_key = yunwu_key or openai_key
                if api_key:
                    self._openai_client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=yunwu_base if yunwu_key else None
                    )
                else:
                    self.logger.warning("No API key found, using mock client")
            except ImportError:
                self.logger.warning("openai not installed, using mock client")
        return self._openai_client
    
    def _validate_preconditions(self, state: PODState):
        """验证前置条件"""
        if not state.get("design_prompts"):
            raise AgentError(
                self.name, 
                "No design prompts available. Run trend_analysis first.",
                recoverable=False
            )
    
    async def process(self, state: PODState) -> Dict[str, Any]:
        """
        生成设计图
        
        输入：design_prompts, style, niche
        输出：designs, total_cost更新
        """
        prompts = state["design_prompts"]
        style = state["style"]
        niche = state["niche"]
        
        self.logger.info(f"Generating {len(prompts)} designs...")
        
        # 并发生成设计
        designs = await self._generate_designs_batch(prompts, style, niche)
        
        # 计算成本
        generation_cost = len(designs) * self.DALLE_COST_PER_IMAGE
        
        self.logger.info(f"Generated {len(designs)} designs, cost: ${generation_cost:.2f}")
        
        # 更新成本
        cost_breakdown = state.get("cost_breakdown", {}).copy()
        cost_breakdown["dalle"] = cost_breakdown.get("dalle", 0) + generation_cost
        
        return {
            "designs": designs,
            "total_cost": state["total_cost"] + generation_cost,
            "cost_breakdown": cost_breakdown,
            "current_step": "design_generation_complete"
        }
    
    async def _generate_designs_batch(
        self, 
        prompts: List[str], 
        style: str, 
        niche: str
    ) -> List[DesignData]:
        """批量生成设计，带并发控制"""
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        
        async def generate_with_semaphore(prompt: str, index: int) -> DesignData:
            async with semaphore:
                return await self._generate_single_design(prompt, style, niche, index)
        
        tasks = [
            generate_with_semaphore(prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉失败的结果
        designs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to generate design {i}: {result}")
            elif result is not None:
                designs.append(result)
        
        return designs
    
    @with_retry(max_retries=2, delay=2.0, backoff=2.0)
    async def _generate_single_design(
        self, 
        prompt: str, 
        style: str, 
        niche: str,
        index: int
    ) -> DesignData:
        """生成单个设计"""
        design_id = f"design_{uuid.uuid4().hex[:12]}"
        
        # 增强提示词，确保适合POD打印
        enhanced_prompt = self._enhance_prompt(prompt, style)
        
        # 调用DALL-E API 并保存到本地
        image_url = await self._call_dalle_api(enhanced_prompt, design_id=design_id)
        
        # 从趋势分析的prompt中提取关键词
        keywords = self._extract_keywords(prompt, niche)
        
        return DesignData(
            design_id=design_id,
            prompt=enhanced_prompt,
            image_url=image_url,
            style=style,
            keywords=keywords,
            created_at=datetime.now().isoformat(),
            quality_score=None,  # 将由质量检查Agent填充
            quality_issues=None
        )
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """增强提示词，确保生成适合POD的图像"""
        enhancement = """
High quality, detailed illustration. Clean design with transparent or solid 
background, suitable for print-on-demand products like t-shirts and mugs. 
No text or watermarks. Vector-style clean edges. 
Professional quality, ready for commercial use."""
        
        return f"{prompt} {enhancement}"
    
    async def _call_dalle_api(self, prompt: str, design_id: str = None) -> str:
        """调用图像生成 API (yunwu.ai) 并保存到本地
        
        API文档: https://yunwu.apifox.cn/api-290549047
        - gpt-image-1: 返回 b64_json (base64 编码的图片数据)
        - dall-e-3: 可能返回 url 或 b64_json
        
        Returns:
            本地图片路径 (如 /static/designs/design_xxx.png)
        """
        import os
        import base64
        
        # 确保 design_id 存在
        if not design_id:
            design_id = f"design_{uuid.uuid4().hex[:12]}"
        
        # 本地保存路径
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "designs")
        os.makedirs(save_dir, exist_ok=True)
        local_file = os.path.join(save_dir, f"{design_id}.png")
        local_url = f"/static/designs/{design_id}.png"
        
        if self.openai_client is None:
            # Mock响应 - 创建一个占位图片
            self.logger.warning(f"No API client, using mock image for {design_id}")
            return f"https://example.com/mock_image_{design_id}.png"
        
        image_model = os.getenv("IMAGE_MODEL", "gpt-image-1")
        image_size = os.getenv("IMAGE_SIZE", "1024x1024")
        
        try:
            # 构建基础参数
            params = {
                "model": image_model,
                "prompt": prompt,
                "size": image_size,
                "n": 1
            }
            
            # quality 参数仅 dall-e-3 支持
            if image_model == "dall-e-3":
                image_quality = os.getenv("IMAGE_QUALITY", "standard")
                params["quality"] = image_quality
            
            self.logger.info(f"Calling image API: model={image_model}, size={image_size}")
            
            # 调用 API 生成图片
            response = await self.openai_client.images.generate(**params)
            
            # 获取响应数据
            image_data = response.data[0]
            
            # 处理 base64 编码的图片 (gpt-image-1 返回 b64_json)
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                self.logger.info(f"Received base64 image data, decoding and saving to {local_file}")
                image_bytes = base64.b64decode(image_data.b64_json)
                with open(local_file, 'wb') as f:
                    f.write(image_bytes)
                self.logger.info(f"Image saved to {local_file} ({len(image_bytes)} bytes)")
                return await self._maybe_upload_r2(local_file, design_id, local_url)
            
            # 处理 URL 响应 (dall-e-3 可能返回 url)
            elif hasattr(image_data, 'url') and image_data.url:
                self.logger.info(f"Received image URL, downloading to {local_file}")
                import aiohttp
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_data.url) as resp:
                            if resp.status == 200:
                                with open(local_file, 'wb') as f:
                                    f.write(await resp.read())
                                self.logger.info(f"Image downloaded and saved to {local_file}")
                                return await self._maybe_upload_r2(local_file, design_id, local_url)
                            else:
                                self.logger.warning(f"Failed to download image: HTTP {resp.status}")
                                return image_data.url
                except Exception as download_error:
                    self.logger.warning(f"Image download failed: {download_error}")
                    return image_data.url
            else:
                # 无法获取图片数据
                self.logger.error(f"No image data in response: {response}")
                raise AgentError(self.name, "No image data in API response")
                
        except Exception as e:
            self.logger.error(f"Image API error: {e}")
            raise AgentError(self.name, f"Image generation failed: {e}")
    
    async def _maybe_upload_r2(self, local_file: str, design_id: str, fallback_url: str) -> str:
        """Upload to R2 if configured and not in demo mode. Deletes local file on success."""
        if self.demo_mode:
            return fallback_url
        try:
            from utils.r2_uploader import is_r2_configured, upload_to_r2
            if is_r2_configured(self.config):
                public_url = await upload_to_r2(
                    local_path=local_file,
                    object_key=f"designs/{design_id}.png",
                    config=self.config,
                )
                # Clean up local file now that it's safely in R2
                try:
                    import os
                    os.remove(local_file)
                    self.logger.info(f"Deleted local file after R2 upload: {local_file}")
                except OSError as rm_err:
                    self.logger.warning(f"Could not delete local file {local_file}: {rm_err}")
                return public_url
        except Exception as e:
            self.logger.warning(f"R2 upload failed, keeping local file: {e}")
        return fallback_url

    def _extract_keywords(self, prompt: str, niche: str) -> List[str]:
        """从提示词中提取关键词"""
        # 简单的关键词提取
        words = prompt.lower().split()
        # 过滤常用词
        stop_words = {'a', 'an', 'the', 'for', 'in', 'on', 'with', 'and', 'or', 'of'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # 确保包含niche
        if niche.lower() not in [k.lower() for k in keywords]:
            keywords.insert(0, niche)
        
        return keywords[:10]  # 最多10个关键词


# 便捷方法
def create_design_generation_node(config: Dict[str, Any] = None):
    """创建设计生成节点"""
    agent = DesignGenerationAgent(config=config)
    
    async def node(state: PODState) -> Dict:
        return await agent(state)
    
    return node
