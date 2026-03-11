"""
POD多智能体系统 - Agent基类
定义所有Agent的公共接口和行为

设计原则：
1. 单一职责：每个Agent只负责一个领域
2. 无状态性：Agent本身不保存状态，所有状态通过State流转
3. 幂等性：相同输入多次执行应产生相同结果
4. 容错性：内置错误处理和重试机制
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from functools import wraps

from core.state import PODState, add_error, update_cost

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Agent执行错误"""
    def __init__(self, agent_name: str, message: str, recoverable: bool = True):
        self.agent_name = agent_name
        self.message = message
        self.recoverable = recoverable
        super().__init__(f"[{agent_name}] {message}")


def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    指数退避重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 退避系数
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
            
            raise last_exception
        return wrapper
    return decorator


class BaseAgent(ABC):
    """
    Agent基类
    
    所有Agent必须实现：
    1. name属性：Agent名称
    2. process方法：主要处理逻辑
    
    基类提供：
    1. 统一的执行入口（__call__）
    2. 错误处理和日志记录
    3. 成本追踪
    4. 执行时间记录
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Agent
        
        Args:
            config: Agent配置，如API密钥、模型参数等
        """
        self.config = config or {}
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger(f"agent.{self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent名称，用于日志和状态追踪"""
        pass
    
    @abstractmethod
    async def process(self, state: PODState) -> Dict[str, Any]:
        """
        主要处理逻辑
        
        Args:
            state: 当前工作流状态
        
        Returns:
            需要更新的状态字段
        """
        pass
    
    async def __call__(self, state: PODState) -> Dict[str, Any]:
        """
        统一的执行入口
        
        提供：
        1. 执行前后的日志记录
        2. 执行时间统计
        3. 错误捕获和状态更新
        4. 元数据更新
        """
        start_time = datetime.now()
        self.logger.info(f"Starting {self.name} agent...")
        
        try:
            # 检查前置条件
            self._validate_preconditions(state)
            
            # 执行主要逻辑
            result = await self.process(state)
            
            # 添加通用元数据更新
            result["current_step"] = f"{self.name}_complete"
            result["updated_at"] = datetime.now().isoformat()
            
            # 计算执行时间
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"{self.name} completed in {elapsed:.2f}s")
            
            return result
            
        except AgentError as e:
            self.logger.error(f"Agent error: {e}")
            error_update = add_error(
                state, 
                step=self.name, 
                error_type="agent_error",
                message=str(e)
            )
            
            if not e.recoverable:
                error_update["status"] = "failed"
            
            return error_update
            
        except Exception as e:
            self.logger.exception(f"Unexpected error in {self.name}: {e}")
            return add_error(
                state,
                step=self.name,
                error_type="unexpected_error",
                message=str(e)
            )
    
    def _validate_preconditions(self, state: PODState):
        """
        验证前置条件
        子类可以重写此方法添加特定验证
        """
        pass
    
    def _track_cost(self, service: str, cost: float, state: PODState) -> Dict:
        """追踪API成本"""
        return update_cost(state, service, cost)


class LLMAgent(BaseAgent):
    """
    基于LLM的Agent基类
    
    扩展功能：
    1. LLM客户端管理
    2. Prompt模板支持
    3. 响应解析
    """
    
    # Cost per million tokens (input, output)
    MODEL_PRICING = {
        "claude-sonnet-4-6":        {"input": 3.00,  "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model: str = None,
        temperature: float = None
    ):
        super().__init__(config)
        import os
        # 从环境变量读取模型配置，支持参数覆盖
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
        self.temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self._llm = None
        self.llm_cost = 0.0        # accumulated real cost this agent instance
        self.token_usage = {"input": 0, "output": 0}
    
    @property
    def llm(self):
        """延迟初始化LLM客户端"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self):
        """
        创建LLM客户端
        使用yunwu.ai中转API（OpenAI兼容格式）
        集成Langfuse进行LLM调用监控
        """
        try:
            from langchain_openai import ChatOpenAI
            
            # 尝试初始化Langfuse回调
            callbacks = self._get_langfuse_callbacks()
            
            # 优先使用yunwu API，否则降级到原始Anthropic API
            yunwu_key = self.config.get("yunwu_api_key")
            yunwu_base = self.config.get("yunwu_api_base", "https://yunwu.ai/v1")
            
            if yunwu_key:
                llm = ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    api_key=yunwu_key,
                    base_url=yunwu_base,
                    callbacks=callbacks
                )
                self.logger.info(f"LLM initialized: {self.model} via yunwu.ai" + 
                               (" with Langfuse" if callbacks else ""))
                return llm
            
            # 降级：使用原始Anthropic API
            anthropic_key = self.config.get("anthropic_api_key")
            if anthropic_key:
                try:
                    from langchain_anthropic import ChatAnthropic
                    return ChatAnthropic(
                        model=self.model,
                        temperature=self.temperature,
                        api_key=anthropic_key,
                        callbacks=callbacks
                    )
                except ImportError:
                    pass
            
            self.logger.warning("No valid API key found, using mock LLM")
            return None
        except ImportError:
            self.logger.warning("langchain_openai not installed, using mock LLM")
            return None
    
    def _get_langfuse_callbacks(self):
        """获取Langfuse回调处理器用于LLM监控
        
        新版 Langfuse SDK 自动从环境变量读取配置:
        - LANGFUSE_PUBLIC_KEY
        - LANGFUSE_SECRET_KEY  
        - LANGFUSE_HOST
        """
        try:
            from langfuse.langchain import CallbackHandler
            import os
            
            # 检查Langfuse配置 - 新版SDK从环境变量自动读取
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            
            if public_key and secret_key:
                # 新版SDK只需传递public_key或不传任何参数（使用env vars）
                handler = CallbackHandler()
                self.logger.info(f"Langfuse callback initialized for {self.name}")
                return [handler]
            else:
                self.logger.warning("Langfuse keys not configured, skipping observability")
                return None
        except ImportError as e:
            self.logger.warning(f"Langfuse langchain module not available: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize Langfuse: {e}")
            return None
    
    async def invoke_llm(self, prompt: str) -> str:
        """
        调用LLM
        
        Args:
            prompt: 提示词
        
        Returns:
            LLM响应文本
        """
        if self.llm is None:
            # Mock响应，用于测试
            return self._mock_response(prompt)
        
        response = await self.llm.ainvoke(prompt)

        # Track real token usage and cost
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens  = usage.get("input_tokens",  0)
            output_tokens = usage.get("output_tokens", 0)
            self.token_usage["input"]  += input_tokens
            self.token_usage["output"] += output_tokens
            pricing = self.MODEL_PRICING.get(self.model, {"input": 3.00, "output": 15.00})
            call_cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
            self.llm_cost += call_cost
            self.logger.info(
                f"Tokens: {input_tokens} in / {output_tokens} out — "
                f"call ${call_cost:.4f} | agent total ${self.llm_cost:.4f} [{self.model}]"
            )

        return response.content
    
    def _mock_response(self, prompt: str) -> str:
        """Mock响应，用于测试"""
        return '{"mock": true, "message": "This is a mock response"}'


class ToolAgent(BaseAgent):
    """
    基于外部工具/API的Agent基类
    
    扩展功能：
    1. API客户端管理
    2. 请求重试
    3. 响应验证
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        api_base_url: Optional[str] = None,
        timeout: float = 30.0
    ):
        super().__init__(config)
        self.api_base_url = api_base_url
        self.timeout = timeout
        self._client = None
    
    @property
    def client(self):
        """延迟初始化HTTP客户端"""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self):
        """创建HTTP客户端"""
        try:
            import httpx
            return httpx.AsyncClient(
                base_url=self.api_base_url,
                timeout=self.timeout,
                headers=self._get_headers()
            )
        except ImportError:
            self.logger.warning("httpx not installed, using mock client")
            return None
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头，子类可重写"""
        return {"Content-Type": "application/json"}
    
    @with_retry(max_retries=3, delay=1.0, backoff=2.0)
    async def api_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        发送API请求，带重试
        
        Args:
            method: HTTP方法
            endpoint: API端点
            **kwargs: 其他请求参数
        
        Returns:
            响应数据
        """
        if self.client is None:
            return self._mock_api_response(method, endpoint)
        
        response = await self.client.request(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def _mock_api_response(self, method: str, endpoint: str) -> Dict:
        """Mock API响应，用于测试"""
        return {"mock": True, "method": method, "endpoint": endpoint}


# 用于创建简单节点函数的工厂
def create_agent_node(agent: BaseAgent) -> Callable[[PODState], Dict]:
    """
    将Agent实例转换为LangGraph节点函数
    
    Args:
        agent: Agent实例
    
    Returns:
        可用于StateGraph.add_node的函数
    """
    async def node(state: PODState) -> Dict:
        return await agent(state)
    
    # 同步包装器，兼容非异步调用
    def sync_node(state: PODState) -> Dict:
        return asyncio.run(agent(state))
    
    return sync_node
