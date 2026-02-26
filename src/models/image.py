from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import os
import time
import json
import requests
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from ..utils import get_logger
from ..utils.oss_utils import OSSImageUploader, is_object_key

logger = get_logger(__name__)


class ImageGenModel(ABC):
    """Abstract base class for image generation models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, output_path: str, **kwargs) -> Tuple[str, float]:
        """
        Generates an image from a prompt.
        
        Args:
            prompt: The input text prompt.
            output_path: The path to save the generated image.
            **kwargs: Additional arguments.
            
        Returns:
            A tuple containing:
            - The path to the generated image file.
            - The duration of the API generation process in seconds.
        """
        pass


class OpenRouterImageModel(ImageGenModel):
    """
    Image generation model using OpenRouter (OpenAI-compatible) API.
    Supports text-to-image via the 'modalities' extension.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        
        # 从配置或环境变量中读取 API 密钥和基础 URL
        self.api_key ="sk-or-v1-55555555"
        if not self.api_key:
            logger.warning("OpenRouter API Key not found in config or environment variables.")
        
        self.base_url = config.get('base_url') or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 模型名称：支持 T2I 和 I2I（但 I2I 暂未完全实现，忽略参考图像）
        self.model_name = self.params.get('model_name', 'sourceful/riverflow-v2-fast')  # T2I 默认模型
        self.i2i_model_name = self.params.get('i2i_model_name', self.model_name)  # I2I 暂用相同模型

    def generate(
        self,
        prompt: str,
        output_path: str,
        ref_image_path: Optional[str] = None,
        ref_image_paths: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, float]:
        """
        Generates an image using OpenRouter API.
        
        Args:
            prompt: Text prompt for generation.
            output_path: Local path to save the image.
            ref_image_path: Legacy single reference image path (optional).
            ref_image_paths: List of reference image paths (optional).
            model_name: Override the default model.
            **kwargs: Additional generation parameters (size, n, negative_prompt, etc.)
        
        Returns:
            Tuple (output_path, api_duration)
        """
        # 处理参考图像（OpenRouter 的 modalities 模式目前不支持图像输入，仅警告并忽略）
        all_ref_paths = []
        if ref_image_path:
            all_ref_paths.append(ref_image_path)
        if ref_image_paths:
            all_ref_paths.extend(ref_image_paths)
        all_ref_paths = list(set(all_ref_paths))
        
        if all_ref_paths:
            logger.warning("Reference images are provided but OpenRouter modalities mode does not support image input. Ignoring reference images.")
        
        # 选择模型：优先使用传入的 model_name，否则根据是否有参考图像选择（但 I2I 暂未实现）
        final_model_name = "sourceful/riverflow-v2-fast"
        logger.info(f"Using model: {final_model_name}")
        
        # 提取生成参数
        size = kwargs.pop('size', self.params.get('size', '1024x1024'))
        n = kwargs.pop('n', self.params.get('n', 1))
        negative_prompt = kwargs.pop('negative_prompt', None)
        
        # 构建消息内容（纯文本）
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # 构建 extra_body 参数（OpenRouter 特定的 modalities 和其他参数）
        extra_body = {
            "modalities": ["image"],
            "size": size,
            "n": n
        }
        if negative_prompt:
            extra_body["negative_prompt"] = negative_prompt
        
        # 合并用户自定义的额外参数
        extra_body.update(kwargs)
        
        logger.info(f"Starting image generation with prompt: {prompt}")
        logger.info(f"Model: {final_model_name}, Size: {size}, N: {n}")
        
        try:
            api_start_time = time.time()
            
            # 调用 OpenRouter API
            response = self.client.chat.completions.create(
                model=final_model_name,
                messages=messages,
                extra_body=extra_body
            )
            
            api_end_time = time.time()
            api_duration = api_end_time - api_start_time
            
            # 解析响应，提取图像 URL
            image_urls = self._extract_image_urls(response)
            if not image_urls:
                raise RuntimeError("No images found in response.")
            
            # 目前只取第一张图（原逻辑只保存一张）
            image_url = image_urls[0]
            logger.info(f"Generation success.")
            logger.info(f"API duration: {api_duration:.2f}s")
            import base64
            header, encoded = image_url.split(',', 1)
            image_data = base64.b64decode(encoded)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(image_data)

            # 下载图像
            logger.info(f"Image saved from base64 data to {output_path}")
            return output_path, api_duration
        
        except APIConnectionError as e:
            logger.error(f"Failed to connect to OpenRouter API: {e}")
            raise RuntimeError(f"API connection error: {e}")
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise RuntimeError(f"Rate limit error: {e}")
        except APIStatusError as e:
            logger.error(f"API returned error {e.status_code}: {e.response}")
            raise RuntimeError(f"API error {e.status_code}: {e.response}")
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}", exc_info=True)
            raise RuntimeError(f"Image generation failed: {str(e)}")

    def _extract_image_urls(self, response) -> List[str]:
        """
        从 OpenRouter 响应中提取图像 URL。
        假设响应格式与 OpenAI 兼容，图像位于 response.choices[0].message.images 列表中。
        """
        image_urls = []
        try:
            message = response.choices[0].message
            if hasattr(message, 'images') and message.images:
                for img in message.images:
                    # OpenRouter 返回的 image 可能是字典，包含 'image_url' 字段
                    if isinstance(img, dict) and 'image_url' in img:
                        url = img['image_url'].get('url')
                        if url:
                            image_urls.append(url)
                    elif isinstance(img, str):
                        # 如果直接是字符串 URL
                        image_urls.append(img)
        except (AttributeError, IndexError, KeyError) as e:
            logger.error(f"Failed to parse image URLs from response: {e}")
            logger.debug(f"Full response: {response}")
        return image_urls

    def _download_image(self, url: str, output_path: str):
        """下载图像到本地文件（与原逻辑相同）"""
        logger.info(f"Downloading image to {output_path}...")
        
        # 设置重试策略
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        temp_path = output_path + ".tmp"
        try:
            response = http.get(url, stream=True, timeout=60, verify=False)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            os.rename(temp_path, output_path)
            logger.info("Download complete.")
            
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
WanxImageModel = OpenRouterImageModel
__all__ = ['ImageGenModel', 'OpenRouterImageModel', 'WanxImageModel']
