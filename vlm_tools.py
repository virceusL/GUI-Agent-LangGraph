import os
import time
import numpy as np
import torch
import cv2
import base64
import io
from PIL import Image, ImageGrab
from typing import List, Dict, Optional, Type, Union
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from utils import (
    get_yolo_model, get_semantic_model, get_siglip_model_processor, find_target_elements,
    get_caption_model_processor
)


class VisionToolsConfig:
    """Vision Tools Configuration"""
    def __init__(self):
        self.test_image_source = r"E:\Production\Programing\Proj\agent\XnAgent\test.png"
        self.target_labels = []
        self.yolo_box_threshold = 0.01
        self.yolo_iou_threshold = 0.1
        self.overlap_iou_threshold = 0.9
        self.confidence_threshold = 0.8
        self.text_scale = 0.4
        self.text_padding = 5
        self.use_paddlleocr = True
        self.imgsz = None
        self.batch_size = 128

        self.yolo_model_path = r"E:\Production\Programing\Proj\agent\OmniParser\weights\icon_detect\model.pt"
        self.semantic_model_path = r"E:\Production\models\hgf\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9"
        self.siglip_model_path = r"E:\Production\models\hgf\hub\models--google--siglip-base-patch16-224\snapshots\7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
        self.caption_model_name = "florence2"
        self.caption_model_path = r"E:\Production\Programing\Proj\agent\OmniParser\weights\icon_caption_florence"


class UIMatchingInput(BaseModel):
    query: str = Field(
        ..., 
        description="Input query for UI matching, use comma ',' to separate multiple queries.", 
        examples=["chrome icon, the browser, search box, WeChat app, flight from Shenzhen to Shanghai"]
    )


class UIMatchingTool(BaseTool):
    """UI Matching Tool - YOLO + SigLIP & OCR + MinLM"""
    name: str = "ui_matching_tool"
    description: str = """ONLY FOR UI MATCHING | NO DESCRIPTION CAPABILITY
    Input: the text label of the UI element to be matched. | Output: the information of the matched UI, coordinates, confidence, interactivity, etc.
    """
    args_schema: Optional[Type[BaseModel]] = UIMatchingInput

    config: VisionToolsConfig = None
    yolo_model: torch.nn.Module = None
    semantic_model: torch.nn.Module = None
    siglip_model_processor: torch.nn.Module = None

    def __init__(self, config: VisionToolsConfig = None):
        super().__init__()
        self.config = config
        self._initialize_models()

    def _initialize_models(self):
        """Load YOLO, SigLIP and Semantic Models"""
        try:
            # 加载模型
            self.yolo_model = get_yolo_model(model_path=self.config.yolo_model_path)
            self.semantic_model = get_semantic_model(model_path=self.config.semantic_model_path)
            self.siglip_model_processor = get_siglip_model_processor(model_name_or_path=self.config.siglip_model_path)

        except Exception as e:
            raise Exception(f"Error loading models: {e}")

    def _capture_screen(self) -> Image.Image:
        """Capture screen screenshot"""
        try:
            screenshot = ImageGrab.grab()
            return screenshot
        
        except Exception as e:
            raise Exception(f"Error capturing screen: {e}")
        
    def _save_results(self, img: Image.Image, target_labels: List[str]):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存图像
        # 创建子文件夹
        if not os.path.exists("results_img"):
            os.makedirs("results_img")
        result_path = f"./results_img/result_{'_'.join(target_labels)}.png"
        
        try:
            img.save(result_path)
            return result_path
        except Exception as e:
            print(f"Warning: Failed to save result image: {e}")
            return ""

    def _run(self, query: Optional[str] = None) -> List[Dict]:
        """
        Run the tool, capture the screen and match UI elements based on the provided query.

        Args:
            query: labels to match, separated by commas. If not provided, use the config's target labels.
            
        Returns:
            List of information dictionary of the matched UI elements, including UI interactivity, content, matched label, confidence, centroid, area, etc.
        """
        try:
            # Capture screen
            screenshot = self._capture_screen()
            
            if query:
                target_labels = query.split(',')
            else:
                target_labels = self.config.target_labels

            # Match elements
            ascii_encoded_image, results = find_target_elements(
                screenshot, target_labels,
                self.yolo_model, self.semantic_model, self.siglip_model_processor,
                confidence_threshold=self.config.confidence_threshold,
                yolo_box_threshold=self.config.yolo_box_threshold, yolo_iou_threshold=self.config.yolo_iou_threshold, overlap_iou_threshold=self.config.overlap_iou_threshold,
                use_paddleocr=False
            )

            result_img = Image.open(io.BytesIO(base64.b64decode(ascii_encoded_image)))
            self._save_results(result_img, target_labels)
            
            return results
            
        except Exception as e:
            print(f"Error in UIMatchingTool._run: {e}")
            return []


# 画面描述生成工具（用于视觉验证当前的屏幕状态）
class VisionCaptionInput(BaseModel):
    """No input needed"""
    pass

class VisionCaptionTool(BaseTool):
    """Vision Caption Tool - florence2"""
    name: str = "vision_caption_tool"
    description: str = """ONLY FOR VISION CAPTIONING | NO MATCHING CAPABILITY
    Input: Not needed | Output: the caption of the current screen, including the UI elements and their interactivity, content, etc.
    """
    args_schema: Optional[Type[BaseModel]] = VisionCaptionInput

    config: VisionToolsConfig = None
    caption_model: torch.nn.Module = None
    
    def __init__(self, config: VisionToolsConfig = None):
        super().__init__()
        self.config = config
        self._initialize_model()

    def _initialize_model(self):
        """Load Florence2 Model"""
        try:
            # 加载模型
            self.caption_model = get_caption_model_processor(
                model_name=self.config.caption_model_name, model_name_or_path=self.config.caption_model_path
            )
        except Exception as e:
            raise Exception(f"Error loading models: {e}")
    
    def _run(self, args: Optional[dict] = None) -> str:
        try:
            # Capture screen
            screenshot = self._capture_screen()

            # Generate caption
            caption = ""

            return caption

        except Exception as e:
            print(f"Error in VisionCaptionTool._run: {e}")
            return ""


# 兼容性函数，用于替代原有的demo函数逻辑
def test_ui_matching():
    """
    UI 元素匹配工具测试, 指定目标: WeChat, Chrome
    """
    # 创建工具实例
    ui_matching_config = VisionToolsConfig()
    ui_matching_tool = UIMatchingTool(ui_matching_config)

    # 设置目标标签
    target_labels = ["Chrome", "WeChat", "Edge", "Firefox", "Browser"]
    query = ','.join(target_labels)
    # query = "Chrome,Edge,Firefox,Browser"
    
    # 方式1：自动截屏识别
    print("Running screen capture recognition...")
    start_time = time.time()
    result = ui_matching_tool._run(query)
    print(f"Time taken: {time.time() - start_time} seconds")
    
    if result:
        print('-' * 100)
        print("Matched UI Elements:")
        for element in result:
            print(f" - {element}")
        print('-' * 100)
    else:
        print("No UI elements found.")


# 使用示例
if __name__ == "__main__":
    # 运行测试
    test_ui_matching()