# GUI-Agent-LangGraph

Lightweight GUI agent toolkit for visual UI matching and simple automation.  
Core features: UI element detection (YOLO/OCR), icon/text matching (SigLIP / sentence-transformers), and peripheral input tools.

## Quickstart

1. Install dependencies (PyTorch, ultralytics, SigLIP/transformers, easyocr/paddleocr, langchain, etc.).
2. Configure environment variables in `.env` (API keys / model endpoints).
3. Run tests or demo:
   - [gui_agent.py](gui_agent.py)

## Main modules

- Perception & helpers: [utils.py](utils.py)  
  - Key functions: [`utils.find_target_elements`](utils.py), [`utils.get_yolo_model`](utils.py), [`utils.get_siglip_model_processor`](utils.py), [`utils.get_caption_model_processor`](utils.py)
- Vision tools (LangChain tools): [vlm_tools.py](vlm_tools.py)  
  - Key classes/functions: [`vlm_tools.VisionToolsConfig`](vlm_tools.py), [`vlm_tools.UIMatchingTool`](vlm_tools.py), [`vlm_tools.VisionCaptionTool`](vlm_tools.py), [`vlm_tools.test_ui_matching`](vlm_tools.py)
- Local input/peripheral tools: [peripheral_tools.py](peripheral_tools.py)  
  - Key tools: [`peripheral_tools.MouseTool`](peripheral_tools.py), [`peripheral_tools.KeyboardTool`](peripheral_tools.py)
- Drawing utilities: [box_annotator.py](box_annotator.py)  
  - Key class: [`box_annotator.BoxAnnotator`](box_annotator.py)
- Executor prompt template: [executor.md](executor.md)

## Usage notes

- The LangChain tools are defined in [vlm_tools.py](vlm_tools.py) and [peripheral_tools.py](peripheral_tools.py); bind them to a chat model for end-to-end operation.
- Visual matching pipeline uses: YOLO/OCR (detector), SigLIP (icon matching), and sentence-transformers (semantic matching). See implementations in [utils.py](utils.py).
- For quick local testing, run [gui_agent.py](gui_agent.py)

## File map

- [utils.py](utils.py) — perception helpers and model loaders  
- [vlm_tools.py](vlm_tools.py) — LangChain vision tools and demo runner  
- [peripheral_tools.py](peripheral_tools.py) — mouse & keyboard tools  
- [box_annotator.py](box_annotator.py) — drawing bounding boxes  
- [executor.md](executor.md) — agent prompt template  
- ~~[parser.py](parser.py) — parse model tool-call outputs  [UNUSED]~~
- ~~[tools.json](tools.json) — tool schema for the agent  [UNUSED]~~
