from typing import Optional, Type, Literal, Union, List, Tuple
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
import pyautogui
import sys


class MouseInput(BaseModel):
    action: Literal["move", "drag", "click", "scroll"] = Field(
        ..., 
        description=(
            "鼠标操作类型: move(光标移动), drag(拖动目标), "
            "click(点击), scroll(滚轮滑动)。")
    )
    target: Union[int, Tuple[int, int]] = Field(
        ..., 
        description=(
            "操作的具体目标: move/drag/click 时为坐标元组: (x, y), "
            "scroll 时为滚轮滑动的整数距离 (正数向上,负数向下)。")
    )
    button: Literal["left", "right", "middle", "primary", "secondary"] = Field(
        "primary", 
        description="鼠标按键: left(左键), right(右键), middle(中键), primary(主键, 一般为左键), secondary(次键, 一般为右键)。(该参数仅适用于 click)"
    )
    interval: float = Field(0.05, ge=0.01, le=1.0, description="每次 click 需要间隔的时间。(该参数仅适用于 click)")
    clicks: int = Field(1, ge=1, le=10, description="click 动作需要重复的次数。(该参数仅适用于 click)")
    
    @field_validator('target')
    def validate_target(cls, v, info):
        action = info.data.get('action')
        if action == "scroll" and not isinstance(v, int):
            raise ValueError(f"'{action}' action requires integer target")
        if action in ("move", "drag", "click") and not isinstance(v, tuple):
            raise ValueError(f"'{action}' action requires tuple target")
        return v

class MouseTool(BaseTool):
    name: str = "mouse_tool"
    description: str = """支持常见鼠标操作。
    操作: move (移动光标), drag (拖动目标), click (点击), scroll (滚轮滑动)。
    """
    args_schema: Optional[Type[BaseModel]] = MouseInput

    def _run(self, action: str, target: Optional[Tuple[int, int]] = None,
            button: Literal["left", "right", "middle", "primary", "secondary"] = "primary",
            interval: float = 0.05,
            clicks: int = 1) -> str:
        try:
            if action == "click":
                pyautogui.click(*target, clicks=clicks, interval=interval, button=button)
                return f"Action Conducted:\n\t action: {button} click\n\t target: {target}\n\t repeated: {clicks} times"
            elif action == "move":
                pyautogui.moveTo(*target)
            elif action == "drag":
                pyautogui.dragTo(*target)
            elif action == "scroll":
                pyautogui.scroll(target)
            return f"Action Conducted:\n\t action: {action}\n\t target: {target}"
        except Exception as e:
            return f"Action Failed: {str(e)}"


class KeyboardInput(BaseModel):
    action: Literal["write", "press", "hotkey", "sequence", "keydown", "keyup"] = Field(
        ...,
        description=(
            "键盘操作类型: write(输入文本), press(按下单个键), "
            "hotkey(使用组合键), sequence(顺序按下多个键), "
            "keydown(按住单个键), keyup(松开单个键)。")
    )
    target: Union[str, List[str]] = Field(
        ...,
        description=(
            "操作的具体目标: write 则输入为文本, press/keydown/keyup 则输入为单个键名, "
            "hotkey 则输入为按键列表, sequence 则输入为顺序按键列表。")
    )  # 统一目标字段
    modifiers: List[str] = Field([], description="同时按住的修饰键列表, 如 ctrl, alt, shift, command (macOS)。")
    interval: float = Field(0.05, ge=0.01, le=1.0, description="按键间隔时间。(仅适用于 write 和 sequence)")
    repeat: int = Field(1, ge=1, le=10, description="键盘操作需要重复的次数。")
    
    @field_validator('target')
    def validate_target(cls, v, info):
        action = info.data.get('action')
        if action == "write" and not isinstance(v, str):
            raise ValueError(f"'{action}' action requires string target")
        if action in ("hotkey","sequence") and not isinstance(v, list):
            raise ValueError(f"'{action}' actions requires list target")
        return v

class KeyboardTool(BaseTool):
    name: str = "keyboard_tool"
    description: str = """支持常见键盘操作。
    操作: write (文本字符串), press (按下单个键, 如回车/大写键), hotkey (如 ctrl+c 组合键), sequence (顺序按下多个键), keydown (按住单个键), keyup (松开单个键)。
    """
    args_schema: Optional[Type[BaseModel]] = KeyboardInput  # 参数格式

    def _run(self, action: str, target: Union[str, List[str]] = None,
            modifiers: List[str] = [],
            interval: float = 0.05,
            repeat: int = 1) -> str:
        
        # 处理 macOS 上的 ctrl 键
        if sys.platform == 'darwin':
            if 'ctrl' in modifiers: 
                modifiers = ['command' if k=='ctrl' else k for k in modifiers]
            if isinstance(target, str) and target == 'ctrl':
                target = 'command'
            if isinstance(target, list):
                target = ['command' if k=='ctrl' else k for k in target]
        try:
            with self._hold_modifiers(modifiers):
                for _ in range(repeat):
                    if action == "write":
                        pyautogui.write(target, interval=interval)
                    elif action == "press":
                        pyautogui.press(target.lower())
                    elif action == "hotkey":
                        pyautogui.hotkey(*target)
                    elif action == "sequence":
                        for key in target:
                            pyautogui.press(key, interval=interval)
                    elif action == "keydown":
                        pyautogui.keyDown(target.lower())
                    elif action == "keyup":
                        pyautogui.keyUp(target.lower())
            return f"Action Conducted:\n\t action: {action}\n\t target: {target}\n\t repeated: {repeat} times"
        except Exception as e:
            return f"Action Failed: {str(e)}"
    
    def _hold_modifiers(self, modifiers: List[str]):
        """修饰键上下文管理器"""
        class ModifierContext:
            def __enter__(self):
                for mod in modifiers:
                    pyautogui.keyDown(mod)
            def __exit__(self, *args):
                for mod in reversed(modifiers):
                    pyautogui.keyUp(mod)
        return ModifierContext()


if __name__=='__main__':
    mouse_tool = MouseTool()
    keyboard_tool = KeyboardTool()
    tools = [mouse_tool, keyboard_tool]
    print(tools)
