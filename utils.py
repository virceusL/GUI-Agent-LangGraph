# from ultralytics import YOLO
import io
import base64
# utility function
import cv2
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt
import easyocr
from paddleocr import PaddleOCR
import base64
import torch
from typing import List, Union
import supervision as sv
import torchvision.transforms as T
from torchvision.ops import box_convert
from PIL import Image

from box_annotator import BoxAnnotator 

reader = easyocr.Reader(['ch_sim', 'en'])
paddle_ocr = PaddleOCR(
    lang='ch_doc',  # other lang also available
    use_angle_cls=False,
    use_gpu=False,  # using cuda will conflict with pytorch in the same process
    show_log=False,
    max_batch_size=1024,
    use_dilation=True,  # improves accuracy
    det_db_score_mode='slow',  # improves accuracy
    rec_batch_num=1024)


def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    '''
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                # keep yolo boxes + prioritize ocr label
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1): # ocr inside icon
                            # box_added = True
                            # delete the box3_elem from ocr_bbox
                            try:
                                # gather all ocr labels
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            # break
                        elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                    else:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
            else:
                filtered_boxes.append(box1)
    return filtered_boxes # torch.tensor(filtered_boxes)


def predict(model, image, caption, box_threshold, text_threshold):
    """ Use huggingface model to replace the original model
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold, # 0.4,
        text_threshold=text_threshold, # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    # model = model['model']
    if scale_img:
        result = model.predict(
            source=image,
            conf=box_threshold,
            imgsz=imgsz,
            iou=iou_threshold, # default 0.7
        )
    else:
        result = model.predict(
            source=image,
            conf=box_threshold,
            iou=iou_threshold, # default 0.7
        )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


def get_semantic_model(model_path):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path)
    return model


def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp


def check_ocr_box(image_source: Union[str, Image.Image], display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    w, h = image_source.size
    if use_paddleocr: # PaddleOCR
        if easyocr_args is None:
            text_threshold = 0.5
        else:
            text_threshold = easyocr_args['text_threshold']
        result = paddle_ocr.ocr(image_np, cls=False)[0]
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text = [item[1][0] for item in result if item[1][1] > text_threshold]
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(image_np, **easyocr_args)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        #  matplotlib expects RGB
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]

    return (text, bb), goal_filtering


def get_yolo_model(model_path):
    from ultralytics import YOLO
    # Load the model.
    model = YOLO(model_path)
    return model


def get_siglip_model_processor(model_name_or_path="siglip-base-patch16-224", device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    if device == 'cpu':
        model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True
        )
    else:
        model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)

    return {'model': model.to(device), 'processor': processor}


def get_text_elements(
                    image_rgb, use_paddleocr=True, 
                    text_scale=0.4, text_padding=5):
    """
    使用OCR获取文本元素
    
    Args:
        image_source: 图像路径或图像对象
        use_paddleocr: 是否使用PaddleOCR
        text_scale: 文本大小 (暂时未使用)
        text_padding: 文本边距 (暂时未使用)

    Returns:
        文本元素列表
    """
    w, h = image_rgb.size

    # OCR 检测
    ocr_bbox_rslt, _ = check_ocr_box(
        image_rgb, 
        display_img=False, 
        output_bb_format='xyxy', 
        use_paddleocr=use_paddleocr
    )
    ocr_text, ocr_bbox = ocr_bbox_rslt

    # 处理OCR边界框
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox = ocr_bbox.tolist()
    else:
        ocr_bbox = None
    
    # 构建元素列表
    ocr_bbox_elem = [
        {'type': 'text', 'bbox': box, 'interactivity': False, 'content': txt} 
        for box, txt in zip(ocr_bbox, ocr_text) if int_box_area(box, w, h) > 0
    ] if ocr_bbox else []

    return ocr_bbox_elem


def get_text_similarity_scores(batch_texts, semantic_model, target_labels, top_k=5):
    """
    使用语义模型计算文本与目标标签的相似度
    
    Args:
        batch_texts: 待匹配的文本列表
        semantic_model: 语义模型 (已载入的模型)
        target_labels: 目标标签列表，如['chrome']
        top_k: 返回前k个最相似的标签
    
    Returns:
        每个文本的相似度得分和最匹配的多个标签
    """
    import torch.nn.functional as F

    if not batch_texts:
        return []

    text_embeddings = F.normalize(semantic_model.encode(batch_texts, convert_to_tensor=True), p=2, dim=1)
    target_embeddings = F.normalize(semantic_model.encode(target_labels, convert_to_tensor=True), p=2, dim=1)

    # 计算相似度
    text_similarity_matrix = torch.mm(text_embeddings, target_embeddings.t())
    top_scores, top_indices = torch.topk(text_similarity_matrix, min(top_k, len(target_labels)), dim=1)

    # 找到每个label对应前top_k个最匹配的text，返回它们的best_match,best_score和top_matches
    results = []
    for i in range(len(batch_texts)):
        results.append({
            'best_match': target_labels[top_indices[i, 0].item()],
            'best_score': top_scores[i, 0].item(),
            'top_matches': [
                {'label': target_labels[top_indices[i, j].item()], 'score': top_scores[i, j].item()}
                for j in range(top_indices.size(1))
            ]
        })

    return results


def get_icon_elements(
                    image_rgb_np, yolo_model, 
                    box_threshold=0.15, iou_threshold=0.75):
    """
    使用YOLO模型获取图标元素
    
    Args:
        image_rgb_np: 输入RGB图像的numpy数组
        yolo_model: YOLO模型 (已载入的模型)
        box_threshold: 置信度阈值
        iou_threshold: IOU阈值

    Returns:
        图标元素列表
    """
    w, h = image_rgb_np.shape[1], image_rgb_np.shape[0]

    # YOLO 检测
    xyxy, logits, _ = predict_yolo(
        model=yolo_model, 
        image=image_rgb_np, 
        box_threshold=box_threshold, 
        imgsz=(h, w), 
        scale_img=False, 
        iou_threshold=iou_threshold
    )
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    
    # 构建元素列表
    xyxy_elem = [
        {'type': 'icon', 'bbox': box, 'interactivity': True, 'content': None} 
        for box in xyxy.tolist() if int_box_area(box, w, h) > 0
    ]

    return xyxy_elem, logits


@torch.inference_mode()
def get_icon_similarity_scores(icon_boxes, starting_idx, image_source, siglip_model_processor, target_labels: List[str], batch_size=128, top_k=5):
    """
    使用SigLIP计算图标与目标标签的相似度
    
    Args:
        icon_boxes: 图标边界框列表
        starting_idx: 开始处理的索引 (跳过OCR框)
        image_source: 输入图像
        siglip_model_processor: SigLIP模型和处理器
        target_labels: 目标标签列表，如['chrome']
        batch_size: 批处理大小
        top_k: 返回前k个最相似的标签
    
    Returns:
        每个图标元素的相似度得分和最匹配的标签
    """
    from torchvision.transforms import ToPILImage
    import torch.nn.functional as F
    
    to_pil = ToPILImage()
    
    # non-ocr boxes
    if starting_idx is not None and starting_idx >= 0:
        icon_boxes = icon_boxes[starting_idx:]
        
    # crop the icons
    cropped_pil_images = []
    for i, coord in enumerate(icon_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            assert xmin < xmax and ymin < ymax

            # crop
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            
            # transform
            cropped_image = cv2.resize(cropped_image, (224, 224))  # SigLIP 224x224
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB) # SigLIP RGB
            
            cropped_pil_images.append(to_pil(cropped_image))
        except Exception as e:
            print(f"Error processing box {i}: {e}")
            continue
    
    if not cropped_pil_images:
        return []
    
    model, processor = siglip_model_processor['model'], siglip_model_processor['processor']
    device = model.device
    
    # prompt template
    label_texts = [f'The icon of {label}.' for label in target_labels]

    results = []
    for i in range(0, len(cropped_pil_images), batch_size):
        batch_images = cropped_pil_images[i:i+batch_size]
        
        # process texts and images both (padding="max_length" to ensure consistency)
        inputs = processor(
            text=label_texts,
            images=batch_images,
            padding="max_length",
            return_tensors="pt",
            max_length=64,
        ).to(device)
        
        # inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # similarity score (shape: [batch_size, num_labels])
        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)  # SigLIP sigmoid
        
        # top k scores for each icon image
        for j in range(len(batch_images)):
            scores = probs[j]
            
            # get top-k matches
            top_scores, top_indices = torch.topk(scores, min(top_k, len(target_labels)))
            
            results.append({
                'best_match': target_labels[top_indices[0]],
                'best_score': top_scores[0].item(),
                'top_matches': [
                    {'label': target_labels[idx], 'score': score.item()}
                    for score, idx in zip(top_scores, top_indices)
                ]
            })
    
    return results


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
            matched_elements=None, text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in phrases]

    # if matching
    if matched_elements:
        for i in range(len(matched_elements)):
            matched_icon = matched_elements[i]
            labels[matched_icon["index"]] += f" ({matched_icon['matched_label']}={matched_icon['confidence']:.2f})"
        
    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def find_target_elements(
                    image_source: Union[str, Image.Image], target_labels: List[str], 
                    yolo_model, semantic_model, siglip_model_processor, 
                    yolo_box_threshold=0.15, yolo_iou_threshold=0.75, 
                    overlap_iou_threshold=0.9, confidence_threshold=0.8,
                    text_scale=0.4, text_padding=5, 
                    use_paddleocr=True, imgsz=None, batch_size=128):
    """
    根据目标标签匹配UI元素 (图标和文本)
    
    Args:
        image_source: 输入图像路径或图像对象
        target_labels: 目标标签列表
        yolo_model: YOLO模型
        semantic_model: 语义分割模型
        siglip_model_processor: SigLIP模型处理器
        yolo_box_threshold: YOLO 检测框阈值
        yolo_iou_threshold: YOLO IOU阈值
        overlap_iou_threshold: 重叠阈值
        confidence_threshold: 匹配阈值
        text_scale: 文本比例
        text_padding: 文本填充
        use_paddleocr: 是否使用PaddleOCR
        imgsz: 图像尺寸
        batch_size: 批处理大小
    
    Returns:
        匹配的UI元素列表
    
    """
    # 图像预处理
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    image_rgb = image_source.convert("RGB")
    image_rgb_np = np.array(image_rgb)
    w, h = image_rgb.size
    imgsz = imgsz or (h, w)

    # 获取图标和文本元素
    ocr_bbox_elem = get_text_elements(image_rgb, use_paddleocr=use_paddleocr, text_scale=text_scale, text_padding=text_padding)
    xyxy_elem, logits = get_icon_elements(image_rgb_np, yolo_model, box_threshold=yolo_box_threshold, iou_threshold=yolo_iou_threshold)

    # 重叠处理 - 统一元素列表
    filtered_boxes = remove_overlap(
        boxes=xyxy_elem, 
        iou_threshold=overlap_iou_threshold, 
        ocr_bbox=ocr_bbox_elem
    )
    
    # 为所有元素分配唯一索引
    for idx, box in enumerate(filtered_boxes):
        box['index'] = idx

    # 分离元素类型
    icon_elements = [box for box in filtered_boxes if box['type'] == 'icon']
    text_elements = [box for box in filtered_boxes if box['type'] == 'text']

    matched_elements = []
    # 处理图标匹配
    if icon_elements:
        
        # 批处理获取图标相似度
        icon_boxes = [e['bbox'] for e in icon_elements]
        similarity_results = get_icon_similarity_scores(
            icon_boxes,
            starting_idx=None,
            image_source=image_rgb_np,
            siglip_model_processor=siglip_model_processor,
            target_labels=target_labels,
            batch_size=batch_size
        )
        
        # 图标匹配结果
        for element, sim_result in zip(icon_elements, similarity_results):
            if sim_result['best_score'] >= confidence_threshold:
                bbox = element['bbox']
                pixel_coords = [
                    int(bbox[0] * w), int(bbox[1] * h),
                    int(bbox[2] * w), int(bbox[3] * h)
                ]
                
                matched_elements.append(element)
                matched_elements[-1].update({
                    'matched_label': sim_result['best_match'],
                    'confidence': sim_result['best_score'],
                    'bbox_pixel': pixel_coords,
                    'center_point': (
                        (pixel_coords[0] + pixel_coords[2]) // 2,
                        (pixel_coords[1] + pixel_coords[3]) // 2
                    ),
                    'area': (pixel_coords[2] - pixel_coords[0]) * (pixel_coords[3] - pixel_coords[1]),
                })
    
    # 处理文本匹配
    if text_elements:
        
        # 获取文本相似度
        texts = [element['content'] for element in text_elements]
        similarity_results = get_text_similarity_scores(
            texts,
            semantic_model=semantic_model,
            target_labels=target_labels,
        )
        
        # 文本匹配结果
        for element, sim_result in zip(text_elements, similarity_results):
            if sim_result['best_score'] >= confidence_threshold:
                bbox = element['bbox']
                pixel_coords = [
                    int(bbox[0] * w), int(bbox[1] * h),
                    int(bbox[2] * w), int(bbox[3] * h)
                ]
                
                # 保留超过阈值的匹配
                matched_elements.append(element)
                matched_elements[-1].update({
                    'matched_label': sim_result['best_match'],
                    'confidence': sim_result['best_score'],
                    'bbox_pixel': pixel_coords,
                    'center_point': (
                        (pixel_coords[0] + pixel_coords[2]) // 2,
                        (pixel_coords[1] + pixel_coords[3]) // 2
                    ),
                    'area': (pixel_coords[2] - pixel_coords[0]) * (pixel_coords[3] - pixel_coords[1]),
                })
    
    # 合并匹配结果并排序
    matched_elements.sort(key=lambda x: x['index'])
    
    # 标注可视化
    box_overlay_ratio = imgsz[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # 准备标注数据
    filtered_boxes_tensor = torch.tensor([box['bbox'] for box in filtered_boxes])
    filtered_boxes_cxcywh = box_convert(boxes=filtered_boxes_tensor, in_fmt="xyxy", out_fmt="cxcywh")
    phrases = [str(box['index']) for box in filtered_boxes]  # 使用索引作为标签
    

    # TODO 图像标注范围 logits <-> matched_elements
    # 标注图像
    annotated_frame, _ = annotate(
        image_source=image_rgb_np,
        boxes=filtered_boxes_cxcywh,
        logits=logits,
        phrases=phrases,
        matched_elements=matched_elements,
        **draw_bbox_config
    )

    # 编码结果图像
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    ascii_encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    
    return ascii_encoded_image, matched_elements


def get_best_match_coordinates(image_source: Union[str, Image.Image], target_label: str,
                              yolo_model, siglip_model_processor, **kwargs):
    """
    便捷函数：获取单个目标标签的最佳匹配坐标
    
    Args:
        image_source: 输入图像
        target_label: 目标标签，如 'chrome'
        yolo_model: YOLO模型
        siglip_model_processor: SigLIP模型处理器
        **kwargs: 其他参数
    
    Returns:
        最佳匹配的图标信息，如果没有找到则返回None
    """
    matched_icons = find_target_elements(
        image_source, [target_label], yolo_model, siglip_model_processor, **kwargs
    )
    
    if matched_icons:
        return matched_icons[0]  # 返回相似度最高的匹配
    else:
        return None
    
    
def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, device_map=None, torch_dtype=torch.float32
            ) 
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, device_map=None, torch_dtype=torch.float16
            ).to(device)

    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM 
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
            ).to(device)
            
    return {'model': model.to(device), 'processor': processor}