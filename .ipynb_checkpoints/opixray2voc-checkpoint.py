import os
import os.path as osp
import cv2
import xml.etree.ElementTree as ET

# OPIXray 数据集的类别
OPIXray_CLASSES = (
    'Folding_Knife', 'Straight_Knife', 'Scissor', 'Utility_Knife', 'Multi-tool_Knife'
)

# 数据集根目录
root_dir = "../autodl-tmp/OPIXray"
voc_root_dir = osp.join(root_dir, "VOCdevkit", "VOC2007")

def create_voc_annotation(txt_file, image_file, output_file):
    """
    将 OPIXray 的 .txt 标注文件转换为 VOC 格式的 XML 文件
    """
    # 解析 OPIXray 的 .txt 标注文件
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 获取图像的宽度和高度
    if not osp.exists(image_file):
        print(f"Image file {image_file} not found.")
        return
    image = cv2.imread(image_file)
    if image is None:
        print(f"Image file {image_file} corrupted.")
        return
    height, width, _ = image.shape

    # 创建 VOC 格式的 XML 文件
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = "VOC2007"
    filename = ET.SubElement(root, "filename")
    filename.text = osp.basename(image_file)
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "The VOC2007 Database"
    annotation = ET.SubElement(source, "annotation")
    annotation.text = "PASCAL VOC2007"
    image_tag = ET.SubElement(source, "image")
    image_tag.text = "flickr"
    flickrid = ET.SubElement(source, "flickrid")
    flickrid.text = "341012865"
    owner = ET.SubElement(root, "owner")
    flickrid_owner = ET.SubElement(owner, "flickrid")
    flickrid_owner.text = "Fried Camels"
    name_owner = ET.SubElement(owner, "name")
    name_owner.text = "Jinky the Fruit Bat"
    size = ET.SubElement(root, "size")
    width_tag = ET.SubElement(size, "width")
    width_tag.text = str(width)
    height_tag = ET.SubElement(size, "height")
    height_tag.text = str(height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    # 遍历标注文件中的每一行
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            print(f"Skipping invalid annotation line: {line.strip()}")
            continue
        image_name = parts[0]
        class_name = parts[1]
        if class_name not in OPIXray_CLASSES:
            print(f"Skipping unknown class: {class_name}")
            continue
        xmin, ymin, xmax, ymax = map(int, parts[2:6])

        # 创建 object 标签
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_name
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        xmin_tag = ET.SubElement(bndbox, "xmin")
        xmin_tag.text = str(xmin)
        ymin_tag = ET.SubElement(bndbox, "ymin")
        ymin_tag.text = str(ymin)
        xmax_tag = ET.SubElement(bndbox, "xmax")
        xmax_tag.text = str(xmax)
        ymax_tag = ET.SubElement(bndbox, "ymax")
        ymax_tag.text = str(ymax)

    # 写入 XML 文件
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Converted {txt_file} to VOC format and saved as {output_file}")

def convert_dataset(phase):
    """
    转换数据集
    :param phase: 'train' 或 'test'
    """
    annotation_dir = osp.join(root_dir, phase, f"{phase}_annotation")
    image_dir = osp.join(root_dir, phase, f"{phase}_image")
    output_dir = osp.join(voc_root_dir, "Annotations")
    os.makedirs(output_dir, exist_ok=True)
    output_image_dir = osp.join(voc_root_dir, "JPEGImages")
    os.makedirs(output_image_dir, exist_ok=True)

    # 读取索引文件
    index_file = osp.join(root_dir, phase, f"{phase}_knife.txt")
    with open(index_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]

    for img_id in image_ids:
        txt_file = osp.join(annotation_dir, f"{img_id}.txt")
        image_file = osp.join(image_dir, f"{img_id}.jpg")  # 或 .TIFF, .tiff
        output_file = osp.join(output_dir, f"{img_id}.xml")
        output_image_file = osp.join(output_image_dir, f"{img_id}.jpg")

        if not osp.exists(txt_file):
            print(f"Annotation file {txt_file} not found, skipping.")
            continue

        if not osp.exists(image_file):
            print(f"Image file {image_file} not found.")
            continue

        # 复制图像文件到 JPEGImages 目录
        import shutil
        shutil.copy(image_file, output_image_file)

        create_voc_annotation(txt_file, image_file, output_file)

# 转换训练集
convert_dataset("train")

# 转换测试集
convert_dataset("test")