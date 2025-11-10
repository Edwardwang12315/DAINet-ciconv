import os
import glob
from PIL import Image

def convert_coco_to_widerface(coco_labels_dir, image_base_dir, output_file):
    """
    将COCO格式标签转换为WiderFace格式
    
    参数:
        coco_labels_dir: COCO标签文件目录（每个图片一个txt文件）
        image_base_dir: 图片基础目录（用于构建图片路径）
        output_file: 输出的WiderFace格式文件路径
    """
    # 获取所有COCO标签文件
    coco_label_files = glob.glob(os.path.join(coco_labels_dir, "*.txt"))
    
    with open(output_file, 'w') as widerface_file:
        for label_file in coco_label_files:
            # 获取图片文件名（不带扩展名）
            file_basename = os.path.splitext(os.path.basename(label_file))[0]
            image_path = os.path.join(image_base_dir, f"{file_basename}.jpg")
            
            # 检查图片文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图片文件 {image_path} 不存在，跳过")
                continue
            else:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size

            # 读取COCO标签文件
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # 写入WiderFace格式
            # 第一行: 图片路径
            widerface_file.write(image_path + ' ')
            # 第二行: 目标数量
            widerface_file.write(str(len(lines)) + ' ')
            
            # 写入每个目标的边界框和类别
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                top = float(parts[1])
                left = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 转换为绝对坐标（需要图片尺寸，这里假设您有办法获取）
                # 注意: 由于COCO标签是归一化坐标，但WiderFace使用绝对坐标
                # 您需要根据实际情况修改这部分代码
                # 这里假设您已经知道图片尺寸，或者可以从图片文件中读取
                # 以下是一个示例，需要您根据实际情况调整
                
                # 示例: 假设图片尺寸为640x480
                # img_width, img_height = 640, 480
                
                x1 = int(left * img_width)
                y1 = int(top * img_height)
                w = int(width * img_width)
                h = int(height * img_height)
                
                # 写入边界框和类别
                widerface_file.write(f"{x1} {y1} {w} {h} {class_id} ")
            
            widerface_file.write("\n")
    
    print(f"转换完成! 结果保存在 {output_file}")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    coco_labels_dir = "../../dataset/coco/labels/val2025"  # COCO标签文件目录
    image_base_dir = "../../dataset/coco/images/val2025"   # 图片基础目录
    output_file = "coco2widerface.txt"     # 输出文件
    
    # 执行转换
    convert_coco_to_widerface(coco_labels_dir, image_base_dir, output_file)