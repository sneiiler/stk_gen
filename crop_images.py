#!/usr/bin/env python3
"""
图片裁剪脚本
将documents/visualize_figs目录中的所有图片裁剪成与第一张图片相同的尺寸
从右下角裁掉多余的内容（保留左上角区域）
"""

import os
import glob
from PIL import Image
import argparse


def crop_images_to_reference(input_dir, output_dir=None, reference_image=None):
    """
    将目录中的所有图片裁剪成与参考图片相同的尺寸
    
    Args:
        input_dir (str): 输入图片目录
        output_dir (str): 输出目录，如果为None则覆盖原图片
        reference_image (str): 参考图片路径，如果为None则使用第一张图片
    """
    
    # 获取所有PNG图片
    image_pattern = os.path.join(input_dir, "*.png")
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"在目录 {input_dir} 中没有找到PNG图片")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 确定参考图片
    if reference_image is None:
        reference_image = image_files[0]
    
    # 获取参考图片的尺寸
    try:
        with Image.open(reference_image) as ref_img:
            target_width, target_height = ref_img.size
            print(f"参考图片: {os.path.basename(reference_image)}")
            print(f"目标尺寸: {target_width} x {target_height}")
    except Exception as e:
        print(f"无法读取参考图片 {reference_image}: {e}")
        return
    
    # 创建输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 处理每张图片
    processed_count = 0
    skipped_count = 0
    
    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                
                # 检查是否需要裁剪
                if original_width == target_width and original_height == target_height:
                    # 尺寸正确，但如果有输出目录，仍需要复制
                    if output_dir:
                        output_path = os.path.join(output_dir, os.path.basename(image_path))
                        img.save(output_path, 'PNG')
                        print(f"复制 {os.path.basename(image_path)} (尺寸已经正确)")
                        processed_count += 1
                    else:
                        print(f"跳过 {os.path.basename(image_path)} (尺寸已经正确)")
                        skipped_count += 1
                    continue
                
                # 检查图片是否足够大
                if original_width < target_width or original_height < target_height:
                    print(f"警告: {os.path.basename(image_path)} 尺寸 ({original_width}x{original_height}) "
                          f"小于目标尺寸 ({target_width}x{target_height})，跳过")
                    skipped_count += 1
                    continue
                
                # 从左上角裁剪（保留左上角，去掉右下角多余部分）
                cropped_img = img.crop((0, 0, target_width, target_height))
                
                # 确定输出路径
                if output_dir:
                    output_path = os.path.join(output_dir, os.path.basename(image_path))
                else:
                    output_path = image_path
                
                # 保存裁剪后的图片
                cropped_img.save(output_path, 'PNG')
                
                print(f"处理完成: {os.path.basename(image_path)} "
                      f"({original_width}x{original_height} -> {target_width}x{target_height})")
                processed_count += 1
                
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 张图片")
    print(f"跳过: {skipped_count} 张图片")


def main():
    parser = argparse.ArgumentParser(description="将图片裁剪成与参考图片相同的尺寸")
    parser.add_argument("--input-dir", "-i", 
                       default="documents/visualize_figs",
                       help="输入图片目录 (默认: documents/visualize_figs)")
    parser.add_argument("--output-dir", "-o", 
                       default="documents/visualize_figs/cropped",
                       help="输出目录 (默认: documents/visualize_figs/cropped)")
    parser.add_argument("--reference", "-r",
                       default=None,
                       help="参考图片路径 (默认: 使用第一张图片)")
    parser.add_argument("--dry-run", "-d",
                       action="store_true",
                       help="仅显示将要处理的图片，不实际处理")
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 {args.input_dir} 不存在")
        return
    
    if args.dry_run:
        # 干运行模式，仅显示信息
        image_pattern = os.path.join(args.input_dir, "*.png")
        image_files = sorted(glob.glob(image_pattern))
        
        if not image_files:
            print(f"在目录 {args.input_dir} 中没有找到PNG图片")
            return
        
        reference_image = args.reference or image_files[0]
        
        try:
            with Image.open(reference_image) as ref_img:
                target_width, target_height = ref_img.size
                print(f"参考图片: {os.path.basename(reference_image)}")
                print(f"目标尺寸: {target_width} x {target_height}")
                print(f"找到 {len(image_files)} 张图片")
                
                for image_path in image_files[:5]:  # 只显示前5张
                    with Image.open(image_path) as img:
                        w, h = img.size
                        status = "需要裁剪" if (w != target_width or h != target_height) else "尺寸正确"
                        print(f"  {os.path.basename(image_path)}: {w}x{h} - {status}")
                
                if len(image_files) > 5:
                    print(f"  ... 还有 {len(image_files) - 5} 张图片")
                    
        except Exception as e:
            print(f"无法读取参考图片: {e}")
    else:
        # 实际处理
        crop_images_to_reference(args.input_dir, args.output_dir, args.reference)


if __name__ == "__main__":
    main()
