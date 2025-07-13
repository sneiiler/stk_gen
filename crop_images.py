#!/usr/bin/env python3
"""
图片裁剪脚本
将documents/visualize_figs目录中的所有图片裁剪成与最小尺寸图片相同的尺寸
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
        reference_image (str): 参考图片路径，如果为None则使用最小尺寸图片
    """

    # 获取所有PNG图片
    image_pattern = os.path.join(input_dir, "*.png")
    image_files = sorted(glob.glob(image_pattern))

    if not image_files:
        print(f"在目录 {input_dir} 中没有找到PNG图片")
        return

    print(f"找到 {len(image_files)} 张图片")

    # 确定参考图片和目标尺寸
    if reference_image is None:
        # 找到最小尺寸的图片
        min_width = float('inf')
        min_height = float('inf')
        min_size_image = None

        print("正在查找最小尺寸图片...")
        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    print(f"  {os.path.basename(image_path)}: {width}x{height}")

                    # 使用面积来判断最小图片
                    if width * height < min_width * min_height:
                        min_width = width
                        min_height = height
                        min_size_image = image_path
            except Exception as e:
                print(f"无法读取图片 {image_path}: {e}")
                continue

        if min_size_image is None:
            print("无法找到有效的图片")
            return

        reference_image = min_size_image
        target_width, target_height = min_width, min_height
        print(f"\n最小尺寸图片: {os.path.basename(reference_image)}")
        print(f"目标尺寸: {target_width} x {target_height}")
    else:
        # 获取指定参考图片的尺寸
        try:
            with Image.open(reference_image) as ref_img:
                target_width, target_height = ref_img.size
                print(f"参考图片: {os.path.basename(reference_image)}")
                print(f"目标尺寸: {target_width} x {target_height}")
        except Exception as e:
            print(f"无法读取参考图片 {reference_image}: {e}")
            return

    # 获取参考图片的DPI信息
    try:
        with Image.open(reference_image) as ref_img:
            target_dpi = ref_img.info.get('dpi', (72, 72))  # 默认72 DPI
            print(f"目标DPI: {target_dpi[0]} x {target_dpi[1]}")
    except Exception as e:
        print(f"无法读取参考图片DPI信息，使用默认值72 DPI: {e}")
        target_dpi = (72, 72)
    
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
                    # 尺寸正确，但需要确保DPI一致
                    if output_dir:
                        output_path = os.path.join(output_dir, os.path.basename(image_path))
                        img.save(output_path, 'PNG', dpi=target_dpi)
                        print(f"复制 {os.path.basename(image_path)} (尺寸已经正确，统一DPI)")
                        processed_count += 1
                    else:
                        # 即使尺寸正确，也要保存以统一DPI
                        img.save(image_path, 'PNG', dpi=target_dpi)
                        print(f"更新 {os.path.basename(image_path)} (统一DPI)")
                        processed_count += 1
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
                
                # 保存裁剪后的图片，确保DPI一致
                cropped_img.save(output_path, 'PNG', dpi=target_dpi)
                
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
                       default="documents/visualize_figs_scenario_1",
                       help="输入图片目录 (默认: documents/visualize_figs)")
    parser.add_argument("--output-dir", "-o", 
                       default=None,
                       help="输出目录 (默认: documents/visualize_figs/cropped)")
    parser.add_argument("--reference", "-r",
                       default=None,
                       help="参考图片路径 (默认: 使用最小尺寸图片)")

    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 {args.input_dir} 不存在")
        return
    
    # 实际处理
    crop_images_to_reference(args.input_dir, args.output_dir, args.reference)


if __name__ == "__main__":
    main()
