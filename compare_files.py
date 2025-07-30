#!/usr/bin/env python3
"""
对比两个文件的前N行是否相同的脚本
"""

import argparse
import sys
from pathlib import Path


def compare_files(file1_path, file2_path, num_lines=2000000, encoding='utf-8'):
    """
    对比两个文件的前N行是否相同
    
    Args:
        file1_path: 第一个文件路径
        file2_path: 第二个文件路径
        num_lines: 要对比的行数，默认200万行
        encoding: 文件编码，默认utf-8
    
    Returns:
        tuple: (是否相同, 不同的行号, 详细信息)
    """
    file1_path = Path(file1_path)
    file2_path = Path(file2_path)
    
    # 检查文件是否存在
    if not file1_path.exists():
        return False, 0, f"文件1不存在: {file1_path}"
    if not file2_path.exists():
        return False, 0, f"文件2不存在: {file2_path}"
    
    print(f"开始对比文件:")
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    print(f"对比前 {num_lines:,} 行")
    print("-" * 50)
    
    try:
        with open(file1_path, 'r', encoding=encoding) as f1, \
             open(file2_path, 'r', encoding=encoding) as f2:
            
            line_num = 0
            file1_ended = False
            file2_ended = False
            
            for line_num in range(1, num_lines + 1):
                # 读取两个文件的当前行
                line1 = f1.readline()
                line2 = f2.readline()
                
                # 检查是否到达文件末尾
                if not line1:
                    file1_ended = True
                if not line2:
                    file2_ended = True
                
                # 如果两个文件都到达末尾
                if file1_ended and file2_ended:
                    print(f"✓ 两个文件都在第 {line_num-1} 行结束，内容完全相同")
                    return True, 0, "文件完全相同"
                
                # 如果只有一个文件到达末尾
                if file1_ended and not file2_ended:
                    return False, line_num, f"文件1在第 {line_num} 行结束，但文件2还有内容"
                if file2_ended and not file1_ended:
                    return False, line_num, f"文件2在第 {line_num} 行结束，但文件1还有内容"
                
                # 对比行内容
                if line1 != line2:
                    print(f"✗ 在第 {line_num} 行发现差异")
                    print(f"文件1: {repr(line1[:100])}{'...' if len(line1) > 100 else ''}")
                    print(f"文件2: {repr(line2[:100])}{'...' if len(line2) > 100 else ''}")
                    return False, line_num, f"第 {line_num} 行内容不同"
                
                # 每10万行显示一次进度
                if line_num % 100000 == 0:
                    print(f"已对比 {line_num:,} 行...")
            
            # 检查是否还有更多行（超过num_lines的部分）
            next_line1 = f1.readline()
            next_line2 = f2.readline()
            
            if next_line1 and not next_line2:
                print(f"✓ 前 {num_lines:,} 行相同，但文件1还有更多内容")
                return True, 0, f"前 {num_lines:,} 行相同，文件1更长"
            elif next_line2 and not next_line1:
                print(f"✓ 前 {num_lines:,} 行相同，但文件2还有更多内容")
                return True, 0, f"前 {num_lines:,} 行相同，文件2更长"
            elif next_line1 and next_line2:
                print(f"✓ 前 {num_lines:,} 行完全相同，两个文件都还有更多内容")
                return True, 0, f"前 {num_lines:,} 行完全相同"
            else:
                print(f"✓ 前 {num_lines:,} 行完全相同，两个文件长度也相同")
                return True, 0, f"前 {num_lines:,} 行完全相同，文件长度相同"
                
    except UnicodeDecodeError as e:
        return False, 0, f"编码错误: {e}"
    except Exception as e:
        return False, 0, f"读取文件时出错: {e}"


def main():
    parser = argparse.ArgumentParser(description="对比两个文件的前N行是否相同")
    parser.add_argument("file1", help="第一个文件路径",default="C:\\Users\\kkai\\Desktop\\zhejianglab\\stk_gen\\data\\stk_access_result_data\\satellite_target_visibility_data_scenario_5_1s_20250729_231834 - 副本.json")
    parser.add_argument("file2", help="第二个文件路径",default="C:\\Users\\kkai\\Desktop\\zhejianglab\\stk_gen\\data\\stk_access_result_data\\satellite_target_visibility_data_scenario_5_1s_20250729_231834.json")
    parser.add_argument("-n", "--lines", type=int, default=2000000, 
                       help="要对比的行数 (默认: 2,000,000)")
    parser.add_argument("-e", "--encoding", default="utf-8", 
                       help="文件编码 (默认: utf-8)")
    
    args = parser.parse_args()
    
    # 执行对比
    is_same, diff_line, message = compare_files(
        args.file1, args.file2, args.lines, args.encoding
    )
    
    print("\n" + "=" * 50)
    if is_same:
        print("✓ 对比结果: 相同")
        print(f"详情: {message}")
        sys.exit(0)
    else:
        print("✗ 对比结果: 不同")
        print(f"详情: {message}")
        if diff_line > 0:
            print(f"首次发现差异的行号: {diff_line}")
        sys.exit(1)


if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式输入
    if len(sys.argv) == 1:
        print("文件对比脚本")
        print("-" * 30)
        file1 = input("请输入第一个文件路径: ").strip()
        file2 = input("请输入第二个文件路径: ").strip()
        lines_input = input("请输入要对比的行数 (默认2,000,000): ").strip()
        
        if not file1 or not file2:
            print("错误: 必须提供两个文件路径")
            sys.exit(1)
        
        try:
            lines = int(lines_input) if lines_input else 2000000
        except ValueError:
            print("错误: 行数必须是一个整数")
            sys.exit(1)
        
        is_same, diff_line, message = compare_files(file1, file2, lines)
        
        print("\n" + "=" * 50)
        if is_same:
            print("✓ 对比结果: 相同")
            print(f"详情: {message}")
        else:
            print("✗ 对比结果: 不同")
            print(f"详情: {message}")
            if diff_line > 0:
                print(f"首次发现差异的行号: {diff_line}")
    else:
        main()
