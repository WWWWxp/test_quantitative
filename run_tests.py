#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有测试的脚本
"""
import os
import sys
import subprocess

def run_test(script_name, description):
    """运行单个测试脚本"""
    print(f"\n{'='*60}")
    print(f"运行测试: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 测试通过")
            print("输出:")
            print(result.stdout)
        else:
            print("❌ 测试失败")
            print("错误输出:")
            print(result.stderr)
            print("标准输出:")
            print(result.stdout)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"💥 测试异常: {e}")
        return False

def main():
    """主函数"""
    print("开始运行量化分析系统测试...")
    
    tests = [
        ("dataset.py", "数据集测试"),
        ("model.py", "模型测试"),
        ("learner.py", "训练器测试"),
        ("test_quantitative.py", "综合测试")
    ]
    
    passed = 0
    total = len(tests)
    
    for script, description in tests:
        if run_test(script, description):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"测试总结: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        return True
    else:
        print("❌ 部分测试失败，请检查配置和代码。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
