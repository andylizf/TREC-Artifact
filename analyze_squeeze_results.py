import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_path):
    """解析单个log文件，提取epoch和accuracy信息"""
    accuracies = []
    losses = []
    epochs = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # 提取accuracy
        if "Accuracy of the network on the 10000 test images:" in line:
            acc = float(line.split()[-1])
            accuracies.append(acc)
        
        # 提取loss (每个epoch的最后一个batch的loss)
        if "loss:" in line and "batch=  300" in line:
            loss = float(line.split()[-1])
            losses.append(loss)
            
    # 生成对应的epoch数
    epochs = list(range(len(accuracies)))
    
    return epochs, accuracies, losses

def analyze_all_logs():
    # 查找所有实验目录
    base_dir = Path('.')
    exp_dirs = list(base_dir.glob('EXP_squeeze_L*'))
    
    results = []
    training_data = {}
    
    for exp_dir in exp_dirs:
        # 从目录名提取L值
        L = int(str(exp_dir).split('L')[-1])
        
        # 找到log文件
        log_files = list(exp_dir.rglob('log.txt'))
        if not log_files:
            continue
            
        # 解析log文件
        epochs, accuracies, losses = parse_log_file(log_files[0])
        
        # 保存训练过程数据
        training_data[L] = {
            'epochs': epochs,
            'accuracies': accuracies,
            'losses': losses
        }
        
        # 记录最佳结果
        results.append({
            'L': L,
            'best_accuracy': max(accuracies),
            'final_accuracy': accuracies[-1],
            'best_loss': min(losses) if losses else None,
            'final_loss': losses[-1] if losses else None,
            'best_epoch': accuracies.index(max(accuracies))
        })
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 创建可视化
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 训练曲线图
    plt.subplot(2, 2, 1)
    for L in training_data.keys():
        plt.plot(training_data[L]['epochs'], 
                training_data[L]['accuracies'], 
                label=f'L={L}')
    plt.title('Training Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Loss曲线图
    plt.subplot(2, 2, 2)
    for L in training_data.keys():
        plt.plot(training_data[L]['epochs'], 
                training_data[L]['losses'], 
                label=f'L={L}')
    plt.title('Training Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. 最佳准确率vs L值
    plt.subplot(2, 2, 3)
    plt.scatter(df_results['L'], df_results['best_accuracy'])
    plt.plot(df_results['L'], df_results['best_accuracy'], '--')
    for _, row in df_results.iterrows():
        plt.annotate(f'L={int(row["L"])}',
                    (row['L'], row['best_accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    plt.title('Best Accuracy vs L Value', fontsize=14)
    plt.xlabel('L Value', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 4. 最佳Loss vs L值
    plt.subplot(2, 2, 4)
    plt.scatter(df_results['L'], df_results['best_loss'])
    plt.plot(df_results['L'], df_results['best_loss'], '--')
    for _, row in df_results.iterrows():
        plt.annotate(f'L={int(row["L"])}',
                    (row['L'], row['best_loss']),
                    xytext=(5, 5), textcoords='offset points')
    plt.title('Best Loss vs L Value', fontsize=14)
    plt.xlabel('L Value', fontsize=12)
    plt.ylabel('Best Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('squeeze_L_analysis.png', dpi=300, bbox_inches='tight')
    
    # 打印统计结果
    print("\n=== SqueezeNet Results Analysis ===")
    print("\nBest Results by Accuracy:")
    print(df_results.sort_values('best_accuracy', ascending=False)[['L', 'best_accuracy', 'best_loss', 'best_epoch']].to_string(index=False))
    
    print("\nBest Results by Loss:")
    print(df_results.sort_values('best_loss')[['L', 'best_loss', 'best_accuracy', 'best_epoch']].to_string(index=False))
    
    # 计算相关性
    corr_acc = np.corrcoef(df_results['L'], df_results['best_accuracy'])[0,1]
    corr_loss = np.corrcoef(df_results['L'], df_results['best_loss'])[0,1]
    
    print("\nCorrelations:")
    print(f"L vs Accuracy: {corr_acc:.4f}")
    print(f"L vs Loss: {corr_loss:.4f}")

if __name__ == '__main__':
    analyze_all_logs() 