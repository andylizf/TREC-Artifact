import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from collections import defaultdict

def run_training(L_value, epochs=50):
    """运行单个L值的训练"""
    cmd = [
        'python', 'examples/train_model.py',
        '--checkpoint_path', f'EXP_squeeze_L{L_value}',
        '--model_name', 'SqueezeNet',
        '--epochs', str(epochs),
        '--batch_size', '128',
        '--learning_rate', '0.001',
        '--momentum', '0.95',
        '--weight_decay', '0.0001',
        '--grad_clip', '5',
        '--step', '15',
        '--trec', '0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
        '--L', f'9,96,8,{L_value},64,8,72,64,16,144,128,4,144,32,6,54,48,6,216,8,4,288,256,8,288,4',
        '--H', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    return result.stdout

def parse_training_log(log_output):
    """解析训练日志，提取accuracy和loss"""
    accuracies = []
    losses_per_epoch = defaultdict(list)
    current_epoch = 0
    
    # 按行处理日志
    for line in log_output.split('\n'):
        # 提取accuracy
        acc_match = re.search(r'Accuracy of the network on the \d+ test images: ([\d.]+)', line)
        if acc_match:
            acc = float(acc_match.group(1))
            accuracies.append(acc)
        
        # 提取loss
        loss_match = re.search(r'\[epoch=(\d+), batch=\s*\d+\] loss: ([\d.]+)', line)
        if loss_match:
            epoch = int(loss_match.group(1))
            loss = float(loss_match.group(2))
            losses_per_epoch[epoch].append(loss)
    
    # 计算每个epoch的平均loss
    losses = []
    for epoch in range(len(accuracies)):
        if epoch in losses_per_epoch:
            epoch_loss = np.mean(losses_per_epoch[epoch])
            losses.append(epoch_loss)
        else:
            losses.append(float('nan'))
    
    return accuracies, losses

def compare_L_values(test_all=False):
    if test_all:
        L_values = list(range(1, 33))
    else:
        # 最好的几组
        best_L = [10, 16, 20, 25, 32, 40, 50, 64]  # 性能好的L值
        worst_L = [3, 7, 26, 30, 36, 48]       # 性能差的L值
        L_values = best_L + worst_L
    
    epochs = 50
    results = defaultdict(dict)
    
    for L in L_values:
        print(f"\nTraining with L={L}")
        output = run_training(L, epochs)
        
        # 保存原始日志
        with open(f'squeeze_training_log_L{L}.txt', 'w') as f:
            f.write(output)
        
        # 解析结果
        accuracies, losses = parse_training_log(output)
        results[L]['accuracies'] = accuracies
        results[L]['losses'] = losses
        
        print(f"Parsed {len(accuracies)} accuracies and {len(losses)} losses")
    
    # 绘制更清晰的图表
    plt.figure(figsize=(30, 15))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(L_values)))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    for i, L in enumerate(L_values):
        plt.plot(range(1, len(results[L]['accuracies']) + 1), 
                results[L]['accuracies'], 
                linewidth=2,
                color=colors[i],
                label=f'L={L}')
    plt.title('Test Accuracy vs Epochs', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., ncol=2, fontsize=12)
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    for i, L in enumerate(L_values):
        plt.plot(range(1, len(results[L]['losses']) + 1), 
                results[L]['losses'], 
                linewidth=2,
                color=colors[i],
                label=f'L={L}')
    plt.title('Training Loss vs Epochs', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('squeeze_L_training_comparison.png', 
                bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # 保存数值结果
    df_results = pd.DataFrame()
    for L in L_values:
        df_results[f'L{L}_accuracy'] = pd.Series(results[L]['accuracies'])
        df_results[f'L{L}_loss'] = pd.Series(results[L]['losses'])
    df_results.to_csv('squeeze_L_training_results.csv', index=False)
    
    # 保存最终结果的摘要
    summary_data = {
        'L': L_values,
        'final_accuracy': [results[L]['accuracies'][-1] for L in L_values],
        'final_loss': [results[L]['losses'][-1] for L in L_values],
        'best_accuracy': [max(results[L]['accuracies']) for L in L_values],
        'best_loss': [min(results[L]['losses']) for L in L_values]
    }
    pd.DataFrame(summary_data).to_csv('squeeze_L_summary_results.csv', index=False)

if __name__ == "__main__":
    compare_L_values(test_all=False)