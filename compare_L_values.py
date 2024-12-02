import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from collections import defaultdict

def run_training(L_value, epochs=10):
    """运行单个L值的训练"""
    cmd = [
        'python', 'examples/train_model.py',
        '--epochs', str(epochs),
        '--batch_size', '128',
        '--model_name', 'CifarNet',
        '--checkpoint_path', f'EXP_L{L_value}',
        '--trec', '0,1',
        '--L', f'{L_value},{L_value}',
        '--H', '8,8',
        '--learning_rate', '0.1',
        '--momentum', '0.9',
        '--weight_decay', '1e-4'
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
        # 原来测试1-32的代码
        L_values = list(range(1, 33))
    else:
        # 最好的几组
        best_L = [10, 16, 20, 25, 32]  # 性能好的L值
        worst_L = [3, 7, 26, 30]       # 性能差的L值
        L_values = best_L + worst_L
    
    epochs = 10
    results = defaultdict(dict)
    
    for L in L_values:
        print(f"\nTraining with L={L}, H=8")
        output = run_training(L, epochs)
        
        # 保存原始日志
        with open(f'H8_training_log_L{L}.txt', 'w') as f:
            f.write(output)
        
        # 解析结果
        accuracies, losses = parse_training_log(output)
        results[L]['accuracies'] = accuracies
        results[L]['losses'] = losses
        
        print(f"Parsed {len(accuracies)} accuracies and {len(losses)} losses")
    
    # 绘制更清晰的图表
    plt.figure(figsize=(30, 15), dpi=300)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(L_values)))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    for i, L in enumerate(L_values):
        plt.plot(range(1, len(results[L]['accuracies']) + 1), 
                results[L]['accuracies'], 
                linewidth=2,
                color=colors[i],
                label=f'L={L}')
    plt.title('Test Accuracy vs Epochs (H=8)', fontsize=16, pad=20)
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
    plt.title('Training Loss vs Epochs (H=8)', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('H8_L_training_comparison.png', 
                bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    
    # 最终性能对比图
    plt.figure(figsize=(25, 10), dpi=300)
    
    # Final accuracy vs L
    plt.subplot(1, 2, 1)
    final_accuracies = [results[L]['accuracies'][-1] for L in L_values]
    plt.plot(L_values, final_accuracies, marker='o', linewidth=2, markersize=8)
    for i, L in enumerate(L_values):
        plt.annotate(f'L={L}\n{final_accuracies[i]:.4f}',
                    (L, final_accuracies[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    plt.title('Final Accuracy vs L Value (H=8)', fontsize=16, pad=20)
    plt.xlabel('L Value', fontsize=14)
    plt.ylabel('Final Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Final loss vs L
    plt.subplot(1, 2, 2)
    final_losses = [results[L]['losses'][-1] for L in L_values]
    plt.plot(L_values, final_losses, marker='o', linewidth=2, markersize=8, color='orange')
    for i, L in enumerate(L_values):
        plt.annotate(f'L={L}\n{final_losses[i]:.4f}',
                    (L, final_losses[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    plt.title('Final Loss vs L Value (H=8)', fontsize=16, pad=20)
    plt.xlabel('L Value', fontsize=14)
    plt.ylabel('Final Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('H8_L_final_performance.png', 
                bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    
    # 保存数值结果
    df_results = pd.DataFrame()
    for L in L_values:
        df_results[f'L{L}_accuracy'] = pd.Series(results[L]['accuracies'])
        df_results[f'L{L}_loss'] = pd.Series(results[L]['losses'])
    df_results.to_csv('H8_L_training_results.csv', index=False)
    
    # 保存最终结果的摘要
    summary_data = {
        'L': L_values,
        'final_accuracy': final_accuracies,
        'final_loss': final_losses,
        'best_accuracy': [max(results[L]['accuracies']) for L in L_values],
        'best_loss': [min(results[L]['losses']) for L in L_values]
    }
    pd.DataFrame(summary_data).to_csv('H8_L_summary_results.csv', index=False)
    
    # 打印关键结果
    print("\nKey Results (H=8):")
    best_L_acc = L_values[np.argmax(final_accuracies)]
    best_L_loss = L_values[np.argmin(final_losses)]
    print(f"Best L for accuracy: {best_L_acc} (accuracy: {max(final_accuracies):.4f})")
    print(f"Best L for loss: {best_L_loss} (loss: {min(final_losses):.4f})")

if __name__ == "__main__":
    # 设置参数来选择运行模式
    test_selected = True  # True: 只测试选定的L值; False: 测试所有1-32的L值
    compare_L_values(test_all=not test_selected)