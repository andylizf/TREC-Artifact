import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results():
    # 设置更大的图形和更清晰的样式
    plt.style.use('default')
    colors = plt.cm.rainbow(np.linspace(0, 1, 32))
    
    # 读取数据
    df_results = pd.read_csv('L_training_results.csv')
    df_summary = pd.read_csv('L_summary_results.csv')
    
    # 创建训练过程图 (更大的尺寸)
    fig = plt.figure(figsize=(30, 15), dpi=300)
    
    # 训练过程图 - Accuracy
    plt.subplot(1, 2, 1)
    for i, L in enumerate(range(1, 33)):
        acc_col = f'L{L}_accuracy'
        if acc_col in df_results.columns:
            plt.plot(range(1, len(df_results[acc_col]) + 1),
                    df_results[acc_col],
                    linewidth=2,
                    color=colors[i],
                    label=f'L={L}')
    
    plt.title('Test Accuracy vs Epochs', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    # 将图例放在图的右侧，使用多列显示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., ncol=4, fontsize=12)
    
    # 训练过程图 - Loss
    plt.subplot(1, 2, 2)
    for i, L in enumerate(range(1, 33)):
        loss_col = f'L{L}_loss'
        if loss_col in df_results.columns:
            plt.plot(range(1, len(df_results[loss_col]) + 1),
                    df_results[loss_col],
                    linewidth=2,
                    color=colors[i],
                    label=f'L={L}')
    
    plt.title('Training Loss vs Epochs', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., ncol=4, fontsize=12)
    
    # 调整布局以确保图例不被裁剪
    plt.tight_layout()
    # 保存图片时确保有足够的空间显示图例
    plt.savefig('L_training_comparison_detailed.png', 
                bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    
    # 创建最终性能对比图
    plt.figure(figsize=(25, 10), dpi=300)
    
    # Final accuracy vs L
    plt.subplot(1, 2, 1)
    plt.plot(df_summary['L'], df_summary['final_accuracy'], 
            marker='o', linewidth=2, markersize=8)
    for i, row in df_summary.iterrows():
        plt.annotate(f'L={int(row["L"])}',
                    (row['L'], row['final_accuracy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    
    plt.title('Final Accuracy vs L Value', fontsize=16, pad=20)
    plt.xlabel('L Value', fontsize=14)
    plt.ylabel('Final Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Best loss vs L
    plt.subplot(1, 2, 2)
    plt.plot(df_summary['L'], df_summary['best_loss'], 
            marker='o', linewidth=2, markersize=8, color='orange')
    for i, row in df_summary.iterrows():
        plt.annotate(f'L={int(row["L"])}',
                    (row['L'], row['best_loss']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    
    plt.title('Best Loss vs L Value', fontsize=16, pad=20)
    plt.xlabel('L Value', fontsize=14)
    plt.ylabel('Best Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('L_final_performance_detailed.png', 
                bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    
    # 打印关键统计信息
    print("\nKey Statistics:")
    best_L_acc = df_summary.loc[df_summary['final_accuracy'].idxmax()]
    best_L_loss = df_summary.loc[df_summary['best_loss'].idxmin()]
    print(f"Best L for accuracy: {int(best_L_acc['L'])} (accuracy: {best_L_acc['final_accuracy']:.4f})")
    print(f"Best L for loss: {int(best_L_loss['L'])} (loss: {best_L_loss['best_loss']:.4f})")
    
    # 计算相关性
    correlation = np.corrcoef(df_summary['L'], df_summary['final_accuracy'])[0,1]
    print(f"\nCorrelation between L and final accuracy: {correlation:.4f}")

if __name__ == "__main__":
    plot_results_H8()  # 运行新的测试
    # plot_results()  # 原来的测试注释掉