import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_L_performance():
    # 读取数据
    df_summary = pd.read_csv('L_summary_results.csv')
    
    # 计算各种相关系数
    correlations = {
        'Pearson': {
            'accuracy': stats.pearsonr(df_summary['L'], df_summary['final_accuracy']),
            'loss': stats.pearsonr(df_summary['L'], df_summary['best_loss'])
        },
        'Spearman': {
            'accuracy': stats.spearmanr(df_summary['L'], df_summary['final_accuracy']),
            'loss': stats.spearmanr(df_summary['L'], df_summary['best_loss'])
        }
    }
    
    # 创建详细的可视化分析
    plt.style.use('default')  # 使用默认样式替代seaborn
    fig = plt.figure(figsize=(20, 15))
    
    # 1. L vs Accuracy 散点图和回归线
    plt.subplot(2, 2, 1)
    plt.scatter(df_summary['L'], df_summary['final_accuracy'], alpha=0.5)
    z = np.polyfit(df_summary['L'], df_summary['final_accuracy'], 1)
    p = np.poly1d(z)
    plt.plot(df_summary['L'], p(df_summary['L']), "r--", alpha=0.8)
    plt.title('L vs Final Accuracy with Regression Line', fontsize=14)
    plt.xlabel('L Value', fontsize=12)
    plt.ylabel('Final Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 2. L vs Loss 散点图和回归线
    plt.subplot(2, 2, 2)
    plt.scatter(df_summary['L'], df_summary['best_loss'], alpha=0.5)
    z = np.polyfit(df_summary['L'], df_summary['best_loss'], 1)
    p = np.poly1d(z)
    plt.plot(df_summary['L'], p(df_summary['L']), "r--", alpha=0.8)
    plt.title('L vs Best Loss with Regression Line', fontsize=14)
    plt.xlabel('L Value', fontsize=12)
    plt.ylabel('Best Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 3. L值分组分析
    plt.subplot(2, 2, 3)
    df_summary['L_group'] = pd.qcut(df_summary['L'], q=4, labels=['Small', 'Medium-Small', 'Medium-Large', 'Large'])
    group_means = df_summary.groupby('L_group')['final_accuracy'].agg(['mean', 'std'])
    plt.bar(range(4), group_means['mean'], yerr=group_means['std'], alpha=0.6)
    plt.xticks(range(4), ['Small', 'Medium-Small', 'Medium-Large', 'Large'])
    plt.title('Accuracy Distribution by L Groups', fontsize=14)
    plt.xlabel('L Value Group', fontsize=12)
    plt.ylabel('Final Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 4. Loss vs Accuracy 散点图，用L值作为点的大小
    plt.subplot(2, 2, 4)
    plt.scatter(df_summary['best_loss'], df_summary['final_accuracy'], 
                s=df_summary['L']*20, alpha=0.5)
    for i, txt in enumerate(df_summary['L']):
        plt.annotate(f'L={txt}', 
                    (df_summary['best_loss'].iloc[i], df_summary['final_accuracy'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.title('Loss vs Accuracy (point size = L)', fontsize=14)
    plt.xlabel('Best Loss', fontsize=12)
    plt.ylabel('Final Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('L_correlation_analysis.png', dpi=300, bbox_inches='tight')
    
    # 打印详细的统计分析
    print("\n=== Correlation Analysis ===")
    print("\nPearson Correlations:")
    print(f"L vs Accuracy: r={correlations['Pearson']['accuracy'][0]:.4f}, p={correlations['Pearson']['accuracy'][1]:.4f}")
    print(f"L vs Loss: r={correlations['Pearson']['loss'][0]:.4f}, p={correlations['Pearson']['loss'][1]:.4f}")
    
    print("\nSpearman Correlations:")
    print(f"L vs Accuracy: rho={correlations['Spearman']['accuracy'][0]:.4f}, p={correlations['Spearman']['accuracy'][1]:.4f}")
    print(f"L vs Loss: rho={correlations['Spearman']['loss'][0]:.4f}, p={correlations['Spearman']['loss'][1]:.4f}")
    
    # 分组统计
    group_stats = df_summary.groupby('L_group').agg({
        'final_accuracy': ['mean', 'std', 'min', 'max'],
        'best_loss': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\n=== Group Statistics ===")
    print("\nAccuracy by L Groups:")
    print(group_stats['final_accuracy'])
    print("\nLoss by L Groups:")
    print(group_stats['best_loss'])
    
    # 最优L值分析
    best_accuracy = df_summary.loc[df_summary['final_accuracy'].idxmax()]
    best_loss = df_summary.loc[df_summary['best_loss'].idxmin()]
    balanced = df_summary.loc[(df_summary['final_accuracy'] > df_summary['final_accuracy'].mean()) & 
                            (df_summary['best_loss'] < df_summary['best_loss'].mean())]
    
    print("\n=== Optimal L Values ===")
    print(f"\nBest for Accuracy: L={best_accuracy['L']} (Accuracy={best_accuracy['final_accuracy']:.4f}, Loss={best_accuracy['best_loss']:.4f})")
    print(f"Best for Loss: L={best_loss['L']} (Accuracy={best_loss['final_accuracy']:.4f}, Loss={best_loss['best_loss']:.4f})")
    print("\nBalanced Performance L values:")
    for _, row in balanced.iterrows():
        print(f"L={int(row['L'])}: Accuracy={row['final_accuracy']:.4f}, Loss={row['best_loss']:.4f}")

if __name__ == "__main__":
    analyze_L_performance() 