#!/usr/bin/env python3
"""
从已有的对齐结果文件计算实验指标
"""
import pandas as pd
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

def normalize_prediction(pred_text):
    """Convert prediction text to binary label (0=normal, 1=abnormal)"""
    if not pred_text or pd.isna(pred_text):
        return None
    
    pred_lower = str(pred_text).lower().strip()
    
    # Check for abnormal indicators
    if any(keyword in pred_lower for keyword in ['abnormal', 'abnor', '1', 'error', 'alert', 'interrupt', 'exception']):
        # Make sure it's not "normal" that contains these substrings
        if 'normal' not in pred_lower or pred_lower.startswith('abnormal'):
            return 1
    
    # Check for normal indicators
    if any(keyword in pred_lower for keyword in ['normal', 'norm', '0']):
        return 0
    
    # Default: if contains "abnormal" anywhere, it's abnormal
    if 'abnormal' in pred_lower:
        return 1
    
    # Default to normal if unclear
    return 0

def calculate_metrics_from_file(aligned_file, original_csv_file):
    """从对齐结果文件和原始CSV文件计算指标"""
    print(f"读取对齐结果文件: {aligned_file}")
    df_aligned = pd.read_excel(aligned_file)
    
    print(f"读取原始数据集: {original_csv_file}")
    df_original = pd.read_csv(original_csv_file)
    
    # 转换原始标签
    if 'Label' in df_original.columns:
        df_original['label'] = df_original['Label'].apply(lambda x: 0 if str(x).strip() == '-' or str(x).strip() == '' else 1)
    
    print(f"\n对齐结果文件:")
    print(f"  行数: {len(df_aligned)}")
    print(f"  列名: {df_aligned.columns.tolist()}")
    
    # 匹配日志并获取真实标签
    labels = []
    for log in df_aligned['log']:
        matches = df_original[df_original['Content'] == log]
        if len(matches) > 0:
            labels.append(matches.iloc[0]['label'])
        else:
            labels.append(None)
    
    df_aligned['true_label'] = labels
    
    # 归一化预测结果
    if 'pred' in df_aligned.columns:
        df_aligned['pred_label'] = df_aligned['pred'].apply(normalize_prediction)
    elif 'pred_label' in df_aligned.columns:
        pass  # Already normalized
    else:
        print("错误: 找不到预测列 'pred' 或 'pred_label'")
        return
    
    # 过滤有效样本
    valid_mask = (df_aligned['true_label'].notna()) & (df_aligned['pred_label'].notna())
    y_true = df_aligned[valid_mask]['true_label'].tolist()
    y_pred = df_aligned[valid_mask]['pred_label'].tolist()
    
    if len(y_true) == 0:
        print("错误: 没有有效的样本对来计算指标")
        return
    
    # 计算指标
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*60)
    print("实验指标 (Evaluation Metrics)")
    print("="*60)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")
    print(f"有效样本数:         {len(y_true)}/{len(df_aligned)}")
    print("="*60)
    
    print("\n详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'], zero_division=0))
    
    # 保存指标到文件
    metrics_file = aligned_file.replace('.xlsx', '_metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("实验指标 (Evaluation Metrics)\n")
        f.write("="*60 + "\n")
        f.write(f"准确率 (Accuracy):  {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall):    {recall:.4f}\n")
        f.write(f"F1分数 (F1-Score):  {f1:.4f}\n")
        f.write(f"有效样本数:         {len(y_true)}/{len(df_aligned)}\n")
        f.write("="*60 + "\n\n")
        f.write("详细分类报告:\n")
        f.write(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'], zero_division=0))
    
    print(f"\n指标已保存到: {metrics_file}")
    
    # 保存带标签的对齐结果
    output_file = aligned_file.replace('.xlsx', '_with_labels.xlsx')
    df_aligned.to_excel(output_file, index=False)
    print(f"带标签的对齐结果已保存到: {output_file}")

if __name__ == "__main__":
    aligned_file = sys.argv[1] if len(sys.argv) > 1 else "Aligned_result_bgl_full.xlsx"
    original_csv = sys.argv[2] if len(sys.argv) > 2 else "BGL_2k.log_structured.csv"
    
    calculate_metrics_from_file(aligned_file, original_csv)



