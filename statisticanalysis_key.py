import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm
import warnings
import os

# Create directory for EPS files
if not os.path.exists('eps_figures'):
    os.makedirs('eps_figures')

# Ignore font warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set font to default to avoid Chinese character issues
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'eps'

sns.set_style("whitegrid")

# Define column name mapping from Chinese to English
column_mapping = {
    'ID': 'ID',
    '性别': 'Gender',
    '年级': 'Grade',
    '专业': 'Major',
    '使用时长': 'Usage_Duration',
    '使用ChatGPT_闭卷课程': 'LLM_Closed_Book_Course',
    '使用ChatGPT_闭卷成绩': 'LLM_Closed_Book_Score',
    '使用ChatGPT_开卷课程': 'LLM_Open_Book_Course',
    '使用ChatGPT_开卷成绩': 'LLM_Open_Book_Score',
    '不使用ChatGPT_闭卷课程': 'No_LLM_Closed_Book_Course',
    '不使用ChatGPT_闭卷成绩': 'No_LLM_Closed_Book_Score',
    '不使用ChatGPT_开卷课程': 'No_LLM_Open_Book_Course',
    '不使用ChatGPT_开卷成绩': 'No_LLM_Open_Book_Score',
    '主要使用场景': 'Primary_Usage_Scenario',
    '使用态度': 'Usage_Attitude',
    '提问能力': 'Questioning_Ability',
    '信息核实': 'Information_Verification',
    '效率提升': 'Efficiency_Improvement',
    '学习渠道_1': 'Learning_Channel_1',
    '学习渠道_2': 'Learning_Channel_2',
    '学习渠道_3': 'Learning_Channel_3',
    '学习渠道_4': 'Learning_Channel_4',
    '学习渠道_5': 'Learning_Channel_5',
    'Q15_知识联系': 'Q15_Knowledge_Connection',
    'Q16_减轻负担': 'Q16_Workload_Reduction',
    'Q17_反思方法': 'Q17_Reflection_Methods',
    'Q18_批判审视': 'Q18_Critical_Examination',
    'Q19_问题解决': 'Q19_Problem_Solving',
    'Q20_深度思考': 'Q20_Deep_Thinking',
    'Q21_主动查证': 'Q21_Active_Verification',
    'Q22_激发创意': 'Q22_Creativity_Stimulation',
    'Q23_信息素养': 'Q23_Information_Literacy',
    'Q24_掌控感': 'Q24_Sense_of_Control',
    'Q25_胜任感': 'Q25_Competence',
    'Q26_归属感': 'Q26_Belonging',
    'Q27_兴趣提升': 'Q27_Interest_Increase',
    'Q28_减轻焦虑': 'Q28_Anxiety_Reduction',
    'Q29_依赖内疚': 'Q29_Dependency_Guilt',
    'Q30_担心落后': 'Q30_Fear_of_Falling_Behind',
    'Q31_探索乐趣': 'Q31_Exploration_Enjoyment',
    'Q32_弥补漏洞': 'Q32_Gap_Filling',
    'Q33_个性化': 'Q33_Personalization',
    'Q34_情感不可替代': 'Q34_Emotional_Irreplaceability',
    'Q35_教学方法改革': 'Q35_Teaching_Method_Reform',
    'Q36_融入课堂': 'Q36_AI_Integration',
    'Q37_注重过程': 'Q37_Process_Focus',
    'Q38_角色转变': 'Q38_Role_Change',
    'Q39_担心关注减少': 'Q39_Attention_Reduction_Concern',
    'Q40_利大于弊': 'Q40_Benefits_vs_Drawbacks',
    'Q41_平衡使用': 'Q41_Balanced_Usage',
    'Q42_职业重要': 'Q42_Career_Importance',
    'Q43_隐私担忧': 'Q43_Privacy_Concerns',
    'Q44_制定规则': 'Q44_Rule_Establishment',
    'Q45_支持课程': 'Q45_Course_Support',
    'Q46_减少讨论': 'Q46_Discussion_Reduction',
    'Q47_促进公平': 'Q47_Fairness_Promotion',
    'Q48_加剧不平等': 'Q48_Inequality_Exacerbation',
    'Q49_驾驭AI': 'Q49_AI_Mastery',
    'Q50_未来乐观': 'Q50_Future_Optimism',
    'Q51_思维不可替代': 'Q51_Thinking_Irreplaceability'
}

# Read data and rename columns to English
df = pd.read_csv('survey_data.csv')
df = df.rename(columns=column_mapping)

print(f"Data shape: {df.shape}")
print(f"Sample size: {len(df)}")

major_mapping = {
    '信息类（含计算机、电信、通信工程等）': 'IT Majors',
    '人文与艺术类（含文学、历史、哲学、新闻等）': 'Humanities & Arts',
    '经管与社科类（含经济、管理、法律、社会学等）': 'Economics & Social Sciences',
    '外语类': 'Foreign Languages',
    '物理与工程类（含物理、机械、电子、电气等）': 'Physics & Engineering',
    '数学与统计类': 'Mathematics & Statistics',
    '生化环材类（包含环境工程、化学、生物学、材料等）': 'Engineering'
}

usage_scene_mapping = {
    '学习辅助（如解题、写作、翻译等）': 'Learning Assistance',
    '娱乐休闲（如聊天、生成创意内容等）': 'Entertainment',
    '信息查询（如搜索、知识问答等）': 'Information Query',
    '其他': 'Other'
}

verification_mapping = {
    '总是会': 'Always',
    '经常会': 'Often', 
    '有时会': 'Sometimes',
    '很少会': 'Rarely',
    '从不': 'Never'
}

questioning_ability_mapping = {
    '非常强': 'Very Strong',
    '比较强': 'Somewhat Strong',
    '一般': 'Average',
    '比较弱': 'Somewhat Weak',
    '非常弱': 'Very Weak'
}

efficiency_mapping = {
    '提升非常大': 'Significant Improvement',
    '提升较大': 'Moderate Improvement',
    '一般': 'No Change',
    '提升较小': 'Slight Improvement',
    '没有提升': 'No Improvement'
}

usage_duration_mapping = {
    '几乎不用（少于1小时）': 'Rarely (<1h)',
    '较少使用（1-7小时）': 'Occasional (1-7h)',
    '中等使用（8-15小时）': 'Moderate (8-15h)',
    '频繁使用（16小时以上）': 'Frequent (16+h)'
}

attitude_mapping = {
    '非常消极': 'Very Negative',
    '比较消极': 'Somewhat Negative', 
    '中立': 'Neutral',
    '比较积极': 'Somewhat Positive',
    '非常积极': 'Very Positive'
}


# 1. Usage penetration analysis
def analyze_usage_penetration():
    """Analyze LLM usage penetration and dependency"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Usage duration distribution pie chart
    time_dist = df['Usage_Duration'].value_counts()
    time_order = ['几乎不用（少于1小时）', '较少使用（1-7小时）', '中等使用（8-15小时）', '频繁使用（16小时以上）']
    time_dist = time_dist.reindex(time_order)
    
    time_labels = [usage_duration_mapping[x] for x in time_order]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    ax1.pie(time_dist.values, labels=time_labels, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('LLM Usage Duration Distribution', fontsize=14, fontweight='bold')
    
    # Usage attitude distribution
    attitude_order = ['非常消极', '比较消极', '中立', '比较积极', '非常积极']
    attitude_dist = df['Usage_Attitude'].value_counts().reindex(attitude_order)
    attitude_labels = [attitude_mapping[x] for x in attitude_order]
    
    ax2.bar(attitude_labels, attitude_dist.values, color='skyblue', alpha=0.7)
    ax2.set_title('Student Attitudes Towards LLM Usage', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Students')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('eps_figures/usage_penetration.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate key metrics
    frequent_medium = (time_dist['频繁使用（16小时以上）'] + time_dist['中等使用（8-15小时）'])
    penetration_rate = frequent_medium / len(df) * 100
    positive_attitude = (df['Usage_Attitude'].isin(['非常积极', '比较积极'])).sum() / len(df) * 100
    
    print("="*50)
    print("1. LLM Usage Penetration Analysis")
    print("="*50)
    print(f"High frequency usage (8+ hours/week): {frequent_medium} students ({penetration_rate:.1f}%)")
    print(f"Positive attitude ratio: {positive_attitude:.1f}%")
    
    return penetration_rate


# 2. Major differences analysis 
def analyze_major_differences():
    """Analyze usage patterns and frequency differences across all majors"""
    
    print("\n" + "="*60)
    print("DEBUG: 检查原始数据分布")
    print("="*60)
    
    # 首先检查原始数据的分布
    print("原始'Primary_Usage_Scenario'列的值分布:")
    scenario_counts = df['Primary_Usage_Scenario'].value_counts()
    print(scenario_counts)
    print(f"总样本数: {len(df)}")
    print(f"唯一值数量: {df['Primary_Usage_Scenario'].nunique()}")
    
    # 查看前10个样本的具体值
    print("\n前10个样本的使用场景:")
    for i in range(min(10, len(df))):
        print(f"样本 {i+1}: '{df.iloc[i]['Primary_Usage_Scenario']}'")
    
    # 创建临时数据框进行映射
    temp_df = df.copy()
    
    # 根据您提供的信息更新使用场景映射字典
    usage_scene_mapping = {
        # 如果数据中是选项字母
        'A': 'Answering Course Questions',
        'B': 'Generating Ideas & Outlines', 
        'C': 'Text Polishing & Translation',
        'D': 'Programming Assistance',
        'E': 'Literature Review',
        # 如果数据中是完整文本
        '解答课程疑难问题': 'Answering Course Questions',
        '生成创意、获取灵感和写作大纲': 'Generating Ideas & Outlines',
        '文本润色、翻译与内容总结': 'Text Polishing & Translation',
        '编程辅助与代码调试': 'Programming Assistance',
        '搜集和概括文献资料': 'Literature Review',
        # 可能的其他值
        '其他': 'Other',
        '': 'Not Specified',
        None: 'Not Specified',
        np.nan: 'Not Specified',
    }
    
    # 安全的映射函数
    def safe_major_mapping(x):
        if pd.isna(x):
            return 'Other'
        x_str = str(x).strip()
        for key, value in major_mapping.items():
            if key.strip() == x_str:
                return value
        # 尝试部分匹配
        for key, value in major_mapping.items():
            if key in x_str or x_str in key:
                return value
        return 'Other'
    
    def safe_scene_mapping(x):
        if pd.isna(x):
            return 'Not Specified'
        x_str = str(x).strip()
        
        # 首先尝试直接匹配
        for key, value in usage_scene_mapping.items():
            if key.strip() == x_str:
                return value
        
        # 如果包含选项字母（如"A. 解答课程疑难问题"）
        if x_str.startswith(('A.', 'B.', 'C.', 'D.', 'E.')):
            option = x_str[0]
            if option in usage_scene_mapping:
                return usage_scene_mapping[option]
        
        # 尝试部分匹配
        for key, value in usage_scene_mapping.items():
            if key in x_str or x_str in key:
                return value
        
        return 'Other'
    
    temp_df['Major_English'] = temp_df['Major'].apply(safe_major_mapping)
    temp_df['Usage_Scenario_English'] = temp_df['Primary_Usage_Scenario'].apply(safe_scene_mapping)
    
    # 打印映射后的分布
    print(f"\n映射后专业分布:")
    print(temp_df['Major_English'].value_counts())
    print(f"\n映射后使用场景分布:")
    scenario_english_counts = temp_df['Usage_Scenario_English'].value_counts()
    print(scenario_english_counts)
    
    # 检查数据分布情况
    if scenario_english_counts.shape[0] == 1 and 'Other' in scenario_english_counts.index:
        print(f"\n警告：所有数据都被映射为'Other'，尝试直接查看原始值")
        print(f"原始数据前5个值:")
        for val in df['Primary_Usage_Scenario'].dropna().unique()[:5]:
            print(f"  '{val}'")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. 专业分布条形图 - 显示所有专业
    major_dist_english = temp_df['Major_English'].value_counts()
    
    if not major_dist_english.empty:
        # 使用所有可用的颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        bars = ax1.bar(range(len(major_dist_english)), major_dist_english.values, 
                      color=colors[:len(major_dist_english)], alpha=0.7)
        ax1.set_title('Student Distribution by Major', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Students')
        ax1.set_xticks(range(len(major_dist_english)))
        ax1.set_xticklabels(major_dist_english.index, rotation=45, ha='right')
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(major_dist_english.values):
            ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No major data available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Student Distribution by Major\n(No data)', fontsize=14, fontweight='bold')
    
    # 2. 使用场景分布 - 修复后的版本
    scene_dist_english = temp_df['Usage_Scenario_English'].value_counts()
    
    if not scene_dist_english.empty and len(scene_dist_english) > 1:
        # 如果是字母选项，按字母顺序排序
        if all(label[0] in ['A', 'B', 'C', 'D', 'E'] for label in scene_dist_english.index if len(label) == 1):
            # 如果是单个字母选项，按字母排序
            scene_order = sorted(scene_dist_english.index)
            scene_dist_english = scene_dist_english.reindex(scene_order)
        
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd700']
        ax2.barh(range(len(scene_dist_english)), scene_dist_english.values, 
                color=colors[:len(scene_dist_english)], alpha=0.7)
        ax2.set_title('Primary Usage Scenarios Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Students')
        ax2.set_yticks(range(len(scene_dist_english)))
        ax2.set_yticklabels(scene_dist_english.index)
        
        # 在条形图上添加数值标签
        for i, v in enumerate(scene_dist_english.values):
            ax2.text(v + 0.5, i, str(v), ha='left', va='center', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No usage scenario data available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Primary Usage Scenarios Distribution\n(No data)', fontsize=14, fontweight='bold')
    
    # 3. 专业vs使用场景热力图 - 包含所有专业
    if not temp_df.empty and temp_df['Usage_Scenario_English'].nunique() > 1:
        # 创建交叉表，包含所有专业和使用场景
        cross_tab_english = pd.crosstab(temp_df['Major_English'], temp_df['Usage_Scenario_English'], normalize='index') * 100
        
        if not cross_tab_english.empty and cross_tab_english.shape[1] > 1:
            # 确保热力图包含所有专业和使用场景
            sns.heatmap(cross_tab_english, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3, 
                       cbar_kws={'label': 'Percentage (%)'})
            ax3.set_title('Major vs Usage Scenarios (%)\n', fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Insufficient usage scenario data\nfor cross-tabulation', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=10)
            ax3.set_title('Major vs Usage Scenarios (%)\n(Insufficient data)', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No data available for analysis', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=10)
        ax3.set_title('Major vs Usage Scenarios (%)\n(No data)', fontsize=12, fontweight='bold')
    
    ax3.set_xlabel('Usage Scenarios')
    ax3.set_ylabel('Major')
    
    # 4. 所有专业的高使用率比较
    # 使用原始数据进行计算
    cross_time_major = pd.crosstab(df['Major'], df['Usage_Duration'], normalize='index') * 100
    
    if not cross_time_major.empty:
        # 计算每个专业的高使用率（中等使用+频繁使用）
        high_usage_by_major = {}
        
        for major in cross_time_major.index:
            high_usage = 0
            # 安全地获取列数据，避免KeyError
            for col in ['中等使用（8-15小时）', '频繁使用（16小时以上）']:
                if col in cross_time_major.columns:
                    high_usage += cross_time_major.loc[major, col]
            
            # 转换为英文专业名称
            english_major = safe_major_mapping(major)
            high_usage_by_major[english_major] = high_usage
        
        # 按高使用率排序
        if high_usage_by_major:
            sorted_majors = sorted(high_usage_by_major.items(), key=lambda x: x[1], reverse=True)
            majors_english = [item[0] for item in sorted_majors]
            high_usage_rates = [item[1] for item in sorted_majors]
            
            # 创建颜色映射
            colors = []
            for major in majors_english:
                if 'IT' in major:
                    colors.append('#1f77b4')  # 蓝色突出IT专业
                elif 'Humanities' in major:
                    colors.append('#ff7f0e')  # 橙色突出人文专业
                elif 'Engineering' in major:
                    colors.append('#2ca02c')  # 绿色突出工程专业
                else:
                    colors.append('#9467bd')  # 紫色表示其他专业
            
            # 绘制水平条形图
            y_pos = range(len(majors_english))
            bars = ax4.barh(y_pos, high_usage_rates, color=colors, alpha=0.7)
            ax4.set_xlabel('High Usage Rate (%)')
            ax4.set_ylabel('Major')
            ax4.set_title('High Usage Rate (8+ hours/week) by Major', fontsize=12, fontweight='bold')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(majors_english)
            
            # 在条形图上添加数值标签
            for i, v in enumerate(high_usage_rates):
                ax4.text(v + 0.5, i, f'{v:.1f}%', ha='left', va='center', fontweight='bold')
            
            # 添加图例说明颜色含义
            from matplotlib.patches import Patch
            legend_elements = []
            if any('IT' in major for major in majors_english):
                legend_elements.append(Patch(facecolor='#1f77b4', alpha=0.7, label='IT Majors'))
            if any('Humanities' in major for major in majors_english):
                legend_elements.append(Patch(facecolor='#ff7f0e', alpha=0.7, label='Humanities & Arts'))
            if any('Engineering' in major for major in majors_english):
                legend_elements.append(Patch(facecolor='#2ca02c', alpha=0.7, label='Engineering'))
            if legend_elements:
                ax4.legend(handles=legend_elements, loc='lower right')
    else:
        ax4.text(0.5, 0.5, 'No usage frequency data by major available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title('High Usage Rate by Major\n(No data)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eps_figures/major_differences.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 详细分析输出
    print("\n" + "="*50)
    print("2. Major Differences Analysis - All Majors")
    print("="*50)
    
    # 输出每个专业的使用场景分布
    print("\nPrimary Usage Scenarios by Major:")
    for major in temp_df['Major_English'].unique():
        major_data = temp_df[temp_df['Major_English'] == major]
        if not major_data.empty and len(major_data) > 0:
            scene_counts = major_data['Usage_Scenario_English'].value_counts()
            print(f"\n{major} (Total: {len(major_data)} students):")
            for scene, count in scene_counts.items():
                percentage = count / len(major_data) * 100
                print(f"  - {scene}: {count} students ({percentage:.1f}%)")
    
    # 输出高使用率分析
    if 'high_usage_by_major' in locals() and high_usage_by_major:
        print(f"\nHigh Usage Rate (8+ hours/week) by Major:")
        if high_usage_by_major:
            max_usage = max(high_usage_by_major.values())
            min_usage = min(high_usage_by_major.values())
            
            for major, rate in sorted(high_usage_by_major.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {major}: {rate:.1f}%")
            
            print(f"\nUsage Frequency Range: {min_usage:.1f}% to {max_usage:.1f}%")
            print(f"Usage Frequency Difference: {max_usage - min_usage:.1f} percentage points")
            
            # 识别最高和最低使用率的专业
            highest_major = max(high_usage_by_major.items(), key=lambda x: x[1])
            lowest_major = min(high_usage_by_major.items(), key=lambda x: x[1])
            
            print(f"\nHighest usage: {highest_major[0]} ({highest_major[1]:.1f}%)")
            print(f"Lowest usage: {lowest_major[0]} ({lowest_major[1]:.1f}%)")
            
            return highest_major[1], lowest_major[1]
    
    return 0, 0
    
# 3. Critical information literacy analysis
def analyze_critical_literacy():
    """Analyze students' critical information literacy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Information verification frequency distribution
    verify_order = ['总是会', '经常会', '有时会', '很少会', '从不']
    verify_dist = df['Information_Verification'].value_counts().reindex(verify_order)
    verify_labels = [verification_mapping[x] for x in verify_order]
    
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
    bars = ax1.bar(verify_labels, verify_dist.values, color=colors, alpha=0.7)
    ax1.set_title('Information Verification Behavior Frequency', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Students')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Calculate risk ratio
    rarely_never = verify_dist['很少会'] + verify_dist['从不']
    total = verify_dist.sum()
    risk_percent = rarely_never / total * 100
    
    ax1.text(0.02, 0.95, f'Rarely/Never verify: {risk_percent:.1f}%', 
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    
    # Critical thinking related indicators
    critical_cols = ['Q18_Critical_Examination', 'Q21_Active_Verification']
    critical_names = ['Critical Examination', 'Active Verification']
    critical_data = []
    
    for col, name in zip(critical_cols, critical_names):
        low_critical = (df[col].isin(['D', 'E'])).sum()
        critical_percent = low_critical / len(df) * 100
        critical_data.append((name, critical_percent))
    
    x_pos = np.arange(len(critical_data))
    ax2.bar(x_pos, [data[1] for data in critical_data], color=['#e74c3c', '#3498db'], alpha=0.7)
    ax2.set_title('Critical Thinking Indicators: Negative Ratios', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Negative Ratio (%)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([data[0] for data in critical_data])
    
    # Add value labels on bars
    for i, v in enumerate([data[1] for data in critical_data]):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eps_figures/critical_literacy.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("3. Critical Information Literacy Analysis")
    print("="*50)
    print(f"Always verify information: {verify_dist['总是会']} students ({verify_dist['总是会']/total*100:.1f}%)")
    print(f"Often verify information: {verify_dist['经常会']} students ({verify_dist['经常会']/total*100:.1f}%)")
    print(f"Rarely or never verify: {rarely_never} students ({risk_percent:.1f}%)")
    print(f"Insufficient critical examination: {critical_data[0][1]:.1f}%")
    print(f"Lack of active verification: {critical_data[1][1]:.1f}%")
    
    return risk_percent

# 4. Metacognition awareness analysis
def analyze_metacognition():
    """Analyze metacognition awareness changes"""
    metacognition_cols = ['Q17_Reflection_Methods', 'Q18_Critical_Examination', 'Q20_Deep_Thinking', 'Q21_Active_Verification']
    col_names = ['Reflection Methods', 'Critical Examination', 'Deep Thinking', 'Active Verification']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metacognition_data = []
    
    for i, (col, name) in enumerate(zip(metacognition_cols, col_names)):
        dist = df[col].value_counts().reindex(['A', 'B', 'C', 'D', 'E'])
        negative_count = (df[col].isin(['D', 'E'])).sum()
        negative_percent = negative_count / len(df) * 100
        metacognition_data.append((name, negative_percent))
        
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        bars = axes[i].bar(dist.index, dist.values, color=colors, alpha=0.7)
        axes[i].set_title(f'{name}\nNegative Ratio: {negative_percent:.1f}%', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Number of Students')
        axes[i].set_xlabel('Response Scale (A=Strongly Agree, E=Strongly Disagree)')
        
        # Add values on bars
        for j, v in enumerate(dist.values):
            axes[i].text(j, v + 3, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eps_figures/metacognition.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate metacognition decline comprehensive indicator
    negative_responses = 0
    for col in metacognition_cols:
        negative_responses += (df[col].isin(['D', 'E'])).sum()
    
    avg_negative_per_col = negative_responses / len(metacognition_cols)
    overall_negative_percent = (avg_negative_per_col / len(df)) * 100
    
    print("\n" + "="*50)
    print("4. Metacognition Awareness Analysis")
    print("="*50)
    for name, percent in metacognition_data:
        print(f"{name} negative ratio: {percent:.1f}%")
    
    print(f"\nMetacognition awareness decline comprehensive ratio: {overall_negative_percent:.1f}%")
    
    return overall_negative_percent

# 8. Educational inequality analysis
def analyze_educational_inequality():
    """Analyze educational inequality phenomena"""
    # Create a temporary dataframe with English values for plotting
    temp_df = df.copy()
    temp_df['Major_English'] = temp_df['Major'].map(major_mapping)
    temp_df['Usage_Duration_English'] = temp_df['Usage_Duration'].map(usage_duration_mapping)
    temp_df['Questioning_English'] = temp_df['Questioning_Ability'].map(questioning_ability_mapping)
    temp_df['Efficiency_English'] = temp_df['Efficiency_Improvement'].map(efficiency_mapping)
    
    # Filter out rows with NaN values in the mapped columns
    temp_df = temp_df.dropna(subset=['Major_English', 'Usage_Duration_English', 'Questioning_English', 'Efficiency_English'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Major vs usage frequency heatmap - Use English mapping directly
    cross_time_major_english = pd.crosstab(temp_df['Major_English'], temp_df['Usage_Duration_English'], normalize='index') * 100
    
    # Check if cross_time_major_english is empty
    if cross_time_major_english.empty:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Usage Frequency Distribution by Major (%)\n(No data available)', fontsize=12, fontweight='bold')
    else:
        sns.heatmap(cross_time_major_english, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax1)
        ax1.set_title('Usage Frequency Distribution by Major (%)\n', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Usage Duration')
    ax1.set_ylabel('Major')
    
    # Skill difference analysis - Calculate average questioning ability by major
    major_ability_english = temp_df.groupby('Major_English')['Questioning_English'].apply(
        lambda x: (x.isin(['Very Strong', 'Somewhat Strong'])).mean() * 100
    ).sort_values(ascending=False)
    
    # Check if major_ability_english is empty
    if major_ability_english.empty:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Questioning Ability by Major (%)\n(No data available)', fontsize=12, fontweight='bold')
    else:
        colors_ability = ['purple' if 'IT' in major else 'orange' if 'Humanities' in major else 'gray' for major in major_ability_english.index]
        ax2.barh(major_ability_english.index, major_ability_english.values, color=colors_ability, alpha=0.7)
        ax2.set_title('Questioning Ability (Strong+Somewhat Strong) by Major (%)\n', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Ratio (%)')
    ax2.set_ylabel('Major')
    
    # Efficiency improvement major differences - Use English mapping directly
    major_efficiency_english = temp_df.groupby('Major_English')['Efficiency_English'].apply(
        lambda x: (x.isin(['Significant Improvement', 'Moderate Improvement'])).mean() * 100
    ).sort_values(ascending=False)
    
    # Check if major_efficiency_english is empty
    if major_efficiency_english.empty:
        ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Efficiency Improvement by Major (%)\n(No data available)', fontsize=12, fontweight='bold')
    else:
        colors_eff = ['blue' if 'IT' in major else 'red' if 'Humanities' in major else 'gray' for major in major_efficiency_english.index]
        ax3.barh(major_efficiency_english.index, major_efficiency_english.values, color=colors_eff, alpha=0.7)
        ax3.set_title('Efficiency Improvement Perception by Major (%)\n', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Ratio (%)')
    ax3.set_ylabel('Major')
    
    # Digital divide visualization
    info_major = '信息类（含计算机、电信、通信工程等）'
    human_major = '人文与艺术类（含文学、历史、哲学、新闻等）'
    
    # Get original data for calculations
    cross_time_major = pd.crosstab(df['Major'], df['Usage_Duration'], normalize='index') * 100
    major_ability = df.groupby('Major')['Questioning_Ability'].apply(lambda x: (x.isin(['非常强', '比较强'])).mean() * 100)
    major_efficiency = df.groupby('Major')['Efficiency_Improvement'].apply(lambda x: (x.isin(['提升非常大', '提升较大'])).mean() * 100)
    
    categories = ['High Usage Rate', 'Strong Questioning Ability', 'Efficiency Improvement']
    info_scores = [0, 0, 0]
    human_scores = [0, 0, 0]
    
    if info_major in cross_time_major.index:
        info_scores[0] = cross_time_major.loc[info_major, ['频繁使用（16小时以上）', '中等使用（8-15小时）']].sum()
    if human_major in cross_time_major.index:
        human_scores[0] = cross_time_major.loc[human_major, ['频繁使用（16小时以上）', '中等使用（8-15小时）']].sum()
    
    if info_major in major_ability.index:
        info_scores[1] = major_ability[info_major]
    if human_major in major_ability.index:
        human_scores[1] = major_ability[human_major]
    
    if info_major in major_efficiency.index:
        info_scores[2] = major_efficiency[info_major]
    if human_major in major_efficiency.index:
        human_scores[2] = major_efficiency[human_major]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax4.bar(x - width/2, info_scores, width, label='IT Majors', color='blue', alpha=0.7)
    ax4.bar(x + width/2, human_scores, width, label='Humanities & Arts', color='red', alpha=0.7)
    ax4.set_xlabel('Indicator Category')
    ax4.set_ylabel('Ratio (%)')
    ax4.set_title('Digital Divide: IT Majors vs Humanities & Arts')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('eps_figures/analyze_educational_inequality.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    # Calculate inequality indicators
    usage_gap = info_scores[0] - human_scores[0]
    ability_gap = info_scores[1] - human_scores[1]
    efficiency_gap = info_scores[2] - human_scores[2]
    
    print("\n" + "="*50)
    print("8. Educational Inequality Analysis")
    print("="*50)
    print(f"IT majors high usage ratio: {info_scores[0]:.1f}%")
    print(f"Humanities & Arts majors high usage ratio: {human_scores[0]:.1f}%")
    print(f"Usage frequency difference: {usage_gap:.1f} percentage points")
    print(f"\nQuestioning ability difference: {ability_gap:.1f} percentage points")
    print(f"Efficiency improvement perception difference: {efficiency_gap:.1f} percentage points")
    print(f"\nDigital divide comprehensive index: {(abs(usage_gap) + abs(ability_gap) + abs(efficiency_gap)) / 3:.1f} points")
    
    return usage_gap

# 6. Subjective evaluation vs actual ability contradiction analysis
def analyze_contradiction():
    """Analyze contradiction between subjective evaluation and actual ability"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Benefits vs drawbacks evaluation distribution
    benefit_dist = df['Q40_Benefits_vs_Drawbacks'].value_counts().reindex(['A', 'B', 'C', 'D', 'E'])
    positive_benefit = (df['Q40_Benefits_vs_Drawbacks'].isin(['A', 'B'])).sum()
    positive_percent = positive_benefit / len(df) * 100
    
    colors_benefit = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    ax1.bar(benefit_dist.index, benefit_dist.values, color=colors_benefit, alpha=0.7)
    ax1.set_title(f'Benefits vs Drawbacks Evaluation\nPositive Evaluation: {positive_percent:.1f}%', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Students')
    ax1.set_xlabel('Response Scale (A=Strongly Agree, E=Strongly Disagree)')
    
    # Problem-solving ability distribution
    problem_solve_dist = df['Q19_Problem_Solving'].value_counts().reindex(['A', 'B', 'C', 'D', 'E'])
    negative_solve = (df['Q19_Problem_Solving'].isin(['D', 'E'])).sum()
    negative_solve_percent = negative_solve / len(df) * 100
    
    colors_solve = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    ax2.bar(problem_solve_dist.index, problem_solve_dist.values, color=colors_solve, alpha=0.7)
    ax2.set_title(f'Problem-Solving Ability\nAbility Decline: {negative_solve_percent:.1f}%', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Students')
    ax2.set_xlabel('Response Scale (A=Strongly Agree, E=Strongly Disagree)')
    
    plt.tight_layout()
    plt.savefig('eps_figures/contradiction_analysis.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cross-analysis: Contradiction group (positive evaluation but ability decline)
    contradiction_group = df[(df['Q40_Benefits_vs_Drawbacks'].isin(['A', 'B'])) & 
                           (df['Q19_Problem_Solving'].isin(['D', 'E']))]
    contradiction_percent = len(contradiction_group) / len(df) * 100
    
    print("\n" + "="*50)
    print("6. Subjective Evaluation vs Objective Ability Contradiction Analysis")
    print("="*50)
    print(f"Believe LLM benefits outweigh drawbacks: {positive_percent:.1f}%")
    print(f"Problem-solving ability decline: {negative_solve_percent:.1f}%")
    print(f"Contradiction group (positive evaluation but ability decline): {contradiction_percent:.1f}%")
    
    return contradiction_percent

# Main function - Execute all analyses
def main():
    """Execute all analyses"""
    print("Starting comprehensive analysis of LLM usage...")
    print("="*60)
    
    # Execute all analyses and collect results
    results = []
    
    # 1. Usage penetration analysis
    penetration = analyze_usage_penetration()
    results.append(penetration)
    
    # 2. Major differences analysis
    info_high_usage, human_high_usage = analyze_major_differences()  # 解包元组
    usage_gap = abs(info_high_usage - human_high_usage)  # 计算使用频率差异
    results.append(usage_gap)
    
    # 3. Critical information literacy analysis
    info_risk = analyze_critical_literacy()
    results.append(info_risk)
    
    # 4. Metacognition awareness analysis
    metacognition_decline = analyze_metacognition()
    results.append(metacognition_decline)
    
    # 5. Educational inequality analysis
    inequality_gap = analyze_educational_inequality()  # 重命名变量以避免混淆
    results.append(inequality_gap)
    
    # 6. Subjective evaluation vs actual ability contradiction analysis
    contradiction = analyze_contradiction()
    results.append(contradiction)
    
    # Final summary report
    print("\n" + "="*80)
    print("Comprehensive Analysis Summary Report: LLM Usage Among Students")
    print("="*80)
    
    summary_points = [
        f"1. Usage Penetration: {penetration:.1f}% students high frequency usage (8+ hours/week)",
        f"2. Major Differences: {usage_gap:.1f} percentage points usage gap between IT and Humanities majors",
        f"3. Information Literacy: {info_risk:.1f}% students rarely or never verify information",
        f"4. Metacognition Decline: {metacognition_decline:.1f}% students show metacognition awareness decline",
        f"5. Educational Inequality: {inequality_gap:.1f} percentage points questioning ability gap between majors",
        f"6. Subjective-Objective Contradiction: {contradiction:.1f}% students show evaluation-ability mismatch"
    ]
    
    for point in summary_points:
        print(point)
    
    print(f"\nAll figures have been saved as EPS files in the 'eps_figures' directory.")
    print("="*80)

# Execute main function
if __name__ == "__main__":
    main()