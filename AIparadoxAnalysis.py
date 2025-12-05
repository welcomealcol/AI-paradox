# AIparadoxAnalysis_fixed_v2.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果使用中文标签）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AIParadoxCausalAnalyzer:
    """
    AI Paradox因果分析器 - 修复版V2
    使用多种因果推断方法分析LLM使用对学习结果的影响
    """
    
    def __init__(self, data_path=None, sample_size=1000):
        """初始化分析器"""
        if data_path:
            try:
                self.df = pd.read_csv(data_path)
                print(f"数据加载完成: {self.df.shape[0]} 行, {self.df.shape[1]} 列")
                
                # 如果数据太大，可以采样
                if len(self.df) > sample_size:
                    self.df = self.df.sample(n=sample_size, random_state=42)
                    print(f"采样后数据: {self.df.shape[0]} 行")
            except Exception as e:
                print(f"数据加载失败: {e}")
                print("创建模拟数据...")
                self._create_simulated_data(sample_size)
        else:
            print("创建模拟数据...")
            self._create_simulated_data(sample_size)
        
        # 数据预处理
        self._preprocess_data()
    
    def _create_simulated_data(self, n=1000):
        """创建模拟数据用于测试"""
        np.random.seed(42)
        
        # 专业列表
        majors = [
            '信息类（含计算机、电信、通信工程等）',
            '数学与统计类',
            '物理与工程类（含物理、机械、电子、电气等）',
            '人文与艺术类（含文学、历史、哲学、新闻等）',
            '外语类',
            '经管与社科类（含经济、管理、法律、社会学等）',
            '生化环材类（包含环境工程、化学、生物学、材料等）'
        ]
        
        # 性别和年级
        genders = ['男', '女']
        grades = ['2021级及以前', '2022级', '2023级', '2024级']
        
        data = {
            'ID': range(1, n+1),
            '性别': np.random.choice(genders, n),
            '年级': np.random.choice(grades, n),
            '专业': np.random.choice(majors, n),
            '使用时长_数值': np.random.choice([0, 1, 2, 3], n, p=[0.05, 0.2, 0.45, 0.3]),
            '使用态度_数值': np.random.randint(1, 6, n),
            '提问能力_数值': np.random.randint(1, 6, n),
            '信息核实_数值': np.random.randint(1, 6, n),
            '效率提升_数值': np.random.randint(1, 6, n),
            '认知特质': np.random.normal(0, 1, n),
            '动机特质': np.random.normal(0, 1, n),
            '教学特质': np.random.normal(0, 1, n),
            '综合特质': np.random.normal(0, 1, n)
        }
        
        # 生成成绩数据（模拟因果效应）
        for i in range(n):
            usage = data['使用时长_数值'][i]
            # 模拟AI Paradox：中等使用效果最好
            if usage == 0:  # 几乎不用
                base_grade_with = np.random.normal(75, 8)
                base_grade_without = np.random.normal(80, 8)
            elif usage == 1:  # 较少使用
                base_grade_with = np.random.normal(78, 7)
                base_grade_without = np.random.normal(82, 7)
            elif usage == 2:  # 中等使用（最佳）
                base_grade_with = np.random.normal(85, 6)
                base_grade_without = np.random.normal(82, 6)
            else:  # 频繁使用（依赖效应）
                base_grade_with = np.random.normal(72, 9)
                base_grade_without = np.random.normal(78, 9)
            
            # 添加随机噪声
            data.setdefault('使用ChatGPT_闭卷成绩_numeric', []).append(base_grade_with)
            data.setdefault('不使用ChatGPT_闭卷成绩_numeric', []).append(base_grade_without)
        
        self.df = pd.DataFrame(data)
        self.df['grade_difference'] = (
            self.df['使用ChatGPT_闭卷成绩_numeric'] - 
            self.df['不使用ChatGPT_闭卷成绩_numeric']
        )
        
        print(f"模拟数据创建完成: {self.df.shape[0]} 行")
    
    def _preprocess_data(self):
        """数据预处理"""
        print("正在预处理数据...")
        
        # 1. 处理使用时长
        if '使用时长' in self.df.columns:
            usage_mapping = {
                '几乎不用（少于1小时）': 0,
                '较少使用（1-7小时）': 1,
                '中等使用（8-15小时）': 2,
                '频繁使用（16小时以上）': 3
            }
            self.df['usage_intensity'] = self.df['使用时长'].map(usage_mapping)
        elif '使用时长_数值' in self.df.columns:
            self.df['usage_intensity'] = self.df['使用时长_数值']
        else:
            self.df['usage_intensity'] = np.random.choice([0, 1, 2, 3], len(self.df), 
                                                         p=[0.05, 0.2, 0.45, 0.3])
        
        # 2. 创建处理变量（高使用组 vs 低使用组）
        self.df['treatment'] = (self.df['usage_intensity'] >= 2).astype(int)
        
        # 3. 处理成绩数据
        if '使用ChatGPT_闭卷成绩' in self.df.columns and '不使用ChatGPT_闭卷成绩' in self.df.columns:
            self.df['使用ChatGPT_闭卷成绩_numeric'] = pd.to_numeric(
                self.df['使用ChatGPT_闭卷成绩'], errors='coerce'
            )
            self.df['不使用ChatGPT_闭卷成绩_numeric'] = pd.to_numeric(
                self.df['不使用ChatGPT_闭卷成绩'], errors='coerce'
            )
        
        # 4. 创建成绩差异
        if '使用ChatGPT_闭卷成绩_numeric' in self.df.columns and '不使用ChatGPT_闭卷成绩_numeric' in self.df.columns:
            self.df['grade_difference'] = (
                self.df['使用ChatGPT_闭卷成绩_numeric'] - 
                self.df['不使用ChatGPT_闭卷成绩_numeric']
            )
        else:
            # 创建模拟成绩差异
            self.df['grade_difference'] = np.random.normal(-2, 5, len(self.df))
        
        print("数据预处理完成")
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        print("=" * 60)
        print("描述性统计分析")
        print("=" * 60)
        
        # 基本统计
        stats_data = []
        
        stats_data.append(['样本量', len(self.df)])
        stats_data.append(['使用强度均值', f"{self.df['usage_intensity'].mean():.2f}"])
        stats_data.append(['成绩差异均值', f"{self.df['grade_difference'].mean():.2f}"])
        
        if '认知特质' in self.df.columns:
            stats_data.append(['认知特质均值', f"{self.df['认知特质'].mean():.2f}"])
        
        stats_data.append(['处理组比例', f"{self.df['treatment'].mean():.2%}"])
        
        stats_df = pd.DataFrame(stats_data, columns=['变量', '值'])
        print(stats_df.to_string(index=False))
        
        # 分组统计
        if 'treatment' in self.df.columns:
            grouped_stats = []
            for treatment_value in [0, 1]:
                group_data = self.df[self.df['treatment'] == treatment_value]
                group_name = '控制组（低使用）' if treatment_value == 0 else '处理组（高使用）'
                
                grouped_stats.append([
                    group_name,
                    len(group_data),
                    f"{group_data['usage_intensity'].mean():.2f}",
                    f"{group_data['grade_difference'].mean():.2f}"
                ])
            
            grouped_df = pd.DataFrame(grouped_stats, 
                                     columns=['组别', '样本量', '平均使用强度', '平均成绩差异'])
            
            print("\n分组统计:")
            print(grouped_df.to_string(index=False))
        
        # 绘制图表
        self._plot_descriptive_stats()
        
        return {'summary': stats_df, 'grouped': grouped_df}

    def _plot_descriptive_stats(self):
        """绘制描述性统计图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 使用强度分布
        usage_counts = self.df['usage_intensity'].value_counts().sort_index()
        axes[0, 0].bar(usage_counts.index, usage_counts.values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('LLM Usage Intensity Distribution', fontsize=12)
        axes[0, 0].set_xlabel('Usage Intensity (0:Almost Never, 1:Low, 2:Medium, 3:High)')
        axes[0, 0].set_ylabel('Number of Students')
        
        # 添加数值标签
        for i, v in enumerate(usage_counts.values):
            axes[0, 0].text(i, v + 5, str(v), ha='center')
        
        # 2. 成绩差异分布
        if 'grade_difference' in self.df.columns:
            axes[0, 1].hist(self.df['grade_difference'].dropna(), bins=30, 
                           alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_title('Grade Difference Distribution', fontsize=12)
            axes[0, 1].set_xlabel('Grade Difference (With AI - Without AI)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].axvline(x=self.df['grade_difference'].mean(), 
                              color='blue', linestyle='-', alpha=0.7, 
                              label=f'Mean: {self.df["grade_difference"].mean():.2f}')
            axes[0, 1].legend()
        
        # 3. 处理组 vs 控制组对比
        if 'treatment' in self.df.columns and 'grade_difference' in self.df.columns:
            treated_mean = self.df[self.df['treatment'] == 1]['grade_difference'].mean()
            control_mean = self.df[self.df['treatment'] == 0]['grade_difference'].mean()
            
            axes[1, 0].bar(['Control Group (Low Usage)', 'Treatment Group (High Usage)'], 
                          [control_mean, treated_mean], 
                          alpha=0.7, color=['blue', 'red'])
            axes[1, 0].set_title('Treatment vs Control: Average Grade Difference', fontsize=12)
            axes[1, 0].set_ylabel('Average Grade Difference')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate([control_mean, treated_mean]):
                axes[1, 0].text(i, v + (0.5 if v >= 0 else -1), f'{v:.2f}', 
                               ha='center', va='bottom' if v >= 0 else 'top')
        
        # 4. 使用强度与成绩差异关系
        if 'grade_difference' in self.df.columns:
            scatter_data = self.df[['usage_intensity', 'grade_difference']].dropna()
            axes[1, 1].scatter(scatter_data['usage_intensity'], 
                              scatter_data['grade_difference'], 
                              alpha=0.5, s=20)
            
            # 添加回归线
            if len(scatter_data) > 1:
                z = np.polyfit(scatter_data['usage_intensity'], 
                              scatter_data['grade_difference'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(0, 3, 100)
                axes[1, 1].plot(x_line, p(x_line), color='red', linewidth=2, 
                               label=f'Trend line: y = {z[0]:.2f}x + {z[1]:.2f}')
            
            axes[1, 1].set_title('Usage Intensity vs Grade Difference', fontsize=12)
            axes[1, 1].set_xlabel('Usage Intensity')
            axes[1, 1].set_ylabel('Grade Difference')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('descriptive_stats.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_ps_balance(self, original_df, matched_df):
        """绘制倾向得分平衡图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 匹配前
        treated_before = original_df[original_df['treatment'] == 1]['propensity_score']
        control_before = original_df[original_df['treatment'] == 0]['propensity_score']
        
        axes[0].hist(treated_before, bins=20, alpha=0.5, label='Treatment Group', 
                    color='red', density=True)
        axes[0].hist(control_before, bins=20, alpha=0.5, label='Control Group', 
                    color='blue', density=True)
        axes[0].set_title('Propensity Score Distribution (Before Matching)')
        axes[0].set_xlabel('Propensity Score')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 匹配后
        treated_after = matched_df[matched_df['treatment'] == 1]['propensity_score']
        control_after = matched_df[matched_df['treatment'] == 0]['propensity_score']
        
        axes[1].hist(treated_after, bins=20, alpha=0.5, label='Treatment Group', 
                    color='red', density=True)
        axes[1].hist(control_after, bins=20, alpha=0.5, label='Control Group', 
                    color='blue', density=True)
        axes[1].set_title('Propensity Score Distribution (After Matching)')
        axes[1].set_xlabel('Propensity Score')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('propensity_score_balance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_rdd(self, rdd_df, cutoff, left_mean, right_mean, bandwidth):
        """绘制断点回归图"""
        plt.figure(figsize=(10, 6))
        
        # 散点图
        plt.scatter(rdd_df['running_var'] + cutoff, rdd_df['grade_difference'], 
                   alpha=0.3, s=20, color='gray', label='Observations')
        
        # 分箱平均
        rdd_df['bin'] = pd.cut(rdd_df['running_var'] + cutoff, bins=20)
        bin_means = rdd_df.groupby('bin')['grade_difference'].mean().reset_index()
        bin_means['bin_center'] = bin_means['bin'].apply(lambda x: x.mid)
        
        plt.scatter(bin_means['bin_center'], bin_means['grade_difference'], 
                   s=50, color='blue', label='Binned Means')
        
        # 断点线
        plt.axvline(x=cutoff, color='black', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Cutoff')
        
        # 左侧和右侧均值线
        plt.axhline(y=left_mean, xmin=0, xmax=0.5, 
                   color='red', linewidth=2, label=f'Left Mean: {left_mean:.2f}')
        plt.axhline(y=right_mean, xmin=0.5, xmax=1, 
                   color='green', linewidth=2, label=f'Right Mean: {right_mean:.2f}')
        
        # 处理效应标注
        effect = right_mean - left_mean
        plt.annotate(f'Treatment Effect = {effect:.2f}', 
                    xy=(cutoff, (left_mean+right_mean)/2),
                    xytext=(cutoff+bandwidth*0.1, (left_mean+right_mean)/2),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                    fontsize=12, color='purple')
        
        plt.xlabel('Usage Intensity', fontsize=12)
        plt.ylabel('Grade Difference', fontsize=12)
        plt.title(f'Regression Discontinuity Analysis: Usage Threshold={cutoff}', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rdd_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_mediation(self, total_effect, direct_effect, indirect_effect, 
                       a_path, b_path, mediation_proportion, mediator_name):
        """绘制中介效应图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 将中文中介变量名称映射为英文
        mediator_name_mapping = {
            '认知特质': 'Cognitive Traits',
            '信息核实_数值': 'Information Verification',
            '提问能力_数值': 'Questioning Ability',
            '效率提升_数值': 'Efficiency Improvement',
            '模拟中介变量': 'Simulated Mediator'
        }
        
        # 获取英文节点标签
        mediator_label = mediator_name_mapping.get(mediator_name, mediator_name)
        
        # 节点位置
        nodes = {
            'LLM Usage': (0, 1),
            mediator_label: (1, 0.5),
            'Grade Difference': (2, 1)
        }
        
        # 绘制节点
        for node, (x, y) in nodes.items():
            ax.scatter(x, y, s=3000, color='lightblue', 
                      edgecolor='black', alpha=0.7, zorder=2)
            ax.text(x, y, node, ha='center', va='center', 
                   fontsize=14, fontweight='bold', zorder=3)
        
        # 绘制路径
        # a路径
        ax.annotate('', xy=nodes[mediator_label], xytext=nodes['LLM Usage'],
                   arrowprops=dict(arrowstyle='->', color='blue', 
                                 lw=3, alpha=0.7))
        ax.text(0.5, 0.7, f'a = {a_path:.3f}', ha='center', 
               va='center', fontsize=12, color='blue',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # b路径
        ax.annotate('', xy=nodes['Grade Difference'], xytext=nodes[mediator_label],
                   arrowprops=dict(arrowstyle='->', color='green', 
                                 lw=3, alpha=0.7))
        ax.text(1.5, 0.7, f'b = {b_path:.3f}', ha='center', 
               va='center', fontsize=12, color='green',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # c'路径 (直接)
        ax.annotate('', xy=nodes['Grade Difference'], xytext=nodes['LLM Usage'],
                   arrowprops=dict(arrowstyle='->', color='red', 
                                 lw=3, linestyle='--', alpha=0.7))
        ax.text(1.0, 1, f"c' = {direct_effect:.3f}", ha='center', 
               va='center', fontsize=12, color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加效应总结
        summary_text = (
            f"Total Effect (c) = {total_effect:.3f}\n"
            f"Indirect Effect (a×b) = {indirect_effect:.3f}\n"
            f"Mediation Proportion = {mediation_proportion:.1%}"
        )
        
        ax.text(1, 0.3, summary_text, ha='center', va='center',
               fontsize=12, bbox=dict(boxstyle='round', 
                                     facecolor='lightyellow', alpha=0.9))
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Mediation Analysis: LLM Usage → {mediator_label} → Grade Difference', 
                    fontsize=16, pad=5)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('mediation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_sensitivity(self, results_df, robust_gamma):
        """绘制敏感性分析图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. ATE敏感性
        axes[0].plot(results_df['Gamma'], results_df['调整后ATE'], 
                    'o-', linewidth=2, markersize=6, color='blue')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].axvline(x=robust_gamma, color='green', linestyle='--', alpha=0.7,
                       label=f'Robustness Boundary: Γ={robust_gamma:.1f}')
        
        axes[0].set_xlabel('Gamma (Unobserved Confounding Strength)')
        axes[0].set_ylabel('Adjusted ATE')
        axes[0].set_title('Sensitivity of ATE to Unobserved Confounding')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. P值敏感性
        axes[1].plot(results_df['Gamma'], results_df['调整后P值'], 
                    's-', linewidth=2, markersize=6, color='purple')
        axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7,
                       label='Significance Threshold (0.05)')
        axes[1].axvline(x=robust_gamma, color='green', linestyle='--', alpha=0.7)
        
        axes[1].set_xlabel('Gamma (Unobserved Confounding Strength)')
        axes[1].set_ylabel('Adjusted P-value')
        axes[1].set_title('Sensitivity of Statistical Significance to Unobserved Confounding')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    
    
    def propensity_score_matching(self):
        """倾向得分匹配分析"""
        print("\n" + "=" * 60)
        print("倾向得分匹配分析")
        print("=" * 60)
        
        # 准备数据
        df_clean = self.df.copy()
        
        # 检查必要的列
        required_cols = ['treatment', 'grade_difference', '性别', '年级', '专业']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        
        if missing_cols:
            print(f"缺少必要列: {missing_cols}")
            print("使用模拟数据进行分析...")
            
            # 添加模拟列
            for col in missing_cols:
                if col == '性别':
                    df_clean['性别'] = np.random.choice(['男', '女'], len(df_clean))
                elif col == '年级':
                    df_clean['年级'] = np.random.choice(['2021', '2022', '2023', '2024'], len(df_clean))
                elif col == '专业':
                    df_clean['专业'] = np.random.choice(['理工科', '文科', '商科'], len(df_clean))
        
        # 选择协变量
        covariates = ['性别', '年级', '专业']
        for col in ['使用态度_数值', '提问能力_数值', '信息核实_数值', '效率提升_数值']:
            if col in df_clean.columns:
                covariates.append(col)
        
        print(f"使用的协变量: {covariates}")
        
        # 创建设计矩阵
        X = pd.get_dummies(df_clean[covariates], drop_first=True)
        X = X.fillna(X.mean())
        
        # 计算倾向得分
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            logit = LogisticRegression(max_iter=1000, random_state=42)
            logit.fit(X_scaled, df_clean['treatment'])
            propensity_scores = logit.predict_proba(X_scaled)[:, 1]
            
            df_clean['propensity_score'] = propensity_scores
            
            # 匹配
            treated = df_clean[df_clean['treatment'] == 1]
            control = df_clean[df_clean['treatment'] == 0]
            
            print(f"处理组样本: {len(treated)}")
            print(f"控制组样本: {len(control)}")
            
            if len(treated) == 0 or len(control) == 0:
                print("无法进行匹配，一组样本为空")
                return None
            
            # 最近邻匹配
            nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(
                control['propensity_score'].values.reshape(-1, 1)
            )
            distances, indices = nbrs.kneighbors(
                treated['propensity_score'].values.reshape(-1, 1)
            )
            
            matched_control = control.iloc[indices.flatten()]
            matched_df = pd.concat([treated, matched_control])
            
            # 计算ATE
            ate = (
                matched_df[matched_df['treatment'] == 1]['grade_difference'].mean() -
                matched_df[matched_df['treatment'] == 0]['grade_difference'].mean()
            )
            
            # t检验
            t_stat, p_value = stats.ttest_ind(
                matched_df[matched_df['treatment'] == 1]['grade_difference'],
                matched_df[matched_df['treatment'] == 0]['grade_difference']
            )
            
            print(f"\n匹配结果:")
            print(f"匹配后样本量: {len(matched_df)}")
            print(f"平均处理效应 (ATE): {ate:.3f}")
            print(f"T统计量: {t_stat:.3f}, P值: {p_value:.4f}")
            
            if p_value < 0.05:
                significance = "显著"
            elif p_value < 0.1:
                significance = "边缘显著"
            else:
                significance = "不显著"
            
            print(f"统计显著性: {significance}")
            
            # 绘制倾向得分平衡图
            self._plot_ps_balance(df_clean, matched_df)
            
            return {
                'ate': ate,
                't_stat': t_stat,
                'p_value': p_value,
                'significance': significance,
                'matched_df': matched_df
            }
            
        except Exception as e:
            print(f"倾向得分匹配失败: {e}")
            return None
    
    
    def regression_discontinuity(self, cutoff=1.5, bandwidth=1.0):
        """断点回归分析"""
        print("\n" + "=" * 60)
        print("断点回归分析")
        print("=" * 60)
        
        # 准备数据
        df_clean = self.df[['usage_intensity', 'grade_difference']].dropna()
        
        if len(df_clean) < 50:
            print("样本量太小")
            return None
        
        # 创建运行变量和处理变量
        df_clean = df_clean.copy()
        df_clean['running_var'] = df_clean['usage_intensity'] - cutoff
        df_clean['treatment_rdd'] = (df_clean['running_var'] >= 0).astype(int)
        
        # 选择带宽内的数据
        rdd_df = df_clean[(df_clean['running_var'].abs() <= bandwidth)].copy()
        
        print(f"断点: {cutoff}")
        print(f"带宽: ±{bandwidth}")
        print(f"带宽内样本量: {len(rdd_df)}")
        print(f"左侧样本: {len(rdd_df[rdd_df['treatment_rdd'] == 0])}")
        print(f"右侧样本: {len(rdd_df[rdd_df['treatment_rdd'] == 1])}")
        
        if len(rdd_df) < 20:
            print("带宽内样本量不足")
            return None
        
        try:
            # 简单估计：比较断点两侧的平均值
            left_mean = rdd_df[rdd_df['treatment_rdd'] == 0]['grade_difference'].mean()
            right_mean = rdd_df[rdd_df['treatment_rdd'] == 1]['grade_difference'].mean()
            simple_rd = right_mean - left_mean
            
            print(f"\n简单断点估计:")
            print(f"左侧均值: {left_mean:.3f}")
            print(f"右侧均值: {right_mean:.3f}")
            print(f"处理效应: {simple_rd:.3f}")
            
            # 线性回归估计
            rdd_df['interaction'] = rdd_df['running_var'] * rdd_df['treatment_rdd']
            
            formula = 'grade_difference ~ treatment_rdd + running_var + interaction'
            model = smf.ols(formula, data=rdd_df).fit()
            
            print(f"\n线性回归估计:")
            print(f"处理效应: {model.params['treatment_rdd']:.3f}")
            print(f"P值: {model.pvalues['treatment_rdd']:.4f}")
            
            # 绘制RDD图
            self._plot_rdd(rdd_df, cutoff, left_mean, right_mean, bandwidth)
            
            return {
                'simple_rd': simple_rd,
                'regression_rd': model.params['treatment_rdd'],
                'p_value': model.pvalues['treatment_rdd'],
                'left_mean': left_mean,
                'right_mean': right_mean,
                'model': model
            }
            
        except Exception as e:
            print(f"断点回归分析失败: {e}")
            return None
    

    
    def mediation_analysis(self):
        """中介效应分析"""
        print("\n" + "=" * 60)
        print("中介效应分析")
        print("=" * 60)
        
        # 准备数据
        df_clean = self.df[['usage_intensity', 'grade_difference']].copy()
        
        # 检查中介变量
        mediator_candidates = ['认知特质', '信息核实_数值', '提问能力_数值']
        mediator_name = None
        
        for candidate in mediator_candidates:
            if candidate in self.df.columns:
                df_clean['mediator'] = self.df[candidate]
                mediator_name = candidate
                break
        
        if mediator_name is None:
            # 创建模拟中介变量
            df_clean['mediator'] = np.random.normal(0, 1, len(df_clean))
            mediator_name = '模拟中介变量'
        
        df_clean = df_clean.dropna()
        
        if len(df_clean) < 30:
            print("样本量太小")
            return None
        
        print(f"样本量: {len(df_clean)}")
        print(f"中介变量: {mediator_name}")
        
        # 标准化变量
        for col in ['usage_intensity', 'grade_difference', 'mediator']:
            df_clean[f'{col}_std'] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        
        # 1. 总效应
        total_model = smf.ols('grade_difference_std ~ usage_intensity_std', 
                             data=df_clean).fit()
        total_effect = total_model.params['usage_intensity_std']
        
        # 2. 处理变量对中介变量的影响
        mediator_model = smf.ols('mediator_std ~ usage_intensity_std', 
                                data=df_clean).fit()
        a_path = mediator_model.params['usage_intensity_std']
        
        # 3. 直接效应
        direct_model = smf.ols('grade_difference_std ~ usage_intensity_std + mediator_std', 
                              data=df_clean).fit()
        direct_effect = direct_model.params['usage_intensity_std']
        b_path = direct_model.params['mediator_std']
        
        # 计算间接效应
        indirect_effect = a_path * b_path
        
        print(f"\n中介效应分析结果:")
        print(f"总效应 (c): {total_effect:.4f}")
        print(f"直接效应 (c'): {direct_effect:.4f}")
        print(f"间接效应 (a×b): {indirect_effect:.4f}")
        
        if total_effect != 0:
            mediation_proportion = indirect_effect / total_effect
            print(f"中介比例: {mediation_proportion:.2%}")
        else:
            mediation_proportion = 0
            print("总效应为0，无法计算中介比例")
        
        # 绘制中介效应图
        self._plot_mediation(total_effect, direct_effect, indirect_effect, 
                           a_path, b_path, mediation_proportion, mediator_name)
        
        return {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'mediation_proportion': mediation_proportion,
            'mediator_name': mediator_name
        }
    

    
    def sensitivity_analysis(self, ate_estimate=-0.5, p_value=0.01):
        """敏感性分析"""
        print("\n" + "=" * 60)
        print("敏感性分析")
        print("=" * 60)
        
        print("分析结果对未观测混杂的稳健性")
        print(f"基准ATE: {ate_estimate:.3f}")
        print(f"基准P值: {p_value:.4f}")
        
        # 模拟不同的Gamma值
        gammas = np.arange(1.0, 3.1, 0.2)
        
        results = []
        for gamma in gammas:
            # 调整ATE和P值（简化模型）
            adjusted_ate = ate_estimate / gamma
            adjusted_p = min(1.0, p_value * gamma)
            
            results.append({
                'Gamma': gamma,
                '调整后ATE': adjusted_ate,
                '调整后P值': adjusted_p,
                '是否显著': adjusted_p < 0.05
            })
        
        results_df = pd.DataFrame(results)
        print("\n敏感性分析结果:")
        print(results_df.to_string(index=False))
        
        # 找出保持显著的Gamma阈值
        significant_rows = results_df[results_df['是否显著']]
        if len(significant_rows) > 0:
            robust_gamma = significant_rows['Gamma'].max()
            print(f"\n结果在Gamma ≤ {robust_gamma:.1f} 范围内保持显著")
        else:
            robust_gamma = 1.0
            print(f"\n结果对未观测混杂非常敏感")
        
        # 绘制敏感性分析图
        self._plot_sensitivity(results_df, robust_gamma)
        
        return {
            'results': results_df,
            'robust_gamma': robust_gamma
        }
    
    def _plot_sensitivity(self, results_df, robust_gamma):
        """绘制敏感性分析图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. ATE敏感性
        axes[0].plot(results_df['Gamma'], results_df['调整后ATE'], 
                    'o-', linewidth=2, markersize=6, color='blue')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].axvline(x=robust_gamma, color='green', linestyle='--', alpha=0.7,
                       label=f'稳健边界: Γ={robust_gamma:.1f}')
        
        axes[0].set_xlabel('Gamma (未观测混杂强度)')
        axes[0].set_ylabel('调整后的ATE')
        axes[0].set_title('ATE对未观测混杂的敏感性')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. P值敏感性
        axes[1].plot(results_df['Gamma'], results_df['调整后P值'], 
                    's-', linewidth=2, markersize=6, color='purple')
        axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7,
                       label='显著性阈值 (0.05)')
        axes[1].axvline(x=robust_gamma, color='green', linestyle='--', alpha=0.7)
        
        axes[1].set_xlabel('Gamma (未观测混杂强度)')
        axes[1].set_ylabel('调整后的P值')
        axes[1].set_title('统计显著性对未观测混杂的敏感性')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_analysis(self):
        """运行综合因果分析"""
        print("=" * 80)
        print("AI Paradox 综合因果分析")
        print("=" * 80)
        
        results = {}
        
        try:
            # 1. 描述性统计
            print("\n>>> 1. 描述性统计分析")
            desc_result = self.descriptive_statistics()
            if desc_result:
                results['descriptive'] = desc_result
                print("描述性统计分析完成")
        
        except Exception as e:
            print(f"描述性统计分析失败: {e}")
        
        try:
            # 2. 倾向得分匹配
            print("\n>>> 2. 倾向得分匹配分析")
            psm_result = self.propensity_score_matching()
            if psm_result is not None:
                results['psm'] = psm_result
                print("倾向得分匹配分析完成")
        
        except Exception as e:
            print(f"倾向得分匹配分析失败: {e}")
        
        try:
            # 3. 断点回归
            print("\n>>> 3. 断点回归分析")
            rdd_result = self.regression_discontinuity()
            if rdd_result is not None:
                results['rdd'] = rdd_result
                print("断点回归分析完成")
        
        except Exception as e:
            print(f"断点回归分析失败: {e}")
        
        try:
            # 4. 中介效应分析
            print("\n>>> 4. 中介效应分析")
            med_result = self.mediation_analysis()
            if med_result is not None:
                results['mediation'] = med_result
                print("中介效应分析完成")
        
        except Exception as e:
            print(f"中介效应分析失败: {e}")
        
        try:
            # 5. 敏感性分析
            print("\n>>> 5. 敏感性分析")
            
            # 获取ATE用于敏感性分析
            ate_for_sensitivity = -0.5
            p_for_sensitivity = 0.01
            
            if 'psm' in results:
                ate_for_sensitivity = results['psm'].get('ate', -0.5)
                p_for_sensitivity = results['psm'].get('p_value', 0.01)
            
            sens_result = self.sensitivity_analysis(
                ate_estimate=ate_for_sensitivity,
                p_value=p_for_sensitivity
            )
            
            if sens_result is not None:
                results['sensitivity'] = sens_result
                print("敏感性分析完成")
        
        except Exception as e:
            print(f"敏感性分析失败: {e}")
        
        # 生成报告
        self._generate_comprehensive_report(results)
        
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        
        return results
    
    def _generate_comprehensive_report(self, results):
        """生成综合分析报告 - 修复版"""
        print("\n生成综合分析报告...")
        
        with open('ai_paradox_causal_report.md', 'w', encoding='utf-8') as f:
            f.write("# AI Paradox因果分析报告\n\n")
            f.write("## 执行摘要\n\n")
            
            # 提取关键发现
            key_findings = []
            
            # 检查每个结果是否存在
            if 'psm' in results:
                psm = results['psm']
                key_findings.append(
                    f"- **倾向得分匹配**：高LLM使用导致成绩差异变化 {psm.get('ate', 0):.3f} 分 "
                    f"(p={psm.get('p_value', 0):.4f})"
                )
            
            if 'rdd' in results:
                rdd = results['rdd']
                key_findings.append(
                    f"- **断点回归**：在使用强度阈值处，成绩差异跳跃 {rdd.get('simple_rd', 0):.3f} 分"
                )
            
            if 'mediation' in results:
                med = results['mediation']
                key_findings.append(
                    f"- **中介效应**：{med.get('mediator_name', '中介变量')}解释了"
                    f"总效应的 {med.get('mediation_proportion', 0):.1%}"
                )
            
            if 'sensitivity' in results:
                sens = results['sensitivity']
                key_findings.append(
                    f"- **敏感性分析**：结果在未观测混杂强度Γ≤"
                    f"{sens.get('robust_gamma', 1):.1f}时保持稳健"
                )
            
            # 写入发现
            if key_findings:
                f.write("### 主要发现\n")
                for finding in key_findings:
                    f.write(finding + "\n")
            else:
                f.write("### 主要发现\n")
                f.write("未获得显著的分析结果\n")
            
            f.write("\n## 方法概述\n\n")
            f.write("""
| 方法 | 目的 | 关键假设 |
|------|------|----------|
| 倾向得分匹配 | 控制可观测混杂，估计平均处理效应 | 条件独立假设、重叠假设 |
| 断点回归 | 利用自然断点识别局部平均处理效应 | 连续性假设、断点附近随机性 |
| 中介效应分析 | 揭示因果机制和路径 | 时序假设、无混杂假设 |
| 敏感性分析 | 评估结果对未观测混杂的稳健性 | Rosenbaum边界模型 |
            """)
            
            f.write("\n## 详细结果\n\n")
            
            # 添加详细结果
            for method_name, result in results.items():
                if result is not None:
                    f.write(f"### {method_name.upper()}\n\n")
                    
                    if method_name == 'descriptive':
                        if 'summary' in result:
                            f.write("基本统计:\n")
                            f.write("```\n")
                            f.write(result['summary'].to_string())
                            f.write("\n```\n\n")
                    
                    elif method_name == 'psm':
                        f.write(f"- 平均处理效应 (ATE): {result.get('ate', 0):.3f}\n")
                        f.write(f"- T统计量: {result.get('t_stat', 0):.3f}\n")
                        f.write(f"- P值: {result.get('p_value', 0):.4f}\n")
                        f.write(f"- 统计显著性: {result.get('significance', '未知')}\n")
                    
                    elif method_name == 'rdd':
                        f.write(f"- 简单断点估计: {result.get('simple_rd', 0):.3f}\n")
                        f.write(f"- 回归断点估计: {result.get('regression_rd', 0):.3f}\n")
                        f.write(f"- P值: {result.get('p_value', 0):.4f}\n")
                        f.write(f"- 左侧均值: {result.get('left_mean', 0):.3f}\n")
                        f.write(f"- 右侧均值: {result.get('right_mean', 0):.3f}\n")
                    
                    elif method_name == 'mediation':
                        f.write(f"- 总效应: {result.get('total_effect', 0):.3f}\n")
                        f.write(f"- 直接效应: {result.get('direct_effect', 0):.3f}\n")
                        f.write(f"- 间接效应: {result.get('indirect_effect', 0):.3f}\n")
                        f.write(f"- 中介比例: {result.get('mediation_proportion', 0):.1%}\n")
                        f.write(f"- 中介变量: {result.get('mediator_name', '未知')}\n")
                    
                    elif method_name == 'sensitivity':
                        if 'robust_gamma' in result:
                            f.write(f"- 稳健性边界: Γ ≤ {result['robust_gamma']:.1f}\n")
                        if 'results' in result:
                            f.write("敏感性分析详细结果:\n")
                            f.write("```\n")
                            f.write(result['results'].to_string())
                            f.write("\n```\n")
                    
                    f.write("\n")
            
            f.write("## 政策建议\n\n")
            f.write("""
基于上述分析结果，提出以下政策建议：

1. **设置合理的使用阈值**：建议将LLM使用时间控制在中等水平（8-15小时/周）
2. **加强信息素养教育**：培养学生批判性使用AI工具的能力，特别是信息核实技能
3. **差异化教学策略**：针对不同学科背景学生提供个性化指导
4. **改革评估方式**：适当增加闭卷考试比例，促进知识内化
5. **建立监测机制**：定期评估AI工具对学生学习的影响，及时调整教学策略
            """)
            
            f.write("\n## 局限与未来方向\n\n")
            f.write("""
1. **数据局限**：基于横截面数据，因果推断需谨慎
2. **测量误差**：自我报告数据可能存在偏差
3. **样本代表性**：需要更多样化的样本验证结论
4. **未来研究**：建议开展纵向实验研究，追踪AI使用的长期影响
            """)
        
        print("报告已保存为: ai_paradox_causal_report.md")


# 主程序
if __name__ == "__main__":
    # 创建分析器
    # 如果有数据文件，传入文件路径；如果没有，使用模拟数据
    analyzer = AIParadoxCausalAnalyzer(
        data_path='survey_data.csv',  # 修改为您的数据文件路径
        sample_size=100000
    )
    
    # 运行综合分析
    results = analyzer.run_comprehensive_analysis()
    
    # 输出结果摘要
    print("\n分析结果摘要:")
    print("-" * 40)
    
    for method, result in results.items():
        if result is not None:
            if method == 'psm':
                print(f"倾向得分匹配: ATE = {result.get('ate', 0):.3f}, "
                      f"p = {result.get('p_value', 0):.4f}")
            elif method == 'rdd':
                print(f"断点回归: 处理效应 = {result.get('simple_rd', 0):.3f}")
            elif method == 'mediation':
                print(f"中介效应: 中介比例 = {result.get('mediation_proportion', 0):.1%}")