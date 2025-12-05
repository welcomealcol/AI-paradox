import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用支持英文的字体
plt.rcParams['axes.unicode_minus'] = False

# 使用您提供的专业映射
major_mapping = {
    '信息类（含计算机、电信、通信工程等）': 'IT Majors',
    '人文与艺术类（含文学、历史、哲学、新闻等）': 'Humanities & Arts',
    '经管与社科类（含经济、管理、法律、社会学等）': 'Economics & Social Sciences',
    '外语类': 'Foreign Languages',
    '物理与工程类（含物理、机械、电子、电气等）': 'Physics & Engineering',
    '数学与统计类': 'Mathematics & Statistics',
    '生化环材类（包含环境工程、化学、生物学、材料等）': 'Engineering'
}

class AIParadoxAnalyzer:
    def __init__(self, data_path=None, df=None):
        """
        Initialize analyzer
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Must provide data path or DataFrame")
        
        # Data preprocessing
        self._preprocess_data()
        
    def _preprocess_data(self):
        """
        Data preprocessing and variable calculation
        """
        print("Starting data preprocessing...")
        
        # Check and print all column names for debugging
        print(f"DataFrame columns: {list(self.df.columns)}")
        
        # Map major names to English (if major column exists)
        if '专业' in self.df.columns:
            self.df['Major_English'] = self.df['专业'].map(major_mapping)
            print("Created English major mapping column")
        else:
            print("Warning: Major column does not exist, cannot create English mapping")
        
        # Convert Likert scale to numerical values
        likert_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
        
        # Convert all Likert scale questions
        likert_columns = [f'Q{i}_{name}' for i, name in [
            (15, '知识联系'), (16, '减轻负担'), (17, '反思方法'), (18, '批判审视'), 
            (19, '问题解决'), (20, '深度思考'), (21, '主动查证'), (22, '激发创意'),
            (23, '信息素养'), (24, '掌控感'), (25, '胜任感'), (26, '归属感'),
            (27, '兴趣提升'), (28, '减轻焦虑'), (29, '依赖内疚'), (30, '担心落后'),
            (31, '探索乐趣'), (32, '弥补漏洞'), (33, '个性化'), (34, '情感不可替代'),
            (35, '教学方法改革'), (36, '融入课堂'), (37, '注重过程'), (38, '角色转变'),
            (39, '担心关注减少'), (40, '利大于弊'), (41, '平衡使用'), (42, '职业重要'),
            (43, '隐私担忧'), (44, '制定规则'), (45, '支持课程'), (46, '减少讨论'),
            (47, '促进公平'), (48, '加剧不平等'), (49, '驾驭AI'), (50, '未来乐观'),
            (51, '思维不可替代')
        ]]
        
        for col in likert_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].map(likert_mapping)
                print(f"Converted column: {col}")
            else:
                print(f"Warning: Column {col} does not exist")
        
        # Check and convert efficiency improvement column (if exists)
        if '效率提升' in self.df.columns:
            efficiency_mapping = {
                '提升非常大': 5, '提升较大': 4, '一般': 3, 
                '提升较小': 2, '没有提升': 1
            }
            self.df['Efficiency_Improvement'] = self.df['效率提升'].map(efficiency_mapping)
            print("Created efficiency improvement numerical column")
        else:
            print("Warning: Efficiency improvement column does not exist, skipping creation")
        
        # Calculate grade differences (assuming grades are already numerical)
        # First ensure grade columns are numerical type
        grade_columns = [
            '使用ChatGPT_闭卷成绩', '不使用ChatGPT_闭卷成绩',
            '使用ChatGPT_开卷成绩', '不使用ChatGPT_开卷成绩'
        ]
        
        for col in grade_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                print(f"Converted grade column {col} to numerical type")
            else:
                print(f"Warning: Grade column {col} does not exist")
        
        # Calculate grade differences
        if all(col in self.df.columns for col in ['使用ChatGPT_闭卷成绩', '不使用ChatGPT_闭卷成绩']):
            self.df['Closed_Book_Perf_Diff'] = self.df['使用ChatGPT_闭卷成绩'] - self.df['不使用ChatGPT_闭卷成绩']
            print("Calculated closed-book grade difference")
        else:
            print("Warning: Cannot calculate closed-book grade difference, missing necessary grade columns")
            
        if all(col in self.df.columns for col in ['使用ChatGPT_开卷成绩', '不使用ChatGPT_开卷成绩']):
            self.df['Open_Book_Perf_Diff'] = self.df['使用ChatGPT_开卷成绩'] - self.df['不使用ChatGPT_开卷成绩']
            print("Calculated open-book grade difference")
        else:
            print("Warning: Cannot calculate open-book grade difference, missing necessary grade columns")
        
        # Calculate cognitive ability indicators (only calculate existing columns)
        cognitive_metrics = []
        
        if all(col in self.df.columns for col in ['Q18_批判审视', 'Q21_主动查证']):
            self.df['Critical_Thinking_Score'] = self.df['Q18_批判审视'] + self.df['Q21_主动查证']
            cognitive_metrics.append('Critical_Thinking_Score')
        
        if all(col in self.df.columns for col in ['Q17_反思方法', 'Q20_深度思考']):
            self.df['Metacognition_Score'] = self.df['Q17_反思方法'] + (6 - self.df['Q20_深度思考'])  # Reverse scoring
            cognitive_metrics.append('Metacognition_Score')
        
        if 'Q23_信息素养' in self.df.columns and '信息核实_数值' in self.df.columns:
            self.df['Information_Literacy_Score'] = self.df['Q23_信息素养'] + self.df['信息核实_数值']
            cognitive_metrics.append('Information_Literacy_Score')
        
        print(f"Calculated cognitive ability indicators: {cognitive_metrics}")
        
        # Calculate psychological motivation indicators
        motivation_metrics = []
        if all(col in self.df.columns for col in ['Q24_掌控感', 'Q25_胜任感', 'Q27_兴趣提升']):
            self.df['Learning_Motivation_Score'] = self.df['Q24_掌控感'] + self.df['Q25_胜任感'] + self.df['Q27_兴趣提升']
            motivation_metrics.append('Learning_Motivation_Score')
        
        if all(col in self.df.columns for col in ['Q29_依赖内疚', 'Q30_担心落后']):
            self.df['Dependency_Anxiety_Score'] = self.df['Q29_依赖内疚'] + self.df['Q30_担心落后']
            motivation_metrics.append('Dependency_Anxiety_Score')
        
        print(f"Calculated psychological motivation indicators: {motivation_metrics}")
        
        # Create usage intensity groups
        if '使用时长_数值' in self.df.columns:
            self.df['Usage_Intensity_Group'] = pd.cut(self.df['使用时长_数值'], 
                                        bins=[0, 1, 2, 3, 4, 5], 
                                        labels=['Rarely', 'Infrequent', 'Moderate', 'Frequent', 'Heavy'])
            print("Created usage intensity groups")
        else:
            print("Warning: Usage duration numerical column does not exist, cannot create usage intensity groups")
        
        # Create attitude groups
        if '使用态度_数值' in self.df.columns:
            self.df['Attitude_Group'] = pd.cut(self.df['使用态度_数值'],
                                    bins=[0, 2, 3, 4, 6],
                                    labels=['Negative', 'Neutral', 'Positive', 'Very Positive'])
            print("Created attitude groups")
        else:
            print("Warning: Usage attitude numerical column does not exist, cannot create attitude groups")
        
        print("Data preprocessing completed")
    
    def analyze_disciplinary_divide(self):
        """
        Step 3: Analyze disciplinary digital divide
        """
        print("\n" + "=" * 50)
        print("Step 3: Disciplinary Digital Divide Analysis")
        print("=" * 50)
        
        # Check if necessary columns exist - using English major column
        required_columns = ['Major_English', '使用时长_数值', 'Closed_Book_Perf_Diff', 'Critical_Thinking_Score']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"Warning: Missing necessary columns {missing_columns}, skipping disciplinary digital divide analysis")
            return None, None
        
        # 1. Analyze usage patterns by major
        agg_dict = {
            '使用时长_数值': 'mean',
            '使用态度_数值': 'mean',
            'Closed_Book_Perf_Diff': 'mean',
            'Critical_Thinking_Score': 'mean'
        }
        
        # If efficiency improvement column exists, add to aggregation dictionary
        if 'Efficiency_Improvement' in self.df.columns:
            agg_dict['Efficiency_Improvement'] = 'mean'
        
        # Group by English major names
        major_analysis = self.df.groupby('Major_English').agg(agg_dict).round(3)
        
        print("Usage Patterns and Benefits by Major:")
        print(major_analysis.sort_values('Closed_Book_Perf_Diff', ascending=False))
        print("\n")
        
        # 2. ANOVA test for differences between majors
        try:
            f_stat_usage, p_usage = f_oneway(
                *[group['使用时长_数值'].values for name, group in self.df.groupby('Major_English')]
            )
            print(f"Usage duration differences between majors: F={f_stat_usage:.3f}, p={p_usage:.3f}")
        except Exception as e:
            print(f"ANOVA for usage duration failed: {e}")
        
        try:
            f_stat_benefit, p_benefit = f_oneway(
                *[group['Closed_Book_Perf_Diff'].values for name, group in self.df.groupby('Major_English')]
            )
            print(f"Grade benefit differences between majors: F={f_stat_benefit:.3f}, p={p_benefit:.3f}")
        except Exception as e:
            print(f"ANOVA for grade benefits failed: {e}")
        
        # 3. Cluster analysis to identify major types
        stem_majors = [
            'IT Majors', 
            'Mathematics & Statistics', 
            'Physics & Engineering',
            'Engineering'
        ]
        
        humanities_majors = [
            'Humanities & Arts',
            'Foreign Languages'
        ]
        
        social_science_majors = [
            'Economics & Social Sciences'
        ]
        
        self.df['Major_Type'] = 'Other'
        
        # Map majors to major types
        for major in stem_majors:
            if major in self.df['Major_English'].values:
                self.df.loc[self.df['Major_English'] == major, 'Major_Type'] = 'STEM'
        
        for major in humanities_majors:
            if major in self.df['Major_English'].values:
                self.df.loc[self.df['Major_English'] == major, 'Major_Type'] = 'Humanities & Arts'
        
        for major in social_science_majors:
            if major in self.df['Major_English'].values:
                self.df.loc[self.df['Major_English'] == major, 'Major_Type'] = 'Social Sciences'
        
        # Analyze differences by major type
        type_agg_dict = {
            '使用时长_数值': 'mean',
            'Closed_Book_Perf_Diff': 'mean'
        }
        
        if 'Critical_Thinking_Score' in self.df.columns:
            type_agg_dict['Critical_Thinking_Score'] = 'mean'
        
        type_analysis = self.df.groupby('Major_Type').agg(type_agg_dict).round(3)
        
        print("\nUsage Patterns and Benefits by Major Type:")
        print(type_analysis)
        
        return major_analysis, type_analysis

    def analyze_threshold_effect(self):
        """
        Step 1: Analyze threshold effect of usage intensity on academic performance
        """
        print("=" * 50)
        print("Step 1: Threshold Effect Analysis")
        print("=" * 50)
        
        # Check necessary columns
        if 'Usage_Intensity_Group' not in self.df.columns or 'Closed_Book_Perf_Diff' not in self.df.columns:
            print("Warning: Missing necessary columns, skipping threshold effect analysis")
            return None, None
        
        # 1. Descriptive statistics
        usage_stats = self.df.groupby('Usage_Intensity_Group').agg({
            'Closed_Book_Perf_Diff': ['mean', 'std', 'count'],
            'Open_Book_Perf_Diff': ['mean', 'std', 'count'],
            '使用ChatGPT_闭卷成绩': 'mean',
            '不使用ChatGPT_闭卷成绩': 'mean'
        }).round(3)
        
        print("Grade Differences by Usage Intensity Groups:")
        print(usage_stats)
        print("\n")
        
        # 2. ANOVA
        try:
            groups = [group['Closed_Book_Perf_Diff'].values for name, group in self.df.groupby('Usage_Intensity_Group')]
            f_stat, p_value = f_oneway(*groups)
            
            print(f"ANOVA for closed-book grade differences: F={f_stat:.3f}, p={p_value:.3f}")
            
            if p_value < 0.05:
                # Tukey HSD post-hoc test
                tukey = pairwise_tukeyhsd(
                    endog=self.df['Closed_Book_Perf_Diff'].dropna(),
                    groups=self.df['Usage_Intensity_Group'].dropna(),
                    alpha=0.05
                )
                print("\nTukey HSD Post-hoc Test:")
                print(tukey)
        except Exception as e:
            print(f"ANOVA failed: {e}")
        
        # 3. Correlation analysis
        if '使用时长_数值' in self.df.columns and 'Closed_Book_Perf_Diff' in self.df.columns:
            try:
                valid_data = self.df[['使用时长_数值', 'Closed_Book_Perf_Diff']].dropna()
                if len(valid_data) > 0:
                    pearson_corr, pearson_p = pearsonr(valid_data['使用时长_数值'], valid_data['Closed_Book_Perf_Diff'])
                    spearman_corr, spearman_p = spearmanr(valid_data['使用时长_数值'], valid_data['Closed_Book_Perf_Diff'])
                    
                    print(f"\nPearson correlation: r={pearson_corr:.3f}, p={pearson_p:.3f}")
                    print(f"Spearman correlation: ρ={spearman_corr:.3f}, p={spearman_p:.3f}")
                    
                    # 4. Quadratic relationship test
                    X = valid_data[['使用时长_数值']].copy()
                    X['使用时长平方'] = X['使用时长_数值'] ** 2
                    X = sm.add_constant(X)
                    y = valid_data['Closed_Book_Perf_Diff']
                    
                    model = sm.OLS(y, X).fit()
                    print(f"\nQuadratic Regression Model:")
                    print(model.summary())
                    
                    return usage_stats, model
            except Exception as e:
                print(f"Correlation analysis failed: {e}")
        
        return usage_stats, None

    def analyze_cognitive_decline(self):
        """
        Step 2: Analyze cognitive ability decline
        """
        print("\n" + "=" * 50)
        print("Step 2: Cognitive Decline Analysis")
        print("=" * 50)
        
        # Check necessary columns
        required_cols = ['Usage_Intensity_Group', 'Critical_Thinking_Score', 'Metacognition_Score']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"Warning: Missing necessary columns {missing_cols}, skipping cognitive decline analysis")
            return None, None
        
        # 1. Group comparison of cognitive scores
        cognitive_stats = self.df.groupby('Usage_Intensity_Group').agg({
            'Critical_Thinking_Score': ['mean', 'std'],
            'Metacognition_Score': ['mean', 'std'],
            'Information_Literacy_Score': ['mean', 'std'] if 'Information_Literacy_Score' in self.df.columns else pd.Series(dtype=float)
        }).round(3)
        
        print("Cognitive Ability Scores by Usage Intensity Groups:")
        print(cognitive_stats)
        print("\n")
        
        # 2. Correlation analysis
        cognitive_correlations = {}
        for cognitive_var in ['Critical_Thinking_Score', 'Metacognition_Score', 'Information_Literacy_Score']:
            if cognitive_var in self.df.columns:
                try:
                    corr, p_value = pearsonr(self.df['使用时长_数值'], self.df[cognitive_var])
                    cognitive_correlations[cognitive_var] = (corr, p_value)
                    print(f"{cognitive_var} correlation with usage duration: r={corr:.3f}, p={p_value:.3f}")
                except Exception as e:
                    print(f"{cognitive_var} correlation analysis failed: {e}")
        
        # 3. Regression analysis controlling for other variables
        control_vars = ['使用时长_数值']
        if '使用态度_数值' in self.df.columns:
            control_vars.append('使用态度_数值')
        if '提问能力_数值' in self.df.columns:
            control_vars.append('提问能力_数值')
        
        regression_results = {}
        for outcome in ['Critical_Thinking_Score', 'Metacognition_Score']:
            if outcome in self.df.columns:
                try:
                    X = self.df[control_vars].copy()
                    X = sm.add_constant(X)
                    y = self.df[outcome]
                    
                    model = sm.OLS(y, X).fit()
                    regression_results[outcome] = model
                    print(f"\nMultiple Regression Results for {outcome}:")
                    print(f"R² = {model.rsquared:.3f}")
                    if '使用时长_数值' in model.params:
                        print(f"Usage duration coefficient: {model.params['使用时长_数值']:.3f} (p={model.pvalues['使用时长_数值']:.3f})")
                except Exception as e:
                    print(f"{outcome} regression analysis failed: {e}")
        
        return cognitive_stats, regression_results

    def analyze_perception_reality_gap(self):
        """
        Step 4: Analyze subjective perception vs objective performance contradiction
        """
        print("\n" + "=" * 50)
        print("Step 4: Perception-Reality Gap Analysis")
        print("=" * 50)
        
        # Check necessary columns
        required_cols = ['Attitude_Group', 'Closed_Book_Perf_Diff', 'Critical_Thinking_Score', 'Metacognition_Score']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"Warning: Missing necessary columns {missing_cols}, skipping perception-reality gap analysis")
            return None, None
        
        # 1. Attitude group comparison
        attitude_comparison = self.df.groupby('Attitude_Group').agg({
            'Closed_Book_Perf_Diff': 'mean',
            'Critical_Thinking_Score': 'mean',
            'Metacognition_Score': 'mean',
            '使用时长_数值': 'mean',
            'Efficiency_Improvement': 'mean' if 'Efficiency_Improvement' in self.df.columns else pd.Series(dtype=float)
        }).round(3)
        
        print("Objective Performance by Attitude Groups:")
        print(attitude_comparison)
        print("\n")
        
        # 2. Correlation analysis: subjective attitude vs objective performance
        try:
            attitude_objective_corr, attitude_objective_p = pearsonr(
                self.df['使用态度_数值'], self.df['Closed_Book_Perf_Diff']
            )
            print(f"Subjective attitude vs grade benefit correlation: r={attitude_objective_corr:.3f}, p={attitude_objective_p:.3f}")
        except Exception as e:
            print(f"Attitude vs grade correlation analysis failed: {e}")
        
        try:
            attitude_cognitive_corr, attitude_cognitive_p = pearsonr(
                self.df['使用态度_数值'], self.df['Critical_Thinking_Score']
            )
            print(f"Subjective attitude vs critical thinking correlation: r={attitude_cognitive_corr:.3f}, p={attitude_cognitive_p:.3f}")
        except Exception as e:
            print(f"Attitude vs critical thinking correlation analysis failed: {e}")
        
        # 3. Identify contradiction group (positive attitude but poor performance)
        if '使用态度_数值' in self.df.columns and 'Closed_Book_Perf_Diff' in self.df.columns:
            high_attitude = self.df[self.df['使用态度_数值'] >= 4]  # Positive and very positive
            if len(high_attitude) > 0:
                contradiction_group = high_attitude[high_attitude['Closed_Book_Perf_Diff'] < 0]
                contradiction_rate = len(contradiction_group) / len(high_attitude) * 100
                print(f"\nContradiction group percentage (positive attitude but declining grades): {contradiction_rate:.1f}%")
            else:
                print("No positive attitude samples")
                contradiction_group = pd.DataFrame()
        else:
            contradiction_group = pd.DataFrame()
        
        return attitude_comparison, contradiction_group

    def comprehensive_analysis(self):
        """
        Comprehensive analysis and AI paradox validation
        """
        print("=" * 60)
        print("AI Paradox Comprehensive Validation Analysis")
        print("=" * 60)
        
        # Execute all analyses
        threshold_results = self.analyze_threshold_effect()
        cognitive_results = self.analyze_cognitive_decline()
        divide_results = self.analyze_disciplinary_divide()
        perception_results = self.analyze_perception_reality_gap()
        
        # Create visualizations
        self.create_visualizations()
        
        # Comprehensive evaluation of AI paradox evidence
        print("\n" + "=" * 60)
        print("AI Paradox Validation Results Summary")
        print("=" * 60)
        
        evidence_count = 0
        total_evidence = 4
        
        # Evidence 1: Threshold effect
        if threshold_results is not None and threshold_results[1] is not None:
            quadratic_model = threshold_results[1]
            if '使用时长平方' in quadratic_model.params:
                usage_sq_coef = quadratic_model.params['使用时长平方']
                usage_sq_p = quadratic_model.pvalues['使用时长平方']
                
                if usage_sq_coef < 0 and usage_sq_p < 0.05:
                    print("✅ Evidence 1 (Threshold Effect): Significant inverted U-shaped relationship found")
                    evidence_count += 1
                else:
                    print("❌ Evidence 1 (Threshold Effect): No significant inverted U-shaped relationship")
            else:
                print("❌ Evidence 1 (Threshold Effect): Cannot test quadratic relationship")
        else:
            print("❌ Evidence 1 (Threshold Effect): Analysis failed")
        
        # Evidence 2: Cognitive decline
        cognitive_negative = False
        if '使用时长_数值' in self.df.columns:
            for cognitive_var in ['Critical_Thinking_Score', 'Metacognition_Score']:
                if cognitive_var in self.df.columns:
                    try:
                        valid_data = self.df[['使用时长_数值', cognitive_var]].dropna()
                        if len(valid_data) > 0:
                            corr, p = pearsonr(valid_data['使用时长_数值'], valid_data[cognitive_var])
                            if corr < -0.1 and p < 0.05:
                                cognitive_negative = True
                                break
                    except:
                        pass
        
        if cognitive_negative:
            print("✅ Evidence 2 (Cognitive Decline): Significant negative correlation found")
            evidence_count += 1
        else:
            print("❌ Evidence 2 (Cognitive Decline): No significant negative correlation")
        
        # Evidence 3: Disciplinary divide
        if divide_results is not None and divide_results[0] is not None:
            major_analysis_df = divide_results[0]
            
            # Check if there's enough data for between-major comparison
            if len(major_analysis_df) >= 2:  # At least two majors
                # Calculate ranges for usage duration and grade differences
                usage_range = major_analysis_df['使用时长_数值'].max() - major_analysis_df['使用时长_数值'].min()
                benefit_range = abs(major_analysis_df['Closed_Book_Perf_Diff'].max() - major_analysis_df['Closed_Book_Perf_Diff'].min())
                
                # If usage duration difference is large or grade difference is large, consider disciplinary divide exists
                if usage_range > 0.3 or benefit_range > 1.0:
                    print("✅ Evidence 3 (Disciplinary Divide): Significant between-major differences found")
                    evidence_count += 1
                else:
                    print("❌ Evidence 3 (Disciplinary Divide): No significant between-major differences")
            else:
                print("❌ Evidence 3 (Disciplinary Divide): Insufficient number of majors")
        else:
            print("❌ Evidence 3 (Disciplinary Divide): Analysis failed")
        
        # Evidence 4: Perception bias
        if '使用态度_数值' in self.df.columns and 'Closed_Book_Perf_Diff' in self.df.columns:
            try:
                valid_data = self.df[['使用态度_数值', 'Closed_Book_Perf_Diff']].dropna()
                if len(valid_data) > 0:
                    attitude_performance_corr, attitude_performance_p = pearsonr(
                        valid_data['使用态度_数值'], valid_data['Closed_Book_Perf_Diff']
                    )
                    if attitude_performance_corr < 0.1:  # Weak or negative correlation
                        print("✅ Evidence 4 (Perception Bias): Subjective attitude and objective performance inconsistent")
                        evidence_count += 1
                    else:
                        print("❌ Evidence 4 (Perception Bias): No significant inconsistency")
                else:
                    print("❌ Evidence 4 (Perception Bias): No valid data")
            except Exception as e:
                print(f"❌ Evidence 4 (Perception Bias): Correlation analysis failed - {e}")
        else:
            print("❌ Evidence 4 (Perception Bias): Missing necessary data")
        
        # Overall conclusion
        print(f"\nAI Paradox Validation Strength: {evidence_count}/{total_evidence}")
        if evidence_count >= 3:
            print("🎯 Conclusion: Strong support for AI paradox existence")
        elif evidence_count >= 2:
            print("⚠️ Conclusion: Moderate support for AI paradox existence")
        else:
            print("❓ Conclusion: Limited support for AI paradox existence")
        
        return {
            'threshold_evidence': threshold_results,
            'cognitive_evidence': cognitive_results,
            'divide_evidence': divide_results,
            'perception_evidence': perception_results,
            'paradox_strength': evidence_count / total_evidence
        }

    def create_visualizations(self):
        """
        Create key results visualizations
        """
        print("\nCreating visualization charts...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Usage intensity vs performance relationship
            if 'Usage_Intensity_Group' in self.df.columns and 'Closed_Book_Perf_Diff' in self.df.columns:
                usage_performance = self.df.groupby('Usage_Intensity_Group')['Closed_Book_Perf_Diff'].mean()
                axes[0, 0].plot(usage_performance.index, usage_performance.values, 'o-', linewidth=2, markersize=8)
                axes[0, 0].set_title('Usage Intensity vs Closed-Book Performance', fontsize=14, fontweight='bold')
                axes[0, 0].set_xlabel('Usage Intensity')
                axes[0, 0].set_ylabel('Average Performance Difference')
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, 'Missing necessary data', ha='center', va='center')
                axes[0, 0].set_title('Usage Intensity vs Performance', fontsize=14, fontweight='bold')
            
            # 2. Major usage patterns and benefits - using English major names
            if 'Major_English' in self.df.columns:
                major_comparison = self.df.groupby('Major_English').agg({
                    '使用时长_数值': 'mean',
                    'Closed_Book_Perf_Diff': 'mean'
                })
                
                x = np.arange(len(major_comparison))
                width = 0.35
                
                if len(major_comparison) > 0:
                    rects1 = axes[0, 1].bar(x - width/2, major_comparison['使用时长_数值'], width, 
                                           label='Average Usage Duration', alpha=0.7)
                    rects2 = axes[0, 1].bar(x + width/2, major_comparison['Closed_Book_Perf_Diff'], width, 
                                           label='Average Performance Difference', alpha=0.7)
                    
                    axes[0, 1].set_xlabel('Major')
                    axes[0, 1].set_ylabel('Score')
                    axes[0, 1].set_title('AI Usage Patterns and Benefits by Major', fontsize=14, fontweight='bold')
                    axes[0, 1].set_xticks(x)
                    axes[0, 1].set_xticklabels(major_comparison.index, rotation=45, ha='right')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'Missing major data', ha='center', va='center')
                    axes[0, 1].set_title('Major Comparison', fontsize=14, fontweight='bold')
            else:
                axes[0, 1].text(0.5, 0.5, 'Missing major data', ha='center', va='center')
                axes[0, 1].set_title('Major Comparison', fontsize=14, fontweight='bold')
            
            # 3. Cognitive scores by usage intensity
            if 'Usage_Intensity_Group' in self.df.columns:
                cognitive_cols = []
                if 'Critical_Thinking_Score' in self.df.columns:
                    cognitive_cols.append('Critical_Thinking_Score')
                if 'Metacognition_Score' in self.df.columns:
                    cognitive_cols.append('Metacognition_Score')
                
                if cognitive_cols:
                    cognitive_by_usage = self.df.groupby('Usage_Intensity_Group')[cognitive_cols].mean()
                    cognitive_by_usage.plot(kind='line', ax=axes[1, 0], marker='o')
                    axes[1, 0].set_title('Cognitive Abilities by Usage Intensity', fontsize=14, fontweight='bold')
                    axes[1, 0].set_xlabel('Usage Intensity')
                    axes[1, 0].set_ylabel('Average Score')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].legend()
                else:
                    axes[1, 0].text(0.5, 0.5, 'Missing cognitive ability data', ha='center', va='center')
                    axes[1, 0].set_title('Cognitive Ability Changes', fontsize=14, fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'Missing usage intensity data', ha='center', va='center')
                axes[1, 0].set_title('Cognitive Ability Changes', fontsize=14, fontweight='bold')
            
            # 4. Subjective attitude vs objective performance scatter plot
            if '使用态度_数值' in self.df.columns and 'Closed_Book_Perf_Diff' in self.df.columns:
                axes[1, 1].scatter(self.df['使用态度_数值'], self.df['Closed_Book_Perf_Diff'], alpha=0.6)
                axes[1, 1].set_xlabel('Subjective Attitude')
                axes[1, 1].set_ylabel('Objective Performance Difference')
                axes[1, 1].set_title('Subjective Attitude vs Objective Performance', fontsize=14, fontweight='bold')
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add trend line
                try:
                    z = np.polyfit(self.df['使用态度_数值'], self.df['Closed_Book_Perf_Diff'], 1)
                    p = np.poly1d(z)
                    axes[1, 1].plot(self.df['使用态度_数值'], p(self.df['使用态度_数值']), "r--", alpha=0.8)
                except:
                    pass
            else:
                axes[1, 1].text(0.5, 0.5, 'Missing attitude or grade data', ha='center', va='center')
                axes[1, 1].set_title('Subjective Attitude vs Objective Performance', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('AI_paradox_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Visualization charts saved as AI_paradox_analysis.png")
            
        except Exception as e:
            print(f"Visualization creation failed: {e}")

# Usage example
if __name__ == "__main__":
    # Assume data file path
    analyzer = AIParadoxAnalyzer(data_path="survey_data.csv")
    
    
    # Or use DataFrame directly
    # analyzer = AIParadoxAnalyzer(df=your_dataframe)
    
    # Execute comprehensive analysis
    results = analyzer.comprehensive_analysis()
    
    print("AI paradox analyzer defined. Please provide data file path or DataFrame to initialize analyzer.")