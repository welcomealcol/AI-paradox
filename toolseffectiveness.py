import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.linalg import eigh
import scipy.stats as stats

class SurveyReliabilityValidity:
    def __init__(self, data):
        self.data = data
        self.scales = self.define_scales_by_domains()
        self.preprocess_data()
        
    def define_scales_by_domains(self):
        """根据问卷设计的五个领域定义量表维度"""
        scales = {
            '认知过程与能力': [
                'Q15_知识联系', 'Q16_减轻负担', 'Q17_反思方法', 'Q18_批判审视', 
                'Q19_问题解决', 'Q20_深度思考', 'Q21_主动查证', 'Q22_激发创意', 'Q23_信息素养'
            ],
            '学习动机与心理感受': [
                'Q24_掌控感', 'Q25_胜任感', 'Q26_归属感', 'Q27_兴趣提升', 
                'Q28_减轻焦虑', 'Q29_依赖内疚', 'Q30_担心落后', 'Q31_探索乐趣'
            ],
            '教学与教师角色': [
                'Q32_弥补漏洞', 'Q33_个性化', 'Q34_情感不可替代', 'Q35_教学方法改革',
                'Q36_融入课堂', 'Q37_注重过程', 'Q38_角色转变', 'Q39_担心关注减少'
            ],
            '综合影响与未来展望': [
                'Q40_利大于弊', 'Q41_平衡使用', 'Q42_职业重要', 'Q43_隐私担忧',
                'Q44_制定规则', 'Q45_支持课程', 'Q46_减少讨论', 'Q47_促进公平',
                'Q48_加剧不平等', 'Q49_驾驭AI', 'Q50_未来乐观', 'Q51_思维不可替代'
            ]
        }
        return scales
    
    def convert_likert_to_numeric(self):
        """根据计分规则将李克特量表的字母答案转换为数值分数"""
        # 正向题目映射：A=5, B=4, C=3, D=2, E=1
        positive_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
        
        # 负向题目映射：A=1, B=2, C=3, D=4, E=5
        #negative_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
        negative_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
        # 正向题目列表（根据您提供的规则）
        positive_items = [
            'Q15_知识联系', 'Q16_减轻负担', 'Q17_反思方法', 'Q18_批判审视', 'Q19_问题解决',
            'Q21_主动查证', 'Q22_激发创意', 'Q23_信息素养', 'Q24_掌控感', 'Q25_胜任感',
            'Q26_归属感', 'Q27_兴趣提升', 'Q28_减轻焦虑', 'Q31_探索乐趣', 'Q32_弥补漏洞',
            'Q33_个性化', 'Q35_教学方法改革', 'Q36_融入课堂', 'Q37_注重过程', 'Q38_角色转变',
            'Q40_利大于弊', 'Q41_平衡使用', 'Q42_职业重要', 'Q45_支持课程', 'Q47_促进公平',
            'Q49_驾驭AI', 'Q50_未来乐观', 'Q51_思维不可替代'
        ]
        
        # 负向题目列表（根据您提供的规则）
        negative_items = [
            'Q20_深度思考', 'Q29_依赖内疚', 'Q30_担心落后', 'Q34_情感不可替代',
            'Q39_担心关注减少', 'Q43_隐私担忧', 'Q44_制定规则', 'Q46_减少讨论', 'Q48_加剧不平等'
        ]
        
        # 处理所有李克特题目
        for col in self.data.columns:
            if col.startswith('Q') and col[1:3].isdigit():
                if col in positive_items and col in self.data.columns:
                    self.data[col] = self.data[col].map(positive_mapping)
                elif col in negative_items and col in self.data.columns:
                    self.data[col] = self.data[col].map(negative_mapping)
        
        print(f"已根据计分规则转换李克特题目")
    
    def preprocess_data(self):
        """预处理数据"""
        # 将李克特量表转换为数值
        self.convert_likert_to_numeric()
        
        # 处理其他数值型列
        numeric_columns = ['使用时长_数值', '使用态度_数值', '提问能力_数值', '信息核实_数值', '效率提升_数值']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        print("数据预处理完成")
    
    def cronbach_alpha(self, scale_data):
        """手动计算Cronbach's Alpha"""
        n_items = scale_data.shape[1]
        if n_items < 2:
            return np.nan
        
        # 确保数据是数值型
        scale_data = scale_data.apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(scale_data) < 2:
            return np.nan
            
        # 计算总方差和各项方差
        total_variance = scale_data.sum(axis=1).var()
        item_variances = scale_data.var(axis=0).sum()
        
        # 避免除以零
        if total_variance == 0:
            return np.nan
            
        # Cronbach's Alpha公式
        alpha = (n_items / (n_items - 1)) * (1 - (item_variances / total_variance))
        return alpha
    
    def calculate_reliability(self):
        """计算各维度的信度系数"""
        reliability_results = []
        
        for scale_name, items in self.scales.items():
            # 检查所有题目是否在数据中
            available_items = [item for item in items if item in self.data.columns]
            
            if len(available_items) >= 2:  # 至少需要2个题目才能计算信度
                try:
                    # 提取该维度的所有题目数据
                    scale_data = self.data[available_items].dropna()
                    
                    if len(scale_data) > 10:  # 确保有足够的样本
                        # 计算Cronbach's alpha
                        alpha = self.cronbach_alpha(scale_data)
                        
                        # 计算项目-总分相关系数
                        total_scores = scale_data.mean(axis=1)
                        item_total_corrs = []
                        for item in available_items:
                            corr = scale_data[item].corr(total_scores)
                            if not pd.isna(corr):
                                item_total_corrs.append(corr)
                        
                        avg_item_total_corr = np.mean(item_total_corrs) if item_total_corrs else np.nan
                        
                        reliability_results.append({
                            '量表维度': scale_name,
                            '题目数量': len(available_items),
                            'Cronbachs_Alpha': round(alpha, 3),
                            '平均项目-总分相关': round(avg_item_total_corr, 3),
                            '样本量': len(scale_data)
                        })
                except Exception as e:
                    print(f"计算{scale_name}信度时出错: {e}")
                    continue
        
        return pd.DataFrame(reliability_results)
    
    def calculate_kmo(self, data):
        """计算KMO取样适切性量数（修正版本）"""
        # 确保数据是数值型
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(data) < 2 or data.shape[1] < 2:
            return 0
            
        corr_matrix = data.corr()
        n_vars = data.shape[1]
        
        # 检查相关矩阵是否满秩
        if np.linalg.matrix_rank(corr_matrix) < n_vars:
            print("警告: 相关矩阵不是满秩的，可能存在完全共线性的变量")
            return 0
        
        try:
            # 计算偏相关系数矩阵（修正方法）
            # 使用逆矩阵计算偏相关系数
            inv_corr_matrix = np.linalg.inv(corr_matrix)
            
            # 创建偏相关系数矩阵
            partial_corr = np.zeros((n_vars, n_vars))
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        # 偏相关系数公式: -inv_corr[i,j] / sqrt(inv_corr[i,i] * inv_corr[j,j])
                        denominator = np.sqrt(inv_corr_matrix[i, i] * inv_corr_matrix[j, j])
                        if denominator != 0:
                            partial_corr[i, j] = -inv_corr_matrix[i, j] / denominator
                        else:
                            partial_corr[i, j] = 0
            
            # 计算KMO（修正公式）
            # 对每个变量计算KMO_i，然后取平均
            kmo_per_var = np.zeros(n_vars)
            
            for i in range(n_vars):
                # 计算变量i与其他变量的简单相关系数平方和
                simple_corr_sq_sum = 0
                # 计算变量i与其他变量的偏相关系数平方和
                partial_corr_sq_sum = 0
                
                for j in range(n_vars):
                    if i != j:
                        simple_corr_sq_sum += corr_matrix.iloc[i, j]**2
                        partial_corr_sq_sum += partial_corr[i, j]**2
                
                # 计算单个变量的KMO
                if (simple_corr_sq_sum + partial_corr_sq_sum) != 0:
                    kmo_per_var[i] = simple_corr_sq_sum / (simple_corr_sq_sum + partial_corr_sq_sum)
            
            # 整体KMO是各变量KMO的平均值
            kmo = np.mean(kmo_per_var[kmo_per_var > 0]) if np.any(kmo_per_var > 0) else 0
            
            return kmo
            
        except np.linalg.LinAlgError:
            print("警告: 无法计算矩阵的逆，使用备用方法计算KMO")
            # 备用方法：简化的KMO计算
            corr_values = corr_matrix.values
            # 计算上三角（不包括对角线）的相关系数平方和
            n = n_vars
            simple_corr_sq_sum = np.sum(np.triu(corr_values**2, 1))
            
            # 对于备用方法，我们使用一个近似的偏相关估计
            # 这里使用每个变量的多元R²作为近似
            partial_corr_sq_approx = 0
            for i in range(n_vars):
                # 排除当前变量
                other_vars = [j for j in range(n_vars) if j != i]
                if len(other_vars) > 0:
                    # 计算当前变量与其他变量的多重相关
                    X = data.iloc[:, other_vars].values
                    y = data.iloc[:, i].values
                    
                    # 添加截距项
                    X_with_intercept = np.column_stack([np.ones(len(X)), X])
                    
                    try:
                        # 计算回归系数
                        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                        # 计算预测值
                        y_pred = X_with_intercept @ beta
                        # 计算R²
                        ss_total = np.sum((y - np.mean(y))**2)
                        ss_residual = np.sum((y - y_pred)**2)
                        if ss_total != 0:
                            r_squared = 1 - (ss_residual / ss_total)
                            # 偏相关系数的平方近似为1 - R²
                            partial_corr_sq_approx += (1 - r_squared)
                    except:
                        continue
            
            # 计算近似的整体KMO
            if simple_corr_sq_sum > 0:
                approx_kmo = simple_corr_sq_sum / (simple_corr_sq_sum + partial_corr_sq_approx)
                return approx_kmo
            else:
                return 0
    
    def bartlett_sphericity(self, data):
        """计算Bartlett球形检验"""
        # 确保数据是数值型
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(data) < 2:
            return 0, 1
            
        corr_matrix = data.corr()
        n = len(data)
        p = data.shape[1]
        
        chi_square = - (n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(corr_matrix))
        df = p * (p - 1) / 2
        p_value = 1 - chi2.cdf(chi_square, df)
        
        return chi_square, p_value
    
    def calculate_validity(self):
        """计算结构效度指标"""
        validity_results = {}
        
        # 提取所有量表题目进行因子分析
        all_scale_items = []
        for items in self.scales.values():
            all_scale_items.extend([item for item in items if item in self.data.columns])
        
        if all_scale_items:
            scale_data = self.data[all_scale_items].dropna()
            
            if len(scale_data) > 0:
                # KMO和Bartlett球形检验
                kmo_value = self.calculate_kmo(scale_data)
                chi_square, p_value = self.bartlett_sphericity(scale_data)
                
                validity_results['KMO取样适切性'] = round(kmo_value, 3)
                validity_results['Bartlett球形检验_卡方'] = round(chi_square, 1)
                validity_results['Bartlett球形检验_p值'] = p_value
                
                # 探索性因子分析
                corr_matrix = scale_data.corr()
                eigenvalues, eigenvectors = eigh(corr_matrix)
                
                # 按特征值排序
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                
                # 计算方差解释率
                total_variance = np.sum(eigenvalues)
                variance_explained = eigenvalues / total_variance
                cumulative_variance = np.cumsum(variance_explained)
                
                validity_results['特征值'] = eigenvalues
                validity_results['方差解释率'] = variance_explained
                validity_results['累计方差解释率'] = cumulative_variance
                
        return validity_results
    
    def calculate_discriminant_validity(self):
        """计算区分效度 - 各维度间的相关性"""
        scale_scores = {}
        
        # 计算各维度总分
        for scale_name, items in self.scales.items():
            available_items = [item for item in items if item in self.data.columns]
            if available_items:
                scale_scores[scale_name] = self.data[available_items].mean(axis=1)
        
        if scale_scores:
            scale_df = pd.DataFrame(scale_scores).dropna()
            
            # 计算维度间相关系数矩阵
            correlation_matrix = scale_df.corr()
            return correlation_matrix
        else:
            return pd.DataFrame()
    
    def comprehensive_analysis(self):
        """执行综合分析并生成报告"""
        print("=== 测量工具信度与效度分析报告 ===\n")
        
        # 1. 信度分析
        print("1. 信度分析结果:")
        reliability_df = self.calculate_reliability()
        if not reliability_df.empty:
            print(reliability_df.to_string(index=False))
        else:
            print("无法计算信度 - 数据不足")
        print()
        
        # 2. 效度分析
        print("2. 效度分析结果:")
        validity_results = self.calculate_validity()
        if validity_results:
            for key, value in validity_results.items():
                if key not in ['特征值', '方差解释率', '累计方差解释率']:
                    print(f"{key}: {value}")
                elif key in ['方差解释率', '累计方差解释率']:
                    print(f"{key} (前5个因子): {[round(x, 3) for x in value[:5]]}")
        else:
            print("无法计算效度 - 数据不足")
        print()
        
        # 3. 区分效度
        print("3. 区分效度 - 各维度间相关系数矩阵:")
        corr_matrix = self.calculate_discriminant_validity()
        if not corr_matrix.empty:
            print(corr_matrix.round(3))
        else:
            print("无法计算区分效度 - 数据不足")
        
        return {
            'reliability': reliability_df,
            'validity': validity_results,
            'discriminant_validity': corr_matrix
        }

# 使用示例
# df = pd.read_csv('your_survey_data.csv')
# analyzer = SurveyReliabilityValidity(df)
# results = analyzer.comprehensive_analysis()


df = pd.read_csv('survey_data.csv')

analyzer = SurveyReliabilityValidity(df)
results = analyzer.comprehensive_analysis()