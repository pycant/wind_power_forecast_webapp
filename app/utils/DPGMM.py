"""
基于DPGMM的风电数据异常检测改进方案
版本：v2.3（修复标签越界与阈值问题，增加评估模块）
最后更新：2025-2-15
"""

# ========================
# 1. 依赖库与配置
# ========================
import pandas as pd
import numpy as np
import json
import logging
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import chi2, gaussian_kde, loguniform
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
# 配置日志模块
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('detection.log'), logging.StreamHandler()]
)

class EnhancedWindTurbineAnalyzer:
    def __init__(self,data,config_path='utils/p_configure.json'):
        # ========================
        # 1.1 数据加载与预处理
        # ========================
        self.config = self._load_config(config_path)
        
        self.log_path = self.config['DPGMM_settings']['log_path']
        self.output_path = self.config['DPGMM_settings']['output_path']
        
        try:
            self.data = data
            logging.info(f"成功加载数据，共 {len(self.data)} 条记录")
        except Exception as e:
            logging.critical(f"数据加载失败: {str(e)}")
            raise

        # 功率单位转换
        # self.data['Power (kW)'] = self.data['Power (MW)'] * 1000
        
        # ========================
        # 1.2 动态参数配置
        # ========================
        self.adaptive_params = self.config['DPGMM_settings']['adaptive_params']
        if [self.config['DPGMM_settings']['more_function_settings']['active']] == 'true':
            self.rated_power = self.config['DPGMM_settings']['more_function_settings']['rated_power']    # 额定功率 (kW)
            self.cut_in_speed = self.config['DPGMM_settings']['more_function_settings']['cut_in_speed']  # 切入风速 (m/s)
            self.cut_out_speed = self.config['DPGMM_settings']['more_function_settings']['cut_out_speed']  # 切出风速 (m/s)
            self.min_samples_per_bin = 50 
        
        # ========================
        # 1.3 初始化结果列
        # ========================
        self.data['Is Anomaly'] = False
        self.data['Anomaly Type'] = 'Normal'
    
    def _load_config(self, path):
        with open(path) as f:
            return json.load(f)
    
    # ========================
    # 2. 核心处理流程
    # ========================
    def analyze(self):
        """主分析流程"""
        try:
            self._power_binning()
            self._feature_engineering()
            self._hybrid_anomaly_detection()
            self._post_processing()
            return self.data
        except Exception as e:
            logging.error(f"分析流程中断: {str(e)}", exc_info=True)
            raise
    
    # ========================
    # 3. 功率区间划分
    # ========================
    def _power_binning(self):
        """动态分箱策略"""
        try:
            # 分位数分箱（每5%一个区间）
            bin_edges = np.percentile(
                self.data['Power (kW)'].dropna(), 
                np.linspace(0, 100, 21)
            )
            self.data['Power Bin'] = pd.cut(
                self.data['Power (kW)'], 
                bins=bin_edges, 
                include_lowest=True
            )
            
            # 过滤无效区间
            valid_bins = []
            for bin_value in self.data['Power Bin'].unique():
                bin_count = len(self.data[self.data['Power Bin'] == bin_value])
                if bin_count >= self.min_samples_per_bin:
                    valid_bins.append(bin_value)
                else:
                    logging.warning(f"分箱 {bin_value} 样本不足 ({bin_count} < {self.min_samples_per_bin})")
            
            self.data = self.data[self.data['Power Bin'].isin(valid_bins)].copy()
            logging.info(f"有效分箱数: {len(valid_bins)}")
        except Exception as e:
            logging.error(f"分箱失败: {str(e)}")
            raise
    
    # ========================
    # 4. 特征工程
    # ========================
    def _feature_engineering(self):
        """增强型特征构造"""
        try:
            # 时间窗口特征
            self.data['Power_1h_std'] = self.data['Power (kW)'].rolling(4, min_periods=1).std()
            self.data['Wind_1h_mean'] = self.data['Wind speed - at the height of wheel hub (m/s)'].rolling(4, min_periods=1).mean()
            
            # 物理特征
            self.data['Wind_Power_Ratio'] = self.data['Power (kW)'] / (
                self.data['Wind speed - at the height of wheel hub (m/s)'] + 1e-6)
            
            # 理论功率残差
            self.data['Power Residual'] = self.data['Power (kW)'] - self._theoretical_power()
        except Exception as e:
            logging.error(f"特征工程失败: {str(e)}")
            raise
    
    def _theoretical_power(self):
        """修正后的理论功率计算"""
        try:
            v = self.data['Wind speed - at the height of wheel hub (m/s)'].values
            def power_formula(x):
                return self.rated_power * (x**3 - self.cut_in_speed**3) / (
                    self.cut_out_speed**3 - self.cut_in_speed**3)
            
            # 三阶段计算：低于切入、正常区间、高于切出
            theoretical_power = np.piecewise(
                v,
                [v < self.cut_in_speed, 
                 (v >= self.cut_in_speed) & (v < self.cut_out_speed), 
                 v >= self.cut_out_speed],
                [0, power_formula, 0]
            )
            return theoretical_power
        except Exception as e:
            logging.error(f"理论功率计算失败: {str(e)}")
            raise
    
    # ========================
    # 5. 混合异常检测
    # ========================
    def _hybrid_anomaly_detection(self):
        """三级检测体系"""
        try:
            self._rule_based_detection()
            self._dpgmm_detection()
            self._ensemble_validation()
        except Exception as e:
            logging.error(f"异常检测失败: {str(e)}")
            raise
    
    def _rule_based_detection(self):
        """基于物理规则的检测"""
        try:
            # 过载检测
            overload_mask = self.data['Power (kW)'] > self.rated_power * 1.15
            self.data.loc[overload_mask, 'Anomaly Type'] = 'Overload'
            
            # 零功率异常
            zero_power_mask = (self.data['Power (kW)'] == 0) & (
                self.data['Wind speed - at the height of wheel hub (m/s)'] > self.cut_in_speed)
            self.data.loc[zero_power_mask, 'Anomaly Type'] = 'ZeroPower'
            
            # 更新异常标记
            rule_based = self.data['Anomaly Type'] != 'Normal'
            self.data['Is Anomaly'] = rule_based
            logging.info(f"规则检测发现异常: {rule_based.sum()} 条")
        except Exception as e:
            logging.error(f"规则检测失败: {str(e)}")
            raise
    
    def _dpgmm_detection(self):
        """基于DPGMM的统计检测"""
        features = [
            'Wind speed - at the height of wheel hub (m/s)',
            'Power (kW)',
            'Power Residual',
            'Wind_Power_Ratio'
        ]
        
        for bin_value in self.data['Power Bin'].cat.categories:
            try:
                bin_data = self.data[self.data['Power Bin'] == bin_value]
                X = bin_data[features].dropna()
                
                if len(X) < self.min_samples_per_bin:
                    logging.warning(f"分箱 {bin_value} 有效样本不足 ({len(X)} < {self.min_samples_per_bin})")
                    continue
                    
                # 动态参数选择
                best_model = self._select_best_model(X)
                best_model.fit(X)
                labels = best_model.predict(X)
                
                # 标签有效性验证
                valid_labels = [l for l in np.unique(labels) if l < best_model.n_components]
                if len(valid_labels) == 0:
                    logging.warning(f"分箱 {bin_value} 未找到有效聚类")
                    continue
                
                # 马氏距离计算
                dists = self._enhanced_mahalanobis(X, labels, best_model)
                
                # 动态阈值
                threshold = self._adaptive_threshold(dists, bin_value)
                anomalies = dists > threshold
                
                # 标记异常
                self.data.loc[X.index, 'Is Anomaly'] |= anomalies
                self.data.loc[X.index[anomalies], 'Anomaly Type'] = 'Statistical'
                logging.info(f"分箱 {bin_value} 发现统计异常: {anomalies.sum()} 条")
                
            except Exception as e:
                logging.error(f"分箱 {bin_value} 处理失败: {str(e)}")
                continue
    
    def _select_best_model(self, X):
        """贝叶斯优化选择模型参数"""
        try:
            param_dist = {
                'n_components': range(*self.adaptive_params['n_components_range']),
                'covariance_type': self.adaptive_params['covariance_types'],
                'weight_concentration_prior': loguniform(*self.adaptive_params['alpha_range'])
            }
            
            search = RandomizedSearchCV(
                BayesianGaussianMixture(
                    weight_concentration_prior_type='dirichlet_process',
                    random_state=42,
                    max_iter=1000
                ),
                param_dist,
                n_iter=10,
                scoring=self._bic_score,
                cv=3,
                n_jobs=-1
            )
            search.fit(X)
            logging.info(f"最佳模型参数: {search.best_params_}")
            return search.best_estimator_
        except Exception as e:
            logging.error(f"模型选择失败: {str(e)}")
            raise
    
    def _bic_score(self, estimator, X):
        """自定义BIC评分"""
        try:
            estimator.fit(X)
            log_likelihood = estimator.score(X)
            d = X.shape[1]
            num_params = estimator.n_components * (d + d * (d + 1) // 2 + 1) - 1
            bic = -2 * log_likelihood + num_params * np.log(X.shape[0])
            return bic
        except Exception as e:
            logging.error(f"BIC计算失败: {str(e)}")
            return np.inf
    
    def _enhanced_mahalanobis(self, X, labels, model):
        """鲁棒马氏距离计算"""
        try:
            n_components_actual = model.covariances_.shape[0]
            valid_labels = [l for l in np.unique(labels) if l < n_components_actual]
            
            if len(valid_labels) == 0:
                logging.warning(f"无有效聚类标签")
                return np.zeros(len(X))
            
            # 标签过滤与重映射
            valid_mask = np.isin(labels, valid_labels)
            X_filtered = X[valid_mask]
            labels_filtered = labels[valid_mask]
            
            if len(X_filtered) == 0:
                return np.zeros(len(X))
            
            # 标签重映射
            label_mapping = {old: new for new, old in enumerate(np.unique(labels_filtered))}
            labels_remapped = np.array([label_mapping[l] for l in labels_filtered])
            
            dists = np.full(len(X), np.inf)
            unique_labels_remapped = np.unique(labels_remapped)
            
            # 协方差正则化
            epsilon = 1e-3
            cov_invs = {}
            for label in unique_labels_remapped:
                original_label = valid_labels[label]
                cov = model.covariances_[original_label] + epsilon * np.eye(model.covariances_.shape[2]) * np.trace(model.covariances_[original_label])
                
                # 条件数检查
                cond_num = np.linalg.cond(cov)
                if cond_num > 1e6:
                    logging.warning(f"协方差矩阵条件数过高 ({cond_num:.1e})，增加正则化")
                    cov += np.eye(cov.shape[0]) * 1e-3
                
                cov_invs[label] = np.linalg.pinv(cov)
            
            # 向量化计算
            for label in unique_labels_remapped:
                mask = (labels_remapped == label)
                X_sub = X_filtered[mask]
                original_label = valid_labels[label]
                mean = model.means_[original_label]
                cov_inv = cov_invs[label]
                
                diff = X_sub - mean
                dist_sub = np.sqrt(np.einsum('...i,...i', diff @ cov_inv, diff))
                
                if len(dist_sub) != mask.sum():
                    raise ValueError(f"距离计算维度不匹配 {len(dist_sub)} vs {mask.sum()}")
                
                dists[valid_mask][mask] = dist_sub
            
            # 处理无效值
            dists[np.isnan(dists)] = np.inf
            return dists
            
        except Exception as e:
            logging.error(f"马氏距离计算错误: {str(e)}", exc_info=True)
            return np.zeros(len(X))
    
    def _adaptive_threshold(self, dists, bin_value):
        """动态阈值策略"""
        try:
            valid_dists = dists[np.isfinite(dists)]
            
            # 小样本处理
            if len(valid_dists) < 10:
                logging.warning(f"分箱 {bin_value} 有效样本不足 ({len(valid_dists)})")
                return np.inf
            
            # 低功率区使用核密度估计
            if bin_value.right < 0.2 * self.rated_power:
                kde = gaussian_kde(valid_dists)
                x = np.linspace(np.min(valid_dists), np.max(valid_dists), 1000)
                cdf = kde.integrate_box_1d(-np.inf, x)
                threshold = x[np.where(cdf >= 0.995)[0][0]]
            # 高功率区使用混合阈值
            else:
                threshold = max(
                    chi2.ppf(0.99, df=4),  # 置信度提升到99%
                    np.mean(valid_dists) + 3 * np.std(valid_dists)
                )
            return threshold
        except Exception as e:
            logging.error(f"阈值计算失败: {str(e)}")
            return np.inf
    
    def _ensemble_validation(self):
        """集成模型验证"""
        try:
            features = [
                'Wind speed - at the height of wheel hub (m/s)',
                'Power (kW)',
                'Power Residual'
            ]
            X = self.data[features].dropna()
            
            # 孤立森林检测
            iso_forest = IsolationForest(contamination=0.02, random_state=42)
            iso_pred = iso_forest.fit_predict(X)
            
            # LOF检测
            lof = LocalOutlierFactor(novelty=True, contamination=0.02)
            lof.fit(X)
            lof_pred = lof.predict(X)
            
            # 逻辑或验证
            final_anomaly = self.data['Is Anomaly'] | (iso_pred == -1) | (lof_pred == -1)
            self.data['Is Anomaly'] = final_anomaly
            
            # 有效性检查
            anomaly_count = self.data['Is Anomaly'].sum()
            if anomaly_count == 0:
                logging.warning("未检测到任何异常，请检查阈值设置")
            else:
                logging.info(f"集成验证后异常总数: {anomaly_count}")
        except Exception as e:
            logging.error(f"集成验证失败: {str(e)}")
            raise
    
    # ========================
    # 6. 后处理与评估
    # ========================
    def _post_processing(self):
        """结果后处理"""
        try:
            # 异常类型修正
            overload_mask = self.data['Anomaly Type'] == 'Overload'
            self.data.loc[overload_mask, 'Is Anomaly'] = True
            
            self._visualize_results()
            # self._save_results()
        except Exception as e:
            logging.error(f"后处理失败: {str(e)}")
            raise
    
    def _visualize_results(self):
        """增强可视化"""
        try:
            plt.figure(figsize=(15, 8))
            colors = {'Normal': 'gray', 'Overload': 'red', 'ZeroPower': 'blue', 'Statistical': 'orange'}
            
            for anomaly_type, color in colors.items():
                mask = self.data['Anomaly Type'] == anomaly_type
                plt.scatter(
                    self.data.loc[mask, 'Wind speed - at the height of wheel hub (m/s)'],
                    self.data.loc[mask, 'Power (kW)'],
                    c=color,
                    label=anomaly_type,
                    alpha=0.6
                )
                
            plt.xlabel('Wind Speed (m/s)')
            plt.ylabel('Power (kW)')
            plt.title('Enhanced Anomaly Detection Results')
            plt.legend()
            plt.savefig('detection_results.png')
            plt.close()
            logging.info("可视化结果已保存")
        except Exception as e:
            logging.error(f"可视化失败: {str(e)}")
    
    def _save_results(self):
        """结果保存与性能评估"""
        try:
            # 过滤异常数据
            anomalies = self.data[self.data['Is Anomaly']]

            # 保存异常数据到 CSV 文件
            anomalies.to_csv('enhanced_anomalies.csv')
            logging.info(f"异常结果已保存，共 {self.data['Is Anomaly'].sum()} 条异常记录")

            # 评估指标计算（假设数据包含真实标签列 'True_Anomaly'）
            if 'True_Anomaly' in self.data.columns:
                y_true = self.data['True_Anomaly'].astype(int)
                y_pred = self.data['Is Anomaly'].astype(int)
                
                # 生成分类报告
                report = classification_report(
                    y_true,
                    y_pred,
                    target_names=['Normal', 'Anomaly'],
                    output_dict=True
                )
                
                # 保存详细评估指标
                with open('detection_metrics.txt', 'w') as metrics_file:
                    metrics_file.write("=== 分类评估报告 ===\n")
                    metrics_file.write(f"精确率 (Precision): {report['Anomaly']['precision']:.4f}\n")
                    metrics_file.write(f"召回率 (Recall): {report['Anomaly']['recall']:.4f}\n")
                    metrics_file.write(f"F1分数 (F1-score): {report['Anomaly']['f1-score']:.4f}\n")
                    metrics_file.write(f"准确率 (Accuracy): {report['accuracy']:.4f}\n\n")
                    
                    metrics_file.write("=== 详细分类报告 ===\n")
                    metrics_file.write(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
                    
                logging.info("模型评估指标已保存到 detection_metrics.txt")
            else:
                logging.warning("数据中未找到真实标签列 'True_Anomaly'，跳过评估指标计算")

            # 基础统计信息
            with open('detection_statistics.txt', 'w') as stats_file:
                stats_file.write(self.data[self.data['Is Anomaly']]['Anomaly Type'].value_counts().to_string())
                stats_file.write(f"\n异常比例: {self.data['Is Anomaly'].mean():.2%}\n")
                stats_file.write(f"总样本数: {len(self.data)}\n")
                stats_file.write(f"异常样本数: {self.data['Is Anomaly'].sum()}\n")

            logging.info("异常统计信息已保存到 detection_statistics.txt")
            logging.info("所有结果已成功保存")
        except Exception as e:
            logging.error(f"结果保存失败: {str(e)}")


