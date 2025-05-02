import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.impute import KNNImputer
from datetime import datetime
import time
import json
from utils.DPGMM import EnhancedWindTurbineAnalyzer as EWTA
import logging

p_result_path = []
m_result_path = []

class DataCleaner:
    def __init__(self, config_path='utils/p_configure.json'):
        self.config = self._load_config(config_path)
        self.raw_data = pd.read_csv(f"data/raw/{self.config['current_file']}")
        self.processed_data = None
        self.cleaned_data = None
        self.message = {'status': [], 'warnings': [], 'operations': [],'errors': []}
        print(self.raw_data.head(10))
        
    def _load_config(self, path):
        with open(path) as f:
            return json.load(f)

    def _detect_columns(self):
        """根据位置识别列类型"""
        try:
            # 时间列固定为第一列
            self.time_col = self.raw_data.columns[0]
            self.time_data = self.raw_data[self.time_col].copy()
            # 因变量列固定为最后一列
            self.target_col = self.raw_data.columns[-1]
            self.target_data = self.raw_data[self.target_col].copy()
            # 中间列为因素列
            self.factor_cols = self.raw_data.columns[1:-1].tolist()
            print(self.raw_data.columns[0],self.time_col)
            
            self.message['status'].append(f"列识别完成 | 时间列: {self.time_col} | 因变量列: {self.target_col} | 因素列数: {len(self.factor_cols)}")
        except IndexError:
                raise ValueError("数据列数不足，至少需要2列（时间列+因变量列）")
    def _validate_data(self):
        # """强化时间列验证"""
        # try:
        # 转换时间列（允许无效值）
        self.time_data= pd.to_datetime(
            self.time_data,
            format=self.config['data_format'].get('time_format', '%Y-%m-%d %H:%M:%S'),
            errors='coerce'  # 关键修改：无效时间转为NaT
        )
            
            # 检查有效时间记录
        valid_mask = self.time_data.notna()
        # print(valid_mask)
        print("有效时间记录数：",valid_mask.sum())
        try:
            if valid_mask.sum() == 0:

                raise ValueError("时间列无有效数据，请检查时间格式配置")
                
            # 记录无效时间数量
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                self.message['warnings'].append(
                    f"发现{invalid_count}条无效时间记录，已自动删除"
                )
                # print(self.raw_data.head(10))
                self.raw_data = self.raw_data[valid_mask].copy()
            print(self.raw_data)  
            # 确保时间连续性
            self._ensure_time_continuity()
            # print(self.raw_data.head(10))
        except Exception as e:
            self.message['errors'].append(f"时间列验证失败：{str(e)}")
            raise
    

    def _handle_missing_values(self):
        """基于位置的空值处理"""
        strategy = self.config['data_format'].get('missing_value_strategy', 'interpolate')
        max_missing = self.config['data_format'].get('max_missing_ratio', 0.3)
        
        # 时间列特殊处理
        time_missing = self.raw_data[self.time_col].isnull().sum()
        if time_missing > 0:
            self.message['warnings'].append(f"时间列存在 {time_missing} 个空值，已删除对应行")
            self.raw_data.dropna(subset=[self.time_col], inplace=True)

        # 因素列处理
        cols_to_drop = []
        # print([self.target_col] + self.factor_cols) 
        # print(self.raw_data.head(10))
        for idx, col in enumerate([self.target_col] + self.factor_cols):
            # print(col)
            
            # print(self.raw_data[col].head(10))
            missing_ratio = self.raw_data[col].isnull().mean()
            # print(f"列 {col} 空值率 {missing_ratio:.2%}")
            
            if missing_ratio > max_missing:
                if idx == 0:  # 因变量列
                    raise ValueError("因变量列空值比例超过阈值")
                cols_to_drop.append(col)
                self.message['operations'].append(f"删除列 {col} | 空值率 {missing_ratio:.2%}")
                continue
                
            if strategy == 'drop':
                initial_count = len(self.raw_data)
                self.raw_data.dropna(subset=[col], inplace=True)
                dropped = initial_count - len(self.raw_data)
                self.message['operations'].append(f"删除 {dropped} 行 | 列 {col}")
            elif strategy == 'interpolate':
                filled = self.raw_data[col].isnull().sum()
                self._smart_interpolate(col)
                self.message['operations'].append(f"填补 {filled} 空值 | 列 {col}")
            elif strategy == 'knn':
                self._knn_impute(col)

        # 执行列删除
        self.raw_data.drop(cols_to_drop, axis=1, inplace=True)
        # 更新因素列
        self.factor_cols = [c for c in self.factor_cols if c not in cols_to_drop]

    def _smart_interpolate(self, col):
        """智能插值策略"""
        # 因变量列使用前向填充
        if col == self.target_col:
            self.raw_data[col] = self.raw_data[col].ffill()
        # 因素列使用线性插值
        else:
            self.raw_data[col] = self.raw_data[col].interpolate(
                method='linear', 
                limit_direction='both'
            )

    def _ensure_time_continuity(self):
        """生成连续时间序列（修复NaT问题）"""
        # 获取有效时间范围
        # print(self.raw_data[self.time_col].head(10))
        # print("原始数据时间范围：")
        start_time = self.raw_data[self.time_col].min()
        end_time = self.raw_data[self.time_col].max()
        # print("start_time:",start_time,"end_time:",end_time)
        # 检测时间频率
        freq = self._detect_time_freq()
        # print("freq:",freq)
        # 生成完整时间范围
        full_range = pd.date_range(
                start=start_time,
                end=end_time,
                freq=pd.infer_freq(self.raw_data[self.time_col].head(10)),
                inclusive='both'
            )
        # print("a1",self.raw_data)
        # 重新索引并插值
        # print('full_range:',full_range)
        # numeric_cols = self.raw_data.select_dtypes(include=np.number).columns
        # self.raw_data[numeric_cols] = (
        #     self.raw_data.set_index(self.time_col)[numeric_cols]
        #     .reindex(full_range)
        #     .interpolate(method='time')
        #     .values
        #     )
        
        # print("a2",self.raw_data)
        self.raw_data.rename(columns={'index': self.time_col}, inplace=True)
        # print("a3",self.raw_data)
        self.message['operations'].append(f"时间序列补全 | 新增 {len(full_range)-len(self.raw_data)} 个时间点")

    def _detect_time_freq(self):
        """更稳健的频率检测"""
        try:
            diffs = self.raw_data[self.time_col].diff().dropna()
            if len(diffs) == 0:
                return '15ME'  # 默认15分钟频率
            return diffs.mode()[0]
        except:
            return '15ME'  # 容错默认值

    def format_data(self):
        """数据处理主流程"""
        try:
            self._detect_columns()
            self._validate_data()
            self._handle_missing_values()
            self._ensure_time_continuity()
            
            # 最终整理
            self.processed_data = self.raw_data[[self.time_col] + self.factor_cols + [self.target_col]]
            self.processed_data.sort_values(self.time_col, inplace=True)
            self.processed_data.reset_index(drop=True, inplace=True)
            
            self.message['status'].append("数据处理完成")
            return self.processed_data
            
        except Exception as e:
            self.message['errors'] = str(e)
            raise

    def save_processed_data(self):
        """保存数据并返回处理摘要"""
        if self.processed_data is not None:
            save_path = f"data/processed/{self.config['current_file'][:-4]}{time.time()}.csv"
            self.processed_data.to_csv(save_path, index=False)
            
            report = {
                'original_shape': self.raw_data.shape,
                'processed_shape': self.processed_data.shape,
                'remaining_factors': len(self.factor_cols),
                'message': self.message
            }
            return save_path, report
        raise ValueError("数据处理未完成")
    
    
    # def drop_error_data(self,method='DPGMM'):
    #     """删除异常数据"""
    #     if method=='DPGMM':
    #         log_path = "log/"
    #         logging.info(f"开始加载数据文件: {log_path}")
    #         dpgmm = DPGMM(self.processed_data,self.config)
    #         dpgmm.fit()
    
    
    
    
    
    
# a=DataCleaner()
# a.format_data()
# a.save_processed_data()
# print(a.message)