import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import logging

class BehaviorFeatureExtractor(Model):
    """特徵提取器：從不同遊戲數據中提取行為特徵"""
    
    def __init__(self, input_dims, embedding_dim=64):
        super(BehaviorFeatureExtractor, self).__init__()
        self.input_dims = input_dims
        
        # 不同模態的特徵提取器
        self.time_feature_extractor = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(embedding_dim, activation='relu')
        ])
        
        self.error_feature_extractor = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(embedding_dim, activation='relu')
        ])
        
        self.pattern_feature_extractor = tf.keras.Sequential([
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dense(embedding_dim, activation='relu')
        ])
    
    def call(self, inputs):
        # 假設輸入是一個字典，包含不同類型的行為數據
        time_features = self.time_feature_extractor(inputs['response_times'])
        error_features = self.error_feature_extractor(inputs['error_patterns'])
        pattern_features = self.pattern_feature_extractor(inputs['interaction_sequences'])
        
        # 合併所有特徵
        combined_features = tf.concat([time_features, error_features, pattern_features], axis=1)
        return combined_features


class BehaviorAnalysisModel(Model):
    """行為分析模型：分析用戶行為並識別認知特徵"""
    
    def __init__(self, input_dims, num_cognitive_indicators=8):
        super(BehaviorAnalysisModel, self).__init__()
        self.feature_extractor = BehaviorFeatureExtractor(input_dims)
        
        # 行為分析層
        self.analysis_layers = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
        ])
        
        # 認知指標預測
        self.cognitive_indicator_predictor = layers.Dense(
            num_cognitive_indicators, activation='sigmoid')
        
        # 異常行為檢測
        self.anomaly_detector = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        analyzed_features = self.analysis_layers(features)
        
        # 輸出認知能力指標（注意力、記憶力等）
        cognitive_indicators = self.cognitive_indicator_predictor(analyzed_features)
        
        # 檢測異常行為模式
        anomaly_score = self.anomaly_detector(analyzed_features)
        
        return {
            'cognitive_indicators': cognitive_indicators,
            'anomaly_score': anomaly_score,
            'feature_vector': analyzed_features
        }
    
    def train_step(self, data):
        # 實現自定義訓練步驟（可包含多任務學習）
        x, y = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            
            # 計算多個損失（認知指標損失和異常檢測損失）
            cognitive_loss = tf.keras.losses.binary_crossentropy(
                y['cognitive_indicators'], predictions['cognitive_indicators'])
            anomaly_loss = tf.keras.losses.binary_crossentropy(
                y['anomaly_label'], predictions['anomaly_score'])
            
            # 總損失是兩個任務損失的加權和
            total_loss = cognitive_loss * 0.7 + anomaly_loss * 0.3
            
        # 計算梯度並更新模型參數
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 更新指標
        self.compiled_metrics.update_state(y, predictions)
        
        # 返回包含損失和指標的字典
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            'cognitive_loss': cognitive_loss,
            'anomaly_loss': anomaly_loss
        })
        
        return results


class GameSpecificBehaviorAnalyzer:
    """特定遊戲行為分析器：為每種遊戲類型定制分析方法"""
    
    def __init__(self, base_model, game_type):
        self.base_model = base_model
        self.game_type = game_type
        self.logger = logging.getLogger(f"behavior_analyzer_{game_type}")
        
        # 針對特定遊戲類型的特殊處理邏輯
        self.game_specific_processors = {
            'attention_calculation': self._process_attention_calculation,
            'time_recognition': self._process_time_recognition,
            'short_term_memory': self._process_short_term_memory,
            'drawing': self._process_drawing,
            'repeat_language': self._process_repeat_language,
            'naming': self._process_naming,
            'object_matching': self._process_object_matching,
            'spatial_concept': self._process_spatial_concept
        }
    
    def analyze(self, game_data):
        """分析特定遊戲中的用戶行為"""
        
        # 前處理：根據遊戲類型進行特定處理
        if self.game_type in self.game_specific_processors:
            processed_data = self.game_specific_processors[self.game_type](game_data)
        else:
            self.logger.warning(f"未知的遊戲類型: {self.game_type}，使用默認處理")
            processed_data = self._default_process(game_data)
        
        # 通過基礎模型進行分析
        analysis_results = self.base_model(processed_data)
        
        # 後處理：添加遊戲特定的解釋
        enriched_results = self._enrich_results(analysis_results, game_data)
        
        return enriched_results
    
    def _default_process(self, game_data):
        """默認數據處理方法"""
        # 確保生成適合模型輸入的數據形狀
        # 獲取性能指標
        metrics = game_data.get('performance_metrics', {})
        
        # 提取響應時間並確保其為2D張量
        response_time = metrics.get('response_time', 0.0)
        response_times = np.array([response_time] * 10).reshape(1, -1)  # 擴展為10個元素
        
        # 提取錯誤模式並確保其為2D張量
        error_rate = metrics.get('error_rate', 0.0)
        error_patterns = np.array([error_rate] * 10).reshape(1, -1)  # 擴展為10個元素
        
        # 創建模擬交互序列 (3D張量: batch_size, sequence_length, features)
        # 這裡我們使用一個簡單的重複序列來避免形狀不匹配錯誤
        interaction_sequences = np.zeros((1, 10, 10))  # batch_size=1, sequence_length=10, features=10
        
        return {
            'response_times': response_times,
            'error_patterns': error_patterns,
            'interaction_sequences': interaction_sequences
        }
    
    def _process_attention_calculation(self, game_data):
        """處理注意力和計算遊戲數據"""
        processed_data = self._default_process(game_data)
        
        # 額外處理：計算任務中的錯誤模式分類
        if 'calculation_errors' in game_data:
            calculation_errors = np.array(game_data['calculation_errors'])
            # 將計算錯誤轉換為分類特徵（例如：進位錯誤、運算符錯誤等）
            error_categories = self._categorize_calculation_errors(calculation_errors)
            processed_data['error_patterns'] = error_categories
            
        return processed_data
    
    def _process_time_recognition(self, game_data):
        """處理時間辨認遊戲數據"""
        # 實現特定處理邏輯
        processed_data = self._default_process(game_data)
        return processed_data
    
    def _process_short_term_memory(self, game_data):
        """處理短期記憶遊戲數據"""
        # 實現特定處理邏輯
        processed_data = self._default_process(game_data)
        
        # 額外處理：提取記憶模式特徵
        if 'memory_sequence' in game_data and 'user_response' in game_data:
            memory_features = self._extract_memory_features(
                game_data['memory_sequence'], 
                game_data['user_response']
            )
            processed_data['memory_features'] = memory_features
            
        return processed_data
    
    def _process_drawing(self, game_data):
        """處理圖形繪製遊戲數據"""
        # 實現特定處理邏輯
        processed_data = self._default_process(game_data)
        return processed_data
    
    def _process_repeat_language(self, game_data):
        """處理重複言語遊戲數據"""
        # 實現特定處理邏輯
        processed_data = self._default_process(game_data)
        return processed_data
    
    def _process_naming(self, game_data):
        """處理命名遊戲數據"""
        # 實現特定處理邏輯
        processed_data = self._default_process(game_data)
        return processed_data
    
    def _process_object_matching(self, game_data):
        """處理物件配對遊戲數據"""
        # 實現特定處理邏輯
        processed_data = self._default_process(game_data)
        return processed_data
    
    def _process_spatial_concept(self, game_data):
        """處理空間概念遊戲數據"""
        # 實現特定處理邏輯
        processed_data = self._default_process(game_data)
        
        # 額外處理：提取空間導航特徵
        if 'navigation_path' in game_data:
            spatial_features = self._extract_spatial_features(game_data['navigation_path'])
            processed_data['spatial_features'] = spatial_features
            
        return processed_data
    
    def _categorize_calculation_errors(self, calculation_errors):
        """將計算錯誤分類為不同類型"""
        # 實現分類邏輯
        # 這裡是示例實現
        error_categories = np.zeros((len(calculation_errors), 5))  # 假設有5種錯誤類型
        
        # 對每個錯誤進行分類處理
        # ...
        
        return error_categories
    
    def _extract_memory_features(self, original_sequence, user_response):
        """從記憶測試中提取特徵"""
        # 實現特徵提取邏輯
        # 例如：計算正確記憶的項目數量、錯誤的位置、替換模式等
        
        # 這裡是示例實現
        features = np.zeros(10)  # 假設提取10個特徵
        
        # 特徵提取處理
        # ...
        
        return features
    
    def _extract_spatial_features(self, navigation_path):
        """從空間導航路徑中提取特徵"""
        # 實現特徵提取邏輯
        # 例如：分析導航效率、繞路情況、方向感等
        
        # 這裡是示例實現
        features = np.zeros(8)  # 假設提取8個特徵
        
        # 特徵提取處理
        # ...
        
        return features
    
    def _enrich_results(self, analysis_results, game_data):
        """為分析結果添加遊戲特定的解釋和建議"""
        
        # 將TensorFlow張量轉換為Python字典
        python_results = {}
        for key, value in analysis_results.items():
            if hasattr(value, 'numpy'):
                # 轉換TensorFlow張量為NumPy數組
                numpy_value = value.numpy()
                # 對於標量轉換為簡單的Python對象
                if numpy_value.size == 1:
                    python_results[key] = float(numpy_value)
                else:
                    python_results[key] = numpy_value
            else:
                python_results[key] = value
        
        enriched_results = python_results.copy()
        
        # 根據遊戲類型和分析結果添加解釋
        if self.game_type == 'attention_calculation':
            # 檢查認知指標數組的第一個元素（注意力）
            cognitive_indicators = enriched_results.get('cognitive_indicators', None)
            if cognitive_indicators is not None and isinstance(cognitive_indicators, np.ndarray):
                if cognitive_indicators.size > 0 and cognitive_indicators.flat[0] < 0.5:
                    enriched_results['interpretations'] = ["用戶在持續注意力方面可能存在困難"]
                    enriched_results['suggestions'] = ["建議降低干擾元素，增加視覺提示"]
                else:
                    enriched_results['interpretations'] = ["用戶注意力表現在正常範圍內"]
                    enriched_results['suggestions'] = ["可以適度增加挑戰性"]
        elif self.game_type == 'short_term_memory':
            # 檢查認知指標數組的第二個元素（記憶力）
            cognitive_indicators = enriched_results.get('cognitive_indicators', None)
            if cognitive_indicators is not None and isinstance(cognitive_indicators, np.ndarray):
                if cognitive_indicators.size > 1 and cognitive_indicators.flat[1] < 0.5:
                    enriched_results['interpretations'] = ["用戶在短期記憶方面可能存在困難"]
                    enriched_results['suggestions'] = ["建議提供更多重複練習和記憶輔助"]
                else:
                    enriched_results['interpretations'] = ["用戶記憶力表現在正常範圍內"]
                    enriched_results['suggestions'] = ["可以適度增加記憶長度"]
        
        # 對於未知的遊戲類型，添加默認解釋
        if 'interpretations' not in enriched_results:
            enriched_results['interpretations'] = [f"用戶在{self.game_type}遊戲中表現平均"]
            enriched_results['suggestions'] = ["繼續當前訓練計劃"]
        
        return enriched_results


def create_behavior_analysis_model(input_dims, num_cognitive_indicators=8):
    """創建行為分析模型實例"""
    model = BehaviorAnalysisModel(input_dims, num_cognitive_indicators)
    return model


def load_pretrained_model(model_path):
    """載入預訓練的行為分析模型"""
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"成功載入預訓練模型: {model_path}")
        return model
    except Exception as e:
        logging.error(f"載入模型時出錯: {e}")
        logging.info("創建新模型...")
        return create_behavior_analysis_model({
            'response_times': 10,
            'error_patterns': 20,
            'interaction_sequences': (30, 5)  # 序列長度為30，每個時間步5個特徵
        })


def get_game_analyzer(game_type, model_path=None):
    """獲取特定遊戲類型的行為分析器"""
    
    # 載入基礎模型
    if model_path:
        base_model = load_pretrained_model(model_path)
    else:
        base_model = create_behavior_analysis_model({
            'response_times': 10,
            'error_patterns': 20,
            'interaction_sequences': (30, 5)
        })
    
    # 創建遊戲特定的分析器
    analyzer = GameSpecificBehaviorAnalyzer(base_model, game_type)
    
    return analyzer 