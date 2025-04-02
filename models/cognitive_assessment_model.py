import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import logging
from collections import defaultdict


class CognitiveFeatureExtractor(Model):
    """認知特徵提取器：從各類型遊戲數據中提取認知功能特徵"""
    
    def __init__(self, feature_sizes):
        super(CognitiveFeatureExtractor, self).__init__()
        
        # 為不同類型的特徵創建不同的處理網絡
        self.feature_networks = {}
        
        # 注意力特徵網絡
        if 'attention' in feature_sizes:
            self.feature_networks['attention'] = tf.keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(feature_sizes['attention'],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu')
            ])
        
        # 記憶力特徵網絡
        if 'memory' in feature_sizes:
            self.feature_networks['memory'] = tf.keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(feature_sizes['memory'],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu')
            ])
        
        # 語言能力特徵網絡
        if 'language' in feature_sizes:
            self.feature_networks['language'] = tf.keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(feature_sizes['language'],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu')
            ])
        
        # 視空間能力特徵網絡
        if 'visuospatial' in feature_sizes:
            self.feature_networks['visuospatial'] = tf.keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(feature_sizes['visuospatial'],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu')
            ])
        
        # 執行功能特徵網絡
        if 'executive' in feature_sizes:
            self.feature_networks['executive'] = tf.keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(feature_sizes['executive'],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu')
            ])
            
        # 融合層
        fusion_input_size = sum(32 for _ in self.feature_networks)
        self.fusion_network = tf.keras.Sequential([
            layers.Dense(fusion_input_size, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu')
        ])
    
    def call(self, inputs, training=False):
        feature_outputs = []
        
        # 通過相應的網絡處理每種特徵
        for feature_type, network in self.feature_networks.items():
            if feature_type in inputs:
                feature_output = network(inputs[feature_type], training=training)
                feature_outputs.append(feature_output)
        
        # 如果沒有特徵，返回空張量
        if not feature_outputs:
            return tf.zeros((inputs.get(list(inputs.keys())[0]).shape[0], 64))
        
        # 合併所有特徵
        combined_features = tf.concat(feature_outputs, axis=1)
        
        # 通過融合網絡
        fused_features = self.fusion_network(combined_features, training=training)
        
        return fused_features


class CognitiveAssessmentModel(Model):
    """認知功能評估模型：評估用戶的認知功能狀態"""
    
    def __init__(self, feature_sizes, num_cognitive_domains=5):
        super(CognitiveAssessmentModel, self).__init__()
        
        # 特徵提取器
        self.feature_extractor = CognitiveFeatureExtractor(feature_sizes)
        
        # 認知功能評分層
        self.assessment_layers = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3)
        ])
        
        # 各認知域評分
        self.domain_scorers = {}
        cognitive_domains = [
            'attention', 'memory', 'language', 
            'visuospatial', 'executive'
        ][:num_cognitive_domains]
        
        for domain in cognitive_domains:
            self.domain_scorers[domain] = layers.Dense(1, activation='sigmoid')
        
        # 整體評分
        self.overall_scorer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # 提取特徵
        features = self.feature_extractor(inputs, training=training)
        
        # 高級特徵表示
        assessment_features = self.assessment_layers(features, training=training)
        
        # 計算各認知域評分
        domain_scores = {}
        for domain, scorer in self.domain_scorers.items():
            domain_scores[domain] = scorer(assessment_features)
        
        # 計算整體評分
        overall_score = self.overall_scorer(assessment_features)
        
        # 輸出結果
        results = {
            'domain_scores': domain_scores,
            'overall_score': overall_score,
            'features': assessment_features
        }
        
        return results
    
    def train_step(self, data):
        """自定義訓練步驟，處理多任務學習"""
        x, y = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            
            # 計算各域損失
            domain_losses = {}
            for domain in self.domain_scorers.keys():
                if domain in y:
                    domain_losses[domain] = tf.keras.losses.binary_crossentropy(
                        y[domain], predictions['domain_scores'][domain])
            
            # 如果有整體評分目標，計算該損失
            if 'overall' in y:
                overall_loss = tf.keras.losses.binary_crossentropy(
                    y['overall'], predictions['overall_score'])
            else:
                # 否則，整體損失為所有域損失的平均
                overall_loss = tf.reduce_mean(list(domain_losses.values()))
            
            # 總損失是域損失和整體損失的加權和
            domain_loss_avg = tf.reduce_mean(list(domain_losses.values()))
            total_loss = domain_loss_avg * 0.7 + overall_loss * 0.3
        
        # 計算梯度並更新模型
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 更新指標
        self.compiled_metrics.update_state(y, predictions)
        
        # 返回損失和指標
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            'loss': total_loss,
            'domain_loss': domain_loss_avg,
            'overall_loss': overall_loss
        })
        for domain, loss in domain_losses.items():
            results[f'{domain}_loss'] = loss
        
        return results


class GameSpecificFeatureExtractor:
    """遊戲特定特徵提取器：從不同遊戲中提取認知相關特徵"""
    
    def __init__(self):
        self.logger = logging.getLogger("cognitive_feature_extractor")
        
        # 特徵提取方法映射
        self.feature_extractors = {
            'attention_calculation': self._extract_attention_calculation_features,
            'time_recognition': self._extract_time_recognition_features,
            'short_term_memory': self._extract_short_term_memory_features,
            'drawing': self._extract_drawing_features,
            'repeat_language': self._extract_repeat_language_features,
            'naming': self._extract_naming_features,
            'object_matching': self._extract_object_matching_features,
            'spatial_concept': self._extract_spatial_concept_features
        }
    
    def extract_features(self, game_data):
        """從遊戲數據中提取認知特徵"""
        
        # 按認知域組織特徵
        features = {
            'attention': [],
            'memory': [],
            'language': [],
            'visuospatial': [],
            'executive': []
        }
        
        # 添加基本表現數據
        if 'performance_metrics' in game_data:
            metrics = game_data['performance_metrics']
            
            # 共通指標
            if 'accuracy' in metrics:
                for domain in features.keys():
                    features[domain].append(metrics['accuracy'])
            
            if 'response_time' in metrics:
                normalized_time = self._normalize_response_time(
                    metrics['response_time'], game_data.get('game_type', ''))
                features['attention'].append(normalized_time)
                features['executive'].append(normalized_time)
        
        # 按遊戲類型提取特定特徵
        if 'game_type' in game_data and game_data['game_type'] in self.feature_extractors:
            game_type = game_data['game_type']
            game_features = self.feature_extractors[game_type](game_data)
            
            # 合併特徵
            for domain in features.keys():
                if domain in game_features:
                    features[domain].extend(game_features[domain])
        
        # 轉換為numpy數組
        for domain in features.keys():
            if features[domain]:
                features[domain] = np.array(features[domain], dtype=np.float32)
            else:
                features[domain] = np.array([], dtype=np.float32)
        
        return features
    
    def _normalize_response_time(self, response_time, game_type):
        """標準化響應時間"""
        # 根據遊戲類型設定基準時間
        if game_type == 'attention_calculation':
            baseline = 5.0  # 秒
        elif game_type in ['naming', 'repeat_language']:
            baseline = 3.0
        else:
            baseline = 4.0
        
        # 標準化: 快速響應接近1，慢響應接近0
        normalized = max(0, 1 - (response_time / (baseline * 2)))
        return normalized
    
    def _extract_attention_calculation_features(self, game_data):
        """提取注意力和計算遊戲特徵"""
        features = defaultdict(list)
        
        if 'calculation_data' in game_data:
            calc_data = game_data['calculation_data']
            
            # 計算錯誤率
            if 'errors' in calc_data and 'total_problems' in calc_data:
                error_rate = calc_data['errors'] / max(1, calc_data['total_problems'])
                features['attention'].append(1 - error_rate)
                features['executive'].append(1 - error_rate)
            
            # 計算速度
            if 'problems_per_minute' in calc_data:
                speed = min(calc_data['problems_per_minute'] / 10, 1.0)  # 標準化
                features['attention'].append(speed)
                features['executive'].append(speed)
                
            # 專注持續能力
            if 'attention_drops' in calc_data:
                sustained_attention = 1 - min(calc_data['attention_drops'] / 5, 1.0)
                features['attention'].append(sustained_attention)
        
        return features
    
    def _extract_time_recognition_features(self, game_data):
        """提取時間辨認遊戲特徵"""
        features = defaultdict(list)
        
        if 'time_recognition_data' in game_data:
            time_data = game_data['time_recognition_data']
            
            # 時間概念理解
            if 'time_concept_score' in time_data:
                features['memory'].append(time_data['time_concept_score'])
                features['executive'].append(time_data['time_concept_score'])
            
            # 環境線索識別能力
            if 'cue_recognition_score' in time_data:
                features['visuospatial'].append(time_data['cue_recognition_score'])
                features['memory'].append(time_data['cue_recognition_score'])
        
        return features
    
    def _extract_short_term_memory_features(self, game_data):
        """提取短期記憶遊戲特徵"""
        features = defaultdict(list)
        
        if 'memory_data' in game_data:
            memory_data = game_data['memory_data']
            
            # 記憶廣度
            if 'memory_span' in memory_data:
                normalized_span = min(memory_data['memory_span'] / 7, 1.0)
                features['memory'].append(normalized_span)
            
            # 記憶保持率
            if 'retention_rate' in memory_data:
                features['memory'].append(memory_data['retention_rate'])
            
            # 序列順序準確性
            if 'sequence_accuracy' in memory_data:
                features['memory'].append(memory_data['sequence_accuracy'])
                features['executive'].append(memory_data['sequence_accuracy'])
        
        return features
    
    def _extract_drawing_features(self, game_data):
        """提取圖形繪製遊戲特徵"""
        features = defaultdict(list)
        
        if 'drawing_data' in game_data:
            drawing_data = game_data['drawing_data']
            
            # 視覺構造能力
            if 'construction_score' in drawing_data:
                features['visuospatial'].append(drawing_data['construction_score'])
            
            # 視覺記憶
            if 'memory_component' in drawing_data:
                features['memory'].append(drawing_data['memory_component'])
                features['visuospatial'].append(drawing_data['memory_component'])
            
            # 細節關注度
            if 'detail_attention' in drawing_data:
                features['attention'].append(drawing_data['detail_attention'])
                features['visuospatial'].append(drawing_data['detail_attention'])
        
        return features
    
    def _extract_repeat_language_features(self, game_data):
        """提取重複言語遊戲特徵"""
        features = defaultdict(list)
        
        if 'language_data' in game_data:
            lang_data = game_data['language_data']
            
            # 語言理解能力
            if 'comprehension_score' in lang_data:
                features['language'].append(lang_data['comprehension_score'])
            
            # 短期語言記憶
            if 'verbal_memory_score' in lang_data:
                features['memory'].append(lang_data['verbal_memory_score'])
                features['language'].append(lang_data['verbal_memory_score'])
            
            # 語音處理
            if 'phonological_score' in lang_data:
                features['language'].append(lang_data['phonological_score'])
        
        return features
    
    def _extract_naming_features(self, game_data):
        """提取命名遊戲特徵"""
        features = defaultdict(list)
        
        if 'naming_data' in game_data:
            naming_data = game_data['naming_data']
            
            # 命名能力
            if 'naming_accuracy' in naming_data:
                features['language'].append(naming_data['naming_accuracy'])
            
            # 語義記憶
            if 'semantic_memory' in naming_data:
                features['memory'].append(naming_data['semantic_memory'])
                features['language'].append(naming_data['semantic_memory'])
            
            # 詞彙檢索
            if 'retrieval_speed' in naming_data:
                # 標準化檢索速度（快=高分）
                retrieval_speed = 1 - min(naming_data['retrieval_speed'] / 5, 1.0)
                features['language'].append(retrieval_speed)
                features['executive'].append(retrieval_speed)
        
        return features
    
    def _extract_object_matching_features(self, game_data):
        """提取物件配對遊戲特徵"""
        features = defaultdict(list)
        
        if 'matching_data' in game_data:
            match_data = game_data['matching_data']
            
            # 視覺辨別能力
            if 'visual_discrimination' in match_data:
                features['visuospatial'].append(match_data['visual_discrimination'])
                features['attention'].append(match_data['visual_discrimination'])
            
            # 決策速度
            if 'decision_speed' in match_data:
                # 標準化決策速度（快=高分）
                speed_score = 1 - min(match_data['decision_speed'] / 5, 1.0)
                features['executive'].append(speed_score)
            
            # 物體識別
            if 'object_recognition' in match_data:
                features['visuospatial'].append(match_data['object_recognition'])
        
        return features
    
    def _extract_spatial_concept_features(self, game_data):
        """提取空間概念遊戲特徵"""
        features = defaultdict(list)
        
        if 'spatial_data' in game_data:
            spatial_data = game_data['spatial_data']
            
            # 空間導航能力
            if 'navigation_score' in spatial_data:
                features['visuospatial'].append(spatial_data['navigation_score'])
                features['executive'].append(spatial_data['navigation_score'])
            
            # 空間記憶
            if 'spatial_memory' in spatial_data:
                features['memory'].append(spatial_data['spatial_memory'])
                features['visuospatial'].append(spatial_data['spatial_memory'])
            
            # 方向感
            if 'orientation_score' in spatial_data:
                features['visuospatial'].append(spatial_data['orientation_score'])
        
        return features


class CognitiveAssessmentService:
    """認知評估服務：整合遊戲數據進行全面認知評估"""
    
    def __init__(self, model_path=None):
        self.logger = logging.getLogger("cognitive_assessment_service")
        
        # 特徵提取器
        self.feature_extractor = GameSpecificFeatureExtractor()
        
        # 評估模型
        self._init_model(model_path)
        
        # 評分標準
        self.scoring_standards = self._get_scoring_standards()
        
        # 使用者歷史數據
        self.user_history = {}
    
    def _init_model(self, model_path):
        """初始化或載入評估模型"""
        feature_sizes = {
            'attention': 10,
            'memory': 10,
            'language': 10,
            'visuospatial': 10,
            'executive': 10
        }
        
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info(f"成功載入認知評估模型: {model_path}")
            except Exception as e:
                self.logger.error(f"載入模型時出錯: {e}")
                self.model = CognitiveAssessmentModel(feature_sizes)
        else:
            self.model = CognitiveAssessmentModel(feature_sizes)
    
    def _get_scoring_standards(self):
        """獲取評分標準"""
        return {
            'attention': {
                'normal': (0.7, 1.0),
                'mild_impairment': (0.5, 0.7),
                'moderate_impairment': (0.3, 0.5),
                'severe_impairment': (0.0, 0.3)
            },
            'memory': {
                'normal': (0.7, 1.0),
                'mild_impairment': (0.5, 0.7),
                'moderate_impairment': (0.3, 0.5),
                'severe_impairment': (0.0, 0.3)
            },
            'language': {
                'normal': (0.7, 1.0),
                'mild_impairment': (0.5, 0.7),
                'moderate_impairment': (0.3, 0.5),
                'severe_impairment': (0.0, 0.3)
            },
            'visuospatial': {
                'normal': (0.7, 1.0),
                'mild_impairment': (0.5, 0.7),
                'moderate_impairment': (0.3, 0.5),
                'severe_impairment': (0.0, 0.3)
            },
            'executive': {
                'normal': (0.7, 1.0),
                'mild_impairment': (0.5, 0.7),
                'moderate_impairment': (0.3, 0.5),
                'severe_impairment': (0.0, 0.3)
            },
            'overall': {
                'normal': (0.7, 1.0),
                'mild_impairment': (0.5, 0.7),
                'moderate_impairment': (0.3, 0.5),
                'severe_impairment': (0.0, 0.3)
            }
        }
    
    def assess_game_performance(self, user_id, game_data):
        """評估單個遊戲表現"""
        
        # 從遊戲數據中提取特徵
        features = self.feature_extractor.extract_features(game_data)
        
        # 確保所有特徵有值且維度正確
        for domain, feature_array in features.items():
            if len(feature_array) == 0:
                features[domain] = np.zeros((1, 10), dtype=np.float32)  # 使用更大的特徵維度
            elif feature_array.ndim == 1:
                # 擴展到2D，並確保特徵維度至少為10
                expanded = np.expand_dims(feature_array, 0)
                if expanded.shape[1] < 10:
                    padded = np.zeros((1, 10), dtype=np.float32)
                    padded[0, :expanded.shape[1]] = expanded[0]
                    features[domain] = padded
                else:
                    features[domain] = expanded
        
        try:
            # 通過模型進行評估
            assessment_results = self.model(features)
            
            # 處理TensorFlow張量輸出
            domain_scores = {}
            for domain, score in assessment_results['domain_scores'].items():
                if hasattr(score, 'numpy'):
                    # 確保我們只提取標量值
                    domain_scores[domain] = float(score.numpy().flatten()[0])
                else:
                    domain_scores[domain] = float(score[0][0])
            
            if hasattr(assessment_results['overall_score'], 'numpy'):
                overall_score = float(assessment_results['overall_score'].numpy().flatten()[0])
            else:
                overall_score = float(assessment_results['overall_score'][0][0])
            
            # 解釋評分
            interpretations = self._interpret_scores(domain_scores, overall_score)
            
            # 存儲用戶歷史數據
            self._update_user_history(user_id, game_data['game_type'], domain_scores, overall_score)
            
        except Exception as e:
            self.logger.error(f"評估時出錯: {e}")
            # 提供默認評估結果
            domain_scores = {domain: 0.5 for domain in self.model.domain_scorers.keys()}
            overall_score = 0.5
            interpretations = {domain: "無法評估" for domain in domain_scores.keys()}
            interpretations['overall'] = "無法評估"
        
        # 返回評估結果
        return {
            'domain_scores': domain_scores,
            'overall_score': overall_score,
            'interpretations': interpretations,
            'timestamp': game_data.get('timestamp', None)
        }
    
    def get_comprehensive_assessment(self, user_id, time_range=None):
        """獲取用戶的綜合認知評估"""
        
        if user_id not in self.user_history:
            self.logger.warning(f"用戶 {user_id} 沒有歷史數據")
            return None
        
        # 獲取指定時間範圍內的評估數據
        assessments = self.user_history[user_id]
        if time_range:
            # 過濾時間範圍
            start_time, end_time = time_range
            assessments = [a for a in assessments if start_time <= a['timestamp'] <= end_time]
        
        if not assessments:
            self.logger.warning(f"用戶 {user_id} 在指定時間範圍內沒有評估數據")
            return None
        
        # 計算平均分數
        domain_scores = defaultdict(list)
        overall_scores = []
        
        for assessment in assessments:
            for domain, score in assessment['domain_scores'].items():
                domain_scores[domain].append(score)
            overall_scores.append(assessment['overall_score'])
        
        avg_domain_scores = {
            domain: np.mean(scores) for domain, scores in domain_scores.items()
        }
        avg_overall_score = np.mean(overall_scores)
        
        # 計算趨勢
        if len(assessments) >= 3:
            trends = self._calculate_trends(assessments)
        else:
            trends = {domain: 'stable' for domain in avg_domain_scores.keys()}
            trends['overall'] = 'stable'
        
        # 解釋評分
        interpretations = self._interpret_scores(avg_domain_scores, avg_overall_score, trends)
        
        # 返回綜合評估
        return {
            'avg_domain_scores': avg_domain_scores,
            'avg_overall_score': avg_overall_score,
            'trends': trends,
            'interpretations': interpretations,
            'data_points': len(assessments),
            'time_range': time_range
        }
    
    def _interpret_scores(self, domain_scores, overall_score, trends=None):
        """解釋評分"""
        interpretations = {}
        
        # 解釋各域評分
        for domain, score in domain_scores.items():
            for level, (lower, upper) in self.scoring_standards[domain].items():
                if lower <= score <= upper:
                    interpretations[domain] = level
                    break
        
        # 解釋整體評分
        for level, (lower, upper) in self.scoring_standards['overall'].items():
            if lower <= overall_score <= upper:
                interpretations['overall'] = level
                break
        
        # 添加趨勢解釋
        if trends:
            for domain, trend in trends.items():
                if trend == 'improving':
                    interpretations[f'{domain}_trend'] = '改善中'
                elif trend == 'declining':
                    interpretations[f'{domain}_trend'] = '退步中'
                else:
                    interpretations[f'{domain}_trend'] = '穩定'
        
        return interpretations
    
    def _update_user_history(self, user_id, game_type, domain_scores, overall_score):
        """更新用戶歷史數據"""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        # 添加新的評估結果
        self.user_history[user_id].append({
            'game_type': game_type,
            'domain_scores': domain_scores,
            'overall_score': overall_score,
            'timestamp': np.datetime64('now')
        })
        
        # 限制歷史數據量
        if len(self.user_history[user_id]) > 100:
            self.user_history[user_id] = self.user_history[user_id][-100:]
    
    def _calculate_trends(self, assessments):
        """計算認知功能趨勢"""
        # 按時間排序
        sorted_assessments = sorted(assessments, key=lambda a: a['timestamp'])
        
        # 獲取數據點
        domains = list(sorted_assessments[0]['domain_scores'].keys())
        
        domain_trends = {}
        
        # 對每個域計算趨勢
        for domain in domains:
            scores = [a['domain_scores'][domain] for a in sorted_assessments]
            domain_trends[domain] = self._detect_trend(scores)
        
        # 計算整體趨勢
        overall_scores = [a['overall_score'] for a in sorted_assessments]
        domain_trends['overall'] = self._detect_trend(overall_scores)
        
        return domain_trends
    
    def _detect_trend(self, scores):
        """檢測分數趨勢"""
        if len(scores) < 3:
            return 'stable'
        
        # 使用簡單線性回歸
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # 計算斜率
        slope = np.polyfit(x, y, 1)[0]
        
        # 根據斜率判斷趨勢
        if slope > 0.02:
            return 'improving'
        elif slope < -0.02:
            return 'declining'
        else:
            return 'stable'


def create_cognitive_assessment_service(model_path=None):
    """創建認知評估服務"""
    service = CognitiveAssessmentService(model_path)
    return service 