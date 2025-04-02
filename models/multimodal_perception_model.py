import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, applications
import logging


class VisionEncoder(Model):
    """視覺編碼器：處理圖像輸入"""
    
    def __init__(self, embedding_dim=256):
        super(VisionEncoder, self).__init__()
        
        # 使用預訓練的視覺骨幹網絡
        base_model = applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # 凍結底層
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        # 添加自定義頂層
        self.base_model = base_model
        self.global_pool = layers.GlobalAveragePooling2D()
        self.projection = layers.Dense(embedding_dim, activation='relu')
        
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_pool(x)
        x = self.projection(x)
        return x


class AudioEncoder(Model):
    """音頻編碼器：處理音頻輸入"""
    
    def __init__(self, embedding_dim=256):
        super(AudioEncoder, self).__init__()
        
        # CNN層用於處理音頻頻譜圖
        self.conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        # 全局池化
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # 投影層
        self.projection = layers.Dense(embedding_dim, activation='relu')
        
    def call(self, inputs, training=False):
        # 假設輸入是頻譜圖 (batch_size, time_steps, freq_bins, channels)
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.global_pool(x)
        x = self.projection(x)
        return x


class SpeechRecognitionModel(Model):
    """語音識別模型：用於重複言語和命名任務"""
    
    def __init__(self, vocab_size, embedding_dim=256):
        super(SpeechRecognitionModel, self).__init__()
        
        # 音頻編碼器
        self.audio_encoder = AudioEncoder(embedding_dim)
        
        # 序列建模層
        self.bidirectional_lstm = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True))
        
        # 注意力機制
        self.attention = layers.Dense(1)
        
        # 輸出層
        self.classifier = layers.Dense(vocab_size, activation='softmax')
        
    def call(self, inputs, training=False):
        # 編碼音頻
        audio_features = self.audio_encoder(inputs, training=training)
        
        # 重塑為序列
        # 假設我們從頻譜圖中提取了序列特徵
        sequence_length = 20  # 這應該根據實際輸入動態確定
        sequence_features = tf.reshape(audio_features, [-1, sequence_length, audio_features.shape[-1] // sequence_length])
        
        # 序列建模
        lstm_features = self.bidirectional_lstm(sequence_features)
        
        # 注意力機制
        attention_weights = tf.nn.softmax(self.attention(lstm_features), axis=1)
        context_vector = tf.reduce_sum(attention_weights * lstm_features, axis=1)
        
        # 分類
        outputs = self.classifier(context_vector)
        
        return outputs


class ObjectMatchingModel(Model):
    """物件匹配模型：用於物件配對任務"""
    
    def __init__(self, embedding_dim=256):
        super(ObjectMatchingModel, self).__init__()
        
        # 共享視覺編碼器處理參考物件和候選物件
        self.vision_encoder = VisionEncoder(embedding_dim)
        
        # 比較網絡
        self.comparison_network = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 輸出相似度分數
        ])
        
    def call(self, inputs, training=False):
        # 解包輸入
        reference_image, candidate_image = inputs
        
        # 編碼參考和候選物件
        reference_embedding = self.vision_encoder(reference_image, training=training)
        candidate_embedding = self.vision_encoder(candidate_image, training=training)
        
        # 計算嵌入差異並餵入比較網絡
        embedding_diff = tf.abs(reference_embedding - candidate_embedding)
        embedding_prod = reference_embedding * candidate_embedding
        
        comparison_features = tf.concat([reference_embedding, candidate_embedding, 
                                        embedding_diff, embedding_prod], axis=1)
        
        similarity_score = self.comparison_network(comparison_features)
        
        return similarity_score


class DrawingAssessmentModel(Model):
    """繪圖評估模型：用於圖形繪製任務"""
    
    def __init__(self, num_features=10):
        super(DrawingAssessmentModel, self).__init__()
        
        # 視覺編碼器處理參考圖形和用戶繪製圖形
        self.vision_encoder = VisionEncoder(256)
        
        # 特徵提取網絡
        self.feature_extractor = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_features)  # 輸出繪圖特徵向量
        ])
        
        # 評分網絡
        self.scoring_network = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 輸出評分 (0-1)
        ])
        
    def call(self, inputs, training=False):
        # 解包輸入
        reference_image, user_drawing = inputs
        
        # 編碼參考和用戶繪圖
        reference_embedding = self.vision_encoder(reference_image, training=training)
        drawing_embedding = self.vision_encoder(user_drawing, training=training)
        
        # 提取特徵
        reference_features = self.feature_extractor(reference_embedding)
        drawing_features = self.feature_extractor(drawing_embedding)
        
        # 計算特徵差異
        feature_diff = tf.abs(reference_features - drawing_features)
        
        # 評分
        score = self.scoring_network(feature_diff)
        
        return {
            'score': score,
            'drawing_features': drawing_features,
            'feature_diff': feature_diff
        }


class TimeRecognitionModel(Model):
    """時間辨認模型：處理聲音/圖像識別對應的時間"""
    
    def __init__(self, num_time_slots=24):
        super(TimeRecognitionModel, self).__init__()
        
        # 視覺和音頻編碼器
        self.vision_encoder = VisionEncoder(256)
        self.audio_encoder = AudioEncoder(256)
        
        # 融合層
        self.fusion_layer = layers.Dense(128, activation='relu')
        
        # 時間槽分類器（粗分類：早上、下午、晚上等）
        self.time_classifier = layers.Dense(num_time_slots, activation='softmax')
        
    def call(self, inputs, training=False):
        # 解包輸入
        has_image = inputs['has_image']
        has_audio = inputs['has_audio']
        
        image_features = None
        audio_features = None
        
        # 處理可用的模態
        if has_image:
            image_features = self.vision_encoder(inputs['image'], training=training)
            
        if has_audio:
            audio_features = self.audio_encoder(inputs['audio'], training=training)
            
        # 融合可用的特徵
        if image_features is not None and audio_features is not None:
            # 多模態融合
            combined_features = tf.concat([image_features, audio_features], axis=1)
        elif image_features is not None:
            combined_features = image_features
        elif audio_features is not None:
            combined_features = audio_features
        else:
            raise ValueError("必須提供至少一種模態的輸入")
            
        # 預測時間
        fused_features = self.fusion_layer(combined_features)
        time_probs = self.time_classifier(fused_features)
        
        return time_probs


class MultimodalPerceptionModel:
    """多模態感知模型：整合不同類型的感知模型"""
    
    def __init__(self, config=None):
        """
        初始化多模態感知模型
        
        參數:
        - config: 配置字典，包含各子模型的參數
        """
        self.logger = logging.getLogger("multimodal_perception")
        self.config = config or {}
        
        # 初始化各個子模型
        self.models = {}
        
        # 視覺和聽覺基礎編碼器
        self.vision_encoder = VisionEncoder(
            embedding_dim=self.config.get('vision_embedding_dim', 256)
        )
        
        self.audio_encoder = AudioEncoder(
            embedding_dim=self.config.get('audio_embedding_dim', 256)
        )
        
    def load_models(self, model_paths):
        """載入預訓練的子模型"""
        
        for model_name, path in model_paths.items():
            try:
                if model_name == 'speech_recognition':
                    vocab_size = self.config.get('vocab_size', 5000)
                    self.models[model_name] = SpeechRecognitionModel(vocab_size)
                    self.models[model_name].load_weights(path)
                    
                elif model_name == 'object_matching':
                    self.models[model_name] = ObjectMatchingModel()
                    self.models[model_name].load_weights(path)
                    
                elif model_name == 'drawing_assessment':
                    self.models[model_name] = DrawingAssessmentModel()
                    self.models[model_name].load_weights(path)
                    
                elif model_name == 'time_recognition':
                    self.models[model_name] = TimeRecognitionModel()
                    self.models[model_name].load_weights(path)
                    
                self.logger.info(f"成功載入模型 {model_name}")
                
            except Exception as e:
                self.logger.error(f"載入模型 {model_name} 失敗: {e}")
                # 初始化新模型
                if model_name == 'speech_recognition':
                    vocab_size = self.config.get('vocab_size', 5000)
                    self.models[model_name] = SpeechRecognitionModel(vocab_size)
                    
                elif model_name == 'object_matching':
                    self.models[model_name] = ObjectMatchingModel()
                    
                elif model_name == 'drawing_assessment':
                    self.models[model_name] = DrawingAssessmentModel()
                    
                elif model_name == 'time_recognition':
                    self.models[model_name] = TimeRecognitionModel()
                    
    def recognize_speech(self, audio_input):
        """進行語音識別"""
        
        if 'speech_recognition' not in self.models:
            self.logger.error("語音識別模型未載入")
            return None
            
        # 將輸入轉換為合適的格式（例如音頻頻譜圖）
        processed_audio = self._preprocess_audio(audio_input)
        
        # 使用語音識別模型進行預測
        predictions = self.models['speech_recognition'](processed_audio)
        
        # 從預測中獲取文本
        text = self._decode_predictions(predictions)
        
        return text
    
    def match_objects(self, reference_image, candidate_images):
        """物體配對任務"""
        
        if 'object_matching' not in self.models:
            self.logger.error("物體配對模型未載入")
            return None
            
        # 預處理圖像
        processed_reference = self._preprocess_image(reference_image)
        processed_candidates = [self._preprocess_image(img) for img in candidate_images]
        
        # 計算每個候選物體與參考物體的相似度
        similarity_scores = []
        for candidate in processed_candidates:
            score = self.models['object_matching']([
                tf.expand_dims(processed_reference, 0),
                tf.expand_dims(candidate, 0)
            ])
            similarity_scores.append(score.numpy()[0][0])
        
        # 找出最匹配的物體
        best_match_idx = np.argmax(similarity_scores)
        
        return {
            'best_match_idx': best_match_idx,
            'similarity_scores': similarity_scores
        }
    
    def assess_drawing(self, reference_image, user_drawing):
        """評估用戶繪圖"""
        
        if 'drawing_assessment' not in self.models:
            self.logger.error("繪圖評估模型未載入")
            return None
            
        # 預處理圖像
        processed_reference = self._preprocess_image(reference_image)
        processed_drawing = self._preprocess_image(user_drawing)
        
        # 評估繪圖
        assessment = self.models['drawing_assessment']([
            tf.expand_dims(processed_reference, 0),
            tf.expand_dims(processed_drawing, 0)
        ])
        
        return {
            'score': assessment['score'].numpy()[0][0],
            'features': assessment['drawing_features'].numpy()[0],
            'feature_differences': assessment['feature_diff'].numpy()[0]
        }
    
    def recognize_time(self, image=None, audio=None):
        """識別圖像或聲音對應的時間"""
        
        if 'time_recognition' not in self.models:
            self.logger.error("時間辨認模型未載入")
            return None
            
        inputs = {
            'has_image': image is not None,
            'has_audio': audio is not None
        }
        
        if image is not None:
            inputs['image'] = tf.expand_dims(self._preprocess_image(image), 0)
            
        if audio is not None:
            inputs['audio'] = tf.expand_dims(self._preprocess_audio(audio), 0)
            
        if not inputs['has_image'] and not inputs['has_audio']:
            self.logger.error("必須提供圖像或音頻輸入")
            return None
            
        # 識別時間
        time_probs = self.models['time_recognition'](inputs)
        
        # 獲取時間槽及其概率
        time_slot = tf.argmax(time_probs, axis=1).numpy()[0]
        confidence = tf.reduce_max(time_probs, axis=1).numpy()[0]
        
        # 時間槽到時間字符串的映射
        time_mapping = self._get_time_mapping()
        time_str = time_mapping.get(time_slot, "未知時間")
        
        return {
            'time_slot': time_slot,
            'time_string': time_str,
            'confidence': confidence,
            'all_probs': time_probs.numpy()[0]
        }
    
    def _preprocess_image(self, image):
        """預處理圖像"""
        # 此處應包含適當的圖像預處理步驟
        # 例如：調整大小、標準化等
        
        # 示例實現
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0  # 標準化到 [0, 1]
        
        return image
    
    def _preprocess_audio(self, audio):
        """預處理音頻數據"""
        # 此處應包含適當的音頻預處理步驟
        # 例如：計算頻譜圖、梅爾頻率倒譜係數等
        
        # 示例實現（假設已經得到頻譜圖）
        return audio
    
    def _decode_predictions(self, predictions):
        """將模型預測解碼為文本"""
        # 此處應包含合適的解碼邏輯
        # 例如：字段到單詞的映射等
        
        # 示例實現
        return "預測文本"
    
    def _get_time_mapping(self):
        """獲取時間槽到時間字符串的映射"""
        # 示例映射
        return {
            0: "凌晨",
            1: "早上",
            2: "上午",
            3: "中午",
            4: "下午",
            5: "傍晚",
            6: "晚上",
            7: "深夜"
        }


def create_multimodal_perception_model(config=None):
    """創建多模態感知模型"""
    model = MultimodalPerceptionModel(config)
    return model


def load_pretrained_perception_model(model_paths=None):
    """載入預訓練的感知模型"""
    config = {
        'vision_embedding_dim': 256,
        'audio_embedding_dim': 256,
        'vocab_size': 5000
    }
    
    model = create_multimodal_perception_model(config)
    
    if model_paths:
        model.load_models(model_paths)
    
    return model 