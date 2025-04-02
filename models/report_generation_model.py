import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64


class ReportGenerationService:
    """報告生成服務：整合遊戲數據並生成評估報告"""
    
    def __init__(self, model_path=None):
        """
        初始化報告生成服務
        
        參數:
        - model_path: 模型路徑，如果指定則載入預訓練模型
        """
        self.logger = logging.getLogger("report_generation")
        
        if model_path:
            self.logger.info(f"從{model_path}載入報告生成模型")
            try:
                self.model = keras.models.load_model(model_path)
            except Exception as e:
                self.logger.warning(f"載入模型失敗: {e}")
                self.model = self._build_model()
        else:
            self.logger.info("創建默認報告生成模型")
            self.model = self._build_model()
    
    def _build_model(self):
        """構建報告生成模型（基於注意力機制的序列模型）"""
        self.logger.info("構建報告生成模型")
        
        try:
            # 簡化的模型，實際應用中會更複雜
            inputs = keras.layers.Input(shape=(None, 64))
            lstm = keras.layers.LSTM(128, return_sequences=True)(inputs)
            attention = keras.layers.Dense(1, activation='tanh')(lstm)
            attention = keras.layers.Flatten()(attention)
            attention = keras.layers.Activation('softmax')(attention)
            attention = keras.layers.RepeatVector(128)(attention)
            attention = keras.layers.Permute([2, 1])(attention)
            
            weighted = keras.layers.Multiply()([lstm, attention])
            context = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(weighted)
            
            dense1 = keras.layers.Dense(256, activation='relu')(context)
            dropout = keras.layers.Dropout(0.3)(dense1)
            output = keras.layers.Dense(128, activation='relu')(dropout)
            
            model = keras.Model(inputs=inputs, outputs=output)
            model.compile(
                optimizer='adam',
                loss='mse'
            )
            return model
        
        except Exception as e:
            self.logger.error(f"構建模型時發生錯誤: {e}")
            # 返回一個假的模型對象，以便服務仍然可以運行
            return type('DummyModel', (), {'predict': lambda x: np.zeros((1, 128))})
    
    def _extract_features(self, game_data_list):
        """從遊戲數據中提取特徵"""
        features = []
        
        for game_data in game_data_list:
            # 提取基本性能指標
            performance = game_data.get('performance_metrics', {})
            game_features = [
                performance.get('accuracy', 0),
                performance.get('response_time', 0),
                performance.get('completion_rate', 0),
            ]
            
            # 根據遊戲類型提取特定特徵
            game_type = game_data.get('game_type', '')
            if game_type == 'attention_calculation':
                calc_data = game_data.get('calculation_data', {})
                game_features.extend([
                    calc_data.get('errors', 0) / max(calc_data.get('total_problems', 1), 1),
                    calc_data.get('problems_per_minute', 0) / 20,  # 歸一化
                    calc_data.get('attention_drops', 0) / 5,       # 歸一化
                ])
            elif game_type == 'short_term_memory':
                memory_data = game_data.get('memory_data', {})
                game_features.extend([
                    memory_data.get('recall_accuracy', 0),
                    memory_data.get('sequence_length', 0) / 10,    # 歸一化
                    memory_data.get('trials', 0) / 5,              # 歸一化
                ])
            
            # 使用自動編碼方式填充至固定長度
            while len(game_features) < 64:
                game_features.append(0)
            
            features.append(game_features[:64])  # 確保不超過特徵維度
        
        return np.array(features).reshape(1, len(features), 64)
    
    def _generate_summary(self, model_output, user_name, game_data_list):
        """生成摘要文本"""
        # 從不同遊戲數據中提取性能
        performances = {}
        for game_data in game_data_list:
            game_type = game_data.get('game_type', 'unknown')
            if game_type not in performances:
                performances[game_type] = []
            performances[game_type].append(game_data.get('performance_metrics', {}))
        
        # 計算平均性能
        avg_performances = {}
        for game_type, perf_list in performances.items():
            avg_perf = {}
            for key in ['accuracy', 'response_time', 'completion_rate']:
                values = [p.get(key, 0) for p in perf_list]
                avg_perf[key] = sum(values) / len(values) if values else 0
            avg_performances[game_type] = avg_perf
        
        # 根據平均性能生成摘要
        game_summaries = []
        overall_score = 0
        
        for game_type, avg_perf in avg_performances.items():
            game_score = avg_perf.get('accuracy', 0) * 0.5 + avg_perf.get('completion_rate', 0) * 0.3 - min(1, avg_perf.get('response_time', 0) / 10) * 0.2
            overall_score += game_score
            
            if game_score > 0.8:
                performance_text = "表現優秀"
            elif game_score > 0.6:
                performance_text = "表現良好"
            elif game_score > 0.4:
                performance_text = "表現中等"
            else:
                performance_text = "需要加強"
            
            game_summaries.append(f"在{game_type}遊戲中{performance_text}（得分：{game_score:.2f}）")
        
        overall_score = overall_score / len(avg_performances) if avg_performances else 0
        
        # 生成最終摘要
        current_date = datetime.now().strftime("%Y年%m月%d日")
        summary = f"{current_date}評估報告\n\n"
        summary += f"{user_name}的綜合認知評估\n\n"
        summary += "遊戲表現摘要：\n" + "\n".join(game_summaries) + "\n\n"
        summary += f"綜合認知分數：{overall_score:.2f}/1.0\n\n"
        
        # 根據總分給出評價
        if overall_score > 0.8:
            summary += "整體評價：認知功能表現優秀，建議維持現有訓練頻率。"
        elif overall_score > 0.6:
            summary += "整體評價：認知功能表現良好，可在特定領域加強訓練。"
        elif overall_score > 0.4:
            summary += "整體評價：認知功能表現中等，建議增加訓練頻率和強度。"
        else:
            summary += "整體評價：認知功能可能存在顯著退化，建議進一步專業評估和有針對性的訓練。"
        
        return summary
    
    def _generate_recommendations(self, game_data_list):
        """生成訓練建議"""
        # 分析表現數據，找出弱項
        game_scores = {}
        for game_data in game_data_list:
            game_type = game_data.get('game_type', 'unknown')
            metrics = game_data.get('performance_metrics', {})
            
            # 簡單加權計算遊戲得分
            score = metrics.get('accuracy', 0) * 0.6 + metrics.get('completion_rate', 0) * 0.4
            
            if game_type not in game_scores:
                game_scores[game_type] = []
            game_scores[game_type].append(score)
        
        # 計算平均得分
        avg_scores = {game: sum(scores)/len(scores) for game, scores in game_scores.items()}
        
        # 基於得分生成建議
        recommendations = ["訓練建議：\n"]
        
        for game, score in sorted(avg_scores.items(), key=lambda x: x[1]):
            if score < 0.4:
                recommendations.append(f"- 加強{game}訓練，建議每日進行至少3次，每次15分鐘")
            elif score < 0.6:
                recommendations.append(f"- 適度加強{game}訓練，建議每日進行1-2次，每次10-15分鐘")
            elif score < 0.8:
                recommendations.append(f"- 維持{game}現有訓練強度，每週3-4次，每次10分鐘")
            else:
                recommendations.append(f"- {game}表現優秀，維持每週2-3次練習即可")
        
        # 添加綜合建議
        recommendations.append("\n綜合建議：")
        
        worst_game = min(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else None
        if worst_game:
            recommendations.append(f"- 重點關注{worst_game}遊戲訓練，這是目前表現最弱的認知領域")
        
        # 添加生活習慣建議
        recommendations.append("- 保持規律作息，確保充足睡眠")
        recommendations.append("- 進行30分鐘中等強度的有氧運動，如散步、游泳等")
        recommendations.append("- 積極參與社交活動，與家人朋友保持互動")
        recommendations.append("- 嘗試新的認知活動，如學習新技能、閱讀新書等")
        
        return "\n".join(recommendations)
    
    def _generate_charts(self, game_data_list):
        """生成報告圖表"""
        charts = {}
        
        try:
            # 按遊戲類型分組數據
            game_data_by_type = {}
            for data in game_data_list:
                game_type = data.get('game_type', 'unknown')
                if game_type not in game_data_by_type:
                    game_data_by_type[game_type] = []
                game_data_by_type[game_type].append(data)
            
            # 生成遊戲表現趨勢圖
            if len(game_data_list) > 1:
                plt.figure(figsize=(10, 6))
                
                for game_type, data_list in game_data_by_type.items():
                    if len(data_list) > 1:
                        # 按時間排序
                        sorted_data = sorted(data_list, key=lambda x: x.get('timestamp', ''))
                        
                        # 提取準確率
                        accuracies = [d.get('performance_metrics', {}).get('accuracy', 0) for d in sorted_data]
                        times = [d.get('timestamp', '') for d in sorted_data]
                        
                        # 將時間轉換為序號
                        x = list(range(len(times)))
                        
                        plt.plot(x, accuracies, 'o-', label=f'{game_type}')
                
                plt.title('遊戲表現趨勢')
                plt.xlabel('時間')
                plt.ylabel('準確率')
                plt.ylim(0, 1)
                plt.legend()
                plt.grid(True)
                
                # 轉換為base64編碼
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png')
                img_data.seek(0)
                charts['performance_trend'] = base64.b64encode(img_data.read()).decode('utf-8')
                plt.close()
            
            # 生成認知領域雷達圖
            if len(game_data_by_type) > 2:
                # 計算各領域平均分數
                domain_scores = {}
                for game_type, data_list in game_data_by_type.items():
                    scores = []
                    for data in data_list:
                        metrics = data.get('performance_metrics', {})
                        score = metrics.get('accuracy', 0) * 0.5 + metrics.get('completion_rate', 0) * 0.5
                        scores.append(score)
                    domain_scores[game_type] = sum(scores) / len(scores) if scores else 0
                
                # 繪製雷達圖
                domains = list(domain_scores.keys())
                scores = [domain_scores[d] for d in domains]
                
                angles = np.linspace(0, 2*np.pi, len(domains), endpoint=False).tolist()
                angles += angles[:1]  # 閉合圖形
                scores += scores[:1]  # 閉合圖形
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                ax.plot(angles, scores, 'o-', linewidth=2)
                ax.fill(angles, scores, alpha=0.25)
                ax.set_thetagrids(np.degrees(angles[:-1]), domains)
                ax.set_ylim(0, 1)
                ax.grid(True)
                plt.title('認知領域評估')
                
                # 轉換為base64編碼
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png')
                img_data.seek(0)
                charts['cognitive_domains'] = base64.b64encode(img_data.read()).decode('utf-8')
                plt.close()
        
        except Exception as e:
            self.logger.error(f"生成圖表時發生錯誤: {e}")
        
        return charts
    
    def generate_report(self, user_id, user_name, game_data_list):
        """生成評估報告"""
        self.logger.info(f"為用戶{user_id}生成評估報告")
        
        if not game_data_list:
            self.logger.warning("沒有遊戲數據可用於生成報告")
            return {
                "summary": f"無法為{user_name}生成報告：沒有遊戲數據",
                "recommendations": "建議完成更多認知訓練遊戲以獲取評估報告"
            }
        
        try:
            # 從遊戲數據中提取特徵
            features = self._extract_features(game_data_list)
            
            # 使用模型生成報告向量
            model_output = self.model.predict(features)
            
            # 生成報告文本
            summary = self._generate_summary(model_output, user_name, game_data_list)
            recommendations = self._generate_recommendations(game_data_list)
            
            # 生成圖表
            charts = self._generate_charts(game_data_list)
            
            report = {
                "summary": summary,
                "recommendations": recommendations
            }
            
            if charts:
                report["charts"] = charts
            
            return report
        
        except Exception as e:
            self.logger.error(f"生成報告時發生錯誤: {e}")
            return {
                "summary": f"為{user_name}生成報告時發生錯誤",
                "recommendations": "系統暫時無法提供詳細建議，請稍後再試"
            }


def create_report_generation_service(model_path=None):
    """創建報告生成服務"""
    return ReportGenerationService(model_path) 