"""
Firebase數據庫整合模組：提供將認知遊戲數據與使用者資料存儲到Firebase的功能
實現長期追蹤、多用戶管理、臨床研究整合、多設備同步和複雜報告生成
"""

import firebase_admin
from firebase_admin import credentials, firestore, storage
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import uuid
import os

from schemas import (
    GameType, UserProfile, GamePerformanceData, SessionData, 
    CognitiveAssessmentData, LongTermProgressData, AssessmentReport
)


class FirebaseManager:
    """Firebase資料庫管理類：處理與Firebase的連接和數據存取"""
    
    def __init__(self, credential_path: Optional[str] = None):
        """初始化Firebase連接"""
        try:
            # 如果已初始化，直接獲取app
            firebase_admin.get_app()
        except ValueError:
            # 否則初始化新的app
            if credential_path:
                cred = credentials.Certificate(credential_path)
            else:
                # 在生產環境中，應通過環境變量或安全存儲獲取認證
                cred = credentials.ApplicationDefault()
            
            firebase_admin.initialize_app(cred, {
                'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET', 'your-bucket-name.appspot.com')
            })
        
        # 獲取Firestore客戶端
        self.db = firestore.client()
        self.storage = storage.bucket()
        print("Firebase連接已初始化")
    
    # ======== 用戶管理功能 ========
    
    def create_user(self, user_profile: UserProfile) -> str:
        """創建新用戶並返回用戶ID"""
        user_dict = {
            'user_id': user_profile.user_id,
            'age': user_profile.age,
            'gender': user_profile.gender,
            'education_level': user_profile.education_level,
            'cognitive_status': user_profile.cognitive_status.value,
            'medical_conditions': user_profile.medical_conditions,
            'cognitive_abilities': user_profile.cognitive_abilities,
            'preferences': user_profile.preferences,
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        
        # 添加到Firestore
        self.db.collection('users').document(user_profile.user_id).set(user_dict)
        return user_profile.user_id
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> None:
        """更新用戶信息"""
        updates['updated_at'] = firestore.SERVER_TIMESTAMP
        self.db.collection('users').document(user_id).update(updates)
    
    def get_user(self, user_id: str) -> Dict[str, Any]:
        """獲取用戶資料"""
        doc_ref = self.db.collection('users').document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        return None
    
    def list_users(self, query_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """獲取符合條件的用戶列表"""
        query = self.db.collection('users')
        
        if query_params:
            for key, value in query_params.items():
                if isinstance(value, list) and len(value) == 2 and value[0] in ['>', '>=', '<', '<=', '==']:
                    # 處理範圍查詢，如 {'age': ['>=', 65]}
                    query = query.where(key, value[0], value[1])
                else:
                    # 簡單相等查詢
                    query = query.where(key, '==', value)
        
        result = query.stream()
        return [doc.to_dict() for doc in result]
    
    # ======== 遊戲會話管理 ========
    
    def save_session(self, session: SessionData) -> str:
        """保存遊戲會話數據"""
        # 將數據轉換為可存儲的字典
        session_dict = {
            'user_id': session.user_id,
            'session_id': session.session_id,
            'game_type': session.game_type.value,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'user_skill': session.user_skill,
            'initial_difficulty': session.initial_difficulty,
            'difficulty_history': session.difficulty_history,
            'performance_history': session.performance_history,
            'satisfaction_history': session.satisfaction_history,
            'cognitive_assessments': session.cognitive_assessments,
            'duration_minutes': session.duration_minutes
        }
        
        # 儲存到Firestore
        self.db.collection('sessions').document(session.session_id).set(session_dict)
        
        # 更新用戶的會話歷史記錄
        self.db.collection('users').document(session.user_id).collection('session_history').document(session.session_id).set({
            'session_id': session.session_id,
            'game_type': session.game_type.value,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'duration_minutes': session.duration_minutes,
            'avg_performance': sum(session.performance_history) / len(session.performance_history) if session.performance_history else 0,
            'final_difficulty': session.difficulty_history[-1] if session.difficulty_history else session.initial_difficulty
        })
        
        return session.session_id
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """獲取會話數據"""
        doc_ref = self.db.collection('sessions').document(session_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        return None
    
    def get_user_sessions(self, user_id: str, game_type: Optional[GameType] = None, 
                         start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None, 
                         limit: int = 50) -> List[Dict[str, Any]]:
        """獲取用戶的會話歷史"""
        query = self.db.collection('users').document(user_id).collection('session_history')
        
        if game_type:
            query = query.where('game_type', '==', game_type.value)
        
        if start_date:
            query = query.where('start_time', '>=', start_date)
            
        if end_date:
            query = query.where('start_time', '<=', end_date)
            
        # 按時間倒序排列
        query = query.order_by('start_time', direction=firestore.Query.DESCENDING).limit(limit)
        
        result = query.stream()
        return [doc.to_dict() for doc in result]
    
    # ======== 遊戲表現數據管理 ========
    
    def save_game_performance(self, performance_data: GamePerformanceData) -> str:
        """保存遊戲表現數據"""
        # 生成唯一ID
        if not performance_data.game_id:
            performance_data.game_id = str(uuid.uuid4())
        
        # 將數據轉換為可存儲的字典
        perf_dict = {
            'user_id': performance_data.user_id,
            'game_id': performance_data.game_id,
            'round_id': performance_data.round_id,
            'game_type': performance_data.game_type.value,
            'timestamp': performance_data.timestamp,
            'difficulty_level': performance_data.difficulty_level,
            'session_id': performance_data.session_id,
            'metrics': {
                'accuracy': performance_data.performance_metrics.accuracy,
                'completion_time': performance_data.performance_metrics.completion_time,
                'correct_responses': performance_data.performance_metrics.correct_responses,
                'incorrect_responses': performance_data.performance_metrics.incorrect_responses,
                'missed_responses': performance_data.performance_metrics.missed_responses,
                'average_response_time': performance_data.performance_metrics.average_response_time,
                'frustration_indicators': performance_data.performance_metrics.frustration_indicators,
                'engagement_level': performance_data.performance_metrics.engagement_level,
                'satisfaction_level': performance_data.performance_metrics.satisfaction_level
            }
        }
        
        # 添加遊戲特定指標
        if performance_data.game_specific_metrics:
            if performance_data.game_specific_metrics.calculation_metrics:
                perf_dict['calculation_metrics'] = {
                    'operation_accuracy': performance_data.game_specific_metrics.calculation_metrics.operation_accuracy,
                    'calculation_speed': performance_data.game_specific_metrics.calculation_metrics.calculation_speed,
                    'error_distribution': performance_data.game_specific_metrics.calculation_metrics.error_distribution
                }
            
            if performance_data.game_specific_metrics.memory_metrics:
                perf_dict['memory_metrics'] = {
                    'recall_accuracy': performance_data.game_specific_metrics.memory_metrics.recall_accuracy,
                    'sequence_length_max': performance_data.game_specific_metrics.memory_metrics.sequence_length_max,
                    'position_accuracy': performance_data.game_specific_metrics.memory_metrics.position_accuracy,
                    'recall_pattern': performance_data.game_specific_metrics.memory_metrics.recall_pattern
                }
            
            if performance_data.game_specific_metrics.spatial_metrics:
                perf_dict['spatial_metrics'] = {
                    'navigation_accuracy': performance_data.game_specific_metrics.spatial_metrics.navigation_accuracy,
                    'spatial_orientation': performance_data.game_specific_metrics.spatial_metrics.spatial_orientation,
                    'landmark_recognition': performance_data.game_specific_metrics.spatial_metrics.landmark_recognition,
                    'path_reconstruction': performance_data.game_specific_metrics.spatial_metrics.path_reconstruction
                }
        
        # 儲存到Firestore
        self.db.collection('game_performances').document(performance_data.game_id).set(perf_dict)
        
        # 同時儲存到會話的子集合
        if performance_data.session_id:
            self.db.collection('sessions').document(performance_data.session_id).collection('performances').document(performance_data.game_id).set(perf_dict)
        
        return performance_data.game_id
    
    # ======== 認知評估數據管理 ========
    
    def save_cognitive_assessment(self, assessment: CognitiveAssessmentData) -> str:
        """保存認知評估數據"""
        assessment_id = f"assessment_{assessment.user_id}_{assessment.assessment_date.strftime('%Y%m%d%H%M%S')}"
        
        # 將數據轉換為可存儲的字典
        assessment_dict = {
            'assessment_id': assessment_id,
            'user_id': assessment.user_id,
            'assessment_date': assessment.assessment_date,
            'overall_score': assessment.overall_score,
            'clinical_notes': assessment.clinical_notes,
            'previous_assessment_id': assessment.previous_assessment_id,
            'change_from_previous': assessment.change_from_previous,
            'cognitive_domains': {}
        }
        
        # 添加認知領域評分
        for domain, score in assessment.cognitive_domains.items():
            assessment_dict['cognitive_domains'][domain] = {
                'score': score.score,
                'subscores': score.subscores,
                'clinical_interpretation': score.clinical_interpretation
            }
        
        # 儲存到Firestore
        self.db.collection('cognitive_assessments').document(assessment_id).set(assessment_dict)
        
        # 同時更新用戶的最新評估記錄
        self.db.collection('users').document(assessment.user_id).update({
            'latest_assessment': assessment_dict,
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        
        return assessment_id
    
    def get_user_assessments(self, user_id: str, start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """獲取用戶的認知評估歷史"""
        query = self.db.collection('cognitive_assessments').where('user_id', '==', user_id)
        
        if start_date:
            query = query.where('assessment_date', '>=', start_date)
            
        if end_date:
            query = query.where('assessment_date', '<=', end_date)
            
        # 按日期排序
        query = query.order_by('assessment_date', direction=firestore.Query.DESCENDING)
        
        result = query.stream()
        return [doc.to_dict() for doc in result]
    
    # ======== 長期進度追蹤 ========
    
    def save_long_term_progress(self, progress_data: LongTermProgressData) -> str:
        """保存長期進度數據"""
        progress_id = f"progress_{progress_data.user_id}_{progress_data.period.start_time.strftime('%Y%m%d')}"
        
        # 將數據轉換為可存儲的字典
        progress_dict = {
            'progress_id': progress_id,
            'user_id': progress_data.user_id,
            'training_program_id': progress_data.training_program_id,
            'period_start': progress_data.period.start_time,
            'period_end': progress_data.period.end_time,
            'session_frequency': progress_data.session_frequency,
            'session_duration_average': progress_data.session_duration_average,
            'engagement_metrics': progress_data.engagement_metrics,
            'overall_progress': [
                {'timestamp': point.timestamp, 
                 'performance_index': point.performance_index,
                 'skill_level': point.skill_level,
                 'difficulty_level': point.difficulty_level}
                for point in progress_data.overall_progress
            ],
            'game_specific_progress': {},
            'skill_transfer': []
        }
        
        # 添加遊戲特定進度
        for game_type, game_progress in progress_data.game_specific_progress.items():
            progress_dict['game_specific_progress'][game_type.value] = {
                'progress': [
                    {'timestamp': point.timestamp, 
                     'performance_index': point.performance_index,
                     'skill_level': point.skill_level,
                     'difficulty_level': point.difficulty_level}
                    for point in game_progress.progress
                ],
                'plateau_periods': game_progress.plateau_periods
            }
        
        # 添加技能遷移數據
        for transfer in progress_data.skill_transfer:
            progress_dict['skill_transfer'].append({
                'primary_game': transfer.primary_game.value,
                'secondary_game': transfer.secondary_game.value,
                'correlation_strength': transfer.correlation_strength,
                'lag_sessions': transfer.lag_sessions
            })
        
        # 儲存到Firestore
        self.db.collection('long_term_progress').document(progress_id).set(progress_dict)
        
        return progress_id
    
    def get_user_progress(self, user_id: str, months: int = 12) -> List[Dict[str, Any]]:
        """獲取用戶的長期進度數據"""
        # 計算開始日期
        start_date = datetime.now() - timedelta(days=30*months)
        
        query = self.db.collection('long_term_progress') \
                      .where('user_id', '==', user_id) \
                      .where('period_start', '>=', start_date) \
                      .order_by('period_start')
        
        result = query.stream()
        return [doc.to_dict() for doc in result]
    
    # ======== 報告生成與管理 ========
    
    def save_assessment_report(self, report: AssessmentReport) -> str:
        """保存評估報告"""
        # 將數據轉換為可存儲的字典
        report_dict = {
            'report_id': report.report_id,
            'user_id': report.user_id,
            'generation_date': report.generation_date,
            'period_covered_start': report.period_covered.start_time,
            'period_covered_end': report.period_covered.end_time,
            'summary': report.summary,
            'domain_assessments': report.domain_assessments,
            'recommendations': report.recommendations,
            'trends': report.trends,
            'clinical_significance': report.clinical_significance,
            'next_assessment_date': report.next_assessment_date
        }
        
        # 儲存報告到Firestore
        self.db.collection('assessment_reports').document(report.report_id).set(report_dict)
        
        # 為用戶添加報告記錄
        self.db.collection('users').document(report.user_id).collection('reports').document(report.report_id).set({
            'report_id': report.report_id,
            'generation_date': report.generation_date,
            'period_covered_start': report.period_covered.start_time,
            'period_covered_end': report.period_covered.end_time,
            'summary': report.summary
        })
        
        # 上傳圖表到Cloud Storage
        for chart_name, chart_path in report.charts.items():
            if os.path.exists(chart_path):
                blob = self.storage.blob(f"reports/{report.user_id}/{report.report_id}/{chart_name}.png")
                blob.upload_from_filename(chart_path)
                
                # 更新報告中的圖表URL
                blob.make_public()
                self.db.collection('assessment_reports').document(report.report_id).update({
                    f'chart_urls.{chart_name}': blob.public_url
                })
        
        return report.report_id
    
    def get_user_reports(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """獲取用戶的評估報告列表"""
        query = self.db.collection('users').document(user_id).collection('reports') \
                      .order_by('generation_date', direction=firestore.Query.DESCENDING) \
                      .limit(limit)
        
        result = query.stream()
        return [doc.to_dict() for doc in result]
    
    def get_report_details(self, report_id: str) -> Dict[str, Any]:
        """獲取完整的評估報告詳情"""
        doc_ref = self.db.collection('assessment_reports').document(report_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        return None
    
    # ======== 臨床研究整合 ========
    
    def save_clinical_data(self, user_id: str, data_type: str, data: Dict[str, Any]) -> str:
        """保存臨床研究相關數據"""
        data_id = f"{user_id}_{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 添加元數據
        data_with_meta = {
            'data_id': data_id,
            'user_id': user_id,
            'data_type': data_type,
            'timestamp': datetime.now(),
            'data': data
        }
        
        # 儲存到Firestore
        self.db.collection('clinical_data').document(data_id).set(data_with_meta)
        
        # 為用戶添加索引
        self.db.collection('users').document(user_id).collection('clinical_data').document(data_id).set({
            'data_id': data_id,
            'data_type': data_type,
            'timestamp': datetime.now()
        })
        
        return data_id
    
    def get_user_clinical_data(self, user_id: str, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """獲取用戶的臨床數據"""
        query = self.db.collection('clinical_data').where('user_id', '==', user_id)
        
        if data_type:
            query = query.where('data_type', '==', data_type)
            
        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        result = query.stream()
        return [doc.to_dict() for doc in result]
    
    # ======== 數據匯出與分析 ========
    
    def export_user_data(self, user_id: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """匯出用戶的完整數據包（適用於研究或數據遷移）"""
        result = {
            'user_profile': self.get_user(user_id),
            'sessions': [],
            'game_performances': [],
            'cognitive_assessments': [],
            'long_term_progress': [],
            'reports': [],
            'clinical_data': []
        }
        
        # 獲取會話數據
        sessions = self.get_user_sessions(user_id, start_date=start_date, end_date=end_date, limit=1000)
        result['sessions'] = sessions
        
        # 獲取遊戲表現數據
        for session in sessions:
            perf_query = self.db.collection('sessions').document(session['session_id']).collection('performances').stream()
            performances = [doc.to_dict() for doc in perf_query]
            result['game_performances'].extend(performances)
        
        # 獲取認知評估數據
        result['cognitive_assessments'] = self.get_user_assessments(user_id, start_date=start_date, end_date=end_date)
        
        # 獲取長期進度數據
        months = 60  # 5年數據
        if start_date and end_date:
            months = int((end_date - start_date).days / 30) + 1
        result['long_term_progress'] = self.get_user_progress(user_id, months=months)
        
        # 獲取報告
        result['reports'] = self.get_user_reports(user_id, limit=100)
        
        # 獲取臨床數據
        result['clinical_data'] = self.get_user_clinical_data(user_id)
        
        return result
    
    def batch_export_data(self, query_params: Dict[str, Any], output_format: str = 'json') -> str:
        """批量匯出多個用戶的數據（用於研究分析）"""
        # 獲取符合條件的用戶
        users = self.list_users(query_params)
        user_ids = [user['user_id'] for user in users]
        
        exported_data = {
            'export_date': datetime.now().isoformat(),
            'query_params': query_params,
            'user_count': len(users),
            'users': users,
            'aggregated_data': {
                'sessions_count': 0,
                'performance_by_game_type': {},
                'average_assessment_scores': {}
            },
            'detailed_data': {}
        }
        
        # 為每個用戶匯出基本數據
        for user_id in user_ids:
            # 獲取會話數據
            sessions = self.get_user_sessions(user_id, limit=100)
            exported_data['aggregated_data']['sessions_count'] += len(sessions)
            
            # 按遊戲類型聚合表現
            for session in sessions:
                game_type = session['game_type']
                if game_type not in exported_data['aggregated_data']['performance_by_game_type']:
                    exported_data['aggregated_data']['performance_by_game_type'][game_type] = {
                        'session_count': 0,
                        'avg_performance': 0,
                        'avg_satisfaction': 0
                    }
                
                exported_data['aggregated_data']['performance_by_game_type'][game_type]['session_count'] += 1
                exported_data['aggregated_data']['performance_by_game_type'][game_type]['avg_performance'] += session.get('avg_performance', 0)
            
            # 獲取認知評估
            assessments = self.get_user_assessments(user_id, limit=10)
            if assessments:
                latest_assessment = assessments[0]
                for domain, data in latest_assessment.get('cognitive_domains', {}).items():
                    if domain not in exported_data['aggregated_data']['average_assessment_scores']:
                        exported_data['aggregated_data']['average_assessment_scores'][domain] = {
                            'total_score': 0,
                            'count': 0
                        }
                    
                    exported_data['aggregated_data']['average_assessment_scores'][domain]['total_score'] += data.get('score', 0)
                    exported_data['aggregated_data']['average_assessment_scores'][domain]['count'] += 1
            
            # 將詳細數據添加到導出
            exported_data['detailed_data'][user_id] = {
                'sessions': sessions,
                'assessments': assessments
            }
        
        # 計算平均值
        for game_type, data in exported_data['aggregated_data']['performance_by_game_type'].items():
            if data['session_count'] > 0:
                data['avg_performance'] /= data['session_count']
                data['avg_satisfaction'] /= data['session_count']
        
        for domain, data in exported_data['aggregated_data']['average_assessment_scores'].items():
            if data['count'] > 0:
                data['average'] = data['total_score'] / data['count']
                del data['total_score']
        
        # 生成唯一的導出文件名
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"data_export_{timestamp}.json"
        
        # 保存到Cloud Storage
        blob = self.storage.blob(f"exports/{filename}")
        blob.upload_from_string(json.dumps(exported_data, default=str))
        
        # 設置公開可訪問
        blob.make_public()
        
        return blob.public_url


# Firebase數據收集及整合使用示例
def demo_firebase_integration():
    """演示Firebase整合的基本用法"""
    
    # 初始化Firebase管理器
    firebase = FirebaseManager('path/to/your/credentials.json')
    
    # 創建用戶
    user_profile = UserProfile(
        user_id="user123",
        age=65,
        gender="female",
        education_level="bachelor",
        cognitive_status=CognitiveStatus.NORMAL,
        medical_conditions=["hypertension"],
        cognitive_abilities={
            "attention": 0.8,
            "memory": 0.7,
            "executive_function": 0.75
        }
    )
    firebase.create_user(user_profile)
    
    # 當完成一個遊戲會話時保存數據
    session = SessionData(
        user_id="user123",
        session_id=f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        game_type=GameType.ATTENTION_CALCULATION,
        start_time=datetime.now() - timedelta(minutes=15),
        end_time=datetime.now(),
        user_skill=0.65,
        difficulty_history=[2, 2, 3, 3, 3],
        performance_history=[0.75, 0.82, 0.78, 0.85, 0.88],
        satisfaction_history=[0.7, 0.75, 0.8, 0.85, 0.9]
    )
    firebase.save_session(session)
    
    # 多用戶查詢示例
    elderly_users = firebase.list_users({
        'age': ['>=', 65]
    })
    print(f"找到 {len(elderly_users)} 位65歲以上的用戶")
    
    # 查詢用戶的長期進度
    progress_data = firebase.get_user_progress("user123", months=6)
    print(f"過去6個月有 {len(progress_data)} 條進度記錄")
    
    # 為研究目的批量導出數據
    export_url = firebase.batch_export_data({
        'cognitive_status': CognitiveStatus.MCI.value,
        'age': ['>=', 60]
    })
    print(f"數據已導出，可從以下URL下載: {export_url}")


if __name__ == "__main__":
    # 運行示例演示
    demo_firebase_integration() 