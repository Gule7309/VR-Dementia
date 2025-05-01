"""
Firebase數據管理模組：提供VR認知訓練應用與Firebase的整合
實現長期用戶數據存儲、多用戶管理、臨床研究整合及多設備同步
"""

import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

from schemas import (
    GameType, UserProfile, GamePerformanceData, SessionData, 
    CognitiveAssessmentData
)


class FirebaseDataManager:
    """Firebase數據管理類：提供與Firebase的連接和數據存取功能"""
    
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
                # 在生產環境中，應通過環境變量獲取認證
                cred = credentials.ApplicationDefault()
            
            firebase_admin.initialize_app(cred)
        
        # 獲取Firestore客戶端
        self.db = firestore.client()
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
            perf_dict['game_specific_metrics'] = {}
            
            if performance_data.game_specific_metrics.calculation_metrics:
                perf_dict['game_specific_metrics']['calculation'] = vars(performance_data.game_specific_metrics.calculation_metrics)
            
            if performance_data.game_specific_metrics.memory_metrics:
                perf_dict['game_specific_metrics']['memory'] = vars(performance_data.game_specific_metrics.memory_metrics)
            
            if performance_data.game_specific_metrics.spatial_metrics:
                perf_dict['game_specific_metrics']['spatial'] = vars(performance_data.game_specific_metrics.spatial_metrics)
        
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
    
    # ======== 數據分析與比較 ========
    
    def get_user_longitudinal_data(self, user_id: str, months: int = 12) -> Dict[str, Any]:
        """獲取用戶長期數據變化"""
        start_date = datetime.now() - timedelta(days=30*months)
        
        # 獲取會話數據
        sessions = self.get_user_sessions(user_id, start_date=start_date, limit=1000)
        
        # 獲取認知評估
        assessments = self.get_user_assessments(user_id, start_date=start_date)
        
        # 按月份組織數據
        monthly_data = {}
        for session in sessions:
            # 獲取會話所屬年月
            session_time = session.get('start_time')
            if isinstance(session_time, str):
                session_time = datetime.fromisoformat(session_time)
            
            month_key = session_time.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'session_count': 0,
                    'avg_performance': 0,
                    'total_performance': 0,
                    'game_types': {},
                    'assessments': []
                }
            
            monthly_data[month_key]['session_count'] += 1
            monthly_data[month_key]['total_performance'] += session.get('avg_performance', 0)
            
            # 按遊戲類型追蹤數據
            game_type = session.get('game_type')
            if game_type not in monthly_data[month_key]['game_types']:
                monthly_data[month_key]['game_types'][game_type] = {
                    'count': 0,
                    'total_performance': 0
                }
            
            monthly_data[month_key]['game_types'][game_type]['count'] += 1
            monthly_data[month_key]['game_types'][game_type]['total_performance'] += session.get('avg_performance', 0)
        
        # 添加認知評估數據
        for assessment in assessments:
            assessment_time = assessment.get('assessment_date')
            if isinstance(assessment_time, str):
                assessment_time = datetime.fromisoformat(assessment_time)
            
            month_key = assessment_time.strftime('%Y-%m')
            
            if month_key in monthly_data:
                monthly_data[month_key]['assessments'].append({
                    'assessment_id': assessment.get('assessment_id'),
                    'overall_score': assessment.get('overall_score'),
                    'date': assessment_time
                })
        
        # 計算平均值和整理數據
        result = []
        for month, data in monthly_data.items():
            if data['session_count'] > 0:
                data['avg_performance'] = data['total_performance'] / data['session_count']
                del data['total_performance']
            
            for game_type, game_data in data['game_types'].items():
                if game_data['count'] > 0:
                    game_data['avg_performance'] = game_data['total_performance'] / game_data['count']
                    del game_data['total_performance']
            
            # 轉換為列表格式
            result.append({
                'month': month,
                'data': data
            })
        
        # 按時間排序
        result.sort(key=lambda x: x['month'])
        
        return {
            'user_id': user_id,
            'period': f"過去{months}個月",
            'data_points': len(result),
            'monthly_data': result
        }
    
    def compare_user_groups(self, group_queries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """比較不同用戶組的表現"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'groups': {}
        }
        
        for group_name, query in group_queries.items():
            # 獲取該組的用戶
            users = self.list_users(query)
            user_ids = [user['user_id'] for user in users]
            
            group_data = {
                'user_count': len(users),
                'avg_age': 0,
                'gender_distribution': {'male': 0, 'female': 0, 'other': 0},
                'avg_performance': {},
                'cognitive_domains': {}
            }
            
            # 計算用戶統計信息
            total_age = 0
            for user in users:
                total_age += user.get('age', 0)
                gender = user.get('gender', 'other')
                group_data['gender_distribution'][gender] = group_data['gender_distribution'].get(gender, 0) + 1
            
            if len(users) > 0:
                group_data['avg_age'] = total_age / len(users)
            
            # 收集所有會話和評估數據
            all_sessions = []
            all_assessments = []
            
            for user_id in user_ids:
                # 獲取最近3個月的數據
                three_months_ago = datetime.now() - timedelta(days=90)
                sessions = self.get_user_sessions(user_id, start_date=three_months_ago)
                all_sessions.extend(sessions)
                
                assessments = self.get_user_assessments(user_id, start_date=three_months_ago)
                all_assessments.extend(assessments)
            
            # 按遊戲類型聚合表現
            game_type_performance = {}
            for session in all_sessions:
                game_type = session.get('game_type')
                if game_type not in game_type_performance:
                    game_type_performance[game_type] = {
                        'count': 0,
                        'total_performance': 0
                    }
                
                game_type_performance[game_type]['count'] += 1
                game_type_performance[game_type]['total_performance'] += session.get('avg_performance', 0)
            
            # 計算平均表現
            for game_type, data in game_type_performance.items():
                if data['count'] > 0:
                    group_data['avg_performance'][game_type] = data['total_performance'] / data['count']
            
            # 計算認知領域平均分數
            domain_scores = {}
            for assessment in all_assessments:
                for domain, data in assessment.get('cognitive_domains', {}).items():
                    if domain not in domain_scores:
                        domain_scores[domain] = {
                            'count': 0,
                            'total_score': 0
                        }
                    
                    domain_scores[domain]['count'] += 1
                    domain_scores[domain]['total_score'] += data.get('score', 0)
            
            # 計算每個認知領域的平均分數
            for domain, data in domain_scores.items():
                if data['count'] > 0:
                    group_data['cognitive_domains'][domain] = data['total_score'] / data['count']
            
            results['groups'][group_name] = group_data
        
        return results


# 使用示例
def demo_firebase_usage():
    """展示Firebase數據管理器的基本使用方法"""
    
    # 初始化Firebase管理器
    firebase = FirebaseDataManager('path/to/credentials.json')
    
    # 創建用戶示例
    user_profile = UserProfile(
        user_id="demo_user_123",
        age=72,
        gender="female",
        education_level="bachelor",
        cognitive_status=CognitiveStatus.NORMAL
    )
    firebase.create_user(user_profile)
    
    print("用戶創建成功，可以開始收集認知訓練數據")
    
    # 查詢不同年齡組的用戶
    elderly_users = firebase.list_users({
        'age': ['>=', 65]
    })
    middle_aged_users = firebase.list_users({
        'age': ['>=', 45],
        'age': ['<', 65]
    })
    
    print(f"系統中有 {len(elderly_users)} 位65歲以上用戶和 {len(middle_aged_users)} 位45-65歲用戶")
    
    # 比較用戶組
    comparison = firebase.compare_user_groups({
        '老年組': {'age': ['>=', 65]},
        '中年組': {'age': ['>=', 45], 'age': ['<', 65]}
    })
    
    print("用戶組比較完成，可用於生成報告")
    
    # 獲取某用戶的長期進展數據
    longitudinal_data = firebase.get_user_longitudinal_data("demo_user_123", months=6)
    print(f"獲取到用戶過去6個月的 {longitudinal_data['data_points']} 個月度數據點")


if __name__ == "__main__":
    # 運行示例
    demo_firebase_usage() 