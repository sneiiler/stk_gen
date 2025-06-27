"""
数据验证器模块

该模块提供了用于验证大模型生成的卫星分簇结果的验证器类。
"""

import sys
sys.path.append('/root/stk_gen')
from typing import Dict, Any, List, Set, Tuple, Optional
import json,re
import logging  
from pydantic import BaseModel
from utils.misc_utils import get_data_dir

class ValidationResult(BaseModel):
    is_valid:bool
    errors:List[str]
    warnings:List[str]
    details:Dict[str,Any]

class ClusterDataValidator:
    """卫星分簇结果验证器
    
    用于验证大模型生成的卫星分簇结果是否符合业务规则和约束条件。
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化验证器
        
        Args:
            logger: 日志记录器，如果为None则使用默认配置
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_output(self, input_data: Dict[str, Any]):
        """验证输出结果
        
        Args:
            
        Returns:
            验证结果，包含验证状态、错误信息和警告信息
        """
        validation_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            details={}
        )
        
        try:
            # 1. 基础格式验证
            self._validate_basic_format(input_data, validation_result)
            if not validation_result.is_valid:
                return validation_result
            
            # 2. 目标覆盖验证
            self._validate_target_coverage(output, input_data, validation_result)
            
            # 3. 卫星分配验证
            self._validate_satellite_assignment(output, input_data, validation_result)
            
            # 4. 主节点验证
            self._validate_master_nodes(output, validation_result)
            
            # 5. 策略约束验证
            self._validate_strategy_constraints(output, input_data, validation_result)
            
            # 6. 链路质量验证
            self._validate_link_quality(output, input_data, validation_result)
            
            # 7. 观测质量验证
            self._validate_observation_quality(output, input_data, validation_result)
            
            # 8. 健康度验证
            self._validate_health_constraints(output, input_data, validation_result)
            
        except Exception as e:
            validation_result.is_valid = False
            validation_result.errors.append(f"验证过程出错: {str(e)}")
            self.logger.error(f"验证过程发生异常: {e}", exc_info=True)
        
        return validation_result
    
    def _validate_basic_format(self, output: Dict[str, Any], result: ValidationResult) -> None:
        """验证基础格式
        
        Args:
            output: 输出数据
            result: 验证结果对象
        """
        if not output:
            result.is_valid = False
            result.errors.append("输出为空")
            return
        
        if "clusters" not in output:
            result.is_valid = False
            result.errors.append("输出格式错误：缺少clusters字段")
            return
        
        if not isinstance(output["clusters"], list):
            result.is_valid = False
            result.errors.append("输出格式错误：clusters字段必须是数组")
            return
        
        # 验证每个集群的格式
        for i, cluster in enumerate(output["clusters"]):
            if not isinstance(cluster, dict):
                result.is_valid = False
                result.errors.append(f"集群 {i} 格式错误：必须是对象")
                continue
            
            required_fields = ["cluster_id", "master", "sats", "targets"]
            for field in required_fields:
                if field not in cluster:
                    result.is_valid = False
                    result.errors.append(f"集群 {i} 缺少必需字段: {field}")
    
    def _validate_target_coverage(self, output: Dict[str, Any], input_data: Dict[str, Any], 
                                 result: ValidationResult) -> None:
        """验证目标覆盖完整性
        
        Args:
            output: 输出数据
            input_data: 输入数据
            result: 验证结果对象
        """
        # 提取输入中的所有目标
        input_targets = set()
        for edge in input_data.get("target_edges", []):
            input_targets.add(edge["to"])
        
        # 提取输出中的所有目标
        output_targets = set()
        for cluster in output["clusters"]:
            output_targets.update(cluster.get("targets", []))
        
        # 检查目标覆盖是否完整
        missing_targets = input_targets - output_targets
        extra_targets = output_targets - input_targets
        
        if missing_targets:
            result.is_valid = False
            result.errors.append(f"目标覆盖不完整：缺少目标 {missing_targets}")
        
        if extra_targets:
            result.is_valid = False
            result.errors.append(f"目标覆盖错误：包含不存在的目标 {extra_targets}")
        
        result.details["target_coverage"] = {
            "input_targets": len(input_targets),
            "output_targets": len(output_targets),
            "coverage_rate": len(output_targets & input_targets) / len(input_targets) if input_targets else 0
        }
    
    def _validate_satellite_assignment(self, output: Dict[str, Any], input_data: Dict[str, Any], 
                                     result: ValidationResult) -> None:
        """验证卫星分配
        
        Args:
            output: 输出数据
            input_data: 输入数据
            result: 验证结果对象
        """
        # 提取输入中的所有卫星
        input_satellites = set()
        for attr in input_data.get("sat_attrs", []):
            input_satellites.add(attr["id"])
        
        # 提取输出中的所有卫星
        all_output_sats = []
        for cluster in output["clusters"]:
            all_output_sats.extend(cluster.get("sats", []))
        
        output_satellites = set(all_output_sats)
        
        # 检查重复分配
        if len(all_output_sats) != len(output_satellites):
            result.is_valid = False
            result.errors.append("存在重复分配的卫星")
        
        # 检查是否存在不存在的卫星
        invalid_satellites = output_satellites - input_satellites
        if invalid_satellites:
            result.is_valid = False
            result.errors.append(f"包含不存在的卫星: {invalid_satellites}")
        
        # 检查卫星利用率
        unused_satellites = input_satellites - output_satellites
        if unused_satellites:
            result.warnings.append(f"未使用的卫星: {unused_satellites}")
        
        result.details["satellite_assignment"] = {
            "total_satellites": len(input_satellites),
            "assigned_satellites": len(output_satellites),
            "utilization_rate": len(output_satellites) / len(input_satellites) if input_satellites else 0
        }
    
    def _validate_master_nodes(self, output: Dict[str, Any], result: ValidationResult) -> None:
        """验证主节点
        
        Args:
            output: 输出数据
            result: 验证结果对象
        """
        for cluster in output["clusters"]:
            master = cluster.get("master")
            sats = cluster.get("sats", [])
            
            if master not in sats:
                result.is_valid = False
                result.errors.append(
                    f"集群 {cluster.get('cluster_id')} 的主节点 {master} 不在卫星列表中"
                )
            
            # 检查主节点是否被重复使用
            master_count = sum(1 for c in output["clusters"] if c.get("master") == master)
            if master_count > 1:
                result.is_valid = False
                result.errors.append(f"主节点 {master} 被重复使用")
    
    def _validate_strategy_constraints(self, output: Dict[str, Any], input_data: Dict[str, Any], 
                                     result: ValidationResult) -> None:
        """验证策略约束
        
        Args:
            output: 输出数据
            input_data: 输入数据
            result: 验证结果对象
        """
        strategy = input_data.get("strategy", "balanced")
        
        for cluster in output["clusters"]:
            sat_count = len(cluster.get("sats", []))
            target_count = len(cluster.get("targets", []))
            
            if strategy == "balanced" and sat_count > target_count:
                result.warnings.append(
                    f"集群 {cluster.get('cluster_id')} 卫星数({sat_count})超过目标数({target_count})"
                )
            elif strategy == "quality" and sat_count > 2 * target_count:
                result.warnings.append(
                    f"集群 {cluster.get('cluster_id')} 卫星数({sat_count})超过2倍目标数({target_count})"
                )
            
            # 检查空集群
            if sat_count == 0:
                result.is_valid = False
                result.errors.append(f"集群 {cluster.get('cluster_id')} 没有卫星")
            
            if target_count == 0:
                result.warnings.append(f"集群 {cluster.get('cluster_id')} 没有目标")
    
    def _validate_link_quality(self, output: Dict[str, Any], input_data: Dict[str, Any], 
                             result: ValidationResult) -> None:
        """验证链路质量
        
        Args:
            output: 输出数据
            input_data: 输入数据
            result: 验证结果对象
        """
        # 构建链路映射
        link_map = {}
        for edge in input_data.get("sat_edges", []):
            key = (edge["from"], edge["to"])
            link_map[key] = edge["w"]
        
        total_link_strength = 0
        cluster_count = 0
        
        for cluster in output["clusters"]:
            sats = cluster.get("sats", [])
            if len(sats) < 2:
                continue
            
            cluster_link_strength = 0
            link_count = 0
            
            # 计算集群内链路强度
            for i, sat1 in enumerate(sats):
                for sat2 in sats[i+1:]:
                    key1 = (sat1, sat2)
                    key2 = (sat2, sat1)
                    
                    if key1 in link_map:
                        cluster_link_strength += link_map[key1]
                        link_count += 1
                    elif key2 in link_map:
                        cluster_link_strength += link_map[key2]
                        link_count += 1
            
            if link_count > 0:
                avg_link_strength = cluster_link_strength / link_count
                total_link_strength += avg_link_strength
                cluster_count += 1
                
                if avg_link_strength < 0.3:
                    result.warnings.append(
                        f"集群 {cluster.get('cluster_id')} 平均链路强度较低: {avg_link_strength:.3f}"
                    )
        
        if cluster_count > 0:
            overall_avg_strength = total_link_strength / cluster_count
            result.details["link_quality"] = {
                "overall_avg_strength": overall_avg_strength,
                "cluster_count": cluster_count
            }
    
    def _validate_observation_quality(self, output: Dict[str, Any], input_data: Dict[str, Any], 
                                    result: ValidationResult) -> None:
        """验证观测质量
        
        Args:
            output: 输出数据
            input_data: 输入数据
            result: 验证结果对象
        """
        # 构建观测质量映射
        obs_map = {}
        for edge in input_data.get("target_edges", []):
            key = (edge["from"], edge["to"])
            obs_map[key] = edge["q"]
        
        total_obs_quality = 0
        obs_count = 0
        
        for cluster in output["clusters"]:
            sats = cluster.get("sats", [])
            targets = cluster.get("targets", [])
            
            cluster_obs_quality = 0
            cluster_obs_count = 0
            
            # 计算集群的观测质量
            for sat in sats:
                for target in targets:
                    key = (sat, target)
                    if key in obs_map:
                        cluster_obs_quality += obs_map[key]
                        cluster_obs_count += 1
            
            if cluster_obs_count > 0:
                avg_obs_quality = cluster_obs_quality / cluster_obs_count
                total_obs_quality += avg_obs_quality
                obs_count += 1
                
                if avg_obs_quality < 0.5:
                    result.warnings.append(
                        f"集群 {cluster.get('cluster_id')} 平均观测质量较低: {avg_obs_quality:.3f}"
                    )
        
        if obs_count > 0:
            overall_avg_quality = total_obs_quality / obs_count
            result.details["observation_quality"] = {
                "overall_avg_quality": overall_avg_quality,
                "cluster_count": obs_count
            }
    
    def _validate_health_constraints(self, output: Dict[str, Any], input_data: Dict[str, Any], 
                                   result: ValidationResult) -> None:
        """验证健康度约束
        
        Args:
            output: 输出数据
            input_data: 输入数据
            result: 验证结果对象
        """
        # 构建健康度映射
        health_map = {}
        for attr in input_data.get("sat_attrs", []):
            health_map[attr["id"]] = attr["health"]
        
        for cluster in output["clusters"]:
            master = cluster.get("master")
            sats = cluster.get("sats", [])
            
            # 检查主节点健康度
            if master in health_map:
                master_health = health_map[master]
                if master_health < 0.7:
                    result.warnings.append(
                        f"集群 {cluster.get('cluster_id')} 主节点 {master} 健康度较低: {master_health:.3f}"
                    )
            
            # 检查集群整体健康度
            cluster_healths = [health_map.get(sat, 0) for sat in sats]
            avg_health = sum(cluster_healths) / len(cluster_healths) if cluster_healths else 0
            
            if avg_health < 0.6:
                result.warnings.append(
                    f"集群 {cluster.get('cluster_id')} 平均健康度较低: {avg_health:.3f}"
                )

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        数据列表
    """
    raw_data = []
    input_user_data = []
    output_resoning_data = []
    output_result_data = []
    
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            # 移除行尾的换行符
            cleaned_line = line.strip()
            if cleaned_line:  # 确保非空行
                # 解析JSON
                data = json.loads(cleaned_line)
                raw_data.append(data)

    for line_data in raw_data:
        for message in line_data:
            if message['role'] == "user":
                input_user_data.append(message['content'])
            if message['role'] == "assistant":
                # 使用正则表达式匹配 </thionk> 后面到下一个 ``` 之间的内容
                pattern = r'<think>(.*?)</think>(.*?)\[(.*)\]$$'
                match = re.search(pattern, message['content'], re.DOTALL)
                if match:
                    try:
                        output_resoning_data.append(match.group(1))
                        output_result_data.append(json.loads("["+match.group(3)+"]"))
                    except json.JSONDecodeError as e:
                        print("JSON 解析失败:", e)
                        return None
    

def main():
    data_path = get_data_dir() / "distilled_training_data_v20250626_sharegpt_format_v2.jsonl"
    load_data(data_path)


if __name__ == "__main__":
    main()
    print("Done.")