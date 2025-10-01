# 模块级常量和函数
from rknnlite.api import RKNNLite
import numpy as np
import os
import warnings
import logging
from typing import List, Dict, Union, Optional

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    warnings.warn("onnxruntime未安装,只能使用RKNN后端", ImportWarning)

# 配置日志
logger = logging.getLogger("somemodelruntime_rknnlite2")
logger.setLevel(logging.ERROR)  # 默认只输出错误信息
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# ONNX Runtime日志级别到Python logging级别的映射
_LOGGING_LEVEL_MAP = {
    0: logging.DEBUG,    # Verbose
    1: logging.INFO,     # Info
    2: logging.WARNING,  # Warning
    3: logging.ERROR,    # Error
    4: logging.CRITICAL  # Fatal
}

# 检查环境变量中的日志级别设置
try:
    env_log_level = os.getenv('ZTU_MODELRT_RKNNL2_LOG_LEVEL')
    if env_log_level is not None:
        log_level = int(env_log_level)
        if log_level in _LOGGING_LEVEL_MAP:
            logger.setLevel(_LOGGING_LEVEL_MAP[log_level])
            logger.info(f"从环境变量设置日志级别: {log_level}")
        else:
            logger.warning(f"环境变量ZTU_MODELRT_RKNNL2_LOG_LEVEL的值无效: {log_level}, 应该是0-4之间的整数")
except ValueError:
    logger.warning(f"环境变量ZTU_MODELRT_RKNNL2_LOG_LEVEL的值无效: {env_log_level}, 应该是0-4之间的整数")


def set_default_logger_severity(level: int) -> None:
    """
    Sets the default logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
    
    Args:
        level: 日志级别(0-4)
    """
    if level not in _LOGGING_LEVEL_MAP:
        raise ValueError(f"无效的日志级别: {level}, 应该是0-4之间的整数")
    logger.setLevel(_LOGGING_LEVEL_MAP[level])

def set_default_logger_verbosity(level: int) -> None:
    """
    Sets the default logging verbosity level. To activate the verbose log, 
    you need to set the default logging severity to 0:Verbose level.
    
    Args:
        level: 日志级别(0-4)
    """
    set_default_logger_severity(level)

# RKNN tensor type到numpy dtype的映射
RKNN_DTYPE_MAP = {
    0: np.float32,  # RKNN_TENSOR_FLOAT32
    1: np.float16,  # RKNN_TENSOR_FLOAT16
    2: np.int8,     # RKNN_TENSOR_INT8
    3: np.uint8,    # RKNN_TENSOR_UINT8
    4: np.int16,    # RKNN_TENSOR_INT16
    5: np.uint16,   # RKNN_TENSOR_UINT16
    6: np.int32,    # RKNN_TENSOR_INT32
    7: np.uint32,   # RKNN_TENSOR_UINT32
    8: np.int64,    # RKNN_TENSOR_INT64
    9: bool,        # RKNN_TENSOR_BOOL
    10: np.int8,    # RKNN_TENSOR_INT4 (用int8表示)
}

def get_available_providers() -> List[str]:
    """
    获取可用的设备提供者列表(为保持接口兼容性的占位函数)
    
    Returns:
        list: 可用的设备提供者列表,总是返回["CPUExecutionProvider", "somemodelruntime_rknnlite2_ExecutionProvider"]
    """
    return ["CPUExecutionProvider", "somemodelruntime_rknnlite2_ExecutionProvider"]


def get_device() -> str:
    """
    获取当前设备

    Returns:
        str: 当前设备
    """
    return "RKNN2"

def get_version_info() -> Dict[str, str]:
    """
    获取版本信息
    
    Returns:
        dict: 包含API和驱动版本信息的字典
    """
    runtime = RKNNLite()
    version = runtime.get_sdk_version()
    return {
        "api_version": version.split('\n')[2].split(': ')[1].split(' ')[0],
        "driver_version": version.split('\n')[3].split(': ')[1]
    }

class IOTensor:
    """输入/输出张量的信息封装类"""
    def __init__(self, name, shape, type=None):
        self.name = name.decode() if isinstance(name, bytes) else name
        self.shape = shape
        self.type = type

    def __str__(self):
        return f"IOTensor(name='{self.name}', shape={self.shape}, type={self.type})"

class SessionOptions:
    """会话选项类"""
    def __init__(self):
        self.enable_profiling = False  # 是否使用性能分析
        self.intra_op_num_threads = 1  # 设置RKNN的线程数, 对应rknn的core_mask
        self.log_severity_level = -1 # 另一个设置日志级别的参数
        self.log_verbosity_level = -1 # 另一个设置日志级别的参数


class InferenceSession:
    """
    RKNNLite运行时封装类,API风格类似ONNX Runtime
    """

    def __new__(cls, model_path: str, sess_options: Optional[SessionOptions] = None, **kwargs):
        processed_path = InferenceSession._process_model_path(model_path, sess_options)
        if isinstance(processed_path, str) and processed_path.lower().endswith('.onnx'):
            logger.info("使用ONNX Runtime加载模型")
            if not HAS_ORT:
                raise RuntimeError("未安装onnxruntime,无法加载ONNX模型")
            return ort.InferenceSession(processed_path, sess_options=sess_options, **kwargs)
        else:
            # 如果不是 ONNX 模型，则调用父类的 __new__ 创建 InferenceSession 实例
            instance = super().__new__(cls)
            # 保存处理后的路径
            instance._processed_path = processed_path
            return instance

    def __init__(self, model_path: str, sess_options: Optional[SessionOptions] = None, **kwargs):
        """
        初始化运行时并加载模型
        
        Args:
            model_path: 模型文件路径(.rknn或.onnx)
            sess_options: 会话选项
            **kwargs: 其他初始化参数
        """
        options = sess_options or SessionOptions()

        # 只在未设置环境变量时使用SessionOptions中的日志级别
        if os.getenv('ZTU_MODELRT_RKNNL2_LOG_LEVEL') is None:
            if options.log_severity_level != -1:
                set_default_logger_severity(options.log_severity_level)
            if options.log_verbosity_level != -1:
                set_default_logger_verbosity(options.log_verbosity_level)
            
        # 使用__new__中处理好的路径
        model_path = getattr(self, '_processed_path', model_path)
        if isinstance(model_path, str) and model_path.lower().endswith('.onnx'):
            # 避免重复加载 ONNX 模型
            return

        # ... 现有的 RKNN 模型加载和初始化代码 ...
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.runtime = RKNNLite(verbose=options.enable_profiling)

        logger.debug(f"正在加载模型: {self.model_path}")
        ret = self.runtime.load_rknn(self.model_path)
        if ret != 0:
            logger.error(f"加载RKNN模型失败: {self.model_path}")
            raise RuntimeError(f'加载RKNN模型失败: {self.model_path}')
        logger.debug("模型加载成功")


        if options.intra_op_num_threads == 1:
            core_mask = RKNNLite.NPU_CORE_AUTO
        elif options.intra_op_num_threads == 2:
            core_mask = RKNNLite.NPU_CORE_0_1
        elif options.intra_op_num_threads == 3:
            core_mask = RKNNLite.NPU_CORE_0_1_2
        else:
            raise ValueError(f"intra_op_num_threads的值无效: {options.intra_op_num_threads}, 只能是1,2或3")

        logger.debug("正在初始化运行时环境")
        ret = self.runtime.init_runtime(core_mask=core_mask)
        if ret != 0:
            logger.error("初始化运行时环境失败")
            raise RuntimeError('初始化运行时环境失败')
        logger.debug("运行时环境初始化成功")

        self._init_io_info()
        self.options = options

    def get_performance_info(self) -> Dict[str, float]:
        """
        获取性能信息
        
        Returns:
            dict: 包含性能信息的字典
        """
        if not self.options.perf_debug:
            raise RuntimeError("性能分析未启用,请在SessionOptions中设置perf_debug=True")
            
        perf = self.runtime.rknn_runtime.get_run_perf()
        return {
            "run_duration": perf.run_duration / 1000.0  # 转换为毫秒
        }

    def set_core_mask(self, core_mask: int) -> None:
        """
        设置NPU核心使用模式
        
        Args:
            core_mask: NPU核心掩码,使用NPU_CORE_*常量
        """
        ret = self.runtime.rknn_runtime.set_core_mask(core_mask)
        if ret != 0:
            raise RuntimeError("设置NPU核心模式失败")

    @staticmethod
    def _process_model_path(model_path, sess_options):
        """
        处理模型路径,支持.onnx和.rknn文件
        
        Args:
            model_path: 模型文件路径
        """
        # 如果是ONNX文件,检查是否需要自动加载RKNN
        if model_path.lower().endswith('.onnx'):
            logger.info("检测到ONNX模型文件")
            
            # 获取需要跳过自动加载的模型列表
            skip_models = os.getenv('ZTU_MODELRT_RKNNL2_SKIP', '').strip()
            if skip_models:
                skip_list = [m.strip() for m in skip_models.split(',')]
                # 获取模型文件名(不含路径)用于匹配
                model_name = os.path.basename(model_path)
                if model_name.lower() in [m.lower() for m in skip_list]:
                    logger.info(f"模型{model_name}在跳过列表中,将使用ONNX Runtime")
                    return model_path
            
            # 构造RKNN文件路径
            rknn_path = os.path.splitext(model_path)[0] + '.rknn'
            if os.path.exists(rknn_path):
                logger.info(f"找到对应的RKNN模型,将使用RKNN: {rknn_path}")
                return rknn_path
            else:
                logger.info("未找到对应的RKNN模型,将使用ONNX Runtime")
                return model_path
            
        return model_path
        
    def _convert_nhwc_to_nchw(self, shape):
        """将NHWC格式的shape转换为NCHW格式"""
        if len(shape) == 4:
            # NHWC -> NCHW
            n, h, w, c = shape
            return [n, c, h, w]
        return shape
        
    def _init_io_info(self):
        """初始化模型的输入输出信息"""
        runtime = self.runtime.rknn_runtime
        
        # 获取输入输出数量
        n_input, n_output = runtime.get_in_out_num()
        
        # 获取输入信息
        self.input_tensors = []
        for i in range(n_input):
            attr = runtime.get_tensor_attr(i)
            shape = [attr.dims[j] for j in range(attr.n_dims)]
            # 对四维输入进行NHWC到NCHW的转换
            shape = self._convert_nhwc_to_nchw(shape)
            # 获取dtype
            dtype = RKNN_DTYPE_MAP.get(attr.type, None)
            tensor = IOTensor(attr.name, shape, dtype)
            self.input_tensors.append(tensor)
            
        # 获取输出信息
        self.output_tensors = []
        for i in range(n_output):
            attr = runtime.get_tensor_attr(i, is_output=True)
            shape = runtime.get_output_shape(i)
            # 获取dtype
            dtype = RKNN_DTYPE_MAP.get(attr.type, None)
            tensor = IOTensor(attr.name, shape, dtype)
            self.output_tensors.append(tensor)
        
    def get_inputs(self):
        """
        获取模型输入信息
        
        Returns:
            list: 包含输入信息的列表
        """
        return self.input_tensors
        
    def get_outputs(self):
        """
        获取模型输出信息
        
        Returns:
            list: 包含输出信息的列表
        """
        return self.output_tensors
        
    def run(self, output_names=None, input_feed=None, data_format="nchw", **kwargs):
        """
        执行模型推理
        
        Args:
            output_names: 输出节点名称列表,指定需要返回哪些输出
            input_feed: 输入数据字典或列表
            data_format: 输入数据格式,"nchw"或"nhwc"
            **kwargs: 其他运行时参数
            
        Returns:
            list: 模型输出结果列表,如果指定了output_names则只返回指定的输出
        """
        if input_feed is None:
            logger.error("input_feed不能为None")
            raise ValueError("input_feed不能为None")
            
        # 准备输入数据
        if isinstance(input_feed, dict):
            # 如果是字典,按照模型输入顺序排列
            inputs = []
            input_map = {tensor.name: i for i, tensor in enumerate(self.input_tensors)}
            for tensor in self.input_tensors:
                if tensor.name not in input_feed:
                    raise ValueError(f"缺少输入: {tensor.name}")
                inputs.append(input_feed[tensor.name])
        elif isinstance(input_feed, (list, tuple)):
            # 如果是列表,确保长度匹配
            if len(input_feed) != len(self.input_tensors):
                raise ValueError(f"输入数量不匹配: 期望{len(self.input_tensors)}, 实际{len(input_feed)}")
            inputs = list(input_feed)
        else:
            logger.error("input_feed必须是字典或列表类型")
            raise ValueError("input_feed必须是字典或列表类型")
            
        # 执行推理
        try:
            logger.debug("开始执行推理")
            all_outputs = self.runtime.inference(inputs=inputs, data_format=data_format)
            
            # 如果没有指定output_names,返回所有输出
            if output_names is None:
                return all_outputs
                
            # 获取指定的输出
            output_map = {tensor.name: i for i, tensor in enumerate(self.output_tensors)}
            selected_outputs = []
            for name in output_names:
                if name not in output_map:
                    raise ValueError(f"未找到输出节点: {name}")
                selected_outputs.append(all_outputs[output_map[name]])
                    
            return selected_outputs
            
        except Exception as e:
            logger.error(f"推理执行失败: {str(e)}")
            raise RuntimeError(f"推理执行失败: {str(e)}")
        
    def close(self):
        """
        关闭会话,释放资源
        """
        if self.runtime is not None:
            logger.info("正在释放运行时资源")
            self.runtime.release()
            self.runtime = None
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def end_profiling(self) -> Optional[str]:
        """
        结束性能分析的存根方法
        
        Returns:
            Optional[str]: None
        """
        warnings.warn("end_profiling()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return None
        
    def get_profiling_start_time_ns(self) -> int:
        """
        获取性能分析开始时间的存根方法
        
        Returns:
            int: 0
        """
        warnings.warn("get_profiling_start_time_ns()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return 0
        
    def get_modelmeta(self) -> Dict[str, str]:
        """
        获取模型元数据的存根方法
        
        Returns:
            Dict[str, str]: 空字典
        """
        warnings.warn("get_modelmeta()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}
        
    def get_session_options(self) -> SessionOptions:
        """
        获取会话选项
        
        Returns:
            SessionOptions: 当前会话选项
        """
        return self.options
        
    def get_providers(self) -> List[str]:
        """
        获取当前使用的providers的存根方法
        
        Returns:
            List[str]: ["CPUExecutionProvider"]
        """
        warnings.warn("get_providers()是存根方法,始终返回CPUExecutionProvider", RuntimeWarning, stacklevel=2)
        return ["CPUExecutionProvider"]
        
    def get_provider_options(self) -> Dict[str, Dict[str, str]]:
        """
        获取provider选项的存根方法
        
        Returns:
            Dict[str, Dict[str, str]]: 空字典
        """
        warnings.warn("get_provider_options()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {} 

    def get_session_config(self) -> Dict[str, str]:
        """
        获取会话配置的存根方法
        
        Returns:
            Dict[str, str]: 空字典
        """
        warnings.warn("get_session_config()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}

    def get_session_state(self) -> Dict[str, str]:
        """
        获取会话状态的存根方法
        
        Returns:
            Dict[str, str]: 空字典
        """
        warnings.warn("get_session_state()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}

    def set_session_config(self, config: Dict[str, str]) -> None:
        """
        设置会话配置的存根方法
        
        Args:
            config: 会话配置字典
        """
        warnings.warn("set_session_config()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)

    def get_memory_info(self) -> Dict[str, int]:
        """
        获取内存使用信息的存根方法
        
        Returns:
            Dict[str, int]: 空字典
        """
        warnings.warn("get_memory_info()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}

    def set_memory_pattern(self, enable: bool) -> None:
        """
        设置内存模式的存根方法
        
        Args:
            enable: 是否启用内存模式
        """
        warnings.warn("set_memory_pattern()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)

    def disable_memory_pattern(self) -> None:
        """
        禁用内存模式的存根方法
        """
        warnings.warn("disable_memory_pattern()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)

    def get_optimization_level(self) -> int:
        """
        获取优化级别的存根方法
        
        Returns:
            int: 0
        """
        warnings.warn("get_optimization_level()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return 0

    def set_optimization_level(self, level: int) -> None:
        """
        设置优化级别的存根方法
        
        Args:
            level: 优化级别
        """
        warnings.warn("set_optimization_level()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)

    def get_model_metadata(self) -> Dict[str, str]:
        """
        获取模型元数据的存根方法(与get_modelmeta不同的接口)
        
        Returns:
            Dict[str, str]: 空字典
        """
        warnings.warn("get_model_metadata()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}

    def get_model_path(self) -> str:
        """
        获取模型路径
        
        Returns:
            str: 模型文件路径
        """
        return self.model_path

    def get_input_type_info(self) -> List[Dict[str, str]]:
        """
        获取输入类型信息的存根方法
        
        Returns:
            List[Dict[str, str]]: 空列表
        """
        warnings.warn("get_input_type_info()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return []

    def get_output_type_info(self) -> List[Dict[str, str]]:
        """
        获取输出类型信息的存根方法
        
        Returns:
            List[Dict[str, str]]: 空列表
        """
        warnings.warn("get_output_type_info()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return [] 