import logging
from typing import Optional, Dict, Union

class ColorFormatter(logging.Formatter):
    """增强的彩色格式化器，支持多种样式选项。
    
    特性:
    - 不同日志级别使用不同颜色
    - 支持文本样式（粗体、下划线等）
    - 支持时间戳格式化
    - 可自定义消息格式
    """
    
    COLORS = {
        logging.DEBUG: "\033[36m",     # 青色
        logging.INFO: "\033[32m",      # 绿色
        logging.WARNING: "\033[33m",   # 黄色
        logging.ERROR: "\033[31m",     # 红色
        logging.CRITICAL: "\033[41m"   # 红色背景
    }
    
    STYLES = {
        'bold': "\033[1m",
        'dim': "\033[2m",
        'underline': "\033[4m",
        'blink': "\033[5m",
        'reverse': "\033[7m"
    }
    
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录。

        Args:
            record: 日志记录对象

        Returns:
            str: 格式化后的日志字符串
        """
        if not hasattr(record, 'style'):
            record.style = ''
        
        color = self.COLORS.get(record.levelno, self.RESET)
        style = ''.join(self.STYLES.get(s, '') for s in record.style.split(',') if s)
        
        formatted_msg = super().format(record)
        return f"{color}{style}{formatted_msg}{self.RESET}"

def setup_logger(
    name: str = "IKSolve",
    level: int = logging.INFO,
    format_str: str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt: str = '%H:%M:%S'
) -> logging.Logger:
    """设置带有颜色和格式化的日志记录器。

    Args:
        name: 日志记录器名称
        level: 日志级别
        format_str: 日志格式字符串
        datefmt: 时间格式字符串

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        formatter = ColorFormatter(format_str, datefmt=datefmt)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

def create_loggers() -> Dict[str, logging.Logger]:
    """创建常用的日志记录器。

    Returns:
        Dict[str, logging.Logger]: 日志记录器字典
    """
    return {
        'ik': setup_logger("IKSolve", level=logging.INFO),
        'control': setup_logger("Control", level=logging.INFO),
        'collision': setup_logger("Collision", level=logging.INFO),
        'performance': setup_logger("Performance", level=logging.INFO)
    }

def log_collision_info(logger: logging.Logger, value: float) -> None:
    """记录碰撞信息。

    Args:
        logger: 碰撞日志记录器
        value: 碰撞距离值
    """
    status = "YES" if value > 0.0 else "NO"
    record = logging.LogRecord(
        "Collision",
        logging.WARNING if value > 0.0 else logging.INFO,
        "", 0,
        f"Distance: {value:.6f} m | Collision: {status}",
        (), None
    )
    record.style = 'bold' if value > 0.0 else ''
    logger.handle(record)

def log_performance_stats(
    logger: logging.Logger,
    stats: Dict[str, Union[int, float]],
    prefix: str = ""
) -> None:
    """记录性能统计信息。

    Args:
        logger: 性能日志记录器
        stats: 性能统计数据字典
        prefix: 消息前缀
    """
    if prefix:
        logger.info(f"{prefix}:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"{key:15s}: {value:8.3f}")
        else:
            logger.info(f"{key:15s}: {value:8d}")




# 示例使用
if __name__ == "__main__":
    # 创建日志记录器
    loggers = create_loggers()
    
    # 使用示例
    loggers['ik'].info("IK solver initialized")
    loggers['control'].warning("Control warning message")
    loggers['collision'].error("Collision detected!")
    
    # 记录碰撞信息
    log_collision_info(loggers['collision'], 0.05)
    
    # 记录性能统计
    stats = {
        'opt_step_count': 1000,
        'oneLoop': 15.5,
        'OptimizeTime': 10.2
    }
    log_performance_stats(loggers['performance'], stats, "Performance Statistics")