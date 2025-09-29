import logging

# 配置日志
def setup_logging(log_enabled=True):
    # 创建 logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # 可以根据需要设置级别

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 可以根据需要设置级别

    # 创建格式器并将其添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(ch)

    # 根据开关控制日志输出
    if not log_enabled:
        logger.disabled = True

    return logger

# 使用日志
logger = setup_logging(log_enabled=True)  # 设置为 False 可关闭日志

logger.debug("这是一条调试信息。")
logger.info("这是一条普通信息。")
logger.warning("这是一条警告信息。")
logger.error("这是一条错误信息。")
logger.critical("这是一条严重错误信息。")
