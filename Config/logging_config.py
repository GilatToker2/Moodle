# """
# Logging configuration for the Academic Content Processing API
# Provides structured logging with different levels and formatters
# """
#
# import logging
# import logging.handlers
# import os
# from datetime import datetime
# import sys
#
#
# def setup_logging(log_level=logging.INFO, log_to_file=True, log_dir="logs"):
#     """
#     Setup comprehensive logging configuration
#
#     Args:
#         log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#         log_to_file: Whether to log to file in addition to console
#         log_dir: Directory to store log files
#     """
#
#     # Create logs directory if it doesn't exist
#     if log_to_file and not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#
#     # Create formatter
#     formatter = logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
#
#     # Get root logger
#     root_logger = logging.getLogger()
#     root_logger.setLevel(log_level)
#
#     # Clear any existing handlers
#     root_logger.handlers.clear()
#
#     # Console handler
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(log_level)
#     console_handler.setFormatter(formatter)
#     root_logger.addHandler(console_handler)
#
#     if log_to_file:
#         # File handler for all logs
#         today = datetime.now().strftime("%Y-%m-%d")
#         log_file = os.path.join(log_dir, f"app_{today}.log")
#
#         file_handler = logging.handlers.RotatingFileHandler(
#             log_file,
#             maxBytes=10 * 1024 * 1024,  # 10MB
#             backupCount=5
#         )
#         file_handler.setLevel(log_level)
#         file_handler.setFormatter(formatter)
#         root_logger.addHandler(file_handler)
#
#         # Error file handler for errors only
#         error_log_file = os.path.join(log_dir, f"errors_{today}.log")
#         error_handler = logging.handlers.RotatingFileHandler(
#             error_log_file,
#             maxBytes=10 * 1024 * 1024,  # 10MB
#             backupCount=5
#         )
#         error_handler.setLevel(logging.ERROR)
#         error_handler.setFormatter(formatter)
#         root_logger.addHandler(error_handler)
#
#     # Set specific loggers for different modules
#     loggers = {
#         'main': logging.getLogger('main'),
#         'video_processor': logging.getLogger('video_processor'),
#         'document_processor': logging.getLogger('document_processor'),
#         'blob_manager': logging.getLogger('blob_manager'),
#         'summarizer': logging.getLogger('summarizer'),
#         'indexer': logging.getLogger('indexer'),
#         'subject_detector': logging.getLogger('subject_detector'),
#         'rag_system': logging.getLogger('rag_system'),
#         'video_indexer_client': logging.getLogger('video_indexer_client')
#     }
#
#     return loggers
#
#
# def get_logger(name):
#     """Get a logger with the specified name"""
#     return logging.getLogger(name)
#
#
# # Performance logging decorator
# def log_performance(logger):
#     """Decorator to log function execution time"""
#
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             import time
#             start_time = time.time()
#             logger.info(f"🚀 Starting {func.__name__} with args: {len(args)} positional, {len(kwargs)} keyword")
#
#             try:
#                 result = func(*args, **kwargs)
#                 execution_time = time.time() - start_time
#                 logger.info(f"✅ {func.__name__} completed successfully in {execution_time:.2f} seconds")
#                 return result
#             except Exception as e:
#                 execution_time = time.time() - start_time
#                 logger.error(f"❌ {func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
#                 raise
#
#         return wrapper
#
#     return decorator
#
#
# # Request logging decorator for API endpoints
# def log_api_request(logger):
#     """Decorator to log API request details"""
#
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             import time
#             start_time = time.time()
#
#             # Extract request details if available
#             request_info = ""
#             if args and hasattr(args[0], 'dict'):
#                 try:
#                     request_data = args[0].dict()
#                     request_info = f" - Request: {request_data}"
#                 except:
#                     pass
#
#             logger.info(f"🌐 API Request: {func.__name__}{request_info}")
#
#             try:
#                 result = func(*args, **kwargs)
#                 execution_time = time.time() - start_time
#                 logger.info(f"✅ API Response: {func.__name__} completed in {execution_time:.2f}s")
#                 return result
#             except Exception as e:
#                 execution_time = time.time() - start_time
#                 logger.error(f"❌ API Error: {func.__name__} failed after {execution_time:.2f}s: {str(e)}")
#                 raise
#
#         return wrapper
#
#     return decorator
import logging
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging with file rotation and multiple levels"""

    # Create logger
    logger = logging.getLogger('academic_api')
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate logs if logger already exists
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler with rotation (max 10MB, keep 5 files)
    file_handler = RotatingFileHandler(
        'logs/academic_api.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Error file handler (only errors and critical)
    error_handler = RotatingFileHandler(
        'logs/errors.log',
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger