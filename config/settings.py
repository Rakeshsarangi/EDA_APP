import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Ollama settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Security settings - Convert to int
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", ".csv,.xlsx,.xls,.json").split(",")
    
    # Analysis settings - Convert to int
    MAX_ROWS_FOR_ANALYSIS = int(os.getenv("MAX_ROWS_FOR_ANALYSIS", "1000000"))
    SAMPLE_SIZE_FOR_LLM = int(os.getenv("SAMPLE_SIZE_FOR_LLM", "100"))
    
    # Visualization settings - Convert to int
    DEFAULT_PLOT_HEIGHT = int(os.getenv("DEFAULT_PLOT_HEIGHT", "400"))
    DEFAULT_PLOT_WIDTH = int(os.getenv("DEFAULT_PLOT_WIDTH", "600"))
    COLOR_PALETTE = os.getenv("COLOR_PALETTE", "viridis").split(",")

settings = Settings()