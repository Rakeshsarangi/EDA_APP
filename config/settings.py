import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Ollama settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Security settings
    MAX_FILE_SIZE_MB = 100
    ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.json']
    
    # Analysis settings
    MAX_ROWS_FOR_ANALYSIS = 100000
    SAMPLE_SIZE_FOR_LLM = 100
    
    # Visualization settings
    DEFAULT_PLOT_HEIGHT = 400
    DEFAULT_PLOT_WIDTH = 600
    COLOR_PALETTE = "viridis"

settings = Settings()