# 🤖 LangGraph Chatbot

A modern chatbot built with LangGraph, Ollama, and Streamlit, featuring a modular tools registry for easy extensibility.

## Features

- **LangGraph-powered**: Structured conversation flow with state management
- **Ollama Integration**: Run LLMs locally with no API keys needed
- **Tools Registry**: Easy-to-use system for adding custom tools
- **Modern UI**: Beautiful Streamlit interface with dark theme
- **Configurable**: Adjust model and temperature settings on the fly

## Quick Start

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai), then pull a model:

```bash
ollama pull llama3.2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Create `.env` file

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### 4. Run the App

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, start the app
streamlit run app.py
```

## Adding Custom Tools

Open `chatbot/tools.py` and add your tools using the registry:

```python
from chatbot.tools import tools_registry

@tools_registry.register
def search_web(query: str) -> str:
    """Search the web for information."""
    # Your implementation here
    return f"Results for: {query}"

@tools_registry.register
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

Or register existing LangChain tools:

```python
from langchain_core.tools import tool

@tool
def my_tool(arg: str) -> str:
    """Tool description."""
    return "result"

tools_registry.register_tool(my_tool)
```

## Project Structure

```
chatbot/
├── app.py              # Streamlit frontend
├── requirements.txt    # Dependencies
├── .env.example        # Environment template
└── chatbot/
    ├── __init__.py     # Package exports
    ├── graph.py        # LangGraph chatbot definition
    └── tools.py        # Tools registry
```

## Configuration

In the Streamlit sidebar, you can:

- Set the Ollama server URL (default: `http://localhost:11434`)
- Select the model (llama3.2, mistral, etc.)
- Enter a custom model name
- Adjust temperature (0.0 - 2.0)
- View registered tools
- Clear chat history

## License

MIT
