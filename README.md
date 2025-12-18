[![Version](https://img.shields.io/github/v/release/marc-shade/VIdeo-Transcription?style=flat-square)](https://github.com/marc-shade/VIdeo-Transcription/releases)
[![Stars](https://img.shields.io/github/stars/marc-shade/VIdeo-Transcription?style=flat-square)](https://github.com/marc-shade/VIdeo-Transcription/stargazers)
[![Forks](https://img.shields.io/github/forks/marc-shade/VIdeo-Transcription?style=flat-square)](https://github.com/marc-shade/VIdeo-Transcription/network/members)
[![License](https://img.shields.io/github/license/marc-shade/VIdeo-Transcription?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.ai)

<img src="https://github.com/marc-shade/VIdeo-Transcription/blob/main/assets/dragon.png" align="right" style="width: 300px;" />
# 🎥 AI Video Transcription with Persona Generation

A powerful video transcription tool that not only transcribes videos but also generates AI personas that can engage in conversations about the content. Built with Streamlit and powered by Ollama for local AI processing.

## ✨ Features

- 🎬 Video to text transcription using Whisper
- 🌐 Multi-language translation support
- 🤖 AI persona generation from transcripts
- 💬 Interactive chat with generated personas
- 🔄 Dynamic model selection from local Ollama installation
- 📊 Client and transcription management
- 🔒 Local AI processing with Ollama

## 🚀 Recent Updates

- Added dynamic Ollama model selection
- Improved persona generation and chat interface
- Added ability to regenerate personas
- Enhanced error handling and feedback
- Improved database management with migrations

## 🛠️ Prerequisites

1. Python 3.11 or higher
2. FFmpeg installed on your system
3. [Ollama](https://ollama.ai/) installed and running
4. At least one Ollama model pulled (e.g., `ollama pull mistral:instruct`)

## 📦 Setup and Installation

### Prerequisites
- Python 3.10+
- pip or conda
- FFmpeg

### Virtual Environment Setup

#### Using venv (Python's built-in virtual environment)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Using Conda
```bash
# Create a new conda environment
conda create -n video-transcription python=3.11

# Activate the environment
conda activate video-transcription

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional conda-specific packages if needed
conda install ffmpeg
```

### System Dependencies
- **FFmpeg**: Required for audio/video processing
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Windows: Download from [FFmpeg official site](https://ffmpeg.org/download.html)

## 🚀 Usage

1. Start the Ollama server:
```bash
ollama serve
```

2. Run the application:
```bash
streamlit run main.py
```

3. Access the web interface at `http://localhost:8501`

## 💡 Features in Detail

### Video Transcription
- Upload video files
- Automatic transcription using Whisper
- Optional timestamp inclusion
- Support for multiple video formats

### Translation
- Translate transcriptions to multiple languages
- Powered by deep-translator
- Maintains formatting and structure

### AI Persona Generation
- Analyzes speaking patterns and content
- Creates context-aware personas
- Generates detailed system prompts
- Supports multiple Ollama models
- Regenerate personas as needed

### Interactive Chat
- Chat with generated personas
- Context-aware responses
- Maintains chat history
- Real-time response generation

## 🔧 Configuration

The application uses several environment variables that can be set in a `.env` file:

```env
OLLAMA_API_BASE=http://localhost:11434
DEFAULT_MODEL=mistral:instruct
```

## 📝 Database Schema

The application uses SQLite with the following main tables:
- clients: Store client information
- transcriptions: Store video transcriptions
- persona_prompts: Store generated AI personas

## 📁 Project Structure

```
video_transcription/
├── main.py             # Primary Streamlit application
├── database.py         # Database management
├── utils.py            # Audio/video processing utilities
├── ai_persona.py       # AI persona generation
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

### Key Components
- **main.py**: Central Streamlit interface for video transcription
- **database.py**: SQLite database operations for clients and transcripts
- **utils.py**: Core utility functions for audio extraction and transcription
- **ai_persona.py**: AI-powered persona analysis and generation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for transcription
- Ollama for local AI processing
- Streamlit for the web interface
- All other open-source contributors

## Troubleshooting

### PyArrow Installation Issues

If you encounter problems installing PyArrow (a Streamlit dependency), try the following:

1. Use pre-built wheels:
```bash
pip install --only-binary=:all: pyarrow
```

2. If you're on an older system or experiencing build errors, you can:
   - Upgrade pip and setuptools
   - Install build dependencies
   - Try specifying a specific version

Example:
```bash
pip install --upgrade pip setuptools wheel
pip install "pyarrow[build]"
# Or specify an exact version
pip install pyarrow==14.0.2
```

Note: PyArrow can be sensitive to system configurations and Python versions. The pre-built wheel method is often the most reliable.
