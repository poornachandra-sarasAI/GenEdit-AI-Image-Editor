# GenEdit — AI Image Editor (MVP)

An intelligent image editing platform that combines natural language processing with AI-powered image manipulation. GenEdit allows users to upload, organize, search, and edit images using simple natural language commands.

## 🚀 Features

### Core Functionality
- **AI-Powered Image Editing**: Edit images using natural language instructions powered by Google's Gemini models
- **Automatic Caption Generation**: AI-generated captions for better image organization and searchability  
- **Semantic Search**: Find images using natural language queries with OpenAI embeddings
- **Version Control**: Track all edits with complete version history and restore capability
- **Image Library Management**: Organize and browse your image collection with an intuitive interface

### Key Capabilities
- **Natural Language Editing**: Commands like "Remove background", "Add tree in background", "Make more vibrant"
- **Smart Search**: Search using descriptions like "beach photos", "product images", "person with dog"
- **Preset Actions**: Quick-access buttons for common editing tasks
- **Version History**: View and restore any previous version of an edited image
- **Metadata Management**: Automatic tagging and embedding generation for each image

## 🛠️ Technology Stack

- **Frontend**: Streamlit for web interface
- **Image Processing**: PIL (Python Imaging Library)
- **AI Models**: 
  - Google Gemini 2.5 Flash for image editing and caption generation
  - OpenAI text-embedding-3-small for semantic search
- **Storage**: JSON-based metadata with file system storage
- **Search**: Vector similarity search using numpy

## 📋 Prerequisites

- Python 3.8 or higher
- Google AI Studio API key (for Gemini models)
- OpenAI API key (for embeddings)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Final Project"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEN_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🖥️ Usage

### Adding Images
1. Navigate to the "Add Image" page
2. Upload an image file (PNG, JPG, JPEG)
3. Optionally add a title
4. The system automatically generates captions and creates searchable embeddings

### Searching Images
1. Go to the "Library" page
2. Use natural language in the search box (e.g., "sunset photos", "people smiling")
3. Browse results or view all images in your library

### Editing Images
1. Open any image from the library
2. Use preset buttons for common edits:
   - Remove background
   - Add tree in background  
   - Make vibrant
3. Or write custom editing instructions in natural language
4. View version history and restore previous versions

### Managing Versions
- Each edit creates a new version while preserving the original
- Access complete version history from the image detail page
- Restore any previous version with one click
- Each version includes the edit prompt for reference

## 📁 Project Structure

```
Final Project/
├── streamlit_app.py      # Main Streamlit application
├── storage.py            # Data storage and retrieval functions
├── ai_utils.py          # AI model integrations (Gemini, OpenAI)
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .env                # Environment variables (create this)
└── data/               # Storage directory
    ├── metadata.json   # Image metadata and embeddings
    └── [image-folders] # Individual image version folders
```

## 🔑 API Keys Setup

### Google AI Studio (Gemini)
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create an API key
3. Add it to your `.env` file as `GEMINI_API_KEY` and `GEN_API_KEY`

### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an API key
3. Add it to your `.env` file as `OPENAI_API_KEY`

## 🎯 Use Cases

- **Content Creation**: Quick image edits for social media, blogs, marketing
- **Photo Organization**: AI-powered tagging and search for large photo collections  
- **Design Iteration**: Version control for design experiments
- **Batch Processing**: Natural language commands for consistent edits across images
- **Personal Photo Management**: Smart organization and search of personal photos

## 🚧 Current Limitations

- Single-user application (no multi-user support)
- Local storage only (no cloud integration)
- Limited to supported image formats (PNG, JPG, JPEG)
- Requires active internet connection for AI features
- API rate limits apply based on your service plans

## 🔮 Future Enhancements

- [ ] Multi-user support with authentication
- [ ] Cloud storage integration (AWS S3, Google Cloud)
- [ ] Batch editing capabilities
- [ ] Advanced search filters and sorting
- [ ] Export/import functionality
- [ ] Mobile-responsive design
- [ ] Integration with additional AI models
- [ ] Real-time collaboration features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the console output for error messages
2. Ensure all API keys are correctly set in the `.env` file
3. Verify that all dependencies are installed
4. Check that the `data/` directory has proper read/write permissions

## 🙏 Acknowledgments

- Google Gemini for powerful image editing capabilities
- OpenAI for semantic search embeddings
- Streamlit for the intuitive web interface
- The open-source community for the underlying libraries

---

**GenEdit MVP** - Making AI-powered image editing accessible through natural language interactions.