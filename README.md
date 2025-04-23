# AI Assistant Platform

An intelligent customer service platform that integrates AI assistants with Telegram for seamless communication.

![AI Assistant Platform](https://via.placeholder.com/800x400?text=AI+Assistant+Platform)

## Features

- 🤖 AI-powered chat assistants
- 📱 Telegram integration
- 💬 Multi-channel support
- 🧠 Custom knowledge base configuration
- 🔒 User authentication
- 📊 Chat history and analytics

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python with FastAPI
- **AI**: Google Gemini AI (via generativeai library)
- **Messaging**: Telegram Bot API
- **Database**: SQLite (local development)
- **Deployment**: Vercel

## Getting Started

### Prerequisites

- Python 3.8+
- A Google AI (Gemini) API key
- A Telegram Bot token (obtain from [BotFather](https://t.me/botfather))

### Local Development

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ai-assistant-platform.git
   cd ai-assistant-platform
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following content:
   ```
   GOOGLE_API_KEY=your_google_api_key
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=password
   ```

5. Run the application
   ```bash
   uvicorn backend:app --reload
   ```

6. Visit `http://localhost:8000` in your browser

### Deployment to Vercel

1. Fork or push this repository to your GitHub account

2. Sign up for a [Vercel account](https://vercel.com/signup) if you don't have one

3. Import your repository in Vercel:
   - Click "Add New" > "Project"
   - Select your repository
   - Configure the project:
     - Framework Preset: Other
     - Build Command: None
     - Output Directory: None
     - Install Command: `pip install -r requirements.txt`

4. Add environment variables in Vercel project settings:
   - `GOOGLE_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `ADMIN_USERNAME`
   - `ADMIN_PASSWORD`

5. Deploy the project

6. Set up Telegram webhook:
   - Replace the polling code with webhook setup
   - Use your Vercel deployment URL: `https://your-project.vercel.app/api/telegram-webhook`

## Using the Platform

### Authentication

The default credentials are:
- Username: `admin`
- Password: `password`

You can change these by setting the `ADMIN_USERNAME` and `ADMIN_PASSWORD` environment variables.

### Creating AI Agents

1. Log in to the platform
2. Navigate to the Agents tab
3. Click "Add Agent"
4. Configure your agent:
   - Name
   - Type
   - System prompt
   - Knowledge base (optional)

### Configuring Telegram

1. Create a bot using [BotFather](https://t.me/botfather)
2. Get the token and set it as `TELEGRAM_BOT_TOKEN`
3. After deployment, your bot will automatically connect

### Adding Knowledge Bases

1. Navigate to the Knowledge tab
2. Click "Add Knowledge Base"
3. Enter a name and content
4. Associate the knowledge base with an agent

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Google Generative AI](https://ai.google.dev/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Vercel](https://vercel.com/)
