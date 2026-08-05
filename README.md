# Octagon IQ

Octagon IQ is a UFC research chat application backed by OpenAI's `gpt-5-nano`
model. It combines a streaming chat interface with a LangChain agent
that can research fighter biographies, career statistics, recent MMA news, and
specific fight previews.

An internet connection and the relevant API keys are required for model access
and some research tools.

## How it works

```text
React chat UI
    |
    | POST /chat (streamed text response)
    v
FastAPI backend
    |
    v
LangChain agent  <-->  UFC research tools
    |
    v
OpenAI API  -->  gpt-5-nano
```

The Vite development server serves the browser application and proxies `/chat`
requests to FastAPI. FastAPI passes the conversation to the LangChain agent and
streams the generated response back to the browser. The agent uses the
OpenAI API for chat completion requests.

## Launch with Docker

### Prerequisites

- Docker
- An OpenAI API key
- A [Tavily](https://tavily.com/) API key for news and fight-description tools
- A CitoAPI key for the fighter statistics endpoint

### 1. Build the image

From the repository root:

```bash
docker build -t octagon-iq .
```

### 2. Configure the application

Create `backend/.env` from the included example and add your API keys:

```bash
cp backend/.env.example backend/.env
```

The `.env` file is ignored by Git and excluded from the Docker image.

### 3. Run the application

```bash
docker run --rm \
  --name octagon-iq \
  -p 5173:5173 \
  -p 8000:8000 \
  --env-file backend/.env \
  octagon-iq
```

Open [http://localhost:5173](http://localhost:5173) to use the chat interface.
The FastAPI service is available at `http://localhost:8000`.

### 4. Stop or inspect the container

The example runs in the foreground; press `Ctrl+C` to stop it. If it is running
detached, use:

```bash
docker logs -f octagon-iq
docker stop octagon-iq
```

## Configuration

Docker reads the application configuration from `backend/.env` through the
`docker run --env-file` option.

| Variable                | Default                 | Purpose                                              |
| ----------------------- | ----------------------- | ---------------------------------------------------- |
| `OPENAI_API_KEY`        | Required                | Authenticates requests to the OpenAI API             |
| `TAVILY_API_KEY`        | None                    | Enables MMA news search and fight-description lookup |
| `FIGHTER_STATS_API_KEY` | None                    | API key sent to the CitoAPI fighter stats service    |
| `LLM_MODEL`             | `gpt-5-nano`            | OpenAI model used by the agent                       |
| `LLM_MAX_TOKENS`        | `2048`                  | Maximum generated tokens                             |
| `SHOW_MODEL_RESPONSE_JSON` | `false`               | Shows model response JSON in chat errors for testing |
| `BACKEND_HOST`          | `0.0.0.0`               | FastAPI bind address                                 |
| `BACKEND_PORT`          | `8000`                  | FastAPI port                                         |
| `BACKEND_ENV_FILE`      | `/app/backend/.env`     | Alternate backend env-file path used by `launch.sh`  |
| `FRONTEND_HOST`         | `0.0.0.0`               | Vite bind address                                    |
| `FRONTEND_PORT`         | `5173`                  | Vite port                                            |
| `VITE_API_TARGET`       | `http://127.0.0.1:8000` | Backend target for the Vite `/chat` proxy            |

No temperature is configured, so the model uses the OpenAI API default.

For a non-Docker launch, `launch.sh` also reads backend configuration from
`backend/.env`, or from the path specified by `BACKEND_ENV_FILE`.

## Tools exposed to the model

| Tool                | Input                              | Data source                  | What it provides                                                       |
| ------------------- | ---------------------------------- | ---------------------------- | ---------------------------------------------------------------------- |
| `fighter_news`      | A focused search query             | Tavily Search API            | Recent MMA reporting, announcements, statements, and interviews        |
| `fighter_stats`     | Fighter name                       | CitoAPI UFC fighter endpoint | Structured career, measurement, and performance statistics             |
| `fighter_wikipedia` | Fighter name                       | Wikipedia API                | A fighter biography, background summary, and source URL                |
| `fight_description` | Both fighter names and `live blog` | Tavily and MMA Fighting      | The matchup description from a relevant MMA Fighting live-blog article |

The agent may make at most three `fighter_news` calls and one
`fight_description` call for one request. Tool errors are reported to the user
instead of being replaced with invented data.

## Technologies

### Frontend

- React 19
- TypeScript
- Vite
- CSS
- Oxlint

### Backend and agent

- Python
- FastAPI and Uvicorn
- LangChain
- `langchain-openai` and the OpenAI API
- Pydantic
- HTTPX and Requests
- Beautiful Soup

### Infrastructure

- OpenAI `gpt-5-nano`
- Docker
- Node.js 24
- Pytest and the Node.js test runner

## API

The backend exposes one application endpoint:

```http
POST /chat
Content-Type: application/json
```

Example request body:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Compare the records of Georges St-Pierre and Kamaru Usman."
    }
  ]
}
```

The response body is streamed as newline-delimited JSON. Successful records
contain `{"content": "..."}`. If generation fails, the backend logs the
exception and sends an `error` record with a user-safe message and a
`retryable` flag so the interface can ask the user to resend the message.

## Development and tests

`launch.sh` starts FastAPI and Vite together. For local use, configure
`backend/.env`, install the Python and frontend dependencies, and run:

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
npm ci --prefix frontend
./launch.sh
```

Run the automated checks with:

```bash
source backend/.venv/bin/activate
pytest backend
npm test --prefix frontend
npm run lint --prefix frontend
npm run build --prefix frontend
```

## Project structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── services/
│   └── tests/
├── frontend/
│   ├── src/
│   └── tests/
├── Dockerfile
└── launch.sh
```

## Troubleshooting

- **OpenAI requests fail**: confirm that `OPENAI_API_KEY` is present and valid.
- **A news or fight-description tool fails**: confirm that `TAVILY_API_KEY` is
  present and valid.
- **Fighter statistics fail**: confirm network access and, when required, a
  valid `FIGHTER_STATS_API_KEY`.
