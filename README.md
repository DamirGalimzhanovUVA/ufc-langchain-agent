# Octagon IQ

Octagon IQ is a UFC research chat application backed by a locally hosted
language model. It combines a streaming chat interface with a LangChain agent
that can research fighter biographies, career statistics, recent MMA news, and
specific fight previews.

The application runs the model locally from a GGUF file with `llama-server`.
The model can still call external data sources, so an internet connection and
the relevant API keys are required for some research tools.

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
llama-server  -->  local GGUF model
```

The Vite development server serves the browser application and proxies `/chat`
requests to FastAPI. FastAPI passes the conversation to the LangChain agent and
streams the generated response back to the browser. The agent uses the
OpenAI-compatible API exposed by `llama-server`.

## Launch with Docker

### Prerequisites

- Docker
- A local, tool-capable chat model in GGUF format
- A [Tavily](https://tavily.com/) API key for news and fight-description tools
- A CitoAPI key for the fighter statistics endpoint

### 1. Build the image

From the repository root:

```bash
docker build -t octagon-iq .
```

### 2. Run the application

Replace `/absolute/path/to/model.gguf` with the location of your model on the
host:

```bash
docker run --rm \
  --name octagon-iq \
  -p 5173:5173 \
  -p 8000:8000 \
  -v "/absolute/path/to/model.gguf:/models/model.gguf:ro" \
  -e MODEL_PATH=/models/model.gguf \
  -e TAVILY_API_KEY=your-tavily-api-key \
  -e FIGHTER_STATS_API_KEY=your-citoapi-key \
  octagon-iq
```

Open [http://localhost:5173](http://localhost:5173) to use the chat interface.
The FastAPI service is available at `http://localhost:8000`.

`MODEL_PATH` must be a path inside the container. The volume mount in the
example makes the host model available at `/models/model.gguf`.

If GPU offloading is unavailable, add `-e LLAMA_GPU_LAYERS=0` to run the model
on the CPU. CPU inference can be substantially slower.

### 3. Stop or inspect the container

The example runs in the foreground; press `Ctrl+C` to stop it. If it is running
detached, use:

```bash
docker logs -f octagon-iq
docker stop octagon-iq
```

## Configuration

Docker environment variables can be passed individually with `-e` or collected
in an env file and supplied with `docker run --env-file`.

| Variable                | Default                    | Purpose                                              |
| ----------------------- | -------------------------- | ---------------------------------------------------- |
| `MODEL_PATH`            | Required                   | Path to the GGUF model as seen by `llama-server`     |
| `TAVILY_API_KEY`        | None                       | Enables MMA news search and fight-description lookup |
| `FIGHTER_STATS_API_KEY` | None                       | API key sent to the CitoAPI fighter stats service    |
| `LLAMA_SERVER_HOST`     | `127.0.0.1`                | Host used by the local model server                  |
| `LLAMA_SERVER_PORT`     | `8080`                     | Port used by the local model server                  |
| `LLAMA_CONTEXT_SIZE`    | `8192`                     | Model context-window size                            |
| `LLAMA_GPU_LAYERS`      | `99`                       | Number of model layers offloaded to the GPU          |
| `LLM_BASE_URL`          | `http://127.0.0.1:8080/v1` | OpenAI-compatible model API URL                      |
| `LLM_MODEL`             | `local-model`              | Model name sent in chat-completion requests          |
| `LLM_API_KEY`           | `local-key`                | API key sent to the model API                        |
| `LLM_TEMPERATURE`       | `0.7`                      | Sampling temperature                                 |
| `LLM_MAX_TOKENS`        | `2048`                     | Maximum generated tokens                             |
| `BACKEND_HOST`          | `0.0.0.0`                  | FastAPI bind address                                 |
| `BACKEND_PORT`          | `8000`                     | FastAPI port                                         |
| `BACKEND_ENV_FILE`      | `/app/backend/.env`        | Alternate backend env-file path used by `launch.sh`  |
| `FRONTEND_HOST`         | `0.0.0.0`                  | Vite bind address                                    |
| `FRONTEND_PORT`         | `5173`                     | Vite port                                            |
| `VITE_API_TARGET`       | `http://127.0.0.1:8000`    | Backend target for the Vite `/chat` proxy            |

For a non-Docker launch, `launch.sh` also reads backend configuration from
`backend/.env`, or from the path specified by `BACKEND_ENV_FILE`.

## Tools exposed to the model

| Tool                | Input                              | Data source                  | What it provides                                                       |
| ------------------- | ---------------------------------- | ---------------------------- | ---------------------------------------------------------------------- |
| `fighter_news`      | A focused search query             | Tavily Search API            | Recent MMA reporting, announcements, statements, and interviews        |
| `fighter_stats`     | Fighter name                       | CitoAPI UFC fighter endpoint | Structured career, measurement, and performance statistics             |
| `fighter_wikipedia` | Fighter name                       | Wikipedia API                | A fighter biography, background summary, and source URL                |
| `fight_description` | Both fighter names and `live blog` | Tavily and MMA Fighting      | The matchup description from a relevant MMA Fighting live-blog article |

The agent may make at most three `fighter_news` calls for one request. Tool
errors are reported to the user instead of being replaced with invented data.

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
- `langchain-openai` and the OpenAI-compatible chat-completions protocol
- Pydantic
- HTTPX and Requests
- Beautiful Soup

### Model and infrastructure

- llama.cpp `llama-server`
- GGUF models
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

The response body is streamed as plain text.

## Development and tests

`launch.sh` starts `llama-server`, FastAPI, and Vite together. For local use,
install the Python and frontend dependencies, make `llama-server` available on
`PATH`, set `MODEL_PATH`, and run:

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
npm ci --prefix frontend
MODEL_PATH=/absolute/path/to/model.gguf ./launch.sh
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
│   ├── tests/
│   └── start-llama-server.sh
├── frontend/
│   ├── src/
│   └── tests/
├── Dockerfile
└── launch.sh
```

## Troubleshooting

- **`MODEL_PATH must be set`**: pass `MODEL_PATH` and mount the corresponding
  GGUF file into the container.
- **The local model server is unavailable**: confirm that the GGUF file is
  readable, the model process started successfully, and `LLM_BASE_URL` matches
  the llama-server address.
- **A news or fight-description tool fails**: confirm that `TAVILY_API_KEY` is
  present and valid.
- **Fighter statistics fail**: confirm network access and, when required, a
  valid `FIGHTER_STATS_API_KEY`.
- **The container runs out of memory**: use a smaller or more heavily quantized
  GGUF model, reduce `LLAMA_CONTEXT_SIZE`, or reduce GPU offloading.
