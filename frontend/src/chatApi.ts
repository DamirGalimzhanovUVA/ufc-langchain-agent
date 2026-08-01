export type ChatRequestMessage = {
  role: 'assistant' | 'user'
  content: string
}

type ChatError = {
  message: string
  retryable: boolean
}

type ChatEvent = {
  content?: string
  error?: ChatError
}

const fallbackErrorMessage =
  "We couldn't complete your request. Please send your message again."

async function getErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as ChatEvent
    return body.error?.message || fallbackErrorMessage
  } catch {
    return fallbackErrorMessage
  }
}

export async function streamChat(
  messages: ChatRequestMessage[],
  onChunk: (content: string) => void,
  fetchRequest: typeof fetch = fetch,
): Promise<string> {
  const response = await fetchRequest('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  })

  if (!response.ok) {
    throw new Error(await getErrorMessage(response))
  }
  if (!response.body) {
    throw new Error('The chat service did not return a response.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let content = ''
  let bufferedText = ''

  function processEvent(line: string) {
    if (!line.trim()) return

    let event: ChatEvent
    try {
      event = JSON.parse(line) as ChatEvent
    } catch {
      throw new Error(fallbackErrorMessage)
    }

    if (event.error) {
      throw new Error(event.error.message || fallbackErrorMessage)
    }
    if (typeof event.content === 'string') {
      content += event.content
      onChunk(content)
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    bufferedText += decoder.decode(value, { stream: !done })

    const lines = bufferedText.split('\n')
    bufferedText = lines.pop() || ''
    for (const line of lines) processEvent(line)

    if (done) {
      processEvent(bufferedText)
      break
    }
  }

  return content
}
