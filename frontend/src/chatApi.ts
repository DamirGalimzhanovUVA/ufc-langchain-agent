export type ChatRequestMessage = {
  role: 'assistant' | 'user'
  content: string
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
    throw new Error(`The chat service returned ${response.status}.`)
  }
  if (!response.body) {
    throw new Error('The chat service did not return a response.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let content = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      content += decoder.decode()
      break
    }
    content += decoder.decode(value, { stream: true })
    onChunk(content)
  }

  return content
}
