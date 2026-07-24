import assert from 'node:assert/strict'
import test from 'node:test'
import { streamChat } from '../src/chatApi.ts'

test('streamChat sends the conversation and emits streamed content', async () => {
  const messages = [{ role: 'user' as const, content: 'Who is the champion?' }]
  const chunks: string[] = []
  let requestUrl = ''
  let requestInit: RequestInit | undefined

  const fetchRequest = async (
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    requestUrl = input.toString()
    requestInit = init
    const encoder = new TextEncoder()
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('The current '))
          controller.enqueue(encoder.encode('champion is...'))
          controller.close()
        },
      }),
    )
  }

  const content = await streamChat(
    messages,
    (streamedContent) => chunks.push(streamedContent),
    fetchRequest,
  )

  assert.equal(requestUrl, '/chat')
  assert.equal(requestInit?.method, 'POST')
  assert.equal(requestInit?.body, JSON.stringify({ messages }))
  assert.deepEqual(chunks, ['The current ', 'The current champion is...'])
  assert.equal(content, 'The current champion is...')
})

test('streamChat reports an unsuccessful response', async () => {
  const fetchRequest = async (): Promise<Response> =>
    new Response(null, { status: 503 })

  await assert.rejects(
    streamChat([], () => undefined, fetchRequest),
    /chat service returned 503/,
  )
})
