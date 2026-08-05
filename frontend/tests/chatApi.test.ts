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
          controller.enqueue(
            encoder.encode('{"content":"The current "}\n'),
          )
          controller.enqueue(
            encoder.encode('{"content":"champion is..."}\n'),
          )
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

test('streamChat reports the backend error from an unsuccessful response', async () => {
  const fetchRequest = async (): Promise<Response> =>
    Response.json(
      {
        error: {
          message: 'Please send your message again.',
          retryable: true,
        },
      },
      { status: 503 },
    )

  await assert.rejects(
    streamChat([], () => undefined, fetchRequest),
    /Please send your message again/,
  )
})

test('streamChat handles an error after partial streamed content', async () => {
  const encoder = new TextEncoder()
  const fetchRequest = async (): Promise<Response> =>
    new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('{"content":"Partial"}\n{"err'))
          controller.enqueue(
            encoder.encode(
              'or":{"message":"Please send the message again.","retryable":true}}\n',
            ),
          )
          controller.close()
        },
      }),
    )

  await assert.rejects(
    streamChat([], () => undefined, fetchRequest),
    /Please send the message again/,
  )
})

test('streamChat displays model response JSON from a debug error', async () => {
  const fetchRequest = async (): Promise<Response> =>
    Response.json(
      {
        error: {
          message: 'The model returned no text.',
          retryable: true,
          modelResponse: {
            content: '',
            response_metadata: { finish_reason: 'length' },
          },
        },
      },
      { status: 500 },
    )

  await assert.rejects(
    streamChat([], () => undefined, fetchRequest),
    (error: Error) => {
      assert.match(error.message, /The model returned no text/)
      assert.match(error.message, /"finish_reason": "length"/)
      return true
    },
  )
})
