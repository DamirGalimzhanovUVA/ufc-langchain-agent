import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { streamChat } from './chatApi'
import './App.css'

type Message = {
  role: 'assistant' | 'user'
  content: string
}

const starterPrompts = [
  'Compare two fighters',
  'Explain the scoring system',
  'Who has the most title defenses?',
]

const initialMessages: Message[] = [
  {
    role: 'assistant',
    content:
      "I'm your UFC fight companion. Ask me about fighters, matchups, records, rules, or the history of the Octagon.",
  },
]

function FighterMark() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true">
      <path d="M9 12h30l4 7-9 20H14L5 19l4-7Z" fill="currentColor" opacity=".16" />
      <path d="m13 17 5 3-2 5 5 2-3 7M35 17l-5 3 2 5-5 2 3 7M18 12l6 6 6-6" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m5 12 14-7-4 14-3-6-7-1Z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function App() {
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage(messageText: string) {
    const content = messageText.trim()
    if (!content || isLoading) return

    const userMessage: Message = { role: 'user', content }
    const requestMessages = [...messages, userMessage]

    setInput('')
    setError('')
    setIsLoading(true)
    setMessages([...requestMessages, { role: 'assistant', content: '' }])

    try {
      const assistantContent = await streamChat(requestMessages, (streamedContent) => {
        setMessages([...requestMessages, { role: 'assistant', content: streamedContent }])
      })

      setMessages([
        ...requestMessages,
        {
          role: 'assistant',
          content: assistantContent || 'I could not generate a response. Please try again.',
        },
      ])
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "We couldn't complete your request. Please send your message again."
      setMessages(messages)
      setInput(content)
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void sendMessage(input)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  function startNewChat() {
    if (isLoading) return
    setMessages(initialMessages)
    setInput('')
    setError('')
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand" aria-label="UFC AI">
          <span className="brand-word">UFC</span>
          <span className="brand-divider" />
          <span className="brand-ai">AI</span>
        </div>

        <div className="sidebar-copy">
          <span className="eyebrow">Fight intelligence</span>
          <h1>Know the fight.<br />Own the conversation.</h1>
          <p>Stats, stories, matchups, and more — powered by AI.</p>
        </div>

        <div className="octagon-art" aria-hidden="true">
          <span className="octagon-ring ring-one" />
          <span className="octagon-ring ring-two" />
          <span className="octagon-ring ring-three" />
        </div>

        <div className="sidebar-footer">
          <span className="live-dot" />
          AI CORNER ONLINE
        </div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div className="assistant-title">
            <div className="fighter-mark"><FighterMark /></div>
            <div>
              <h2>Octagon IQ</h2>
              <p>Your AI fight companion</p>
            </div>
          </div>
          <button className="new-chat-button" type="button" onClick={startNewChat}>
            <span>+</span> New chat
          </button>
        </header>

        <div className="conversation" aria-live="polite" aria-busy={isLoading}>
          <div className="date-divider"><span>Today</span></div>

          {messages.map((message, index) => (
            <div className={`message-row ${message.role}`} key={`${message.role}-${index}`}>
              {message.role === 'assistant' && (
                <div className="avatar"><FighterMark /></div>
              )}
              <div className="message-content">
                <span className="message-label">
                  {message.role === 'assistant' ? 'Octagon IQ' : 'You'}
                </span>
                <div className={`message-bubble ${!message.content ? 'typing-bubble' : ''}`}>
                  {message.content || (
                    <span className="typing" aria-label="Octagon IQ is typing">
                      <i /><i /><i />
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}

          {messages.length === 1 && (
            <div className="suggestions" aria-label="Suggested questions">
              <p>Try asking</p>
              <div>
                {starterPrompts.map((prompt) => (
                  <button type="button" onClick={() => void sendMessage(prompt)} key={prompt}>
                    <span>↗</span>{prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {error && <div className="error-message" role="alert">{error}</div>}
          <div ref={endRef} />
        </div>

        <div className="composer-wrap">
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              aria-label="Message Octagon IQ"
              placeholder="Ask about a fighter, matchup, or UFC history..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={isLoading}
            />
            <button
              className="send-button"
              type="submit"
              aria-label="Send message"
              disabled={!input.trim() || isLoading}
            >
              <SendIcon />
            </button>
          </form>
          <p className="composer-note">AI can make mistakes. Check important fight stats.</p>
        </div>
      </section>
    </main>
  )
}

export default App
