import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/** Compact previews contain no interactive elements inside navigation buttons. */
export function MarkdownContent({ content, className = '', preview = false }: { content: string; className?: string; preview?: boolean }) {
  return (
    <div className={`rc-md ${preview ? 'rc-md-preview' : ''} ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={preview ? {
        a: ({ children }) => <span>{children}</span>,
        img: ({ alt }) => <span>{alt}</span>,
        input: ({ checked }) => <span aria-hidden="true">{checked ? '☑ ' : '☐ '}</span>,
      } : undefined}>{content}</ReactMarkdown>
    </div>
  )
}
