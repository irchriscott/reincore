const requestTags = new Set(['userrequest', 'userquery', 'usermessage', 'userprompt', 'userinput', 'usercontent', 'maincontent', 'mainmessage', 'request', 'prompt', 'task'])
const metadataTags = new Set([
  'currentdatetime', 'taggedfiles', 'toolschangednotice', 'environmentinfo', 'workspaceinfo',
  'editorcontext', 'context', 'attachments', 'attachment', 'instructions', 'reminderinstructions',
  'usermemory', 'sessionmemory', 'repomemory', 'systemreminder', 'toolresult', 'files', 'selection',
  'diff', 'reincorefilehistorydata', 'reincorecontinuationdata', 'conversationsummary', 'contextsummary',
])
// These are Markdown/HTML formatting, rather than client request envelopes.
const htmlTags = new Set('a abbr address article aside audio b blockquote br caption code col colgroup dd del details div dl dt em figcaption figure footer h1 h2 h3 h4 h5 h6 header hr i img input ins kbd li main mark nav ol p picture pre s samp section small source span strong sub summary sup table tbody td th thead time tr u ul video wbr'.split(' '))
const normalizedTag = (name: string) => name.toLowerCase().replace(/[-_:]/g, '')

type RequestTag = { name: string; start: number; bodyStart: number; bodyEnd: number; end: number }

/** Mask code with spaces so tag offsets still refer to the original Markdown. */
function maskCode(content: string): string {
  const ranges: [number, number][] = []
  const fences = /^ {0,3}(`{3,}|~{3,})[^\n]*(?:\n|$)/gm
  for (let opening = fences.exec(content); opening; opening = fences.exec(content)) {
    const fence = opening[1]
    const closing = new RegExp(`^ {0,3}${fence[0]}{${fence.length},}(?=[ \\t]*(?:\\r?$|\\\\?<\\/[a-z]))`, 'gmi')
    closing.lastIndex = fences.lastIndex
    const end = closing.exec(content)
    ranges.push([opening.index, end ? closing.lastIndex : content.length])
    fences.lastIndex = end ? closing.lastIndex : content.length
  }
  let masked = content
  for (const [start, end] of ranges.reverse()) {
    masked = masked.slice(0, start) + masked.slice(start, end).replace(/[^\r\n]/g, ' ') + masked.slice(end)
  }
  return masked.replace(/(`+)[\s\S]*?\1(?!`)|^(?: {4}|\t).*$/gm, (code) => code.replace(/[^\r\n]/g, ' '))
}

function requestTagBlocks(content: string): { blocks: RequestTag[]; masked: string } {
  const masked = maskCode(content)
  const tags = /\\?<\s*(\/?)\s*([a-z][\w:.-]*)(?:\s+(?:"[^"]*"|'[^']*'|[^'"<>])*?)?\s*(\/?)>/gi
  const stack: Omit<RequestTag, 'bodyEnd' | 'end'>[] = []
  const blocks: RequestTag[] = []
  for (const tag of masked.matchAll(tags)) {
    const name = normalizedTag(tag[2])
    if (htmlTags.has(name)) continue
    const start = tag.index
    const end = start + tag[0].length
    if (tag[1]) {
      const index = stack.findLastIndex((opening) => opening.name === name)
      if (index >= 0) {
        blocks.push({ ...stack[index], bodyEnd: start, end })
        stack.splice(index)
      } else if (metadataTags.has(name)) {
        blocks.push({ name, start, bodyStart: end, bodyEnd: end, end })
      }
    } else if (tag[3]) {
      blocks.push({ name, start, bodyStart: end, bodyEnd: end, end })
    } else {
      stack.push({ name, start, bodyStart: end })
    }
  }
  // Truncated client envelopes can occur in stored audit previews.
  for (const opening of stack) {
    if (metadataTags.has(opening.name) || requestTags.has(opening.name)) {
      blocks.push({ ...opening, bodyEnd: content.length, end: content.length })
    }
  }
  return { blocks, masked }
}

function withoutMetadata(content: string): string {
  const { blocks, masked } = requestTagBlocks(content)
  const ranges = blocks.filter((block) => metadataTags.has(block.name) || (
    // Unknown, standalone envelopes are client scaffolding. Inline XML prose stays literal.
    /(?:^|\n|>)\s*$/.test(masked.slice(0, block.start))
    && /^\s*(?:\n|<|$)/.test(masked.slice(block.end))
  )).sort((a, b) => a.start - b.start || b.end - a.end)
  let message = ''
  let cursor = 0
  for (const range of ranges) {
    if (range.start > cursor) message += content.slice(cursor, range.start)
    cursor = Math.max(cursor, range.end)
  }
  return (message + content.slice(cursor)).trim()
}

/** Prefer the main user message; keep client context out of display and the audit payload intact. */
export function parseRequestContent(content?: string): { message: string; taggedFiles: string[] } {
  const raw = content ?? ''
  const { blocks } = requestTagBlocks(raw)
  const taggedFiles = blocks.filter((block) => block.name === 'taggedfiles')
    .sort((a, b) => a.start - b.start)
    .flatMap((block) => raw.slice(block.bodyStart, block.bodyEnd).split(/\r?\n/)
      .map((file) => file.trim().replace(/^(?:[-*+]\s+|\d+[.)]\s+)/, ''))
      .filter(Boolean))
  const main = blocks.filter((block) => requestTags.has(block.name)
    && raw.slice(block.bodyStart, block.bodyEnd).trim()
    && !blocks.some((parent) => parent.start < block.start && parent.end >= block.end
      && metadataTags.has(parent.name)))
    .sort((a, b) => b.start - a.start)[0]
  return { message: withoutMetadata(main ? raw.slice(main.bodyStart, main.bodyEnd) : raw), taggedFiles }
}

export function requestMessage(content?: string): string {
  return parseRequestContent(content).message
}
