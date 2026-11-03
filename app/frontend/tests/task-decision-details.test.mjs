import assert from 'node:assert/strict'
import test from 'node:test'
import { createServer } from 'vite'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

test('decision detail cards show evidence, status and references and escape untrusted text', async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' })
  try {
    const { TaskDecisionDetails } = await server.ssrLoadModule('/src/components/TaskDecisionDetails.tsx')
    const html = renderToStaticMarkup(createElement(TaskDecisionDetails, { payload: {
      task_id: 'task-one', operation: 'capture', entries: [{
        id: 'decision-1', kind: 'permission', status: 'revoked', value: '<script>unsafe()</script>',
        scope: 'project-one::production', question: 'May I deploy?', expires_at: '2026-09-09T10:00:00Z',
        source: { role: 'user', quote: 'Do not deploy', message_hash: 'hash-1', truncated: true },
        supersedes: ['older-1'], group_only: true,
      }],
    } }))
    for (const text of ['revoked', 'project-one::production', 'May I deploy?', 'Do not deploy', 'hash-1', 'older-1', 'Evidence truncated', 'Request group only']) assert.ok(html.includes(text), text)
    assert.ok(html.includes('&lt;script&gt;'))
    assert.ok(!html.includes('<script>'))
    const empty = renderToStaticMarkup(createElement(TaskDecisionDetails, { payload: {} }))
    assert.ok(empty.includes('No decision entries are available'))
  } finally { await server.close() }
})
