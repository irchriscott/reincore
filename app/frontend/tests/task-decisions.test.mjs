import assert from 'node:assert/strict'
import test from 'node:test'
import { decisionEntries, taskDecisionPreview, taskDecisionSummary } from '../src/utils/task-decisions.ts'

test('legacy events expose saved values and statuses without new metadata', () => {
  const event = { entries: [
    { kind: 'answer', status: 'active', value: 'Use PostgreSQL' },
    { kind: 'permission', status: 'revoked', value: 'Do not deploy' },
  ] }
  assert.equal(taskDecisionSummary(event), '1 answer, 1 permission')
  assert.match(taskDecisionPreview(event), /answer \(active\): Use PostgreSQL/)
  assert.match(taskDecisionPreview(event), /permission \(revoked\): Do not deploy/)
})
test('new summaries and malformed or empty historic entries render safely', () => {
  assert.equal(taskDecisionSummary({ summary: '2 answer', entries: [] }), '2 answer')
  assert.deepEqual(decisionEntries({ entries: [null, 'bad', [], { value: 'saved' }] }), [{ value: 'saved' }])
  assert.match(taskDecisionPreview({}), /No decision details/)
  assert.equal(taskDecisionPreview({ entries: [{ kind: 'missing_information', question: 'Which region?' }] }), 'missing information: Which region?')
})
