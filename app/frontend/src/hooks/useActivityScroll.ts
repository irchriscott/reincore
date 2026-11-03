import { useCallback, useRef } from 'react'

export function activityEventScrollTop(pageTop: number, activityTop: number, eventMax: number) {
  return Math.min(Math.max(0, pageTop - activityTop), Math.max(0, eventMax))
}

function pageScroller(workspace: HTMLElement) {
  for (let parent = workspace.parentElement; parent; parent = parent.parentElement) {
    if (parent === document.body || parent === document.documentElement) break
    if (/(auto|scroll)/.test(getComputedStyle(parent).overflowY)) return parent
  }
  return document.scrollingElement as HTMLElement | null
}

function hasNativeScroller(target: Element, workspace: HTMLElement, events: HTMLElement | null) {
  for (let element: Element | null = target; element && element !== workspace; element = element.parentElement) {
    if (element === events) return false
    if (element instanceof HTMLElement && element.scrollHeight > element.clientHeight + 1
      && /(auto|scroll)/.test(getComputedStyle(element).overflowY)) return true
  }
  return false
}

function bindActivityScroll(workspace: HTMLElement) {
  const region = workspace.closest<HTMLElement>('[data-activity-scroll-region]')
  const range = region?.querySelector<HTMLElement>('[data-activity-scroll-range]')
  const page = pageScroller(workspace)
  if (!region || !range || !page) return () => {}

  const host = workspace.closest('main') ?? workspace
  const scrollTarget = page === document.scrollingElement ? window : page
  let events: HTMLElement | null = null
  let requestRail: HTMLElement | null = null
  let observedContent: Element[] = []
  let eventMax = 0
  let hadEvents = false
  let syncedEventTop = 0

  const activityTop = () => {
    const viewportTop = page === document.scrollingElement ? 0 : page.getBoundingClientRect().top + page.clientTop
    const stickyTop = Number.parseFloat(getComputedStyle(workspace).top) || 0
    return Math.max(0, region.getBoundingClientRect().top - viewportTop + page.scrollTop - stickyTop)
  }

  const syncEvents = () => {
    if (!events) return
    const next = activityEventScrollTop(page.scrollTop, activityTop(), eventMax)
    if (Math.abs(events.scrollTop - next) > 0.5) events.scrollTop = next
    syncedEventTop = events.scrollTop
  }

  const onEventScroll = () => {
    // Focus navigation and deep links can reveal an event programmatically.
    // Keep the main scrollbar at the corresponding position as well.
    if (!events || Math.abs(events.scrollTop - syncedEventTop) <= 1) return
    const top = activityTop()
    const expected = activityEventScrollTop(page.scrollTop, top, eventMax)
    if (Math.abs(events.scrollTop - expected) > 1) {
      page.scrollTo({ top: top + events.scrollTop, behavior: 'instant' })
    }
  }

  const resize = new ResizeObserver(() => updateLayout())
  const updateLayout = () => {
    const nextRail = workspace.querySelector<HTMLElement>('[data-request-rail]')
    if (nextRail !== requestRail) {
      if (requestRail) {
        resize.unobserve(requestRail)
        requestRail.removeAttribute('data-request-overflow')
      }
      requestRail = nextRail
      if (requestRail) resize.observe(requestRail)
    }
    // Only an overflowing navigator should contain wheel/touch input.
    // A list that fits must still let the page drive the event timeline.
    if (requestRail) {
      requestRail.toggleAttribute('data-request-overflow', requestRail.scrollHeight > requestRail.clientHeight + 1)
    }

    const top = activityTop()
    const wasPastActivity = page.scrollTop > top
    const nextEvents = workspace.querySelector<HTMLElement>('[data-event-scroll]')
    const changed = nextEvents !== events
    if (changed) {
      events?.removeEventListener('scroll', onEventScroll)
      if (events) resize.unobserve(events)
      events = nextEvents
      syncedEventTop = events?.scrollTop ?? 0
      if (events) {
        resize.observe(events)
        events.addEventListener('scroll', onEventScroll, { passive: true })
      }
    }

    const nextContent = events ? Array.from(events.children) : []
    if (nextContent.length !== observedContent.length || nextContent.some((element, index) => element !== observedContent[index])) {
      observedContent.forEach((element) => resize.unobserve(element))
      observedContent = nextContent
      observedContent.forEach((element) => resize.observe(element))
    }
    eventMax = events ? Math.max(0, events.scrollHeight - events.clientHeight) : 0
    // Reset before shortening the range, so native scroll clamping cannot
    // restore the previous request's position when its replacement loads.
    if (changed && hadEvents && wasPastActivity) {
      page.scrollTo({ top, behavior: 'instant' })
    }
    // Give the native page scrollbar the event list's full remaining distance.
    // The sticky workspace stays visible while this range passes beneath it.
    const height = `${eventMax}px`
    if (range.style.height !== height) range.style.height = height
    hadEvents ||= Boolean(events)
    syncEvents()
  }

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.defaultPrevented || event.altKey || event.metaKey) return

    const target = event.target
    if (!(target instanceof Element) || !events || !workspace.contains(target)) return
    if (target.closest('input, textarea, select, [role="dialog"]')) return

    const rail = target.closest<HTMLElement>('[data-request-rail]')
    const scroller = rail && rail.scrollHeight > rail.clientHeight + 1 ? rail : page

    if (scroller === page && hasNativeScroller(target, workspace, events)) return

    const distances: Record<string, number> = {
      ArrowDown: 40,
      ArrowUp: -40,
      PageDown: scroller.clientHeight * 0.9,
      PageUp: -scroller.clientHeight * 0.9
    }
    const distance = distances[event.key]
    if (event.ctrlKey && event.key !== 'Home' && event.key !== 'End') return
    if (distance === undefined && event.key !== 'Home' && event.key !== 'End') return

    event.preventDefault()

    const top = event.key === 'Home' ? 0
      : event.key === 'End' ? scroller.scrollHeight - scroller.clientHeight
        : scroller.scrollTop + (distance ?? 0)
    scroller.scrollTo({ top, behavior: 'instant' })
  }

  const mutations = new MutationObserver(updateLayout)
  mutations.observe(workspace, { childList: true, subtree: true, characterData: true })
  resize.observe(workspace)

  scrollTarget.addEventListener('scroll', syncEvents, { passive: true })
  host.addEventListener('keydown', onKeyDown, true)
  window.addEventListener('resize', updateLayout)

  updateLayout()

  return () => {
    mutations.disconnect()
    resize.disconnect()
    requestRail?.removeAttribute('data-request-overflow')
    events?.removeEventListener('scroll', onEventScroll)
    scrollTarget.removeEventListener('scroll', syncEvents)
    host.removeEventListener('keydown', onKeyDown, true)
    window.removeEventListener('resize', updateLayout)
    range.style.height = ''
  }
}

export function useActivityScroll() {
  const cleanup = useRef<(() => void) | null>(null)
  return useCallback((root: HTMLElement | null) => {
    cleanup.current?.()
    const workspace = root?.closest<HTMLElement>('[data-activity-workspace]')
    cleanup.current = workspace ? bindActivityScroll(workspace) : null
  }, [])
}
