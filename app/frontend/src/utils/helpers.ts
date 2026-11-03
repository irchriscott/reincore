/** Inclusive UTC instants for local calendar dates, preserving subsecond events. */
export function presetToDateRange(preset: string, from: string, to: string, now = new Date()): [string | null, string | null] {
    const date = (value: string) => {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null
        const [y, m, d] = value.split('-').map(Number)
        const result = new Date(y, m - 1, d)
        return result.getFullYear() === y && result.getMonth() === m - 1 && result.getDate() === d ? result : null
    }
    const y = now.getFullYear(), m = now.getMonth(), d = now.getDate()
    let start: Date | null = null, end: Date | null = null
    if (preset === 'today') { 
        start = new Date(y, m, d); 
        end = new Date(y, m, d + 1) 
    } else if (preset === 'this_month') { 
        start = new Date(y, m, 1); 
        end = new Date(y, m, d + 1) 
    } else if (preset === 'last_month') { 
        start = new Date(y, m - 1, 1); 
        end = new Date(y, m, 1) 
    } else if (preset === 'custom') {
        start = date(from)
        const last = date(to)
        if (last) 
            end = new Date(last.getFullYear(), last.getMonth(), last.getDate() + 1)
    }
    // Rust stores nanoseconds; subtracting just 1ms would exclude the last 999,999ns.
    return [start?.toISOString() ?? null, end ? new Date(end.getTime() - 1000).toISOString().replace('.000Z', '.999999999Z') : null]
}
