// Helpers for formatting session statuses for display.
//
// Session statuses come from the backend as snake_case or lowercase words
// (e.g. "active", "paused", "completed", "checkpointed"). Display them in
// sentence case in badges and labels for a more polished look.

/**
 * Capitalize the first letter of a status string, preserving the rest.
 * Handles empty strings and single-char statuses safely.
 *
 * Examples: "active" -> "Active", "completed" -> "Completed",
 * "in_progress" -> "In_progress".
 */
export function capitalizeStatus(status: string): string {
  if (!status) return status
  return status.charAt(0).toUpperCase() + status.slice(1)
}