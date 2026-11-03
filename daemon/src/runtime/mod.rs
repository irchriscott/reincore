//! Long-running runtime: daemon, app launching, desktop integration.
//!
//! The daemon lifecycle and HTTP serving, single-provider app launching
//! with launch-plan negotiation, the shared application state the desktop
//! app links against, and desktop notifications.

pub mod app;
pub mod daemon;
pub mod launch;
pub mod launch_routing;
pub mod notifications;
