//! Provider adapters, model records, and pricing data.
//!
//! Everything that talks to a model vendor or describes its models: the
//! provider trait and its adapters, the normalized chat types they
//! translate to and from, the versioned model catalog, and fixed-point
//! pricing schedules.

pub mod model_catalog;
pub mod model_library;
pub mod models;
pub mod pricing;
pub mod provider;
