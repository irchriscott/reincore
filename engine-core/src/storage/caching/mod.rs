//! Reuse: memoization, semantic cache, and prompt-cache support.
//!
//! Everything that avoids paying twice: content-addressed tool-call
//! memoization and its reuse policy, embedding-based semantic response
//! cache with local embeddings, cache-prefix normalization for
//! provider-side prompt caching, and predictive pre-warming.

pub mod cache;
pub mod embedding;
pub mod normalizer;
pub mod prewarm;
pub mod tool_cache;
