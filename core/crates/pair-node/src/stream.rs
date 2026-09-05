//! Shared streaming plumbing for the two inference lanes.

use bytes::Bytes;
use futures::stream::Stream;

/// Wraps a byte stream into an axum body. Deliberately sets **no**
/// `Content-Length`: PAIR's proxies are stock `httputil.ReverseProxy`, whose
/// flush heuristic is "flush per write when the response is `text/event-stream`
/// **or** `Content-Length` is unset" (`docs/pair-contract.md` §3.1). A chunked
/// response therefore reaches the client token by token.
pub(crate) fn chunked_body<S>(stream: S) -> axum::body::Body
where
    S: Stream<Item = Result<Bytes, std::convert::Infallible>> + Send + 'static,
{
    axum::body::Body::from_stream(stream)
}
