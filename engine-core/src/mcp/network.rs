//! Public HTTPS egress for tenant-configured MCP servers. DNS results are checked
//! in the resolver used by the HTTP connection itself, preventing DNS rebinding.
use super::error;
use crate::EngineResult;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

pub fn public_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(a == 0
                || a == 10
                || a == 127
                || a >= 224
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && b == 168)
                || (a == 100 && (64..=127).contains(&b))
                || (a == 192 && b == 0 && (c == 0 || c == 2))
                || (a == 192 && b == 88 && c == 99)
                || (a == 198 && (b == 18 || b == 19))
                || (a == 198 && b == 51 && c == 100)
                || (a == 203 && b == 0 && c == 113))
        }
        IpAddr::V6(ip) => {
            let s = ip.segments();
            // Allow global unicast only; exclude special-purpose, transition,
            // documentation and IPv4-mapped address space.
            (s[0] & 0xe000) == 0x2000
                && !(s[0] == 0x2001 && (s[1] < 0x200 || s[1] == 0xdb8))
                && s[0] != 0x2002
                && !(s[0] == 0x3fff && s[1] < 0x1000)
        }
    }
}

pub fn validate_cloud_url(raw: &str) -> EngineResult<()> {
    let url = url::Url::parse(raw).map_err(|_| error("invalid MCP URL"))?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(error(
            "cloud MCP requires public HTTPS without URL credentials, query or fragment",
        ));
    }
    match url.host() {
        Some(url::Host::Ipv4(ip)) if public_address(ip.into()) => Ok(()),
        Some(url::Host::Ipv6(ip)) if public_address(ip.into()) => Ok(()),
        Some(url::Host::Domain(name))
            if name.contains('.')
                && !name.ends_with('.')
                && !["localhost", "local", "internal", "test", "invalid"]
                    .iter()
                    .any(|suffix| name == *suffix || name.ends_with(&format!(".{suffix}"))) =>
        {
            Ok(())
        }
        _ => Err(error("cloud MCP destination is not public")),
    }
}

/// Validates an OAuth endpoint the engine calls itself (token, registration or
/// metadata). Cloud hosts reach only public HTTPS; local hosts may also use
/// numeric loopback HTTP. Unlike MCP endpoints, a query string is allowed.
pub fn validate_endpoint_url(raw: &str, public_only: bool) -> EngineResult<url::Url> {
    let url = url::Url::parse(raw).map_err(|_| error("invalid authorization endpoint"))?;
    if raw.len() > 2048
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(error(
            "authorization endpoints cannot embed credentials or a fragment",
        ));
    }
    if public_only {
        let mut target = url.clone();
        target.set_query(None);
        validate_cloud_url(target.as_str())?;
    } else if !(url.scheme() == "https" || url.scheme() == "http" && loopback(&url)) {
        return Err(error(
            "authorization endpoints require HTTPS or numeric loopback HTTP",
        ));
    }
    Ok(url)
}

pub(super) fn loopback(url: &url::Url) -> bool {
    url.host_str()
        .is_some_and(|host| host == "127.0.0.1" || host == "[::1]")
}

/// HTTP client for upstream MCP and OAuth traffic: never follows redirects or
/// proxies, and resolves only public addresses when `public_only`.
pub(super) fn client(public_only: bool) -> EngineResult<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .connect_timeout(std::time::Duration::from_secs(10));
    if public_only {
        builder = builder.dns_resolver(resolver());
    }
    builder
        .build()
        .map_err(|_| error("MCP HTTP client initialization failed"))
}

pub(super) struct PublicResolver;

impl Resolve for PublicResolver {
    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(async move {
            let addresses: Vec<SocketAddr> =
                tokio::net::lookup_host((name.as_str(), 0)).await?.collect();
            if addresses.is_empty()
                || addresses
                    .iter()
                    .any(|address| !public_address(address.ip()))
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "MCP DNS destination denied",
                )
                .into());
            }
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

pub(super) fn resolver() -> Arc<PublicResolver> {
    Arc::new(PublicResolver)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn denies_private_special_and_encoded_destinations() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "192.168.1.1",
            "198.18.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "fe80::1",
            "2002:7f00:1::",
        ] {
            assert!(!public_address(ip.parse().unwrap()), "{ip}");
        }
        for url in [
            "https://2130706433/mcp",
            "https://127.1/mcp",
            "https://[::1]/mcp",
            "http://example.com/mcp",
            "https://x@public.example/mcp",
            "https://example.com/?secret=x",
            "https://metadata.google.internal/mcp",
        ] {
            assert!(validate_cloud_url(url).is_err(), "{url}");
        }
        assert!(public_address("8.8.8.8".parse().unwrap()));
        assert!(public_address("2606:4700:4700::1111".parse().unwrap()));
        assert!(validate_cloud_url("https://mcp.example.com/mcp").is_ok());
    }
    #[tokio::test]
    async fn resolver_rejects_loopback_dns_answers() {
        assert!(PublicResolver
            .resolve("localhost".parse().unwrap())
            .await
            .is_err());
    }
}
